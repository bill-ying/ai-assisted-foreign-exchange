"""
ComplianceFxAssistant — FxAssistant subclass with LangGraph compliance validation.

GoF Patterns:
- Template Method (inherited): BaseFxTool's validate→execute→format pipeline
  is reused unchanged by the graph nodes.
- Strategy (inherited + extended): ValidationStrategy is injected here and
  passed into ComplianceGraphBuilder. Swapping strategies requires no changes
  to the assistant or the graph.
- Builder: ComplianceGraphBuilder constructs the LangGraph StateGraph.
  ComplianceFxAssistant acts as the Director — it configures and triggers
  the builder, but has no knowledge of graph internals.
- Facade (inherited): The public surface remains chat() / clear_history() /
  close(). Clients are completely unaware of LangGraph running underneath.
- Open/Closed: FxAssistant is extended, not modified. All existing tests
  and clients of FxAssistant continue to work unchanged.

Design note on subclassing vs. composition:
  Subclassing is chosen here because ComplianceFxAssistant IS an FxAssistant
  (Liskov Substitution Principle holds — it can replace FxAssistant anywhere).
  It needs access to the protected LLM, tool registry, history, and event bus
  that are already present on the parent, and overrides only chat() to route
  through the compliance graph instead of the direct loop.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage

from ai.compliance import ComplianceGraphBuilder, LenientValidationStrategy, ValidationStrategy
from ai.compliance.state import ComplianceStatus
from ai.events import AssistantEvent, EventType
from ai.fx_assistant import FxAssistant, ChatResult
from fx_service.rate_provider import RateProvider
from ai.result_formatter import ResultFormatter
from ai.chat_history import ChatHistory

logger = logging.getLogger(__name__)


class ComplianceFxAssistant(FxAssistant):
    """
    FX assistant with LangGraph-powered compliance validation (GoF Facade).

    Extends FxAssistant by routing every chat() call through a LangGraph
    StateGraph that validates the LLM response against the raw tool data
    before delivering it to the user. If validation fails, the graph
    automatically attempts LLM correction up to max_corrections times.

    The public API is identical to FxAssistant — this is a drop-in upgrade.

    Usage:
        with ComplianceFxAssistant.create() as assistant:
            response = assistant.chat("What was USD/CAD on 2024-01-15?")
            # response is guaranteed to have passed compliance checks
    """

    def __init__(
        self,
        llm,
        tool_registry,
        chat_history,
        event_bus,
        fx_service,
        compliance_graph,
        owns_service: bool = False,
    ) -> None:
        """
        Initialize with all FxAssistant dependencies plus the compiled graph.

        Prefer using the create() factory method for typical usage.

        Args:
            llm:              LangChain chat model with tools bound
            tool_registry:    Registry of available tools
            chat_history:     Chat history storage strategy
            event_bus:        Observer bus for audit events
            fx_service:       FX rate service instance
            compliance_graph: Compiled LangGraph StateGraph
            owns_service:     Whether this assistant owns the fx_service lifecycle
        """
        super().__init__(
            llm=llm,
            tool_registry=tool_registry,
            chat_history=chat_history,
            event_bus=event_bus,
            fx_service=fx_service,
            owns_service=owns_service,
        )
        self._compliance_graph = compliance_graph

    @classmethod
    def create(
        cls,
        model_name: str = "google/gemma-4-31b-it:free",
        temperature: float = 0.1,
        rate_provider: Optional[RateProvider] = None,
        formatter: Optional[ResultFormatter] = None,
        chat_history: Optional[ChatHistory] = None,
        enable_audit: bool = True,
        validation_strategy: Optional[ValidationStrategy] = None,
        max_corrections: int = 2,
    ) -> 'ComplianceFxAssistant':
        """
        Factory Method: create a fully configured ComplianceFxAssistant.

        Delegates base component assembly to the parent create() logic,
        then builds the compliance graph on top using ComplianceGraphBuilder.

        Args:
            model_name:          OpenRouter model name (default: google/gemma-4-31b-it:free)
            temperature:         LLM temperature (0 = deterministic)
            rate_provider:       Rate data source strategy
            formatter:           Result formatting strategy
            chat_history:        History storage strategy
            enable_audit:        Whether to attach the AuditLogger observer
            validation_strategy: Compliance validation strategy
                                 (default: LenientValidationStrategy)
            max_corrections:     Max correction loop iterations (default: 2)

        Returns:
            Fully configured ComplianceFxAssistant instance
        """
        # Build the base assistant (reuse all parent wiring)
        base: FxAssistant = FxAssistant.create(
            model_name=model_name,
            temperature=temperature,
            rate_provider=rate_provider,
            formatter=formatter,
            chat_history=chat_history,
            enable_audit=enable_audit,
        )

        # Build the compliance graph via the Builder, injecting the fallback-aware
        # invoke callable so the graph gets automatic 429 retries.
        strategy = validation_strategy or LenientValidationStrategy()

        # We need a temporary instance to bind _invoke_with_fallback, so build it
        # first, then pass its method into the graph builder.
        instance = cls(
            llm=base._llm,
            tool_registry=base._tool_registry,
            chat_history=base._history,
            event_bus=base._event_bus,
            fx_service=base._fx_service,
            compliance_graph=None,  # temporary placeholder
            owns_service=True,
        )
        # Restore the selected model (super().__init__ resets to MODELS[0])
        instance._current_model = model_name or cls.MODELS[0]
        instance._api_key = base._api_key
        instance._temperature = base._temperature

        compliance_graph = (
            ComplianceGraphBuilder(
                llm_invoke=instance._invoke_with_fallback,
                tool_registry=base._tool_registry,
                event_bus=base._event_bus,
                system_prompt=cls.SYSTEM_PROMPT,
            )
            .with_validation_strategy(strategy)
            .with_max_corrections(max_corrections)
            .build()
        )
        instance._compliance_graph = compliance_graph
        return instance

    def chat(self, user_message: str) -> ChatResult:
        """
        Process a user message through the compliance validation graph.

        Overrides FxAssistant.chat() to route through the LangGraph StateGraph.
        The graph executes: invoke_llm → [execute_tools →] invoke_llm_final
                           → validate → [correct →]* emit

        The conversation history is updated after the graph completes,
        consistent with how FxAssistant manages it.

        Args:
            user_message: Natural language query from the user

        Returns:
            ChatResult dict with keys ``answer`` (compliance-validated response)
            and ``model`` (model that produced the final answer)
        """
        self._event_bus.publish(
            AssistantEvent(EventType.QUERY_RECEIVED, data={"query": user_message})
        )

        try:
            self._history.add_message(HumanMessage(content=user_message))

            initial_state = {
                "messages":           self._build_messages(),
                "user_message":       user_message,
                "tool_results":       {},
                "tool_args":          {},
                "llm_response":       "",
                "validation_result":  None,
                "correction_attempts": 0,
                "final_response":     "",
                "compliance_passed":  False,
            }

            final_state = self._compliance_graph.invoke(initial_state)

            response = final_state["final_response"]
            passed = final_state.get("compliance_passed", False)

            if not passed:
                logger.warning(
                    "Response delivered with compliance disclaimer after "
                    "%d correction attempt(s).",
                    final_state.get("correction_attempts", 0),
                )

            return ChatResult(answer=response, model=self._current_model)

        except Exception as e:
            logger.error("Error processing message through compliance graph: %s", e)
            self._event_bus.publish(
                AssistantEvent(EventType.ERROR_OCCURRED, data={"error": str(e)})
            )
            return ChatResult(
                answer=f"I encountered an error processing your request: {str(e)}",
                model=self._current_model,
            )

    def get_graph_mermaid(self) -> str:
        """
        Returns the Mermaid representation of the underlying LangGraph.
        
        Useful for generating visualizations of the compliance pipeline for documentation
        or debugging purposes.
        """
        return self._compliance_graph.get_graph().draw_mermaid()
