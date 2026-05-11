"""
LangGraph compliance validation graph (GoF Builder pattern).

GoF Patterns:
- Builder: ComplianceGraphBuilder constructs the StateGraph step by step.
  The director (caller) calls build() to get a compiled, ready-to-run graph.
  This separates graph topology construction from the graph's runtime logic.
- Strategy: the ValidationStrategy is injected, keeping node logic decoupled
  from the specific rules and severity interpretation used.
- Observer: graph nodes publish to the EventBus so all compliance events
  appear in the audit log alongside normal assistant events.

Graph topology:
                      ┌─────────────┐
                      │  invoke_llm │  (first LLM call)
                      └──────┬──────┘
                             │
              ┌──────────────┴──────────────┐
        tool_calls?                    no tool calls
              │                             │
    ┌─────────▼──────────┐                  │
    │   execute_tools    │                  │
    └─────────┬──────────┘                  │
              │                             │
    ┌─────────▼──────────┐                  │
    │  invoke_llm_final  │◄─────────────────┘
    └─────────┬──────────┘  (also target of correction loop)
              │
    ┌─────────▼──────────┐
    │      validate      │
    └─────────┬──────────┘
              │
     ┌────────┴────────┐
   passed?          failed?
     │                 │
  ┌──▼──┐      ┌───────▼──────┐
  │emit │      │    correct   │  (inject correction prompt)
  └──┬──┘      └───────┬──────┘
     │                 │
     │     max retries exceeded?
     │         ├─ yes → emit (with compliance warning prepended)
     │         └─ no  → invoke_llm_final  (retry loop)
   [END]
"""

import logging
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from ai.events import AssistantEvent, EventBus, EventType
from ai.tools import ToolRegistry
from .state import ComplianceState, ComplianceStatus, ValidationResult
from .validator import LenientValidationStrategy, ValidationStrategy

logger = logging.getLogger(__name__)

_MAX_CORRECTIONS_DEFAULT = 2

# Routing literals used in conditional edges
_ROUTE_TOOLS = "tools"
_ROUTE_DONE = "done"
_ROUTE_PASS = "pass"
_ROUTE_FAIL = "fail"
_ROUTE_MAX_RETRIES = "max_retries"


class ComplianceGraphBuilder:
    """
    Builder for the LangGraph compliance validation graph (GoF Builder).

    Constructs the StateGraph topology and wires all node functions.
    Inject dependencies once at construction time; call build() to get
    a compiled, executable graph.

    Usage:
        graph = (
            ComplianceGraphBuilder(llm, tool_registry, event_bus)
            .with_validation_strategy(StrictValidationStrategy())
            .with_max_corrections(3)
            .build()
        )
        result = graph.invoke(initial_state)
    """

    def __init__(
        self,
        llm,
        tool_registry: ToolRegistry,
        event_bus: EventBus,
        system_prompt: str,
    ) -> None:
        """
        Args:
            llm:           LangChain chat model with tools already bound
            tool_registry: Registry of available domain tools
            event_bus:     Observer bus for audit event publishing
            system_prompt: System prompt injected at the start of every invocation
        """
        self._llm = llm
        self._tool_registry = tool_registry
        self._event_bus = event_bus
        self._system_prompt = system_prompt
        self._validator: ValidationStrategy = LenientValidationStrategy()
        self._max_corrections: int = _MAX_CORRECTIONS_DEFAULT

    # ── Fluent builder methods ──────────────────────────────────────────────

    def with_validation_strategy(self, strategy: ValidationStrategy) -> 'ComplianceGraphBuilder':
        """Override the default LenientValidationStrategy."""
        self._validator = strategy
        return self

    def with_max_corrections(self, max_corrections: int) -> 'ComplianceGraphBuilder':
        """Override the default maximum correction attempts (default: 2)."""
        self._max_corrections = max_corrections
        return self

    # ── Public build ────────────────────────────────────────────────────────

    def build(self):
        """
        Construct and compile the compliance StateGraph.

        Returns:
            A compiled LangGraph runnable ready to be invoked with a
            ComplianceState dict.
        """
        graph = StateGraph(ComplianceState)

        graph.add_node("invoke_llm",       self._node_invoke_llm)
        graph.add_node("execute_tools",    self._node_execute_tools)
        graph.add_node("invoke_llm_final", self._node_invoke_llm_final)
        graph.add_node("validate",         self._node_validate)
        graph.add_node("correct",          self._node_correct)
        graph.add_node("emit",             self._node_emit)

        graph.set_entry_point("invoke_llm")

        graph.add_conditional_edges(
            "invoke_llm",
            self._route_after_first_llm,
            {_ROUTE_TOOLS: "execute_tools", _ROUTE_DONE: "invoke_llm_final"},
        )
        graph.add_edge("execute_tools",    "invoke_llm_final")
        graph.add_edge("invoke_llm_final", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                _ROUTE_PASS:        "emit",
                _ROUTE_FAIL:        "correct",
                _ROUTE_MAX_RETRIES: "emit",
            },
        )
        graph.add_edge("correct", "invoke_llm_final")
        graph.add_edge("emit", END)

        return graph.compile()

    # ── Node implementations ─────────────────────────────────────────────────

    def _node_invoke_llm(self, state: ComplianceState) -> Dict[str, Any]:
        """
        First LLM invocation.

        Builds the full message list (system prompt + history) and invokes
        the LLM. Stores the raw AIMessage in `messages` for downstream nodes.
        If the LLM requests tool calls they will be handled by execute_tools;
        otherwise the graph routes directly to invoke_llm_final.
        """
        messages = [SystemMessage(content=self._system_prompt)] + state["messages"]
        response = self._llm.invoke(messages)
        return {
            "messages":    [response],
            "llm_response": response.content,
        }

    def _node_execute_tools(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Execute all tool calls requested by the LLM.

        Iterates tool_calls on the last AIMessage, dispatches each through
        the ToolRegistry, and appends ToolMessages to the state. Accumulates
        raw tool args and formatted results for the validator.
        """
        last_ai_message: AIMessage = state["messages"][-1]

        tool_results: Dict[str, str] = {}
        tool_args: Dict[str, Any] = {}
        tool_messages = []

        for tool_call in last_ai_message.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]

            self._event_bus.publish(AssistantEvent(
                EventType.TOOL_CALLED,
                data={"tool_name": tool_name, "args": args},
            ))

            domain_tool = self._tool_registry.get_tool(tool_name)
            result = domain_tool.run(**args) if domain_tool else f"Unknown tool: {tool_name}"

            self._event_bus.publish(AssistantEvent(
                EventType.TOOL_RESULT,
                data={"tool_name": tool_name, "result": result},
            ))

            tool_results[tool_name] = result
            tool_args[tool_name] = args
            tool_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

        return {
            "messages":    tool_messages,
            "tool_results": tool_results,
            "tool_args":    tool_args,
        }

    def _node_invoke_llm_final(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Final LLM invocation — produces the natural language response.

        Called after tool execution (to synthesise tool results into prose)
        and after each correction attempt (to incorporate correction guidance).
        The full accumulated message list is passed to preserve context.
        """
        messages = [SystemMessage(content=self._system_prompt)] + state["messages"]
        response = self._llm.invoke(messages)
        return {
            "messages":     [response],
            "llm_response":  response.content,
        }

    def _node_validate(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Run the compliance validation strategy against the latest LLM response.

        Compares the response text against the raw tool results and args
        accumulated during execute_tools. Stores the ValidationResult in
        state for routing and audit logging.
        """
        result: ValidationResult = self._validator.validate(
            response=state["llm_response"],
            tool_results=state.get("tool_results", {}),
            tool_args=state.get("tool_args", {}),
        )

        log_level = logging.WARNING if not result.passed else logging.INFO
        logger.log(
            log_level,
            "Compliance validation: %s | violations: %d",
            result.status.name,
            len(result.violations),
        )
        for v in result.violations:
            logger.log(log_level, "  %s", v)

        return {"validation_result": result}

    def _node_correct(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Inject a correction prompt into the message history.

        Appends a HumanMessage containing the compliance failure details
        so that the next invoke_llm_final call has the correction context.
        Increments the correction_attempts counter for loop termination.
        """
        validation_result: ValidationResult = state["validation_result"]
        correction_prompt = validation_result.to_correction_prompt()

        logger.warning(
            "Compliance correction attempt %d/%d",
            state["correction_attempts"] + 1,
            self._max_corrections,
        )

        return {
            "messages":           [HumanMessage(content=correction_prompt)],
            "correction_attempts": state["correction_attempts"] + 1,
        }

    def _node_emit(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Produce the final response for the caller.

        If the response passed compliance, emit it as-is. If max correction
        retries were exceeded, prepend a compliance disclaimer so the user
        is aware the response could not be fully verified.
        """
        validation_result: ValidationResult = state.get("validation_result")
        response = state["llm_response"]
        passed = True

        if (
            validation_result is not None
            and not validation_result.passed
            and state["correction_attempts"] >= self._max_corrections
        ):
            disclaimer = (
                "\n\n⚠️ *Compliance notice: this response could not be fully "
                "verified against the source data after "
                f"{self._max_corrections} correction attempt(s). "
                "Please verify the rate independently via bankofcanada.ca.*"
            )
            response = response + disclaimer
            passed = False
            logger.error(
                "Compliance max retries exceeded. Emitting with disclaimer."
            )

        self._event_bus.publish(AssistantEvent(
            EventType.RESPONSE_GENERATED,
            data={"response": response, "compliance_passed": passed},
        ))

        return {
            "final_response":  response,
            "compliance_passed": passed,
        }

    # ── Routing functions ────────────────────────────────────────────────────

    def _route_after_first_llm(
        self, state: ComplianceState
    ) -> Literal["tools", "done"]:
        """Route to tool execution if the LLM made tool calls, else skip ahead."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return _ROUTE_TOOLS
        return _ROUTE_DONE

    def _route_after_validate(
        self, state: ComplianceState
    ) -> Literal["pass", "fail", "max_retries"]:
        """Route based on validation result and remaining correction budget."""
        result: ValidationResult = state.get("validation_result")
        if result is None or result.passed:
            return _ROUTE_PASS
        if state["correction_attempts"] >= self._max_corrections:
            return _ROUTE_MAX_RETRIES
        return _ROUTE_FAIL
