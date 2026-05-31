"""
FxAssistant — Facade for AI-powered FX rate queries.

GoF Patterns:
- Facade: FxAssistant provides a simplified interface that hides the complexity
  of LLM providers, tool registries, event buses, and chat history.
- Factory Method: create() builds a fully configured assistant with sensible defaults.
- Strategy: LLM, formatter, history, and rate provider are all injected abstractions.
- Observer: EventBus publishes lifecycle events to subscribed observers.

This is the single entry point for the AI layer. Clients (main.py) interact
only with chat() and clear_history() — all internal wiring is hidden.
"""

import logging
import os
from typing import Optional, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)

from fx_service import FxRateService
from fx_service.rate_provider import RateProvider
from fx_service.mcp_provider import McpProvider
from ai.events import EventBus, EventType, AssistantEvent, AuditLogger
from ai.result_formatter import ResultFormatter, LLMResultFormatter
from ai.chat_history import ChatHistory, InMemoryChatHistory
from ai.tools import ToolRegistry, FxRateTool, create_langchain_tool

logger = logging.getLogger(__name__)


class FxAssistant:
    """
    AI-powered assistant for USD/CAD exchange rate queries (GoF Facade).

    Composes multiple subsystems behind a simple interface:
    - LLM (via LangChain ChatOpenAI pointed at OpenRouter, swappable via Strategy)
    - ToolRegistry (manages FX tools and LangChain bindings)
    - ChatHistory (Strategy for conversation storage)
    - EventBus (Observer for audit and monitoring)
    - ResultFormatter (Strategy for output formatting)

    Usage:
        with FxAssistant.create() as assistant:
            response = assistant.chat("What was USD/CAD on 2024-01-15?")
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant. Answer general knowledge questions directly.
For USD/CAD exchange rates, ALWAYS use the get_fx_rate tool — do NOT attempt to determine whether a date is valid, in the future, a weekend, or a holiday before calling the tool. Call the tool for every date the user provides; the server will determine if a rate is available. If the tool returns no rate, inform the user that no rate is available for that date."""

    def __init__(
        self,
        llm,
        tool_registry: ToolRegistry,
        chat_history: ChatHistory,
        event_bus: EventBus,
        fx_service: FxRateService,
        owns_service: bool = False,
    ):
        """
        Initialize with fully configured dependencies (Dependency Injection).

        Prefer using the create() factory method for typical usage.

        Args:
            llm: LangChain chat model with tools bound
            tool_registry: Registry of available tools
            chat_history: Chat history storage
            event_bus: Event dispatcher for observers
            fx_service: FX rate service instance
            owns_service: Whether this assistant owns (and should close) the service
        """
        self._llm = llm
        self._tool_registry = tool_registry
        self._history = chat_history
        self._event_bus = event_bus
        self._fx_service = fx_service
        self._owns_service = owns_service

    @classmethod
    def create(
        cls,
        model_name: str = "google/gemma-4-31b-it:free",
        temperature: float = 0.1,
        rate_provider: Optional[RateProvider] = None,
        formatter: Optional[ResultFormatter] = None,
        chat_history: Optional[ChatHistory] = None,
        enable_audit: bool = True,
    ) -> 'FxAssistant':
        """
        Factory Method: create a fully configured FxAssistant.

        Assembles all components with sensible defaults. This is the
        recommended way to create an assistant.

        Args:
            model_name: OpenRouter model name (default: google/gemma-4-31b-it:free)
            temperature: LLM temperature (0 = deterministic)
            rate_provider: Rate data source strategy (default: Bank of Canada)
            formatter: Result formatting strategy (default: LLM formatter)
            chat_history: History storage strategy (default: in-memory)
            enable_audit: Whether to attach the AuditLogger observer

        Returns:
            Fully configured FxAssistant instance
        """
        # Strategy: rate provider (default: MCP server)
        provider = rate_provider or McpProvider()
        fx_service = FxRateService(provider=provider)

        # Strategy: result formatter
        fmt = formatter or LLMResultFormatter()

        # Template Method: create domain tool + LangChain bridge
        fx_rate_tool = FxRateTool(fx_service, formatter=fmt)
        lc_tool = create_langchain_tool(fx_rate_tool)

        # Registry
        registry = ToolRegistry()
        registry.register("get_fx_rate", fx_rate_tool, lc_tool)

        # LLM with tools bound (LangChain native tool calling) via OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Get one at https://openrouter.ai/keys"
            )
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
        llm_with_tools = llm.bind_tools(registry.langchain_tools)

        # Strategy: chat history
        history = chat_history or InMemoryChatHistory()

        # Observer: event bus
        event_bus = EventBus()
        if enable_audit:
            event_bus.subscribe(AuditLogger())

        return cls(
            llm=llm_with_tools,
            tool_registry=registry,
            chat_history=history,
            event_bus=event_bus,
            fx_service=fx_service,
            owns_service=True,
        )

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.

        Orchestration flow:
        1. Publish QUERY_RECEIVED event
        2. Add user message to history
        3. Invoke LLM (with tools bound)
        4. If LLM requests tool calls → execute tools → re-invoke LLM
        5. Publish RESPONSE_GENERATED event
        6. Return final response text

        Args:
            user_message: Natural language query from the user

        Returns:
            The assistant's response as a string
        """
        self._event_bus.publish(
            AssistantEvent(EventType.QUERY_RECEIVED, data={"query": user_message})
        )

        try:
            self._history.add_message(HumanMessage(content=user_message))

            messages = self._build_messages()
            response = self._llm.invoke(messages)

            # Handle tool calls (LangChain native — no regex needed)
            if response.tool_calls:
                response_text = self._handle_tool_calls(response, messages)
            else:
                self._history.add_message(AIMessage(content=response.content))
                response_text = response.content

            self._event_bus.publish(
                AssistantEvent(
                    EventType.RESPONSE_GENERATED,
                    data={"response": response_text}
                )
            )
            return response_text

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._event_bus.publish(
                AssistantEvent(EventType.ERROR_OCCURRED, data={"error": str(e)})
            )
            return f"I encountered an error processing your request: {str(e)}"

    def _handle_tool_calls(self, ai_response, messages: list) -> str:
        """
        Execute tool calls from the LLM response and get the final answer.

        Uses LangChain's native tool-calling protocol:
        1. AI response contains structured tool_calls
        2. Execute each tool
        3. Add ToolMessages with results
        4. Re-invoke LLM for final natural language response

        Args:
            ai_response: The AIMessage containing tool_calls
            messages: Current message list for re-invocation

        Returns:
            Final response text after tool execution
        """
        self._history.add_message(ai_response)
        messages.append(ai_response)

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            self._event_bus.publish(
                AssistantEvent(
                    EventType.TOOL_CALLED,
                    data={"tool_name": tool_name, "args": tool_args}
                )
            )

            # Execute through the registry's domain tool
            domain_tool = self._tool_registry.get_tool(tool_name)
            if domain_tool:
                result = domain_tool.run(**tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            self._event_bus.publish(
                AssistantEvent(
                    EventType.TOOL_RESULT,
                    data={"tool_name": tool_name, "result": result}
                )
            )

            tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
            self._history.add_message(tool_msg)
            messages.append(tool_msg)

        # Re-invoke LLM with tool results for final response
        final_response = self._llm.invoke(messages)
        self._history.add_message(AIMessage(content=final_response.content))
        return final_response.content

    def _build_messages(self) -> list:
        """Build the full message list: system prompt + chat history."""
        return [SystemMessage(content=self.SYSTEM_PROMPT)] + self._history.get_messages()

    def clear_history(self) -> None:
        """Clear conversation history and notify observers."""
        self._history.clear()
        self._event_bus.publish(AssistantEvent(EventType.HISTORY_CLEARED))

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus to subscribe/unsubscribe observers."""
        return self._event_bus

    def close(self) -> None:
        """Release resources owned by this assistant."""
        if self._owns_service:
            self._fx_service.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
