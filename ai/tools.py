"""
Tool definitions and registry (GoF Template Method + Registry patterns).

GoF Patterns:
- Template Method: BaseFxTool defines the algorithm skeleton (validate → execute → format);
  concrete tools override the individual steps.
- Registry: ToolRegistry provides centralized tool management and LangChain integration.

The tools bridge between LangChain's @tool decorator (which the LLM sees)
and our domain logic (FxRateService), keeping concerns cleanly separated.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from langchain_core.tools import tool

from fx_service import FxRateService, ExchangeResult
from ai.result_formatter import ResultFormatter, LLMResultFormatter

logger = logging.getLogger(__name__)


class BaseFxTool(ABC):
    """
    Abstract base for FX tools (GoF Template Method pattern).

    Defines the algorithm skeleton for tool execution:
      1. validate — check input arguments
      2. execute — perform the business logic
      3. format  — render the result as a string

    Subclasses override these steps to implement specific tool behavior.
    The run() method is final — it defines the invariant algorithm.
    """

    def __init__(self, formatter: Optional[ResultFormatter] = None):
        """
        Args:
            formatter: Strategy for formatting results. Defaults to LLMResultFormatter.
        """
        self._formatter = formatter or LLMResultFormatter()

    def run(self, **kwargs) -> str:
        """
        Template method: validate → execute → format.

        This method defines the invariant algorithm. Do not override.
        Override _validate, _execute, and _format instead.
        """
        try:
            self._validate(kwargs)
            result = self._execute(kwargs)
            return self._format_result(result)
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing tool: {str(e)}"

    @abstractmethod
    def _validate(self, args: Dict[str, Any]) -> None:
        """
        Validate input arguments. Raise ValueError if invalid.

        Args:
            args: Dictionary of tool arguments
        """
        ...

    @abstractmethod
    def _execute(self, args: Dict[str, Any]) -> Any:
        """
        Execute the core tool logic.

        Args:
            args: Validated dictionary of tool arguments

        Returns:
            The raw result (type depends on concrete tool)
        """
        ...

    @abstractmethod
    def _format_result(self, result: Any) -> str:
        """
        Format the result into a string for the LLM or user.

        Args:
            result: The raw result from _execute

        Returns:
            Formatted string
        """
        ...


class FxRateTool(BaseFxTool):
    """
    Concrete Template Method implementation for FX rate lookups.

    Validates currency/date inputs, delegates to FxRateService,
    and formats the result using the injected ResultFormatter strategy.
    """

    REQUIRED_ARGS = ('from_currency', 'to_currency', 'date')

    def __init__(
        self,
        fx_service: FxRateService,
        formatter: Optional[ResultFormatter] = None
    ):
        super().__init__(formatter)
        self._fx_service = fx_service

    def _validate(self, args: Dict[str, Any]) -> None:
        """Ensure required arguments are present."""
        for key in self.REQUIRED_ARGS:
            if key not in args:
                raise ValueError(f"Missing required argument: {key}")

    def _execute(self, args: Dict[str, Any]) -> Optional[ExchangeResult]:
        """Fetch the rate from the FxRateService."""
        return self._fx_service.get_rate_for_date(
            from_currency=args['from_currency'].upper(),
            to_currency=args['to_currency'].upper(),
            lookup_date=args['date'],
            amount=args.get('amount')
        )

    def _format_result(self, result: Optional[ExchangeResult]) -> str:
        """Format using the injected ResultFormatter strategy."""
        if result is None:
            return (
                "No exchange rate data available for that date. "
                "This might be a weekend, holiday, or a date outside "
                "the available data range."
            )
        return self._formatter.format(result)


def create_langchain_tool(fx_rate_tool: FxRateTool):
    """
    Factory function: creates a LangChain-compatible @tool that delegates
    to our FxRateTool (Template Method).

    This bridges LangChain's tool-calling mechanism with our domain logic.
    The LLM sees the @tool's docstring and schema; execution flows through
    the Template Method pipeline (validate → execute → format).
    """

    @tool
    def get_fx_rate(
        from_currency: str,
        to_currency: str,
        date: str,
        amount: Optional[float] = None
    ) -> str:
        """Get the exchange rate between USD and CAD for a specific date from Bank of Canada.

        Use this tool whenever the user asks about exchange rates, currency conversion,
        or how much one currency is worth in another.

        Args:
            from_currency: The source currency to convert from (USD or CAD)
            to_currency: The target currency to convert to (USD or CAD)
            date: The date for the exchange rate in YYYY-MM-DD format
            amount: Optional amount to convert. If not specified, returns the rate for 1 unit

        Returns:
            Exchange rate information as a formatted string
        """
        kwargs = {
            'from_currency': from_currency,
            'to_currency': to_currency,
            'date': date,
        }
        if amount is not None:
            kwargs['amount'] = amount
        return fx_rate_tool.run(**kwargs)

    return get_fx_rate


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides centralized tool management with both domain-level access
    (BaseFxTool instances) and LangChain-level access (for bind_tools).
    """

    def __init__(self):
        self._tools: Dict[str, BaseFxTool] = {}
        self._langchain_tools = []

    def register(self, name: str, fx_tool: BaseFxTool, langchain_tool) -> None:
        """
        Register a tool with its LangChain wrapper.

        Args:
            name: Unique tool identifier
            fx_tool: The domain-level tool instance
            langchain_tool: The LangChain @tool wrapper
        """
        self._tools[name] = fx_tool
        self._langchain_tools.append(langchain_tool)

    def get_tool(self, name: str) -> Optional[BaseFxTool]:
        """Look up a domain tool by name."""
        return self._tools.get(name)

    @property
    def langchain_tools(self) -> list:
        """All registered LangChain tool wrappers (for bind_tools)."""
        return list(self._langchain_tools)

    @property
    def tool_names(self) -> List[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())
