"""
Result formatting strategies (GoF Strategy pattern).

Different formatters can be used depending on the output target:
- LLMResultFormatter: concise, factual output for LLM consumption
- HumanResultFormatter: friendly, verbose output for direct display

New formatters (e.g., JSON for APIs, CSV for reports) can be added
without modifying existing code — Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from fx_service.currency import ExchangeResult


class ResultFormatter(ABC):
    """
    Abstract formatter for exchange results (GoF Strategy pattern).

    Subclasses define how an ExchangeResult is rendered to a string.
    The formatter is injected into tools and components that need
    to present results.
    """

    @abstractmethod
    def format(self, result: ExchangeResult) -> str:
        """
        Format an ExchangeResult into a string representation.

        Args:
            result: The exchange result to format

        Returns:
            Formatted string
        """
        ...


class LLMResultFormatter(ResultFormatter):
    """
    Formats results for LLM consumption.

    Concise and factual — gives the model exactly the data it needs
    to compose a natural language response.
    """

    def format(self, result: ExchangeResult) -> str:
        base = (
            f"Exchange rate on {result.rate_date}: "
            f"1 {result.from_currency.value} = {result.rate} {result.to_currency.value}"
        )
        if result.amount is not None and result.converted_amount is not None:
            return (
                f"{base}. "
                f"{result.amount} {result.from_currency.value} = "
                f"{result.converted_amount} {result.to_currency.value}"
            )
        return base


class HumanResultFormatter(ResultFormatter):
    """
    Formats results for direct human consumption.

    Friendly, verbose, with properly formatted dates and numbers.
    """

    def format(self, result: ExchangeResult) -> str:
        date_str = result.rate_date.strftime("%B %d, %Y")
        if result.amount is not None and result.converted_amount is not None:
            return (
                f"On {date_str}, the exchange rate was "
                f"1 {result.from_currency.value} = {result.rate} {result.to_currency.value}.\n"
                f"{result.amount:,.2f} {result.from_currency.value} = "
                f"{result.converted_amount:,.2f} {result.to_currency.value}"
            )
        return (
            f"On {date_str}, 1 {result.from_currency.value} was worth "
            f"{result.rate} {result.to_currency.value}."
        )
