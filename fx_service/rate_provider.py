"""
Rate Provider abstraction (GoF Strategy pattern).

Defines the interface for exchange rate data sources, enabling easy
substitution of providers (Bank of Canada, ECB, mock for testing, etc.)
without modifying client code.

GoF Patterns:
- Strategy: RateProvider defines the interface; concrete providers implement it
- Adapter: BankOfCanadaProvider adapts BankOfCanadaClient to the RateProvider interface
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List, Optional


class RateProvider(ABC):
    """
    Abstract interface for exchange rate data sources.

    GoF Strategy pattern — concrete implementations can be swapped
    without modifying the FxRateService that depends on this interface.

    All implementations must provide rate lookup by date and by date range,
    plus proper resource cleanup.
    """

    @abstractmethod
    def get_rate(self, series_name: str, lookup_date: date) -> Optional[Dict[str, Any]]:
        """
        Fetch a single rate observation for a specific date.

        Args:
            series_name: The rate series identifier (e.g., 'FXUSDCAD')
            lookup_date: The date to look up

        Returns:
            Observation dictionary if found, None otherwise
        """
        ...

    @abstractmethod
    def get_rates(
        self,
        series_name: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """
        Fetch rate observations for a date range.

        Args:
            series_name: The rate series identifier
            start_date: Start of the date range (inclusive)
            end_date: End of the date range (inclusive)

        Returns:
            List of observation dictionaries
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this provider."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BankOfCanadaProvider(RateProvider):
    """
    Concrete Strategy + Adapter: Bank of Canada Valet API.

    Adapts the BankOfCanadaClient (which knows HTTP/API details) to the
    RateProvider interface (which the service layer depends on).

    This dual role (Strategy + Adapter) is a common GoF composition:
    the class both implements the strategy interface AND adapts an
    existing class to conform to it.
    """

    def __init__(self, client=None):
        """
        Initialize with an optional pre-configured client.

        Args:
            client: Optional BankOfCanadaClient instance. Creates one if not provided.
        """
        from .bank_of_canada_client import BankOfCanadaClient
        self._client = client or BankOfCanadaClient()
        self._owns_client = client is None

    def get_rate(self, series_name: str, lookup_date: date) -> Optional[Dict[str, Any]]:
        """Delegate to BankOfCanadaClient for single-date lookup."""
        return self._client.get_rate_for_date(series_name, lookup_date)

    def get_rates(
        self,
        series_name: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Delegate to BankOfCanadaClient for date-range lookup."""
        return self._client.get_observations(series_name, start_date, end_date)

    def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            self._client.close()
