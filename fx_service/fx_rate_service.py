"""
FX Rate Service - Main orchestrator for exchange rate lookups.

Depends on the RateProvider abstraction (GoF Strategy pattern),
not on any concrete API client. This enables easy testing with
mock providers and future expansion to additional data sources.
"""

from datetime import date, datetime
from typing import List, Optional, Union
import logging

from .rate_provider import RateProvider
from .mcp_provider import McpProvider
from .currency import Currency, ExchangeDirection, ExchangeResult

logger = logging.getLogger(__name__)


class FxRateService:
    """
    Service for looking up USD/CAD exchange rates.

    This is the main entry point for all exchange rate operations.
    It orchestrates between the RateProvider (data source) and the
    domain model (Currency, ExchangeDirection, ExchangeResult).

    The provider is injected via constructor (Dependency Injection),
    following the GoF Strategy pattern — any RateProvider implementation
    can be substituted without modifying this class.
    """

    def __init__(self, provider: Optional[RateProvider] = None):
        """
        Initialize the FX Rate Service.

        Args:
            provider: Rate data source (Strategy pattern).
                     Defaults to McpProvider if not provided.
        """
        self._provider = provider or McpProvider()
        self._owns_provider = provider is None

    def get_rate_for_date(
        self,
        from_currency: Union[str, Currency],
        to_currency: Union[str, Currency],
        lookup_date: Union[str, date],
        amount: Optional[float] = None
    ) -> Optional[ExchangeResult]:
        """
        Get the exchange rate for a specific date.

        Args:
            from_currency: Source currency ('USD' or 'CAD')
            to_currency: Target currency ('USD' or 'CAD')
            lookup_date: Date to look up (string in YYYY-MM-DD format or date object)
            amount: Optional amount to convert

        Returns:
            ExchangeResult if rate found, None otherwise

        Raises:
            ValueError: If currency pair is not supported
        """
        # Normalize inputs
        from_curr = self._normalize_currency(from_currency)
        to_curr = self._normalize_currency(to_currency)
        lookup_dt = self._normalize_date(lookup_date)

        # Get exchange direction and series name
        direction = ExchangeDirection.from_currencies(from_curr, to_curr)

        logger.info(f"Looking up {direction.name} rate for {lookup_dt}")

        # Fetch from provider (Strategy pattern — provider is interchangeable)
        observation = self._provider.get_rate(
            direction.series_name,
            lookup_dt
        )

        if not observation:
            logger.warning(f"No rate found for {lookup_dt}")
            return None

        # Extract rate value
        rate = self._extract_rate(observation, direction.series_name)
        if rate is None:
            return None

        return ExchangeResult(
            rate_date=lookup_dt,
            rate=rate,
            from_currency=from_curr,
            to_currency=to_curr,
            amount=amount
        )

    def get_monthly_rates(
        self,
        from_currency: Union[str, Currency],
        to_currency: Union[str, Currency],
        year: int
    ) -> List[ExchangeResult]:
        """
        Get month-end exchange rates for each month of a given year.

        Args:
            from_currency: Source currency
            to_currency: Target currency
            year: The year to fetch rates for

        Returns:
            List of ExchangeResult for each month's last available rate
        """
        from_curr = self._normalize_currency(from_currency)
        to_curr = self._normalize_currency(to_currency)
        direction = ExchangeDirection.from_currencies(from_curr, to_curr)

        logger.info(f"Looking up monthly {direction.name} rates for {year}")

        observations = self._provider.get_rates(
            direction.series_name,
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31)
        )

        if not observations:
            logger.warning(f"No rates found for year {year}")
            return []

        results = []
        for month in range(1, 13):
            month_obs = [
                obs for obs in observations
                if obs['d'].startswith(f"{year}-{month:02d}")
            ]

            if month_obs:
                last_obs = month_obs[-1]
                rate = self._extract_rate(last_obs, direction.series_name)
                if rate is not None:
                    rate_date = datetime.strptime(last_obs['d'], '%Y-%m-%d').date()
                    results.append(ExchangeResult(
                        rate_date=rate_date,
                        rate=rate,
                        from_currency=from_curr,
                        to_currency=to_curr
                    ))

        return results

    def _normalize_currency(self, currency: Union[str, Currency]) -> Currency:
        """Convert string to Currency enum if needed."""
        if isinstance(currency, Currency):
            return currency
        return Currency(currency.upper())

    def _normalize_date(self, lookup_date: Union[str, date]) -> date:
        """Convert string to date if needed."""
        if isinstance(lookup_date, date):
            return lookup_date
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%B %d, %Y', '%b %d, %Y']:
            try:
                return datetime.strptime(lookup_date, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date: {lookup_date}")

    def _extract_rate(self, observation: dict, series_name: str) -> Optional[float]:
        """Extract the rate value from an observation dictionary."""
        rate_data = observation.get(series_name)
        if rate_data and 'v' in rate_data:
            return float(rate_data['v'])
        return None

    def close(self):
        """Close the underlying provider if owned by this service."""
        if self._owns_provider:
            self._provider.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
