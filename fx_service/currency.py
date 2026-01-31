"""
Currency types and exchange result data structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import date


class Currency(Enum):
    """Supported currencies for exchange rate lookup."""
    USD = "USD"
    CAD = "CAD"


class ExchangeDirection(Enum):
    """Direction of currency exchange."""
    USD_TO_CAD = "FXUSDCAD"
    CAD_TO_USD = "FXCADUSD"
    
    @classmethod
    def from_currencies(cls, from_currency: Currency, to_currency: Currency) -> 'ExchangeDirection':
        """
        Determine exchange direction from source and target currencies.
        
        Args:
            from_currency: The source currency
            to_currency: The target currency
            
        Returns:
            The corresponding ExchangeDirection
            
        Raises:
            ValueError: If the currency pair is not supported
        """
        if from_currency == Currency.USD and to_currency == Currency.CAD:
            return cls.USD_TO_CAD
        elif from_currency == Currency.CAD and to_currency == Currency.USD:
            return cls.CAD_TO_USD
        else:
            raise ValueError(f"Unsupported currency pair: {from_currency.value} to {to_currency.value}")
    
    @property
    def series_name(self) -> str:
        """Get the Bank of Canada series name for this exchange direction."""
        return self.value
    
    @property
    def from_currency(self) -> Currency:
        """Get the source currency for this exchange direction."""
        if self == ExchangeDirection.USD_TO_CAD:
            return Currency.USD
        return Currency.CAD
    
    @property
    def to_currency(self) -> Currency:
        """Get the target currency for this exchange direction."""
        if self == ExchangeDirection.USD_TO_CAD:
            return Currency.CAD
        return Currency.USD


@dataclass
class ExchangeResult:
    """
    Result of an exchange rate lookup.
    
    Attributes:
        rate_date: The date for which the rate applies
        rate: The exchange rate value
        from_currency: The source currency
        to_currency: The target currency
        amount: Optional amount to convert
        converted_amount: Optional converted amount (rate * amount)
    """
    rate_date: date
    rate: float
    from_currency: Currency
    to_currency: Currency
    amount: Optional[float] = None
    converted_amount: Optional[float] = None
    
    def __post_init__(self):
        """Calculate converted amount if amount is provided."""
        if self.amount is not None and self.converted_amount is None:
            self.converted_amount = round(self.rate * self.amount, 4)
    
    def __str__(self) -> str:
        """Human-readable representation of the exchange result."""
        if self.amount is not None and self.converted_amount is not None:
            return (f"{self.amount} {self.from_currency.value} = "
                    f"{self.converted_amount} {self.to_currency.value} "
                    f"(rate: {self.rate} on {self.rate_date})")
        return (f"1 {self.from_currency.value} = {self.rate} {self.to_currency.value} "
                f"on {self.rate_date}")
