"""
FX Service Package
Provides exchange rate lookup functionality using Bank of Canada API.
"""

from .currency import Currency, ExchangeDirection, ExchangeResult
from .bank_of_canada_client import BankOfCanadaClient
from .fx_rate_service import FxRateService

__all__ = [
    'Currency',
    'ExchangeDirection',
    'ExchangeResult',
    'BankOfCanadaClient',
    'FxRateService',
]
