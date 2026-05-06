"""
FX Service Package
Provides exchange rate lookup functionality using pluggable rate providers.
"""

from .currency import Currency, ExchangeDirection, ExchangeResult
from .rate_provider import RateProvider, BankOfCanadaProvider
from .bank_of_canada_client import BankOfCanadaClient
from .fx_rate_service import FxRateService

__all__ = [
    'Currency',
    'ExchangeDirection',
    'ExchangeResult',
    'RateProvider',
    'BankOfCanadaProvider',
    'BankOfCanadaClient',
    'FxRateService',
]
