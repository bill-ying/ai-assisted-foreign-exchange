"""
FX Service Package
Provides exchange rate lookup functionality using pluggable rate providers.
"""

from .currency import Currency, ExchangeDirection, ExchangeResult
from .rate_provider import RateProvider, BankOfCanadaProvider
from .mcp_provider import McpProvider
from .bank_of_canada_client import BankOfCanadaClient
from .fx_rate_service import FxRateService

__all__ = [
    'Currency',
    'ExchangeDirection',
    'ExchangeResult',
    'RateProvider',
    'BankOfCanadaProvider',
    'McpProvider',
    'BankOfCanadaClient',
    'FxRateService',
]
