"""
Data Providers Package
Redundant data providers with automatic fallback for the RDT Trading System.
"""

from data.providers.base import DataProvider, Quote, HistoricalData, ProviderError
from data.providers.yfinance_provider import YFinanceProvider
from data.providers.alpha_vantage_provider import AlphaVantageProvider
from data.providers.provider_manager import ProviderManager

__all__ = [
    "DataProvider",
    "Quote",
    "HistoricalData",
    "ProviderError",
    "YFinanceProvider",
    "AlphaVantageProvider",
    "ProviderManager",
]
