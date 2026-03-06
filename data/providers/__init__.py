"""
Data Providers Package
Redundant data providers with automatic fallback for the RDT Trading System.
"""

from data.providers.base import DataProvider, Quote, HistoricalData, ProviderError
from data.providers.provider_manager import ProviderManager

# Optional provider imports (may not be installed in all environments)
try:
    from data.providers.ibkr_provider import IBKRProvider
except ImportError:
    pass

try:
    from data.providers.yfinance_provider import YFinanceProvider
except ImportError:
    pass

try:
    from data.providers.alpha_vantage_provider import AlphaVantageProvider
except ImportError:
    pass

__all__ = [
    "DataProvider",
    "Quote",
    "HistoricalData",
    "ProviderError",
    "ProviderManager",
]
