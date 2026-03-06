"""
YFinance Data Provider — REMOVED

yfinance has been replaced by IBKR as the sole data source for the live
trading system. This stub exists to prevent ImportError in case any code
still references it. All methods raise ProviderError.

For offline scripts (backtesting, ML training), yfinance remains in
requirements.txt and can be used directly.
"""

from loguru import logger

from data.providers.base import (
    DataProvider,
    ProviderError,
)


class YFinanceProvider(DataProvider):
    """Stub — yfinance provider removed. Use IBKRProvider instead."""

    def __init__(self, **kwargs):
        super().__init__(name="yfinance", priority=999, **kwargs)
        logger.info("YFinanceProvider stub loaded (yfinance removed from live trading)")

    def get_quote(self, symbol: str):
        raise ProviderError("yfinance provider removed — use IBKR")

    def get_historical(self, symbol: str, **kwargs):
        raise ProviderError("yfinance provider removed — use IBKR")

    def get_batch_quotes(self, symbols, **kwargs):
        raise ProviderError("yfinance provider removed — use IBKR")

    def get_batch_historical(self, symbols, **kwargs):
        raise ProviderError("yfinance provider removed — use IBKR")

    def is_available(self) -> bool:
        return False
