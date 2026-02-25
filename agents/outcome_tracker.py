"""
Outcome Tracker Agent

Tracks what happened to rejected signals by fetching prices after 1h, 4h, and 1d.
This helps evaluate whether rejection criteria are too strict — if many rejected
signals would have been profitable, the thresholds need loosening.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from agents.base import ScheduledAgent
from agents.events import EventType


class OutcomeTracker(ScheduledAgent):
    """
    Tracks price outcomes for rejected trading signals.

    Runs every hour during market hours. For each rejected signal from the
    last 24 hours that hasn't been fully tracked yet, fetches the current
    price and fills in the appropriate price_after column based on elapsed time.
    """

    def __init__(
        self,
        check_interval: float = 3600.0,  # 1 hour
        lookback_hours: int = 48,
        **kwargs
    ):
        super().__init__(
            name="OutcomeTracker",
            interval_seconds=check_interval,
            **kwargs
        )
        self.lookback_hours = lookback_hours
        self._trades_repo = None

    @property
    def trades_repo(self):
        """Lazy-load trades repository."""
        if self._trades_repo is None:
            from data.database import get_trades_repository
            self._trades_repo = get_trades_repository()
        return self._trades_repo

    async def initialize(self):
        """Initialize outcome tracker."""
        logger.info("OutcomeTracker initialized")

    async def cleanup(self):
        """Cleanup outcome tracker."""
        pass

    def get_subscribed_events(self) -> List[EventType]:
        """No event subscriptions needed — purely scheduled."""
        return []

    async def handle_event(self, event):
        """No events to handle."""
        pass

    async def run_scheduled_task(self):
        """Check rejected signals and update price outcomes."""
        try:
            pending = self.trades_repo.get_rejected_signals_pending_outcome(
                hours=self.lookback_hours
            )

            if not pending:
                return

            logger.info(f"OutcomeTracker: checking {len(pending)} rejected signals")

            # Batch fetch all needed prices in one call (non-blocking)
            symbols = list({s['symbol'] for s in pending})
            prices = await self._batch_fetch_prices(symbols)

            updated = 0
            for signal in pending:
                price = prices.get(signal['symbol'])
                if price is not None and self._update_signal_with_price(signal, price):
                    updated += 1

            if updated:
                logger.info(f"OutcomeTracker: updated {updated}/{len(pending)} signal outcomes")

        except Exception as e:
            logger.error(f"OutcomeTracker error: {e}")

    async def _batch_fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple symbols using yahooquery (non-blocking)."""
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="outcome")
        try:
            return await loop.run_in_executor(executor, self._batch_fetch_prices_sync, symbols)
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def _batch_fetch_prices_sync(symbols: List[str]) -> Dict[str, float]:
        """Synchronous batch price fetch using yahooquery."""
        try:
            from yahooquery import Ticker
            t = Ticker(symbols, asynchronous=False)
            prices = t.price
            result = {}
            for symbol, data in prices.items():
                if isinstance(data, dict):
                    price = data.get("regularMarketPrice")
                    if price is not None:
                        result[symbol] = float(price)
            return result
        except Exception as e:
            logger.debug(f"Batch price fetch failed: {e}")
            return {}

    def _update_signal_with_price(self, signal: Dict, current_price: float) -> bool:
        """Update signal outcome with the given price."""
        try:
            signal_time = signal.get('timestamp')
            if isinstance(signal_time, str):
                signal_time = datetime.fromisoformat(signal_time)

            now = datetime.utcnow()
            elapsed = now - signal_time
            elapsed_hours = elapsed.total_seconds() / 3600

            entry_price = float(signal['price'])
            direction = signal['direction']

            # Calculate would-have-been PnL (per share, as percentage)
            if direction == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            update_data = {}

            # Fill in based on elapsed time
            if elapsed_hours >= 1:
                update_data['price_after_1h'] = current_price
                update_data['would_have_pnl_1h'] = pnl_pct

            if elapsed_hours >= 4:
                update_data['price_after_4h'] = current_price
                update_data['would_have_pnl_4h'] = pnl_pct

            if elapsed_hours >= 24:
                update_data['price_after_1d'] = current_price
                update_data['would_have_pnl_1d'] = pnl_pct

            if update_data:
                return self.trades_repo.update_rejected_signal_outcome(
                    signal['id'], update_data
                )

            return False

        except Exception as e:
            logger.debug(f"Error updating outcome for {signal.get('symbol')}: {e}")
            return False

