"""
Scanner Agent
Scans market for RRS trading signals
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from agents.base import ScheduledAgent, AgentState
from agents.events import Event, EventType
from shared.indicators.rrs import RRSCalculator, check_daily_strength, check_daily_weakness


class ScannerAgent(ScheduledAgent):
    """
    Automated market scanner agent

    Responsibilities:
    - Scan watchlist for RRS signals
    - Filter by daily chart strength
    - Publish signals for analysis
    """

    def __init__(
        self,
        watchlist: List[str],
        data_provider,  # DataProvider instance
        scan_interval: float = 60.0,
        rrs_threshold: float = 2.0,
        **kwargs
    ):
        super().__init__(
            name="ScannerAgent",
            interval_seconds=scan_interval,
            **kwargs
        )

        self.watchlist = watchlist
        self.data_provider = data_provider
        self.rrs_threshold = rrs_threshold

        self.rrs_calculator = RRSCalculator()
        self.last_scan_time: Optional[datetime] = None
        self.signals_found: List[Dict] = []

        # Cooldown to avoid spamming signals
        self.signal_cooldown: Dict[str, datetime] = {}
        self.cooldown_minutes = 15

    async def initialize(self):
        """Initialize scanner resources"""
        logger.info(f"Initializing scanner with {len(self.watchlist)} symbols")
        self.metrics.custom_metrics["watchlist_size"] = len(self.watchlist)
        self.metrics.custom_metrics["scans_completed"] = 0
        self.metrics.custom_metrics["signals_found_total"] = 0

    async def cleanup(self):
        """Cleanup scanner resources"""
        logger.info("Scanner cleanup complete")

    def get_subscribed_events(self) -> List[EventType]:
        """Events this agent listens to"""
        return [
            EventType.MARKET_OPEN,
            EventType.MARKET_CLOSE,
            EventType.SYSTEM_START
        ]

    async def handle_event(self, event: Event):
        """Handle incoming events"""
        if event.event_type == EventType.MARKET_OPEN:
            logger.info("Market opened - scanner active")
            # Clear cooldowns at market open
            self.signal_cooldown.clear()

        elif event.event_type == EventType.MARKET_CLOSE:
            logger.info("Market closed - scanner idle")
            # Could pause scanning during closed hours

    async def run_scheduled_task(self):
        """Run the market scan"""
        await self.scan_market()

    async def scan_market(self):
        """Scan all watchlist symbols for signals"""
        await self.publish(EventType.SCAN_STARTED, {
            "watchlist_size": len(self.watchlist),
            "timestamp": datetime.now().isoformat()
        })

        self.signals_found = []
        strong_rs = []
        strong_rw = []

        try:
            # Get SPY data first (benchmark)
            spy_data = await self._get_spy_data()
            if not spy_data:
                logger.error("Failed to get SPY data, skipping scan")
                return

            # Scan each symbol
            for symbol in self.watchlist:
                try:
                    signal = await self._scan_symbol(symbol, spy_data)
                    if signal:
                        if signal["direction"] == "long":
                            strong_rs.append(signal)
                        else:
                            strong_rw.append(signal)

                except Exception as e:
                    logger.debug(f"Error scanning {symbol}: {e}")
                    continue

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)

            # Sort by RRS strength
            strong_rs.sort(key=lambda x: x["rrs"], reverse=True)
            strong_rw.sort(key=lambda x: x["rrs"])

            # Publish signals
            for signal in strong_rs[:5]:  # Top 5 RS
                await self._publish_signal(signal)

            for signal in strong_rw[:5]:  # Top 5 RW
                await self._publish_signal(signal)

            # Update metrics
            self.last_scan_time = datetime.now()
            self.metrics.custom_metrics["scans_completed"] += 1
            self.metrics.custom_metrics["last_scan"] = self.last_scan_time.isoformat()

            await self.publish(EventType.SCAN_COMPLETED, {
                "strong_rs_count": len(strong_rs),
                "strong_rw_count": len(strong_rw),
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"Scan complete: {len(strong_rs)} RS, {len(strong_rw)} RW")

        except Exception as e:
            logger.error(f"Scan error: {e}")

    async def _get_spy_data(self) -> Optional[Dict]:
        """Get SPY benchmark data"""
        try:
            # This would use the data provider
            # For now, returning mock structure
            data = await self.data_provider.get_stock_data("SPY")
            if data is None:
                return None

            return {
                "current_price": data.get("current_price"),
                "previous_close": data.get("previous_close"),
                "daily_data": data.get("daily_data")
            }
        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return None

    async def _scan_symbol(self, symbol: str, spy_data: Dict) -> Optional[Dict]:
        """Scan a single symbol for signals"""
        # Check cooldown
        if symbol in self.signal_cooldown:
            elapsed = (datetime.now() - self.signal_cooldown[symbol]).total_seconds()
            if elapsed < self.cooldown_minutes * 60:
                return None

        # Get stock data
        stock_data = await self.data_provider.get_stock_data(symbol)
        if not stock_data:
            return None

        # Check filters
        if stock_data.get("volume", 0) < 500000:
            return None
        if stock_data.get("current_price", 0) < 5.0:
            return None

        # Calculate RRS
        rrs_data = self.rrs_calculator.calculate_rrs_current(
            stock_data={
                "current_price": stock_data["current_price"],
                "previous_close": stock_data["previous_close"]
            },
            spy_data={
                "current_price": spy_data["current_price"],
                "previous_close": spy_data["previous_close"]
            },
            stock_atr=stock_data.get("atr", 1.0)
        )

        rrs = rrs_data["rrs"]

        # Check threshold
        if abs(rrs) < self.rrs_threshold:
            return None

        # Check daily chart
        daily_data = stock_data.get("daily_data")
        if daily_data is None:
            return None

        daily_strength = check_daily_strength(daily_data)
        daily_weakness = check_daily_weakness(daily_data)

        # Determine direction
        direction = None
        if rrs > self.rrs_threshold and daily_strength["is_strong"]:
            direction = "long"
        elif rrs < -self.rrs_threshold and daily_weakness["is_weak"]:
            direction = "short"

        if not direction:
            return None

        # Build signal
        signal = {
            "symbol": symbol,
            "direction": direction,
            "rrs": rrs,
            "rrs_status": rrs_data["status"],
            "price": stock_data["current_price"],
            "atr": stock_data.get("atr", 0),
            "volume": stock_data.get("volume", 0),
            "stock_pct_change": rrs_data["stock_pc"],
            "spy_pct_change": rrs_data["spy_pc"],
            "daily_strong": daily_strength["is_strong"],
            "daily_weak": daily_weakness["is_weak"],
            "ema3": daily_strength.get("ema3"),
            "ema8": daily_strength.get("ema8"),
            "timestamp": datetime.now().isoformat()
        }

        return signal

    async def _publish_signal(self, signal: Dict):
        """Publish a trading signal"""
        # Set cooldown
        self.signal_cooldown[signal["symbol"]] = datetime.now()

        # Track signal
        self.signals_found.append(signal)
        self.metrics.custom_metrics["signals_found_total"] += 1

        # Publish event
        await self.publish(EventType.SIGNAL_FOUND, signal)

        logger.info(
            f"Signal: {signal['symbol']} {signal['direction'].upper()} "
            f"RRS={signal['rrs']:.2f} @ ${signal['price']:.2f}"
        )

    def update_watchlist(self, symbols: List[str]):
        """Update the watchlist"""
        self.watchlist = symbols
        self.metrics.custom_metrics["watchlist_size"] = len(symbols)
        logger.info(f"Watchlist updated: {len(symbols)} symbols")

    def add_symbol(self, symbol: str):
        """Add a symbol to watchlist"""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)

    def remove_symbol(self, symbol: str):
        """Remove a symbol from watchlist"""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
