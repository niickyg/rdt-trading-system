"""
Data Collection Agent

Collects ML training data on a 5-minute schedule during market hours:
- Intraday 5-minute bars for watchlist and open position symbols (via IBKR)
- Trade snapshots (MFE/MAE tracking) for open positions
- Options Greeks snapshots for open options positions

Also handles event-driven collection:
- MARKET_CLOSE: daily technical indicators, market regime, sector data
- POSITION_OPENED: initial trade snapshot
- POSITION_CLOSED: finalize MFE/MAE summary on trades table
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from loguru import logger

from agents.base import ScheduledAgent
from agents.events import EventType, Event


# SPDR sector ETFs for relative strength computation
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Energy': 'XLE',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
    'Industrials': 'XLI',
    'Communication Services': 'XLC',
}


class DataCollectionAgent(ScheduledAgent):
    """
    Collects and persists ML training data on a scheduled basis.

    Runs every 5 minutes during market hours (9:30-16:00 ET).
    Handles MARKET_CLOSE, POSITION_OPENED, and POSITION_CLOSED events.
    """

    def __init__(
        self,
        data_provider,
        watchlist: List[str],
        **kwargs
    ):
        super().__init__(
            name="DataCollectionAgent",
            interval_seconds=300,
            **kwargs
        )
        self.data_provider = data_provider
        self.watchlist = watchlist
        self._ml_repo = None
        self._trades_repo = None
        self._historical_cache = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    @property
    def ml_repo(self):
        """Lazy-load ML data repository."""
        if self._ml_repo is None:
            from data.database.ml_repository import get_ml_repository
            self._ml_repo = get_ml_repository()
        return self._ml_repo

    @property
    def trades_repo(self):
        """Lazy-load trades repository."""
        if self._trades_repo is None:
            from data.database import get_trades_repository
            self._trades_repo = get_trades_repository()
        return self._trades_repo

    @property
    def historical_cache(self):
        """Lazy-load historical bar cache."""
        if self._historical_cache is None:
            from data.database.historical_cache import get_historical_cache
            self._historical_cache = get_historical_cache()
        return self._historical_cache

    async def initialize(self):
        """Initialize the data collection agent."""
        logger.info("DataCollectionAgent initialized")

    async def cleanup(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)

    def get_subscribed_events(self) -> List[EventType]:
        """Events this agent listens to."""
        return [
            EventType.MARKET_CLOSE,
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
        ]

    async def handle_event(self, event: Event):
        """Handle incoming events."""
        if event.event_type == EventType.MARKET_CLOSE:
            await self._collect_daily_data()
        elif event.event_type == EventType.POSITION_OPENED:
            await self._on_position_opened(event.data)
        elif event.event_type == EventType.POSITION_CLOSED:
            await self._on_position_closed(event.data)

    async def run_scheduled_task(self):
        """Run scheduled data collection (every 5 minutes)."""
        if not self._is_market_hours():
            return

        try:
            await self._collect_intraday_bars()
        except Exception as e:
            logger.error(f"Error collecting intraday bars: {e}")

        try:
            await self._collect_trade_snapshots()
        except Exception as e:
            logger.error(f"Error collecting trade snapshots: {e}")

        try:
            await self._collect_options_greeks()
        except Exception as e:
            logger.error(f"Error collecting options greeks: {e}")

    # =========================================================================
    # Market Hours Check
    # =========================================================================

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:30-16:00 ET)."""
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        now_et = datetime.now(ZoneInfo("America/New_York"))
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        # Skip weekends
        if now_et.weekday() >= 5:
            return False

        return market_open <= now_et <= market_close

    # =========================================================================
    # Scheduled Collection: Intraday Bars (via IBKR ProviderManager)
    # =========================================================================

    async def _collect_intraday_bars(self):
        """Persist latest 5-minute bars for watchlist + open position symbols."""
        # Combine watchlist with open position symbols
        symbols = set(self.watchlist)
        try:
            open_positions = self.trades_repo.get_open_positions()
            for pos in open_positions:
                symbols.add(pos.get('symbol', ''))
        except Exception as e:
            logger.warning(f"Could not fetch open positions for bar collection: {e}")

        symbols.discard('')
        if not symbols:
            return

        loop = asyncio.get_running_loop()

        def _fetch_bars():
            bars = []
            try:
                from data.providers.provider_manager import get_provider_manager
                pm = get_provider_manager()

                for symbol in symbols:
                    try:
                        result = pm.get_historical(symbol, period="1d", interval="5m")
                        if result is None or result.data.empty:
                            continue
                        df = result.data
                        for ts, row in df.iterrows():
                            bars.append({
                                'symbol': symbol,
                                'timestamp': ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                                'open': float(row['open']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'volume': int(row['volume']),
                                'vwap': None,
                            })
                    except Exception as e:
                        logger.debug(f"Failed to fetch 5m bars for {symbol}: {e}")
            except Exception as e:
                logger.warning(f"ProviderManager not available for intraday bar collection: {e}")
            return bars

        bars = await loop.run_in_executor(self._executor, _fetch_bars)
        if bars:
            saved = self.ml_repo.save_intraday_bars(bars)
            if saved > 0:
                logger.info(f"Collected {saved} new intraday bars for {len(symbols)} symbols")

    # =========================================================================
    # Scheduled Collection: Trade Snapshots
    # =========================================================================

    async def _collect_trade_snapshots(self):
        """Take snapshots of all open trades with running MFE/MAE."""
        try:
            open_trades = self.trades_repo.get_trades(status='open')
        except Exception as e:
            logger.warning(f"Could not fetch open trades for snapshots: {e}")
            return

        if not open_trades:
            return

        for trade in open_trades:
            try:
                await self._take_trade_snapshot(trade)
            except Exception as e:
                logger.debug(f"Failed to snapshot trade {trade.get('id')}: {e}")

    async def _take_trade_snapshot(self, trade: Dict):
        """Take a single trade snapshot with MFE/MAE computation."""
        trade_id = trade['id']
        symbol = trade['symbol']
        entry_price = float(trade['entry_price'])
        direction = trade.get('direction', 'long')
        stop_loss = float(trade['stop_loss']) if trade.get('stop_loss') else None
        take_profit = float(trade['take_profit']) if trade.get('take_profit') else None

        # Get current price
        try:
            quote = await self.data_provider.get_quote(symbol)
            if not quote:
                return
            current_price = float(quote.get('price', 0))
            if current_price <= 0:
                return
        except Exception:
            return

        # Compute unrealized P&L
        if direction == 'long':
            unrealized_pnl = current_price - entry_price
        else:
            unrealized_pnl = entry_price - current_price

        shares = int(trade.get('shares', 1))
        unrealized_pnl_total = unrealized_pnl * shares
        unrealized_pnl_pct = (unrealized_pnl / entry_price) * 100 if entry_price else 0

        # Compute R-multiple
        unrealized_r = None
        if stop_loss and entry_price:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                unrealized_r = unrealized_pnl / risk_per_share

        # Get previous snapshot for running MFE/MAE
        prev = self.ml_repo.get_latest_snapshot(trade_id)
        prev_mfe = float(prev['mfe']) if prev and prev.get('mfe') is not None else 0.0
        prev_mae = float(prev['mae']) if prev and prev.get('mae') is not None else 0.0
        prev_mfe_pct = float(prev['mfe_pct']) if prev and prev.get('mfe_pct') is not None else 0.0
        prev_mae_pct = float(prev['mae_pct']) if prev and prev.get('mae_pct') is not None else 0.0
        prev_bars = int(prev['bars_held']) if prev and prev.get('bars_held') is not None else 0

        # Running MFE/MAE
        mfe = max(prev_mfe, unrealized_pnl_total) if unrealized_pnl_total > 0 else prev_mfe
        mae = min(prev_mae, unrealized_pnl_total) if unrealized_pnl_total < 0 else prev_mae
        mfe_pct = max(prev_mfe_pct, unrealized_pnl_pct) if unrealized_pnl_pct > 0 else prev_mfe_pct
        mae_pct = min(prev_mae_pct, unrealized_pnl_pct) if unrealized_pnl_pct < 0 else prev_mae_pct

        bars_held = prev_bars + 1

        # Distance to stop/target
        distance_to_stop_pct = None
        distance_to_target_pct = None
        if stop_loss and current_price:
            distance_to_stop_pct = ((current_price - stop_loss) / current_price) * 100
        if take_profit and current_price:
            distance_to_target_pct = ((take_profit - current_price) / current_price) * 100

        snapshot_data = {
            'trade_id': trade_id,
            'timestamp': datetime.utcnow(),
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl_total,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'unrealized_r': unrealized_r,
            'mfe': mfe,
            'mae': mae,
            'mfe_pct': mfe_pct,
            'mae_pct': mae_pct,
            'bars_held': bars_held,
            'distance_to_stop_pct': distance_to_stop_pct,
            'distance_to_target_pct': distance_to_target_pct,
        }

        self.ml_repo.save_trade_snapshot(snapshot_data)

    # =========================================================================
    # Scheduled Collection: Options Greeks
    # =========================================================================

    async def _collect_options_greeks(self):
        """Snapshot Greeks for all open options positions."""
        try:
            if hasattr(self.trades_repo, "get_all_options_positions"):
                open_options = self.trades_repo.get_all_options_positions()
            elif hasattr(self.trades_repo, "get_options_positions"):
                open_options = self.trades_repo.get_options_positions()
            else:
                logger.warning(
                    "OPTIONS_WARN[MISSING_OPTIONS_ACCESSOR] Trades repository has no options position accessor"
                )
                return
        except Exception as e:
            logger.debug(f"Could not fetch open options positions: {e}")
            return

        if not isinstance(open_options, list):
            logger.warning(
                "OPTIONS_WARN[INVALID_OPTIONS_POSITIONS] Expected list of open options positions; skipping cycle"
            )
            return

        if not open_options:
            return

        for pos in open_options:
            try:
                if not isinstance(pos, dict):
                    logger.debug("OPTIONS_WARN[INVALID_OPTIONS_POSITION_ROW] Skipping malformed options position row")
                    continue
                symbol = pos.get('symbol', '')
                if not symbol:
                    continue

                # Get underlying quote
                quote = await self.data_provider.get_quote(symbol)
                underlying_price = float(quote.get('price', 0)) if quote else None

                greeks_data = {
                    'options_trade_id': None,  # FK to options_trades; open positions don't have a trade record yet
                    'symbol': symbol,
                    'timestamp': datetime.utcnow(),
                    'underlying_price': underlying_price,
                    'delta': pos.get('entry_delta'),
                    'iv': pos.get('entry_iv'),
                    'premium': pos.get('entry_premium'),
                }

                self.ml_repo.save_options_greeks_snapshot(greeks_data)
            except Exception as e:
                logger.debug(f"Failed to snapshot options greeks for {pos.get('symbol')}: {e}")

    # =========================================================================
    # Event: MARKET_CLOSE → Daily Data Collection
    # =========================================================================

    async def _collect_daily_data(self):
        """Collect end-of-day data: technical indicators, market regime, sectors."""
        logger.info("DataCollectionAgent: collecting daily data on market close")

        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(self._executor, self._compute_technical_indicators)
        except Exception as e:
            logger.error(f"Error computing technical indicators: {e}")

        try:
            await loop.run_in_executor(self._executor, self._compute_market_regime)
        except Exception as e:
            logger.error(f"Error computing market regime: {e}")

        try:
            await loop.run_in_executor(self._executor, self._compute_sector_data)
        except Exception as e:
            logger.error(f"Error computing sector data: {e}")

        logger.info("DataCollectionAgent: daily data collection complete")

    def _compute_technical_indicators(self):
        """Compute and save daily technical indicators for all watchlist symbols."""
        import pandas as pd

        today = date.today()

        for symbol in self.watchlist:
            try:
                df = self.historical_cache.get_daily_bars(symbol, lookback_days=260)
                if df is None or len(df) < 50:
                    continue

                close = df['close']
                high = df['high']
                low = df['low']
                volume = df['volume']

                # RSI-14
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, float('nan'))
                rsi_14 = 100 - (100 / (1 + rs))

                # MACD
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                macd_line = ema_12 - ema_26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                macd_histogram = macd_line - macd_signal

                # Bollinger Bands
                bb_middle = close.rolling(20).mean()
                bb_std = close.rolling(20).std()
                bb_upper = bb_middle + 2 * bb_std
                bb_lower = bb_middle - 2 * bb_std
                bb_width = (bb_upper - bb_lower) / bb_middle

                # EMAs
                ema_9 = close.ewm(span=9, adjust=False).mean()
                ema_21 = close.ewm(span=21, adjust=False).mean()
                ema_50 = close.ewm(span=50, adjust=False).mean()
                ema_200 = close.ewm(span=200, adjust=False).mean()

                # ATR-14
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr_14 = tr.rolling(14).mean()

                # ADX (simplified - using directional movement)
                plus_dm = high.diff()
                minus_dm = -low.diff()
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
                atr_smooth = tr.ewm(span=14, adjust=False).mean()
                plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
                minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan'))
                adx = dx.ewm(span=14, adjust=False).mean()

                # OBV
                obv = (volume * (~close.diff().le(0)).astype(int) - volume * close.diff().lt(0).astype(int)).cumsum()

                # Get latest values
                idx = -1
                indicators = {
                    'symbol': symbol,
                    'date': today,
                    'rsi_14': float(rsi_14.iloc[idx]) if pd.notna(rsi_14.iloc[idx]) else None,
                    'macd_line': float(macd_line.iloc[idx]) if pd.notna(macd_line.iloc[idx]) else None,
                    'macd_signal': float(macd_signal.iloc[idx]) if pd.notna(macd_signal.iloc[idx]) else None,
                    'macd_histogram': float(macd_histogram.iloc[idx]) if pd.notna(macd_histogram.iloc[idx]) else None,
                    'bb_upper': float(bb_upper.iloc[idx]) if pd.notna(bb_upper.iloc[idx]) else None,
                    'bb_middle': float(bb_middle.iloc[idx]) if pd.notna(bb_middle.iloc[idx]) else None,
                    'bb_lower': float(bb_lower.iloc[idx]) if pd.notna(bb_lower.iloc[idx]) else None,
                    'bb_width': float(bb_width.iloc[idx]) if pd.notna(bb_width.iloc[idx]) else None,
                    'ema_9': float(ema_9.iloc[idx]) if pd.notna(ema_9.iloc[idx]) else None,
                    'ema_21': float(ema_21.iloc[idx]) if pd.notna(ema_21.iloc[idx]) else None,
                    'ema_50': float(ema_50.iloc[idx]) if pd.notna(ema_50.iloc[idx]) else None,
                    'ema_200': float(ema_200.iloc[idx]) if pd.notna(ema_200.iloc[idx]) else None,
                    'adx': float(adx.iloc[idx]) if pd.notna(adx.iloc[idx]) else None,
                    'obv': int(obv.iloc[idx]) if pd.notna(obv.iloc[idx]) else None,
                    'atr_14': float(atr_14.iloc[idx]) if pd.notna(atr_14.iloc[idx]) else None,
                    'close_price': float(close.iloc[idx]) if pd.notna(close.iloc[idx]) else None,
                }

                self.ml_repo.save_technical_indicators(indicators)

            except Exception as e:
                logger.debug(f"Failed to compute indicators for {symbol}: {e}")

        logger.info(f"Computed technical indicators for {len(self.watchlist)} symbols")

    def _compute_market_regime(self):
        """Compute and save daily market regime data."""
        today = date.today()

        try:
            # VIX data — use yfinance (sole exception for ^VIX)
            vix_close = None
            try:
                import yfinance as yf
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="5d")
                vix_close = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else None
            except Exception as e:
                logger.debug(f"VIX fetch failed: {e}")

            vix_regime = None
            if vix_close is not None:
                if vix_close < 15:
                    vix_regime = "low"
                elif vix_close < 25:
                    vix_regime = "normal"
                elif vix_close < 35:
                    vix_regime = "elevated"
                else:
                    vix_regime = "extreme"

            # SPY data from DB cache
            spy_df = self.historical_cache.get_daily_bars("SPY", lookback_days=260)
            if spy_df is None or spy_df.empty:
                return

            spy_close = spy_df['close']
            spy_close_val = float(spy_close.iloc[-1])
            spy_ema_50 = float(spy_close.ewm(span=50, adjust=False).mean().iloc[-1])
            spy_ema_200 = float(spy_close.ewm(span=200, adjust=False).mean().iloc[-1])

            spy_above_50ema = spy_close_val > spy_ema_50
            spy_above_200ema = spy_close_val > spy_ema_200

            # Determine trend
            if spy_above_200ema and spy_above_50ema:
                spy_trend = "bullish"
            elif not spy_above_200ema and not spy_above_50ema:
                spy_trend = "bearish"
            else:
                spy_trend = "neutral"

            # Regime label
            regime_label = "unknown"
            if vix_regime in ("low", "normal") and spy_trend == "bullish":
                regime_label = "risk_on"
            elif vix_regime in ("elevated", "extreme") and spy_trend == "bearish":
                regime_label = "risk_off"
            elif vix_regime in ("elevated", "extreme"):
                regime_label = "volatile"
            else:
                regime_label = "transitional"

            regime_data = {
                'date': today,
                'vix_close': vix_close,
                'vix_regime': vix_regime,
                'spy_close': spy_close_val,
                'spy_trend': spy_trend,
                'spy_above_200ema': spy_above_200ema,
                'spy_above_50ema': spy_above_50ema,
                'regime_label': regime_label,
            }

            self.ml_repo.save_market_regime_daily(regime_data)
            vix_str = f"{vix_close:.1f}" if vix_close else "N/A"
            logger.info(f"Saved market regime: {regime_label} (VIX={vix_str}, SPY trend={spy_trend})")

        except Exception as e:
            logger.error(f"Failed to compute market regime: {e}")

    def _compute_sector_data(self):
        """Compute and save daily sector relative strength."""
        today = date.today()

        try:
            # Get SPY returns from DB cache
            spy_df = self.historical_cache.get_daily_bars("SPY", lookback_days=90)
            if spy_df is None or spy_df.empty:
                return

            spy_close = spy_df['close']
            spy_returns_5d = (spy_close.iloc[-1] / spy_close.iloc[-5] - 1) if len(spy_close) >= 5 else 0
            spy_returns_20d = (spy_close.iloc[-1] / spy_close.iloc[-20] - 1) if len(spy_close) >= 20 else 0
            spy_returns_60d = (spy_close.iloc[-1] / spy_close.iloc[-60] - 1) if len(spy_close) >= 60 else 0

            sector_records = []
            for sector_name, etf_symbol in SECTOR_ETFS.items():
                try:
                    etf_df = self.historical_cache.get_daily_bars(etf_symbol, lookback_days=90)
                    if etf_df is None or etf_df.empty:
                        continue

                    etf_close = etf_df['close']
                    close_price = float(etf_close.iloc[-1])
                    daily_return_pct = float((etf_close.iloc[-1] / etf_close.iloc[-2] - 1) * 100) if len(etf_close) >= 2 else None

                    # Relative strength vs SPY
                    rs_5d = None
                    rs_20d = None
                    rs_60d = None

                    if len(etf_close) >= 5:
                        etf_ret_5d = etf_close.iloc[-1] / etf_close.iloc[-5] - 1
                        rs_5d = float((etf_ret_5d - spy_returns_5d) * 100)
                    if len(etf_close) >= 20:
                        etf_ret_20d = etf_close.iloc[-1] / etf_close.iloc[-20] - 1
                        rs_20d = float((etf_ret_20d - spy_returns_20d) * 100)
                    if len(etf_close) >= 60:
                        etf_ret_60d = etf_close.iloc[-1] / etf_close.iloc[-60] - 1
                        rs_60d = float((etf_ret_60d - spy_returns_60d) * 100)

                    sector_records.append({
                        'date': today,
                        'sector': sector_name,
                        'etf_symbol': etf_symbol,
                        'close_price': close_price,
                        'daily_return_pct': daily_return_pct,
                        'relative_strength_5d': rs_5d,
                        'relative_strength_20d': rs_20d,
                        'relative_strength_60d': rs_60d,
                    })
                except Exception as e:
                    logger.debug(f"Failed to compute sector data for {etf_symbol}: {e}")

            # Rank by 20d relative strength
            sector_records.sort(key=lambda x: x.get('relative_strength_20d') or -999, reverse=True)
            for rank, record in enumerate(sector_records, 1):
                record['sector_rank'] = rank

            if sector_records:
                saved = self.ml_repo.save_sector_data_batch(sector_records)
                logger.info(f"Saved sector data for {saved} sectors")

        except Exception as e:
            logger.error(f"Failed to compute sector data: {e}")

    # =========================================================================
    # Event: POSITION_OPENED → Initial Snapshot
    # =========================================================================

    async def _on_position_opened(self, data: Dict):
        """Take an initial trade snapshot when a position is opened."""
        trade_id = data.get('trade_id')
        if not trade_id:
            return

        try:
            trade = self.trades_repo.get_trade_by_id(trade_id)
            if trade:
                snapshot_data = {
                    'trade_id': trade_id,
                    'timestamp': datetime.utcnow(),
                    'current_price': float(trade['entry_price']),
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'unrealized_r': 0.0,
                    'mfe': 0.0,
                    'mae': 0.0,
                    'mfe_pct': 0.0,
                    'mae_pct': 0.0,
                    'bars_held': 0,
                }
                self.ml_repo.save_trade_snapshot(snapshot_data)
                logger.debug(f"Initial snapshot saved for trade {trade_id}")
        except Exception as e:
            logger.error(f"Error creating initial snapshot for trade {trade_id}: {e}")

    # =========================================================================
    # Event: POSITION_CLOSED → Finalize MFE/MAE
    # =========================================================================

    async def _on_position_closed(self, data: Dict):
        """Finalize MFE/MAE summary on the trades table when a position is closed."""
        trade_id = data.get('trade_id')
        if not trade_id:
            return

        try:
            await self._finalize_trade_mfe_mae(trade_id)
        except Exception as e:
            logger.error(f"Error finalizing MFE/MAE for trade {trade_id}: {e}")

    async def _finalize_trade_mfe_mae(self, trade_id: int):
        """Query all snapshots and compute peak MFE/MAE for the trade summary."""
        snapshots = self.ml_repo.get_trade_snapshots(trade_id)
        if not snapshots:
            logger.debug(f"No snapshots found for trade {trade_id}, skipping MFE/MAE finalization")
            return

        # Find peaks across all snapshots
        peak_mfe = max(s.get('mfe', 0) or 0 for s in snapshots)
        peak_mae = min(s.get('mae', 0) or 0 for s in snapshots)
        peak_mfe_pct = max(s.get('mfe_pct', 0) or 0 for s in snapshots)
        peak_mae_pct = min(s.get('mae_pct', 0) or 0 for s in snapshots)
        total_bars = max(s.get('bars_held', 0) or 0 for s in snapshots)

        # Find which bar reached peak MFE
        bars_to_mfe = 0
        for s in snapshots:
            if (s.get('mfe') or 0) == peak_mfe and peak_mfe > 0:
                bars_to_mfe = s.get('bars_held', 0) or 0
                break

        # Compute R-multiples from the trade's risk
        trade = self.trades_repo.get_trade_by_id(trade_id)
        peak_mfe_r = None
        peak_mae_r = None
        if trade:
            entry_price = float(trade.get('entry_price', 0))
            stop_loss = float(trade['stop_loss']) if trade.get('stop_loss') else None
            if stop_loss and entry_price:
                risk_per_share = abs(entry_price - stop_loss)
                shares = int(trade.get('shares', 1))
                total_risk = risk_per_share * shares
                if total_risk > 0:
                    peak_mfe_r = peak_mfe / total_risk
                    peak_mae_r = peak_mae / total_risk

        mfe_mae_data = {
            'peak_mfe': peak_mfe,
            'peak_mae': peak_mae,
            'peak_mfe_pct': peak_mfe_pct,
            'peak_mae_pct': peak_mae_pct,
            'peak_mfe_r': peak_mfe_r,
            'peak_mae_r': peak_mae_r,
            'bars_to_mfe': bars_to_mfe,
            'bars_held': total_bars,
        }

        self.ml_repo.update_trade_mfe_mae(trade_id, mfe_mae_data)
        logger.info(f"Finalized MFE/MAE for trade {trade_id}: MFE=${peak_mfe:.2f}, MAE=${peak_mae:.2f}")
