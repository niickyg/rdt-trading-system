"""
ML Data Repository for the RDT Trading System.

Provides CRUD operations for ML training data tables:
- Intraday bars (5-minute OHLCV)
- Technical indicators (daily)
- Trade snapshots (MFE/MAE tracking)
- Market regime (daily context)
- Sector data (relative strength)
- Options Greeks history
- Earnings calendar
- Trade MFE/MAE summary updates
"""

import threading
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
from sqlalchemy import and_, desc
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .connection import get_db_manager
from .models import (
    IntradayBar, TechnicalIndicator, TradeSnapshot, MarketRegimeDaily,
    SectorData, OptionsGreeksHistory, EarningsCalendar, Trade,
)


class MLDataRepository:
    """Repository for ML training data database operations."""

    def __init__(self):
        self._db_manager = None

    @property
    def db_manager(self):
        """Lazy-load database manager."""
        if self._db_manager is None:
            self._db_manager = get_db_manager()
        return self._db_manager

    # =========================================================================
    # Intraday Bars
    # =========================================================================

    def save_intraday_bars(self, bars: List[Dict[str, Any]]) -> int:
        """
        Save a batch of intraday bars (upsert on symbol+timestamp).

        Returns:
            Number of bars saved.
        """
        if not bars:
            return 0
        try:
            with self.db_manager.get_session() as session:
                saved = 0
                for bar in bars:
                    existing = session.query(IntradayBar).filter(
                        and_(
                            IntradayBar.symbol == bar['symbol'],
                            IntradayBar.timestamp == bar['timestamp'],
                        )
                    ).first()
                    if existing:
                        continue
                    record = IntradayBar(
                        symbol=bar['symbol'],
                        timestamp=bar['timestamp'],
                        open=float(bar['open']),
                        high=float(bar['high']),
                        low=float(bar['low']),
                        close=float(bar['close']),
                        volume=int(bar['volume']),
                        vwap=bar.get('vwap'),
                    )
                    session.add(record)
                    saved += 1
                logger.debug(f"Saved {saved} intraday bars")
                return saved
        except Exception as e:
            logger.error(f"Error saving intraday bars: {e}")
            return 0

    def get_intraday_bars(
        self, symbol: str, start: datetime, end: datetime
    ) -> List[Dict[str, Any]]:
        """Get intraday bars for a symbol within a time range."""
        try:
            with self.db_manager.get_session() as session:
                rows = session.query(IntradayBar).filter(
                    and_(
                        IntradayBar.symbol == symbol,
                        IntradayBar.timestamp >= start,
                        IntradayBar.timestamp <= end,
                    )
                ).order_by(IntradayBar.timestamp).all()
                return [
                    {
                        'id': r.id, 'symbol': r.symbol,
                        'timestamp': r.timestamp,
                        'open': float(r.open), 'high': float(r.high),
                        'low': float(r.low), 'close': float(r.close),
                        'volume': r.volume, 'vwap': float(r.vwap) if r.vwap else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting intraday bars for {symbol}: {e}")
            return []

    # =========================================================================
    # Technical Indicators
    # =========================================================================

    def save_technical_indicators(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Save or update daily technical indicators for a symbol+date."""
        try:
            with self.db_manager.get_session() as session:
                existing = session.query(TechnicalIndicator).filter(
                    and_(
                        TechnicalIndicator.symbol == data['symbol'],
                        TechnicalIndicator.date == data['date'],
                    )
                ).first()
                if existing:
                    for key, value in data.items():
                        if key not in ('id',) and hasattr(existing, key):
                            setattr(existing, key, value)
                    record = existing
                else:
                    record = TechnicalIndicator(**data)
                    session.add(record)
                session.flush()
                return {'id': record.id, 'symbol': record.symbol, 'date': record.date}
        except Exception as e:
            logger.error(f"Error saving technical indicators: {e}")
            return None

    def get_technical_indicators(
        self, symbol: str, start: date, end: date
    ) -> List[Dict[str, Any]]:
        """Get technical indicators for a symbol within a date range."""
        try:
            with self.db_manager.get_session() as session:
                rows = session.query(TechnicalIndicator).filter(
                    and_(
                        TechnicalIndicator.symbol == symbol,
                        TechnicalIndicator.date >= start,
                        TechnicalIndicator.date <= end,
                    )
                ).order_by(TechnicalIndicator.date).all()
                return [
                    {
                        'id': r.id, 'symbol': r.symbol, 'date': r.date,
                        'rsi_14': float(r.rsi_14) if r.rsi_14 else None,
                        'macd_line': float(r.macd_line) if r.macd_line else None,
                        'macd_signal': float(r.macd_signal) if r.macd_signal else None,
                        'macd_histogram': float(r.macd_histogram) if r.macd_histogram else None,
                        'bb_upper': float(r.bb_upper) if r.bb_upper else None,
                        'bb_middle': float(r.bb_middle) if r.bb_middle else None,
                        'bb_lower': float(r.bb_lower) if r.bb_lower else None,
                        'bb_width': float(r.bb_width) if r.bb_width else None,
                        'ema_9': float(r.ema_9) if r.ema_9 else None,
                        'ema_21': float(r.ema_21) if r.ema_21 else None,
                        'ema_50': float(r.ema_50) if r.ema_50 else None,
                        'ema_200': float(r.ema_200) if r.ema_200 else None,
                        'adx': float(r.adx) if r.adx else None,
                        'obv': r.obv,
                        'atr_14': float(r.atr_14) if r.atr_14 else None,
                        'close_price': float(r.close_price) if r.close_price else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting technical indicators for {symbol}: {e}")
            return []

    # =========================================================================
    # Trade Snapshots
    # =========================================================================

    def save_trade_snapshot(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Save a trade snapshot."""
        try:
            with self.db_manager.get_session() as session:
                record = TradeSnapshot(
                    trade_id=data['trade_id'],
                    timestamp=data.get('timestamp', datetime.utcnow()),
                    current_price=float(data['current_price']),
                    unrealized_pnl=data.get('unrealized_pnl'),
                    unrealized_pnl_pct=data.get('unrealized_pnl_pct'),
                    unrealized_r=data.get('unrealized_r'),
                    mfe=data.get('mfe'),
                    mae=data.get('mae'),
                    mfe_pct=data.get('mfe_pct'),
                    mae_pct=data.get('mae_pct'),
                    bars_held=data.get('bars_held'),
                    rsi_at_snapshot=data.get('rsi_at_snapshot'),
                    distance_to_stop_pct=data.get('distance_to_stop_pct'),
                    distance_to_target_pct=data.get('distance_to_target_pct'),
                )
                session.add(record)
                session.flush()
                return {'id': record.id, 'trade_id': record.trade_id}
        except Exception as e:
            logger.error(f"Error saving trade snapshot: {e}")
            return None

    def get_trade_snapshots(self, trade_id: int) -> List[Dict[str, Any]]:
        """Get all snapshots for a trade."""
        try:
            with self.db_manager.get_session() as session:
                rows = session.query(TradeSnapshot).filter(
                    TradeSnapshot.trade_id == trade_id
                ).order_by(TradeSnapshot.timestamp).all()
                return [
                    {
                        'id': r.id, 'trade_id': r.trade_id,
                        'timestamp': r.timestamp,
                        'current_price': float(r.current_price),
                        'unrealized_pnl': float(r.unrealized_pnl) if r.unrealized_pnl else None,
                        'unrealized_pnl_pct': float(r.unrealized_pnl_pct) if r.unrealized_pnl_pct else None,
                        'unrealized_r': float(r.unrealized_r) if r.unrealized_r else None,
                        'mfe': float(r.mfe) if r.mfe else None,
                        'mae': float(r.mae) if r.mae else None,
                        'mfe_pct': float(r.mfe_pct) if r.mfe_pct else None,
                        'mae_pct': float(r.mae_pct) if r.mae_pct else None,
                        'bars_held': r.bars_held,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting trade snapshots for trade {trade_id}: {e}")
            return []

    def get_latest_snapshot(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot for a trade."""
        try:
            with self.db_manager.get_session() as session:
                row = session.query(TradeSnapshot).filter(
                    TradeSnapshot.trade_id == trade_id
                ).order_by(desc(TradeSnapshot.timestamp)).first()
                if not row:
                    return None
                return {
                    'id': row.id, 'trade_id': row.trade_id,
                    'timestamp': row.timestamp,
                    'current_price': float(row.current_price),
                    'mfe': float(row.mfe) if row.mfe else None,
                    'mae': float(row.mae) if row.mae else None,
                    'mfe_pct': float(row.mfe_pct) if row.mfe_pct else None,
                    'mae_pct': float(row.mae_pct) if row.mae_pct else None,
                    'bars_held': row.bars_held,
                }
        except Exception as e:
            logger.error(f"Error getting latest snapshot for trade {trade_id}: {e}")
            return None

    # =========================================================================
    # Market Regime Daily
    # =========================================================================

    def save_market_regime_daily(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Save or update daily market regime data."""
        try:
            with self.db_manager.get_session() as session:
                existing = session.query(MarketRegimeDaily).filter(
                    MarketRegimeDaily.date == data['date']
                ).first()
                if existing:
                    for key, value in data.items():
                        if key not in ('id',) and hasattr(existing, key):
                            setattr(existing, key, value)
                    record = existing
                else:
                    record = MarketRegimeDaily(**data)
                    session.add(record)
                session.flush()
                return {'id': record.id, 'date': record.date}
        except Exception as e:
            logger.error(f"Error saving market regime daily: {e}")
            return None

    def get_market_regime(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Get market regime for a specific date."""
        try:
            with self.db_manager.get_session() as session:
                row = session.query(MarketRegimeDaily).filter(
                    MarketRegimeDaily.date == target_date
                ).first()
                if not row:
                    return None
                return {
                    'id': row.id, 'date': row.date,
                    'vix_close': float(row.vix_close) if row.vix_close else None,
                    'vix_regime': row.vix_regime,
                    'spy_close': float(row.spy_close) if row.spy_close else None,
                    'spy_trend': row.spy_trend,
                    'spy_above_200ema': row.spy_above_200ema,
                    'spy_above_50ema': row.spy_above_50ema,
                    'advance_decline_ratio': float(row.advance_decline_ratio) if row.advance_decline_ratio else None,
                    'new_highs': row.new_highs,
                    'new_lows': row.new_lows,
                    'breadth_thrust': float(row.breadth_thrust) if row.breadth_thrust else None,
                    'put_call_ratio': float(row.put_call_ratio) if row.put_call_ratio else None,
                    'regime_label': row.regime_label,
                }
        except Exception as e:
            logger.error(f"Error getting market regime for {target_date}: {e}")
            return None

    # =========================================================================
    # Sector Data
    # =========================================================================

    def save_sector_data_batch(self, records: List[Dict[str, Any]]) -> int:
        """Save a batch of sector data records (upsert on date+sector)."""
        if not records:
            return 0
        try:
            with self.db_manager.get_session() as session:
                saved = 0
                for data in records:
                    existing = session.query(SectorData).filter(
                        and_(
                            SectorData.date == data['date'],
                            SectorData.sector == data['sector'],
                        )
                    ).first()
                    if existing:
                        for key, value in data.items():
                            if key not in ('id',) and hasattr(existing, key):
                                setattr(existing, key, value)
                    else:
                        record = SectorData(**data)
                        session.add(record)
                    saved += 1
                logger.debug(f"Saved {saved} sector data records")
                return saved
        except Exception as e:
            logger.error(f"Error saving sector data batch: {e}")
            return 0

    def get_sector_data(self, target_date: date) -> List[Dict[str, Any]]:
        """Get all sector data for a specific date."""
        try:
            with self.db_manager.get_session() as session:
                rows = session.query(SectorData).filter(
                    SectorData.date == target_date
                ).order_by(SectorData.sector_rank).all()
                return [
                    {
                        'id': r.id, 'date': r.date,
                        'sector': r.sector, 'etf_symbol': r.etf_symbol,
                        'close_price': float(r.close_price) if r.close_price else None,
                        'daily_return_pct': float(r.daily_return_pct) if r.daily_return_pct else None,
                        'relative_strength_5d': float(r.relative_strength_5d) if r.relative_strength_5d else None,
                        'relative_strength_20d': float(r.relative_strength_20d) if r.relative_strength_20d else None,
                        'relative_strength_60d': float(r.relative_strength_60d) if r.relative_strength_60d else None,
                        'sector_rank': r.sector_rank,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting sector data for {target_date}: {e}")
            return []

    # =========================================================================
    # Options Greeks History
    # =========================================================================

    def save_options_greeks_snapshot(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Save an options Greeks snapshot."""
        try:
            with self.db_manager.get_session() as session:
                record = OptionsGreeksHistory(
                    options_trade_id=data.get('options_trade_id'),
                    symbol=data['symbol'],
                    timestamp=data.get('timestamp', datetime.utcnow()),
                    underlying_price=data.get('underlying_price'),
                    delta=data.get('delta'),
                    gamma=data.get('gamma'),
                    theta=data.get('theta'),
                    vega=data.get('vega'),
                    iv=data.get('iv'),
                    premium=data.get('premium'),
                    dte=data.get('dte'),
                    moneyness=data.get('moneyness'),
                    intrinsic_value=data.get('intrinsic_value'),
                    extrinsic_value=data.get('extrinsic_value'),
                )
                session.add(record)
                session.flush()
                return {'id': record.id, 'symbol': record.symbol}
        except Exception as e:
            logger.error(f"Error saving options greeks snapshot: {e}")
            return None

    def get_options_greeks_history(self, trade_id: int) -> List[Dict[str, Any]]:
        """Get all Greeks snapshots for an options trade."""
        try:
            with self.db_manager.get_session() as session:
                rows = session.query(OptionsGreeksHistory).filter(
                    OptionsGreeksHistory.options_trade_id == trade_id
                ).order_by(OptionsGreeksHistory.timestamp).all()
                return [
                    {
                        'id': r.id, 'options_trade_id': r.options_trade_id,
                        'symbol': r.symbol, 'timestamp': r.timestamp,
                        'underlying_price': float(r.underlying_price) if r.underlying_price else None,
                        'delta': float(r.delta) if r.delta else None,
                        'gamma': float(r.gamma) if r.gamma else None,
                        'theta': float(r.theta) if r.theta else None,
                        'vega': float(r.vega) if r.vega else None,
                        'iv': float(r.iv) if r.iv else None,
                        'premium': float(r.premium) if r.premium else None,
                        'dte': r.dte,
                        'moneyness': float(r.moneyness) if r.moneyness else None,
                        'intrinsic_value': float(r.intrinsic_value) if r.intrinsic_value else None,
                        'extrinsic_value': float(r.extrinsic_value) if r.extrinsic_value else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting options greeks history for trade {trade_id}: {e}")
            return []

    # =========================================================================
    # Earnings Calendar
    # =========================================================================

    def save_earnings_event(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Save or update an earnings calendar event."""
        try:
            with self.db_manager.get_session() as session:
                existing = session.query(EarningsCalendar).filter(
                    and_(
                        EarningsCalendar.symbol == data['symbol'],
                        EarningsCalendar.earnings_date == data['earnings_date'],
                    )
                ).first()
                if existing:
                    for key, value in data.items():
                        if key not in ('id', 'created_at') and hasattr(existing, key):
                            setattr(existing, key, value)
                    record = existing
                else:
                    record = EarningsCalendar(**data)
                    session.add(record)
                session.flush()
                return {'id': record.id, 'symbol': record.symbol, 'earnings_date': record.earnings_date}
        except Exception as e:
            logger.error(f"Error saving earnings event: {e}")
            return None

    def get_upcoming_earnings(self, symbol: str, days: int = 14) -> List[Dict[str, Any]]:
        """Get upcoming earnings events for a symbol within N days."""
        try:
            today = date.today()
            end_date = today + timedelta(days=days)
            with self.db_manager.get_session() as session:
                rows = session.query(EarningsCalendar).filter(
                    and_(
                        EarningsCalendar.symbol == symbol,
                        EarningsCalendar.earnings_date >= today,
                        EarningsCalendar.earnings_date <= end_date,
                    )
                ).order_by(EarningsCalendar.earnings_date).all()
                return [
                    {
                        'id': r.id, 'symbol': r.symbol,
                        'earnings_date': r.earnings_date,
                        'timing': r.timing,
                        'eps_estimate': float(r.eps_estimate) if r.eps_estimate else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting upcoming earnings for {symbol}: {e}")
            return []

    def get_earnings_for_date(self, target_date: date) -> List[Dict[str, Any]]:
        """Get all earnings events for a specific date."""
        try:
            with self.db_manager.get_session() as session:
                rows = session.query(EarningsCalendar).filter(
                    EarningsCalendar.earnings_date == target_date
                ).order_by(EarningsCalendar.symbol).all()
                return [
                    {
                        'id': r.id, 'symbol': r.symbol,
                        'earnings_date': r.earnings_date,
                        'timing': r.timing,
                        'eps_estimate': float(r.eps_estimate) if r.eps_estimate else None,
                        'eps_actual': float(r.eps_actual) if r.eps_actual else None,
                        'eps_surprise_pct': float(r.eps_surprise_pct) if r.eps_surprise_pct else None,
                        'revenue_estimate': float(r.revenue_estimate) if r.revenue_estimate else None,
                        'revenue_actual': float(r.revenue_actual) if r.revenue_actual else None,
                        'revenue_surprise_pct': float(r.revenue_surprise_pct) if r.revenue_surprise_pct else None,
                        'price_change_1d_pct': float(r.price_change_1d_pct) if r.price_change_1d_pct else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting earnings for date {target_date}: {e}")
            return []

    # =========================================================================
    # Trade MFE/MAE Summary
    # =========================================================================

    def update_trade_mfe_mae(self, trade_id: int, data: Dict[str, Any]) -> bool:
        """Update the MFE/MAE summary columns on the trades table."""
        try:
            with self.db_manager.get_session() as session:
                trade = session.query(Trade).filter(Trade.id == trade_id).first()
                if not trade:
                    logger.warning(f"Trade {trade_id} not found for MFE/MAE update")
                    return False
                for key in (
                    'peak_mfe', 'peak_mae', 'peak_mfe_pct', 'peak_mae_pct',
                    'peak_mfe_r', 'peak_mae_r', 'bars_to_mfe', 'bars_held',
                ):
                    if key in data:
                        setattr(trade, key, data[key])
                logger.debug(f"Updated MFE/MAE for trade {trade_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating trade MFE/MAE for {trade_id}: {e}")
            return False


# =============================================================================
# Singleton
# =============================================================================

_ml_repository = None
_ml_repository_lock = threading.Lock()


def get_ml_repository() -> MLDataRepository:
    """Get or create the global ML data repository (thread-safe)."""
    global _ml_repository
    if _ml_repository is None:
        with _ml_repository_lock:
            if _ml_repository is None:
                _ml_repository = MLDataRepository()
    return _ml_repository
