"""
Trades Repository for the RDT Trading System.

Provides CRUD operations for trades and positions with database persistence.
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from .connection import get_db_manager
from .models import (
    Trade, Position, Signal, TradeDirection, TradeStatus, ExitReason, SignalStatus,
    OptionsPosition, OptionsTrade,
    DailyStats, RejectedSignal, EquitySnapshot, ParameterChange,
)


class TradesRepository:
    """Repository for trade and position database operations."""

    def __init__(self):
        self._db_manager = None

    @property
    def db_manager(self):
        """Lazy-load database manager."""
        if self._db_manager is None:
            self._db_manager = get_db_manager()
            # Ensure tables exist
            self._db_manager.create_tables()
        return self._db_manager

    def save_trade(self, trade_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save a new trade to the database.

        Args:
            trade_data: Dictionary containing trade information:
                - symbol: Stock ticker symbol
                - direction: 'long' or 'short'
                - entry_price: Entry price
                - shares: Number of shares
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
                - rrs_at_entry: RRS value at entry (optional)
                - entry_time: Entry timestamp (optional, defaults to now)

        Returns:
            Dictionary with the created trade data or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                # Map direction string to enum
                direction = TradeDirection.LONG if trade_data.get('direction') == 'long' else TradeDirection.SHORT

                trade = Trade(
                    symbol=trade_data['symbol'].upper(),
                    direction=direction,
                    entry_price=float(trade_data['entry_price']),
                    shares=int(trade_data['shares']),
                    entry_time=trade_data.get('entry_time', datetime.utcnow()),
                    stop_loss=trade_data.get('stop_loss'),
                    take_profit=trade_data.get('take_profit'),
                    rrs_at_entry=trade_data.get('rrs_at_entry') or trade_data.get('rrs'),
                    status=TradeStatus.OPEN,
                    # Filter evaluation metadata
                    vix_regime=trade_data.get('vix_regime'),
                    vix_value=trade_data.get('vix_value'),
                    market_regime=trade_data.get('market_regime'),
                    sector_name=trade_data.get('sector_name'),
                    sector_rs=trade_data.get('sector_rs'),
                    spy_trend=trade_data.get('spy_trend'),
                    ml_confidence=trade_data.get('ml_confidence'),
                    signal_strategy=trade_data.get('signal_strategy'),
                    news_sentiment=trade_data.get('news_sentiment'),
                    news_warning=trade_data.get('news_warning'),
                    regime_rrs_threshold=trade_data.get('regime_rrs_threshold'),
                    regime_stop_multiplier=trade_data.get('regime_stop_multiplier'),
                    regime_target_multiplier=trade_data.get('regime_target_multiplier'),
                    vix_position_size_mult=trade_data.get('vix_position_size_mult'),
                    sector_boost=trade_data.get('sector_boost'),
                    first_hour_filtered=trade_data.get('first_hour_filtered'),
                    strategy_name=trade_data.get('strategy_name', 'rrs_momentum'),
                )
                session.add(trade)
                session.flush()  # Get the ID
                trade_id = trade.id
                logger.info(f"Saved trade {trade_id}: {trade.direction.value} {trade.shares} {trade.symbol} @ ${trade.entry_price:.2f}")
                # Return dictionary representation to avoid detached instance issues
                return self._trade_to_dict(trade)
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return None

    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        direction: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trades with optional filters.

        Args:
            symbol: Filter by stock symbol
            status: Filter by status ('open', 'closed', 'cancelled')
            direction: Filter by direction ('long', 'short')
            days: Filter by trades within last N days
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(Trade)

                # Apply filters
                if symbol:
                    query = query.filter(Trade.symbol == symbol.upper())

                if status:
                    status_enum = TradeStatus(status)
                    query = query.filter(Trade.status == status_enum)

                if direction:
                    direction_enum = TradeDirection(direction)
                    query = query.filter(Trade.direction == direction_enum)

                if days:
                    cutoff = datetime.utcnow() - timedelta(days=days)
                    query = query.filter(Trade.entry_time >= cutoff)

                # Order by most recent first
                query = query.order_by(desc(Trade.entry_time)).limit(limit)

                trades = query.all()
                return [self._trade_to_dict(trade) for trade in trades]
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def get_trade_by_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Get a single trade by ID."""
        try:
            with self.db_manager.get_session() as session:
                trade = session.query(Trade).filter(Trade.id == trade_id).first()
                if trade:
                    return self._trade_to_dict(trade)
                return None
        except Exception as e:
            logger.error(f"Error getting trade {trade_id}: {e}")
            return None

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        exit_time: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Close a trade and calculate PnL.

        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price
            exit_reason: Reason for exit ('stop_loss', 'take_profit', 'trailing_stop', 'manual', 'end_of_day')
            exit_time: Exit timestamp (defaults to now)

        Returns:
            Updated trade dictionary or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                trade = session.query(Trade).filter(Trade.id == trade_id).first()
                if not trade:
                    logger.warning(f"Trade {trade_id} not found")
                    return None

                if trade.status != TradeStatus.OPEN:
                    logger.warning(f"Trade {trade_id} is not open (status: {trade.status})")
                    return None

                trade.exit_price = exit_price
                trade.exit_time = exit_time or datetime.utcnow()
                trade.exit_reason = ExitReason(exit_reason)
                trade.status = TradeStatus.CLOSED

                # Calculate PnL
                if trade.direction == TradeDirection.LONG:
                    trade.pnl = (exit_price - trade.entry_price) * trade.shares
                else:  # SHORT
                    trade.pnl = (trade.entry_price - exit_price) * trade.shares

                trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.shares)) * 100

                logger.info(f"Closed trade {trade_id}: PnL=${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
                session.flush()
                return self._trade_to_dict(trade)
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return None

    def close_trade_by_symbol(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        exit_time: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Close an open trade by symbol.

        Args:
            symbol: Stock symbol
            exit_price: Exit price
            exit_reason: Reason for exit
            exit_time: Exit timestamp (defaults to now)

        Returns:
            Updated trade dictionary or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                trade = session.query(Trade).filter(
                    and_(
                        Trade.symbol == symbol.upper(),
                        Trade.status == TradeStatus.OPEN
                    )
                ).first()

                if not trade:
                    logger.warning(f"No open trade found for {symbol}")
                    return None

                trade.exit_price = exit_price
                trade.exit_time = exit_time or datetime.utcnow()
                trade.exit_reason = ExitReason(exit_reason)
                trade.status = TradeStatus.CLOSED

                # Calculate PnL
                if trade.direction == TradeDirection.LONG:
                    trade.pnl = (exit_price - trade.entry_price) * trade.shares
                else:  # SHORT
                    trade.pnl = (trade.entry_price - exit_price) * trade.shares

                trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.shares)) * 100

                logger.info(f"Closed trade for {symbol}: PnL=${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
                session.flush()
                return self._trade_to_dict(trade)
        except Exception as e:
            logger.error(f"Error closing trade for {symbol}: {e}")
            return None

    # =========================================================================
    # Position Operations
    # =========================================================================

    def save_position(self, position_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save or update a position in the database.

        If a position for the symbol already exists, it will be updated.

        Args:
            position_data: Dictionary containing position information

        Returns:
            Dictionary with the created/updated position data or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                symbol = position_data['symbol'].upper()
                direction = TradeDirection.LONG if position_data.get('direction') == 'long' else TradeDirection.SHORT

                # Check if position exists
                position = session.query(Position).filter(Position.symbol == symbol).first()

                if position:
                    # Update existing position
                    position.direction = direction
                    position.entry_price = float(position_data['entry_price'])
                    position.shares = int(position_data['shares'])
                    position.stop_loss = position_data.get('stop_loss') or position_data.get('stop_price')
                    position.take_profit = position_data.get('take_profit') or position_data.get('target_price')
                    position.current_price = position_data.get('current_price')
                    position.unrealized_pnl = position_data.get('unrealized_pnl')
                    position.rrs_at_entry = position_data.get('rrs_at_entry') or position_data.get('rrs')
                    if position_data.get('strategy_name'):
                        position.strategy_name = position_data['strategy_name']
                    logger.info(f"Updated position: {symbol}")
                else:
                    # Create new position
                    position = Position(
                        symbol=symbol,
                        direction=direction,
                        entry_price=float(position_data['entry_price']),
                        shares=int(position_data['shares']),
                        entry_time=position_data.get('entry_time', datetime.utcnow()),
                        stop_loss=position_data.get('stop_loss') or position_data.get('stop_price'),
                        take_profit=position_data.get('take_profit') or position_data.get('target_price'),
                        current_price=position_data.get('current_price'),
                        unrealized_pnl=position_data.get('unrealized_pnl'),
                        rrs_at_entry=position_data.get('rrs_at_entry') or position_data.get('rrs'),
                        strategy_name=position_data.get('strategy_name', 'rrs_momentum'),
                    )
                    session.add(position)
                    logger.info(f"Created position: {direction.value} {position.shares} {symbol} @ ${position.entry_price:.2f}")

                session.flush()
                # Return dictionary representation to avoid detached instance issues
                return self._position_to_dict(position)
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.

        Returns:
            List of position dictionaries
        """
        try:
            with self.db_manager.get_session() as session:
                positions = session.query(Position).order_by(desc(Position.entry_time)).all()
                return [self._position_to_dict(pos) for pos in positions]
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get a position by symbol. Raises on DB error (caller must handle)."""
        with self.db_manager.get_session() as session:
            position = session.query(Position).filter(Position.symbol == symbol.upper()).first()
            if position:
                return self._position_to_dict(position)
            return None

    def update_position(self, position_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a position by ID.

        Args:
            position_id: Position ID
            data: Dictionary of fields to update

        Returns:
            Updated position dictionary or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    logger.warning(f"Position {position_id} not found")
                    return None

                # Update allowed fields
                if 'current_price' in data:
                    position.current_price = float(data['current_price'])
                if 'unrealized_pnl' in data:
                    position.unrealized_pnl = float(data['unrealized_pnl'])
                if 'stop_loss' in data:
                    position.stop_loss = float(data['stop_loss'])
                if 'take_profit' in data:
                    position.take_profit = float(data['take_profit'])
                if 'shares' in data:
                    position.shares = int(data['shares'])

                session.flush()
                logger.info(f"Updated position {position_id}")
                return self._position_to_dict(position)
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {e}")
            return None

    def update_position_price(self, symbol: str, current_price: float, unrealized_pnl: float) -> bool:
        """Update current_price and unrealized_pnl for an open position by symbol."""
        try:
            with self.db_manager.get_session() as session:
                position = session.query(Position).filter(
                    Position.symbol == symbol
                ).first()
                if not position:
                    return False
                position.current_price = current_price
                position.unrealized_pnl = unrealized_pnl
                session.flush()
                return True
        except Exception as e:
            logger.debug(f"Error updating position price for {symbol}: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """
        Close (delete) a position by symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if closed successfully, False otherwise
        """
        try:
            with self.db_manager.get_session() as session:
                position = session.query(Position).filter(Position.symbol == symbol.upper()).first()
                if position:
                    session.delete(position)
                    logger.info(f"Closed position for {symbol}")
                    return True
                logger.warning(f"No position found for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False

    # =========================================================================
    # Options Position Operations
    # =========================================================================

    def save_options_position(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save or update an options position (upsert by symbol).

        Args:
            data: Dict with symbol, strategy_name, direction, contracts,
                  entry_premium, total_premium, entry_iv, entry_delta,
                  order_ids, legs_json, fill_details_json, entry_time.

        Returns:
            Dictionary representation or None on failure.
        """
        try:
            with self.db_manager.get_session() as session:
                symbol = data['symbol'].upper()
                position = session.query(OptionsPosition).filter(
                    OptionsPosition.symbol == symbol
                ).first()

                if position:
                    position.strategy_name = data['strategy_name']
                    position.direction = data['direction']
                    position.contracts = int(data['contracts'])
                    position.entry_premium = float(data.get('entry_premium', 0))
                    position.total_premium = float(data.get('total_premium', 0))
                    position.entry_iv = data.get('entry_iv')
                    position.entry_delta = data.get('entry_delta')
                    position.order_ids = data.get('order_ids')
                    position.legs_json = data['legs_json']
                    position.fill_details_json = data.get('fill_details_json')
                    logger.info(f"Updated options position: {symbol}")
                else:
                    position = OptionsPosition(
                        symbol=symbol,
                        strategy_name=data['strategy_name'],
                        direction=data['direction'],
                        contracts=int(data['contracts']),
                        entry_time=data.get('entry_time', datetime.utcnow()),
                        entry_premium=float(data.get('entry_premium', 0)),
                        total_premium=float(data.get('total_premium', 0)),
                        entry_iv=data.get('entry_iv'),
                        entry_delta=data.get('entry_delta'),
                        order_ids=data.get('order_ids'),
                        legs_json=data['legs_json'],
                        fill_details_json=data.get('fill_details_json'),
                    )
                    session.add(position)
                    logger.info(f"Created options position: {data['strategy_name']} {symbol} x{data['contracts']}")

                session.flush()
                return self._options_position_to_dict(position)
        except Exception as e:
            logger.error(f"Error saving options position: {e}")
            return None

    def get_all_options_positions(self) -> List[Dict[str, Any]]:
        """Get all open options positions."""
        try:
            with self.db_manager.get_session() as session:
                positions = session.query(OptionsPosition).order_by(
                    OptionsPosition.entry_time.desc()
                ).all()
                return [self._options_position_to_dict(p) for p in positions]
        except Exception as e:
            logger.error(f"Error getting options positions: {e}")
            return []

    def get_options_positions(self) -> List[Dict[str, Any]]:
        """Backward-compatible alias used by agents expecting this method name."""
        return self.get_all_options_positions()

    def get_options_position_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get an options position by underlying symbol."""
        try:
            with self.db_manager.get_session() as session:
                position = session.query(OptionsPosition).filter(
                    OptionsPosition.symbol == symbol.upper()
                ).first()
                if position:
                    return self._options_position_to_dict(position)
                return None
        except Exception as e:
            logger.error(f"Error getting options position for {symbol}: {e}")
            return None

    def close_options_position(self, symbol: str) -> bool:
        """Delete an options position by symbol."""
        try:
            with self.db_manager.get_session() as session:
                position = session.query(OptionsPosition).filter(
                    OptionsPosition.symbol == symbol.upper()
                ).first()
                if position:
                    session.delete(position)
                    logger.info(f"Closed options position for {symbol}")
                    return True
                logger.warning(f"No options position found for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error closing options position for {symbol}: {e}")
            return False

    def save_options_trade(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save an options trade record (historical).

        Args:
            data: Dict with symbol, strategy_name, direction, contracts,
                  entry/exit details, pnl, legs_json, etc.

        Returns:
            Dictionary representation or None on failure.
        """
        try:
            with self.db_manager.get_session() as session:
                trade = OptionsTrade(
                    symbol=data['symbol'].upper(),
                    strategy_name=data['strategy_name'],
                    direction=data['direction'],
                    contracts=int(data['contracts']),
                    entry_time=data['entry_time'],
                    exit_time=data.get('exit_time'),
                    entry_premium=float(data.get('entry_premium', 0)),
                    total_premium=float(data.get('total_premium', 0)),
                    exit_premium=data.get('exit_premium'),
                    pnl=data.get('pnl'),
                    pnl_percent=data.get('pnl_percent'),
                    entry_iv=data.get('entry_iv'),
                    exit_iv=data.get('exit_iv'),
                    entry_delta=data.get('entry_delta'),
                    legs_json=data['legs_json'],
                    fill_details_json=data.get('fill_details_json'),
                    exit_reason=data.get('exit_reason'),
                    status=TradeStatus(data.get('status', 'closed')),
                    order_ids=data.get('order_ids'),
                )
                session.add(trade)
                session.flush()
                logger.info(f"Saved options trade: {data['strategy_name']} {data['symbol']} P&L=${data.get('pnl', 0):.2f}")
                return self._options_trade_to_dict(trade)
        except Exception as e:
            logger.error(f"Error saving options trade: {e}")
            return None

    def get_options_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get options trades with optional filters."""
        try:
            with self.db_manager.get_session() as session:
                query = session.query(OptionsTrade)
                if symbol:
                    query = query.filter(OptionsTrade.symbol == symbol.upper())
                if status:
                    query = query.filter(OptionsTrade.status == TradeStatus(status))
                if days:
                    cutoff = datetime.utcnow() - timedelta(days=days)
                    query = query.filter(OptionsTrade.entry_time >= cutoff)
                query = query.order_by(OptionsTrade.entry_time.desc()).limit(limit)
                return [self._options_trade_to_dict(t) for t in query.all()]
        except Exception as e:
            logger.error(f"Error getting options trades: {e}")
            return []

    def _options_position_to_dict(self, pos: 'OptionsPosition') -> Dict[str, Any]:
        """Convert an OptionsPosition to a dictionary."""
        return {
            'id': pos.id,
            'symbol': pos.symbol,
            'strategy_name': pos.strategy_name,
            'direction': pos.direction,
            'contracts': pos.contracts,
            'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
            'entry_premium': float(pos.entry_premium) if pos.entry_premium else 0.0,
            'total_premium': float(pos.total_premium) if pos.total_premium else 0.0,
            'entry_iv': float(pos.entry_iv) if pos.entry_iv is not None else None,
            'entry_delta': float(pos.entry_delta) if pos.entry_delta is not None else None,
            'order_ids': pos.order_ids,
            'legs_json': pos.legs_json,
            'fill_details_json': pos.fill_details_json,
            'current_premium': float(pos.current_premium) if pos.current_premium is not None else None,
            'unrealized_pnl': float(pos.unrealized_pnl) if pos.unrealized_pnl is not None else None,
            'updated_at': pos.updated_at.isoformat() if pos.updated_at else None,
        }

    def update_options_position_price(self, symbol: str, current_premium: float, unrealized_pnl: float) -> bool:
        """Update current_premium and unrealized_pnl for an options position."""
        try:
            with self.db_manager.get_session() as session:
                position = session.query(OptionsPosition).filter(
                    OptionsPosition.symbol == symbol
                ).first()
                if not position:
                    return False
                position.current_premium = current_premium
                position.unrealized_pnl = unrealized_pnl
                session.flush()
                return True
        except Exception as e:
            logger.debug(f"Error updating options position price for {symbol}: {e}")
            return False

    def _options_trade_to_dict(self, trade: 'OptionsTrade') -> Dict[str, Any]:
        """Convert an OptionsTrade to a dictionary."""
        return {
            'id': trade.id,
            'symbol': trade.symbol,
            'strategy_name': trade.strategy_name,
            'direction': trade.direction,
            'contracts': trade.contracts,
            'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'entry_premium': float(trade.entry_premium) if trade.entry_premium else 0.0,
            'total_premium': float(trade.total_premium) if trade.total_premium else 0.0,
            'exit_premium': float(trade.exit_premium) if trade.exit_premium is not None else None,
            'pnl': float(trade.pnl) if trade.pnl is not None else None,
            'pnl_percent': float(trade.pnl_percent) if trade.pnl_percent is not None else None,
            'entry_iv': float(trade.entry_iv) if trade.entry_iv is not None else None,
            'exit_iv': float(trade.exit_iv) if trade.exit_iv is not None else None,
            'entry_delta': float(trade.entry_delta) if trade.entry_delta is not None else None,
            'legs_json': trade.legs_json,
            'fill_details_json': trade.fill_details_json,
            'exit_reason': trade.exit_reason,
            'status': trade.status.value if trade.status else None,
            'order_ids': trade.order_ids,
        }

    # =========================================================================
    # Signal Operations
    # =========================================================================

    def save_signal(self, signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save a new signal to the database.

        Args:
            signal_data: Dictionary containing signal information

        Returns:
            Dictionary with the created signal data or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                direction = TradeDirection.LONG if signal_data.get('direction') == 'long' else TradeDirection.SHORT

                signal = Signal(
                    symbol=signal_data['symbol'].upper(),
                    timestamp=signal_data.get('timestamp', datetime.utcnow()),
                    rrs=float(signal_data['rrs']),
                    status=SignalStatus.PENDING,
                    direction=direction,
                    price=float(signal_data['price']),
                    atr=signal_data.get('atr'),
                    daily_strong=signal_data.get('daily_strong'),
                    daily_weak=signal_data.get('daily_weak')
                )
                session.add(signal)
                session.flush()
                logger.info(f"Saved signal: {signal.symbol} {direction.value} RRS={signal.rrs:.2f}")
                # Return dictionary representation to avoid detached instance issues
                return self._signal_to_dict(signal)
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return None

    def get_active_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get pending/active signals.

        Returns:
            List of signal dictionaries
        """
        try:
            with self.db_manager.get_session() as session:
                signals = session.query(Signal).filter(
                    Signal.status == SignalStatus.PENDING
                ).order_by(desc(Signal.timestamp)).limit(limit).all()
                return [self._signal_to_dict(sig) for sig in signals]
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []

    def get_signals(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get signals with optional filters.

        Args:
            symbol: Filter by stock symbol
            status: Filter by status ('pending', 'triggered', 'expired', 'ignored')
            days: Filter by signals within last N days
            limit: Maximum number of signals to return

        Returns:
            List of signal dictionaries
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(Signal)

                if symbol:
                    query = query.filter(Signal.symbol == symbol.upper())

                if status:
                    status_enum = SignalStatus(status)
                    query = query.filter(Signal.status == status_enum)

                if days:
                    cutoff = datetime.utcnow() - timedelta(days=days)
                    query = query.filter(Signal.timestamp >= cutoff)

                query = query.order_by(desc(Signal.timestamp)).limit(limit)
                signals = query.all()
                return [self._signal_to_dict(sig) for sig in signals]
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []

    def update_signal_status(self, signal_id: int, status: str) -> bool:
        """Update a signal's status."""
        try:
            with self.db_manager.get_session() as session:
                signal = session.query(Signal).filter(Signal.id == signal_id).first()
                if signal:
                    signal.status = SignalStatus(status)
                    logger.info(f"Updated signal {signal_id} status to {status}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating signal {signal_id}: {e}")
            return False

    # =========================================================================
    # Rejected Signal Operations
    # =========================================================================

    def save_rejected_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save a rejected signal with full context for outcome analysis.

        Args:
            data: Dictionary containing rejected signal information

        Returns:
            Dictionary with the created record or None if failed
        """
        try:
            import json
            with self.db_manager.get_session() as session:
                direction = TradeDirection.LONG if data.get('direction') == 'long' else TradeDirection.SHORT
                reasons = data.get('rejection_reasons', [])
                if isinstance(reasons, list):
                    reasons = json.dumps(reasons)

                record = RejectedSignal(
                    symbol=data['symbol'].upper(),
                    direction=direction,
                    rrs=float(data.get('rrs') or 0),
                    price=float(data.get('price') or 0),
                    timestamp=data.get('timestamp', datetime.utcnow()),
                    rejection_reasons=reasons,
                    market_regime=data.get('market_regime'),
                    daily_strong=data.get('daily_strong'),
                    daily_weak=data.get('daily_weak'),
                    atr=data.get('atr'),
                    volume=data.get('volume'),
                    ml_probability=data.get('ml_probability'),
                    ml_confidence=data.get('ml_confidence'),
                )
                session.add(record)
                session.flush()
                logger.info(f"Saved rejected signal: {record.symbol} {direction.value} ({len(data.get('rejection_reasons', []))} reasons)")
                return {
                    'id': record.id,
                    'symbol': record.symbol,
                    'direction': record.direction.value,
                    'rrs': float(record.rrs),
                    'price': float(record.price),
                    'timestamp': record.timestamp.isoformat() if record.timestamp else None,
                    'rejection_reasons': reasons,
                }
        except Exception as e:
            logger.error(f"Error saving rejected signal: {e}")
            return None

    def get_rejected_signals_pending_outcome(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get rejected signals from the last N hours that still need outcome tracking.

        Args:
            hours: Look back window in hours

        Returns:
            List of rejected signal dictionaries
        """
        try:
            with self.db_manager.get_session() as session:
                cutoff = datetime.utcnow() - timedelta(hours=hours)
                records = session.query(RejectedSignal).filter(
                    and_(
                        RejectedSignal.timestamp >= cutoff,
                        RejectedSignal.price_after_1d.is_(None),
                    )
                ).order_by(RejectedSignal.timestamp).all()

                return [{
                    'id': r.id,
                    'symbol': r.symbol,
                    'direction': r.direction.value if r.direction else None,
                    'rrs': float(r.rrs) if r.rrs else 0,
                    'price': float(r.price) if r.price else 0,
                    'timestamp': r.timestamp.isoformat() if r.timestamp else None,
                    'rejection_reasons': r.rejection_reasons,
                    'price_after_1h': r.price_after_1h,
                    'price_after_4h': r.price_after_4h,
                    'price_after_1d': r.price_after_1d,
                } for r in records]
        except Exception as e:
            logger.error(f"Error getting pending rejected signals: {e}")
            return []

    def update_rejected_signal_outcome(self, signal_id: int, data: Dict[str, Any]) -> bool:
        """
        Update a rejected signal with price outcome data.

        Args:
            signal_id: ID of the rejected signal
            data: Dict with price_after_1h, price_after_4h, price_after_1d, etc.

        Returns:
            True if updated successfully
        """
        try:
            with self.db_manager.get_session() as session:
                record = session.query(RejectedSignal).filter(
                    RejectedSignal.id == signal_id
                ).first()
                if not record:
                    return False

                if 'price_after_1h' in data:
                    record.price_after_1h = data['price_after_1h']
                if 'price_after_4h' in data:
                    record.price_after_4h = data['price_after_4h']
                if 'price_after_1d' in data:
                    record.price_after_1d = data['price_after_1d']
                if 'would_have_pnl_1h' in data:
                    record.would_have_pnl_1h = data['would_have_pnl_1h']
                if 'would_have_pnl_4h' in data:
                    record.would_have_pnl_4h = data['would_have_pnl_4h']
                if 'would_have_pnl_1d' in data:
                    record.would_have_pnl_1d = data['would_have_pnl_1d']

                session.flush()
                return True
        except Exception as e:
            logger.error(f"Error updating rejected signal outcome {signal_id}: {e}")
            return False

    # =========================================================================
    # Daily Stats Operations
    # =========================================================================

    def save_daily_stats(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Upsert daily trading statistics.

        Args:
            data: Dictionary with date, starting_balance, ending_balance, pnl, etc.

        Returns:
            Dictionary with the created/updated record or None if failed
        """
        try:
            from datetime import date as date_type
            with self.db_manager.get_session() as session:
                stats_date = data.get('date')
                if isinstance(stats_date, str):
                    stats_date = datetime.fromisoformat(stats_date).date()
                elif isinstance(stats_date, datetime):
                    stats_date = stats_date.date()

                # Upsert: check if record exists for this date
                record = session.query(DailyStats).filter(
                    DailyStats.date == stats_date
                ).first()

                if record:
                    # Update existing
                    record.starting_balance = data.get('starting_balance', record.starting_balance)
                    record.ending_balance = data.get('ending_balance', record.ending_balance)
                    record.pnl = data.get('pnl', record.pnl)
                    record.pnl_percent = data.get('pnl_percent', record.pnl_percent)
                    record.num_trades = data.get('num_trades', record.num_trades)
                    record.winners = data.get('winners', record.winners)
                    record.losers = data.get('losers', record.losers)
                    record.win_rate = data.get('win_rate', record.win_rate)
                    record.avg_win = data.get('avg_win', record.avg_win)
                    record.avg_loss = data.get('avg_loss', record.avg_loss)
                    record.largest_win = data.get('largest_win', record.largest_win)
                    record.largest_loss = data.get('largest_loss', record.largest_loss)
                    record.market_regime = data.get('market_regime', record.market_regime)
                    logger.info(f"Updated daily stats for {stats_date}")
                else:
                    # Create new
                    record = DailyStats(
                        date=stats_date,
                        starting_balance=data.get('starting_balance', 0),
                        ending_balance=data.get('ending_balance', 0),
                        pnl=data.get('pnl', 0),
                        pnl_percent=data.get('pnl_percent'),
                        num_trades=data.get('num_trades', 0),
                        winners=data.get('winners', 0),
                        losers=data.get('losers', 0),
                        win_rate=data.get('win_rate'),
                        avg_win=data.get('avg_win'),
                        avg_loss=data.get('avg_loss'),
                        largest_win=data.get('largest_win'),
                        largest_loss=data.get('largest_loss'),
                        market_regime=data.get('market_regime'),
                    )
                    session.add(record)
                    logger.info(f"Created daily stats for {stats_date}")

                session.flush()
                return {
                    'id': record.id,
                    'date': record.date.isoformat(),
                    'pnl': float(record.pnl) if record.pnl else 0,
                    'num_trades': record.num_trades,
                    'win_rate': float(record.win_rate) if record.win_rate else None,
                }
        except Exception as e:
            logger.error(f"Error saving daily stats: {e}")
            return None

    def get_latest_daily_stats(self) -> Optional[Dict[str, Any]]:
        """Get the most recent daily stats record."""
        try:
            with self.db_manager.get_session() as session:
                record = session.query(DailyStats).order_by(
                    desc(DailyStats.date)
                ).first()
                if not record:
                    return None
                return {
                    'id': record.id,
                    'date': record.date.isoformat(),
                    'starting_balance': float(record.starting_balance),
                    'ending_balance': float(record.ending_balance),
                    'pnl': float(record.pnl),
                    'num_trades': record.num_trades,
                }
        except Exception as e:
            logger.error(f"Error getting latest daily stats: {e}")
            return None

    # =========================================================================
    # Equity Snapshot Operations
    # =========================================================================

    def save_equity_snapshot(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save an equity snapshot.

        Args:
            data: Dictionary with equity_value, cash, positions_value, etc.

        Returns:
            Dictionary with the created record or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                record = EquitySnapshot(
                    timestamp=data.get('timestamp', datetime.utcnow()),
                    equity_value=float(data['equity_value']),
                    cash=data.get('cash'),
                    positions_value=data.get('positions_value'),
                    open_positions_count=data.get('open_positions_count', 0),
                    drawdown_pct=data.get('drawdown_pct'),
                    high_water_mark=data.get('high_water_mark'),
                )
                session.add(record)
                session.flush()
                logger.debug(f"Saved equity snapshot: ${record.equity_value:.2f}")
                return {
                    'id': record.id,
                    'timestamp': record.timestamp.isoformat(),
                    'equity_value': float(record.equity_value),
                    'drawdown_pct': float(record.drawdown_pct) if record.drawdown_pct else None,
                }
        except Exception as e:
            logger.error(f"Error saving equity snapshot: {e}")
            return None

    def get_high_water_mark(self) -> float:
        """Get the highest equity value ever recorded."""
        try:
            from sqlalchemy import func
            with self.db_manager.get_session() as session:
                result = session.query(func.max(EquitySnapshot.equity_value)).scalar()
                return float(result) if result else 0.0
        except Exception as e:
            logger.error(f"Error getting high water mark: {e}")
            return 0.0

    # =========================================================================
    # Parameter Change Operations
    # =========================================================================

    def save_parameter_change(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save a parameter change record.

        Args:
            data: Dictionary with parameter_name, old_value, new_value, reason, etc.

        Returns:
            Dictionary with the created record or None if failed
        """
        try:
            with self.db_manager.get_session() as session:
                record = ParameterChange(
                    timestamp=data.get('timestamp', datetime.utcnow()),
                    parameter_name=data['parameter_name'],
                    old_value=float(data['old_value']),
                    new_value=float(data['new_value']),
                    reason=data.get('reason', 'adaptive_adjustment'),
                    trade_count_basis=data.get('trade_count_basis'),
                    win_rate_at_change=data.get('win_rate_at_change'),
                    regime=data.get('regime'),
                )
                session.add(record)
                session.flush()
                logger.debug(
                    f"Saved parameter change: {record.parameter_name} "
                    f"{record.old_value} -> {record.new_value}"
                )
                return {
                    'id': record.id,
                    'parameter_name': record.parameter_name,
                    'old_value': float(record.old_value),
                    'new_value': float(record.new_value),
                    'reason': record.reason,
                }
        except Exception as e:
            logger.error(f"Error saving parameter change: {e}")
            return None

    # =========================================================================
    # Performance Statistics
    # =========================================================================

    def calculate_performance_stats(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate trading performance statistics from closed trades.

        Args:
            days: Number of days to look back (None for all time)

        Returns:
            Dictionary of performance statistics
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(Trade).filter(Trade.status == TradeStatus.CLOSED)

                if days:
                    cutoff = datetime.utcnow() - timedelta(days=days)
                    query = query.filter(Trade.exit_time >= cutoff)

                trades = query.all()

                if not trades:
                    return {
                        'total_signals': 0,
                        'wins': 0,
                        'losses': 0,
                        'win_rate': 0.0,
                        'avg_win_pct': 0.0,
                        'avg_loss_pct': 0.0,
                        'profit_factor': 0.0,
                        'total_return_pct': 0.0,
                        'max_drawdown_pct': 0.0
                    }

                # Calculate stats
                total_trades = len(trades)
                winners = [t for t in trades if t.pnl and t.pnl > 0]
                losers = [t for t in trades if t.pnl and t.pnl <= 0]

                num_winners = len(winners)
                num_losers = len(losers)
                win_rate = num_winners / total_trades if total_trades > 0 else 0

                # Average win/loss percentages
                avg_win_pct = sum(t.pnl_percent for t in winners) / num_winners if winners else 0
                avg_loss_pct = sum(t.pnl_percent for t in losers) / num_losers if losers else 0

                # Total PnL
                total_pnl = sum(t.pnl for t in trades if t.pnl)
                gross_profit = sum(t.pnl for t in winners if t.pnl)
                gross_loss = abs(sum(t.pnl for t in losers if t.pnl))

                # Profit factor
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

                # Estimate total return (simplified - would need account balance history for accuracy)
                # Using sum of percentage returns as approximation
                total_return_pct = sum(t.pnl_percent for t in trades if t.pnl_percent)

                return {
                    'total_signals': total_trades,
                    'wins': num_winners,
                    'losses': num_losers,
                    'win_rate': round(win_rate, 2),
                    'avg_win_pct': round(avg_win_pct, 2),
                    'avg_loss_pct': round(avg_loss_pct, 2),
                    'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
                    'total_return_pct': round(total_return_pct, 2),
                    'max_drawdown_pct': 0.0,  # Would need equity curve to calculate
                    'total_pnl': round(total_pnl, 2) if total_pnl else 0.0
                }
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'profit_factor': 0.0,
                'total_return_pct': 0.0,
                'max_drawdown_pct': 0.0
            }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert a Trade object to a dictionary."""
        return {
            'id': trade.id,
            'symbol': trade.symbol,
            'direction': trade.direction.value if trade.direction else None,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'shares': trade.shares,
            'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'rrs_at_entry': trade.rrs_at_entry,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'status': trade.status.value if trade.status else None,
            'exit_reason': trade.exit_reason.value if trade.exit_reason else None,
            # Filter evaluation metadata
            'vix_regime': trade.vix_regime,
            'vix_value': float(trade.vix_value) if trade.vix_value is not None else None,
            'market_regime': trade.market_regime,
            'sector_name': trade.sector_name,
            'sector_rs': float(trade.sector_rs) if trade.sector_rs is not None else None,
            'spy_trend': trade.spy_trend,
            'ml_confidence': float(trade.ml_confidence) if trade.ml_confidence is not None else None,
            'signal_strategy': trade.signal_strategy,
            'news_sentiment': float(trade.news_sentiment) if trade.news_sentiment is not None else None,
            'news_warning': trade.news_warning,
            'regime_rrs_threshold': float(trade.regime_rrs_threshold) if trade.regime_rrs_threshold is not None else None,
            'regime_stop_multiplier': float(trade.regime_stop_multiplier) if trade.regime_stop_multiplier is not None else None,
            'regime_target_multiplier': float(trade.regime_target_multiplier) if trade.regime_target_multiplier is not None else None,
            'vix_position_size_mult': float(trade.vix_position_size_mult) if trade.vix_position_size_mult is not None else None,
            'sector_boost': float(trade.sector_boost) if trade.sector_boost is not None else None,
            'first_hour_filtered': trade.first_hour_filtered,
        }

    def _position_to_dict(self, position: Position) -> Dict[str, Any]:
        """Convert a Position object to a dictionary."""
        # Calculate unrealized PnL if we have current price
        pnl = position.unrealized_pnl
        pnl_pct = None

        if position.current_price and position.entry_price:
            if position.direction == TradeDirection.LONG:
                pnl = (position.current_price - position.entry_price) * position.shares
            else:
                pnl = (position.entry_price - position.current_price) * position.shares
            pnl_pct = (pnl / (position.entry_price * position.shares)) * 100

        return {
            'id': position.id,
            'symbol': position.symbol,
            'direction': position.direction.value if position.direction else None,
            'entry_price': position.entry_price,
            'shares': position.shares,
            'entry_time': position.entry_time.isoformat() if position.entry_time else None,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'current_price': position.current_price,
            'pnl': pnl,
            'pnl_pct': round(pnl_pct, 2) if pnl_pct is not None else None,
            'rrs_at_entry': position.rrs_at_entry
        }

    def _signal_to_dict(self, signal: Signal) -> Dict[str, Any]:
        """Convert a Signal object to a dictionary."""
        return {
            'id': signal.id,
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat() if signal.timestamp else None,
            'generated_at': signal.timestamp.isoformat() if signal.timestamp else None,
            'rrs': signal.rrs,
            'status': signal.status.value if signal.status else None,
            'direction': signal.direction.value if signal.direction else None,
            'price': signal.price,
            'entry_price': signal.price,  # Alias for API compatibility
            'atr': signal.atr,
            'daily_strong': signal.daily_strong,
            'daily_weak': signal.daily_weak,
            'strength': 'strong' if signal.daily_strong else 'moderate' if not signal.daily_weak else 'weak',
            'strategy': 'RRS_Momentum'
        }


# Global repository instance
_trades_repository = None
_trades_repository_lock = threading.Lock()


def get_trades_repository() -> TradesRepository:
    """Get or create the global trades repository (thread-safe)."""
    global _trades_repository
    if _trades_repository is None:
        with _trades_repository_lock:
            if _trades_repository is None:
                _trades_repository = TradesRepository()
    return _trades_repository
