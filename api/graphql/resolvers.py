"""
GraphQL Resolvers

Implements resolver functions for all GraphQL queries and mutations.
Uses existing database repositories and services.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger

from api.graphql.auth import require_auth, require_tier, require_feature


# =============================================================================
# Repository Access
# =============================================================================

def get_trades_repository():
    """Get trades repository with fallback"""
    try:
        from data.database import get_trades_repository as get_repo
        return get_repo()
    except ImportError:
        logger.warning("Trades repository not available")
        return None


def get_position_tracker():
    """Get position tracker with fallback"""
    try:
        from trading import get_position_tracker as get_tracker
        return get_tracker()
    except ImportError:
        logger.warning("Position tracker not available")
        return None


# =============================================================================
# Query Resolvers
# =============================================================================

class SignalResolver:
    """Resolver for signal queries"""

    @staticmethod
    def resolve_signals(
        info,
        limit: int = 50,
        offset: int = 0,
        direction: Optional[str] = None,
        strength: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        min_rrs: Optional[float] = None,
        max_rrs: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resolve signals with filtering.

        Args:
            limit: Maximum number of signals to return
            offset: Number of signals to skip
            direction: Filter by direction (long/short)
            strength: Filter by strength (strong/moderate/weak)
            symbol: Filter by symbol
            status: Filter by status
            min_rrs: Minimum RRS value
            max_rrs: Maximum RRS value

        Returns:
            List of signal dictionaries
        """
        repo = get_trades_repository()
        if not repo:
            return []

        # Get signals with basic filters
        signals = repo.get_signals(
            symbol=symbol,
            status=status if status else None,
            days=30,  # Default to last 30 days
            limit=limit + offset,  # Get extra for offset
        )

        # Apply additional filters
        filtered = signals

        if direction:
            direction_val = direction.value if hasattr(direction, 'value') else direction
            filtered = [s for s in filtered if s.get('direction') == direction_val]

        if strength:
            strength_val = strength.value if hasattr(strength, 'value') else strength
            filtered = [s for s in filtered if s.get('strength') == strength_val]

        if min_rrs is not None:
            filtered = [s for s in filtered if s.get('rrs', 0) >= min_rrs]

        if max_rrs is not None:
            filtered = [s for s in filtered if s.get('rrs', 0) <= max_rrs]

        # Apply offset and limit
        return filtered[offset:offset + limit]

    @staticmethod
    def resolve_signal_by_id(info, id: int) -> Optional[Dict[str, Any]]:
        """Resolve a single signal by ID"""
        repo = get_trades_repository()
        if not repo:
            return None

        signals = repo.get_signals(limit=1000)
        for signal in signals:
            if signal.get('id') == id:
                return signal
        return None


class PositionResolver:
    """Resolver for position queries"""

    @staticmethod
    def resolve_positions(
        info,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        profitable_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Resolve open positions with filtering.

        Args:
            status: Filter by status (currently only 'open' supported)
            symbol: Filter by symbol
            direction: Filter by direction
            profitable_only: Only return profitable positions

        Returns:
            List of position dictionaries
        """
        repo = get_trades_repository()
        if not repo:
            return []

        positions = repo.get_open_positions()

        # Apply filters
        filtered = positions

        if symbol:
            filtered = [p for p in filtered if p.get('symbol', '').upper() == symbol.upper()]

        if direction:
            direction_val = direction.value if hasattr(direction, 'value') else direction
            filtered = [p for p in filtered if p.get('direction') == direction_val]

        if profitable_only:
            filtered = [p for p in filtered if (p.get('pnl') or 0) > 0]

        return filtered

    @staticmethod
    def resolve_position_by_symbol(info, symbol: str) -> Optional[Dict[str, Any]]:
        """Resolve a single position by symbol"""
        repo = get_trades_repository()
        if not repo:
            return None

        return repo.get_position_by_symbol(symbol)


class TradeResolver:
    """Resolver for trade queries"""

    @staticmethod
    def resolve_trades(
        info,
        days: int = 30,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Resolve trades with filtering.

        Args:
            days: Number of days to look back
            symbol: Filter by symbol
            status: Filter by status
            direction: Filter by direction
            limit: Maximum trades to return
            offset: Number of trades to skip

        Returns:
            List of trade dictionaries
        """
        repo = get_trades_repository()
        if not repo:
            return []

        status_val = status.value if hasattr(status, 'value') else status
        direction_val = direction.value if hasattr(direction, 'value') else direction

        trades = repo.get_trades(
            symbol=symbol,
            status=status_val,
            direction=direction_val,
            days=days,
            limit=limit + offset,
        )

        return trades[offset:offset + limit]

    @staticmethod
    def resolve_trade_by_id(info, id: int) -> Optional[Dict[str, Any]]:
        """Resolve a single trade by ID"""
        repo = get_trades_repository()
        if not repo:
            return None

        return repo.get_trade_by_id(id)


class PortfolioResolver:
    """Resolver for portfolio queries"""

    @staticmethod
    def resolve_portfolio(info) -> Dict[str, Any]:
        """
        Resolve portfolio summary.

        Returns combined portfolio data including positions and performance.
        """
        repo = get_trades_repository()
        if not repo:
            return {
                'total_value': 0,
                'cash_available': 0,
                'buying_power': 0,
                'day_pnl': 0,
                'day_pnl_percent': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'positions_count': 0,
                'positions': [],
            }

        # Get open positions
        positions = repo.get_open_positions()

        # Calculate totals
        total_position_value = sum(
            (p.get('current_price') or p.get('entry_price', 0)) * p.get('shares', 0)
            for p in positions
        )
        total_unrealized_pnl = sum(p.get('pnl') or 0 for p in positions)

        # Get performance stats
        stats = repo.calculate_performance_stats(days=30)

        return {
            'total_value': total_position_value,
            'cash_available': 0,  # Would need broker integration
            'buying_power': 0,  # Would need broker integration
            'day_pnl': 0,  # Would need real-time calculation
            'day_pnl_percent': 0,
            'total_pnl': stats.get('total_pnl', 0),
            'total_pnl_percent': stats.get('total_return_pct', 0),
            'positions_count': len(positions),
            'positions': positions,
            'performance': stats,
        }


class MarketResolver:
    """Resolver for market status queries"""

    @staticmethod
    def resolve_market_status(info) -> Dict[str, Any]:
        """
        Resolve current market status.

        Returns market open/close status and trading hours info.
        """
        try:
            from utils.timezone import get_market_status, is_market_open, get_eastern_time

            status_str = get_market_status()
            is_open = is_market_open()
            current_time = get_eastern_time()

            return {
                'is_open': is_open,
                'status': status_str,
                'current_time': current_time.isoformat() if current_time else None,
                'next_open': None,  # Would need market calendar
                'next_close': None,
                'timezone': 'America/New_York',
            }
        except ImportError:
            return {
                'is_open': False,
                'status': 'unknown',
                'current_time': datetime.utcnow().isoformat(),
                'next_open': None,
                'next_close': None,
                'timezone': 'UTC',
            }


class ScannerResolver:
    """Resolver for scanner status queries"""

    @staticmethod
    def resolve_scanner_status(info) -> Dict[str, Any]:
        """
        Resolve scanner operational status.

        Returns scanner running state and statistics.
        """
        try:
            from scanner.realtime_scanner import RealTimeScanner
            scanner_available = True
        except ImportError:
            scanner_available = False

        repo = get_trades_repository()
        active_signals = 0
        if repo:
            signals = repo.get_active_signals(limit=100)
            active_signals = len(signals)

        return {
            'is_running': scanner_available,
            'last_scan': datetime.utcnow().isoformat(),
            'symbols_monitored': 175,  # Default watchlist size
            'active_signals': active_signals,
            'scan_interval_seconds': 300,  # 5 minutes
            'scanner_version': '1.0.0',
        }


class DailyStatsResolver:
    """Resolver for daily statistics queries"""

    @staticmethod
    def resolve_daily_stats(
        info,
        days: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resolve daily trading statistics.

        Args:
            days: Number of days to look back
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of daily stats dictionaries
        """
        repo = get_trades_repository()
        if not repo:
            return []

        # Get trades within the date range
        trades = repo.get_trades(days=days, status='closed', limit=1000)

        # Group by date and calculate stats
        from collections import defaultdict
        by_date = defaultdict(list)

        for trade in trades:
            exit_time = trade.get('exit_time')
            if exit_time:
                if isinstance(exit_time, str):
                    trade_date = exit_time[:10]
                else:
                    trade_date = exit_time.strftime('%Y-%m-%d')
                by_date[trade_date].append(trade)

        stats = []
        for date_str, day_trades in sorted(by_date.items(), reverse=True):
            winners = [t for t in day_trades if (t.get('pnl') or 0) > 0]
            losers = [t for t in day_trades if (t.get('pnl') or 0) <= 0]
            total_pnl = sum(t.get('pnl') or 0 for t in day_trades)

            stats.append({
                'date': date_str,
                'num_trades': len(day_trades),
                'winners': len(winners),
                'losers': len(losers),
                'pnl': total_pnl,
                'win_rate': len(winners) / len(day_trades) if day_trades else 0,
            })

        return stats[:days]


class UserResolver:
    """Resolver for user queries"""

    @staticmethod
    def resolve_me(info) -> Optional[Dict[str, Any]]:
        """
        Resolve current authenticated user.

        Returns user information from context.
        """
        context = info.context
        user = context.get('user')

        if not user:
            return None

        return {
            'id': user.user_id,
            'email': user.email,
            'tier': user.subscription_tier.value,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'is_active': user.is_active,
            'rate_limit': user.rate_limit,
            'features': context.get('features', {}),
        }


# =============================================================================
# Mutation Resolvers
# =============================================================================

class AlertMutationResolver:
    """Resolver for alert mutations"""

    @staticmethod
    def create_alert(info, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new price alert.

        Args:
            input: Alert input data

        Returns:
            Created alert data
        """
        context = info.context
        user = context.get('user')

        if not user:
            raise Exception("Authentication required")

        # Get database session
        try:
            from data.database import get_session
            from data.database.models import QueuedAlertRecord
            import uuid

            with get_session() as session:
                # Create alert record
                alert_record = QueuedAlertRecord(
                    alert_id=str(uuid.uuid4()),
                    user_id=user.user_id,
                    title=f"Price Alert: {input.get('symbol', '').upper()}",
                    message=input.get('message') or f"Price reached {input.get('condition')} ${input.get('price')}",
                    priority=input.get('priority', 'normal'),
                    alert_type='price_alert',
                    channels=input.get('notification_method', 'email'),
                    status='pending'
                )
                session.add(alert_record)
                session.commit()

                alert = {
                    'id': alert_record.id,
                    'alert_id': alert_record.alert_id,
                    'user_id': user.user_id,
                    'symbol': input.get('symbol', '').upper(),
                    'price': input.get('price'),
                    'condition': input.get('condition'),
                    'notification_method': input.get('notification_method', 'email'),
                    'message': alert_record.message,
                    'expires_at': input.get('expires_at'),
                    'created_at': alert_record.queued_at.isoformat() if alert_record.queued_at else datetime.utcnow().isoformat(),
                    'is_active': True,
                    'triggered': False,
                }

                logger.info(f"Created alert {alert_record.alert_id} for {alert['symbol']} at ${alert['price']}")
                return alert

        except ImportError:
            # Fallback to in-memory if database not available
            logger.warning("Database not available, using mock alert")
            alert = {
                'id': 1,
                'user_id': user.user_id,
                'symbol': input.get('symbol', '').upper(),
                'price': input.get('price'),
                'condition': input.get('condition'),
                'notification_method': input.get('notification_method', 'email'),
                'message': input.get('message'),
                'expires_at': input.get('expires_at'),
                'created_at': datetime.utcnow().isoformat(),
                'is_active': True,
                'triggered': False,
            }
            logger.info(f"Created alert for {alert['symbol']} at ${alert['price']}")
            return alert


class PositionMutationResolver:
    """Resolver for position mutations"""

    @staticmethod
    def close_position(
        info,
        symbol: str,
        price: float,
        exit_reason: str = 'manual',
    ) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            symbol: Symbol of position to close
            price: Exit price
            exit_reason: Reason for closing

        Returns:
            Closed trade data
        """
        context = info.context
        if not context.get('authenticated'):
            raise Exception("Authentication required")

        repo = get_trades_repository()
        if not repo:
            raise Exception("Database not available")

        # Close the trade
        result = repo.close_trade_by_symbol(
            symbol=symbol.upper(),
            exit_price=price,
            exit_reason=exit_reason,
        )

        if not result:
            raise Exception(f"No open position found for {symbol}")

        # Also remove from positions table
        repo.close_position(symbol.upper())

        logger.info(f"Closed position for {symbol} at ${price}")
        return result


class SettingsMutationResolver:
    """Resolver for settings mutations"""

    @staticmethod
    def update_settings(info, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user settings.

        Args:
            input: Settings input data

        Returns:
            Updated settings
        """
        context = info.context
        user = context.get('user')

        if not user:
            raise Exception("Authentication required")

        # Persist settings to database
        try:
            from data.database import get_session
            from data.database.models import AlertSchedule

            with get_session() as session:
                # Update or create alert schedule with settings
                schedule = session.query(AlertSchedule).filter_by(
                    user_id=user.user_id,
                    name='Default'
                ).first()

                # Build channels list based on settings
                channels = []
                if input.get('email_alerts', True):
                    channels.append('email')
                if input.get('push_alerts', False):
                    channels.append('pushover')
                if input.get('sms_alerts', False):
                    channels.append('sms')

                if schedule:
                    schedule.channels = ','.join(channels) if channels else None
                    schedule.updated_at = datetime.utcnow()
                else:
                    schedule = AlertSchedule(
                        user_id=user.user_id,
                        name='Default',
                        start_time='22:00',
                        end_time='07:00',
                        channels=','.join(channels) if channels else None,
                        is_active=True
                    )
                    session.add(schedule)

                session.commit()

                settings = {
                    'user_id': user.user_id,
                    'email_alerts': 'email' in channels,
                    'push_alerts': 'pushover' in channels,
                    'sms_alerts': 'sms' in channels,
                    'webhook_url': input.get('webhook_url'),
                    'default_position_size': input.get('default_position_size', 1000),
                    'risk_per_trade': input.get('risk_per_trade', 1.0),
                    'max_open_positions': input.get('max_open_positions', 5),
                    'auto_trailing_stop': input.get('auto_trailing_stop', False),
                    'trailing_stop_percent': input.get('trailing_stop_percent', 2.0),
                    'updated_at': datetime.utcnow().isoformat(),
                }

                logger.info(f"Updated settings for user {user.user_id}")
                return settings

        except ImportError:
            # Fallback if database not available
            logger.warning("Database not available, settings not persisted")
            settings = {
                'user_id': user.user_id,
                'email_alerts': input.get('email_alerts', True),
                'push_alerts': input.get('push_alerts', False),
                'sms_alerts': input.get('sms_alerts', False),
                'webhook_url': input.get('webhook_url'),
                'default_position_size': input.get('default_position_size', 1000),
                'risk_per_trade': input.get('risk_per_trade', 1.0),
                'max_open_positions': input.get('max_open_positions', 5),
                'auto_trailing_stop': input.get('auto_trailing_stop', False),
                'trailing_stop_percent': input.get('trailing_stop_percent', 2.0),
                'updated_at': datetime.utcnow().isoformat(),
            }
            logger.info(f"Updated settings for user {user.user_id}")
            return settings


# =============================================================================
# Risk Query Resolvers
# =============================================================================

class RiskResolver:
    """Resolver for risk-related queries"""

    @staticmethod
    def resolve_risk_status(info) -> Dict[str, Any]:
        """
        Resolve current risk status.

        Returns risk metrics and trading status.
        """
        try:
            from risk.risk_manager import RiskManager
            from risk.models import RiskLimits

            # Try to get global risk manager instance
            risk_manager = None
            try:
                from api.v1.routes import get_risk_manager_instance
                risk_manager = get_risk_manager_instance()
            except ImportError:
                pass

            if risk_manager is None:
                # Create a default one for query
                risk_manager = RiskManager(account_size=100000.0)

            return {
                'trading_halted': risk_manager.trading_halted,
                'halt_reason': risk_manager.halt_reason if risk_manager.trading_halted else None,
                'account_size': risk_manager.account_size,
                'current_balance': risk_manager.current_balance,
                'daily_pnl': risk_manager.daily_pnl,
                'daily_pnl_percent': (risk_manager.daily_pnl / risk_manager.daily_start_balance * 100)
                    if risk_manager.daily_start_balance > 0 else 0,
                'current_drawdown': risk_manager.current_drawdown,
                'max_drawdown': risk_manager.max_drawdown,
                'peak_balance': risk_manager.peak_balance,
                'daily_trades': risk_manager.daily_trades,
                'daily_wins': risk_manager.daily_wins,
                'daily_losses': risk_manager.daily_losses,
                'day_trades_today': risk_manager.day_trades_today,
                'open_positions': len(risk_manager.open_positions),
            }

        except ImportError:
            logger.warning("Risk manager not available")
            return {
                'trading_halted': False,
                'halt_reason': None,
                'account_size': 0,
                'current_balance': 0,
                'daily_pnl': 0,
                'daily_pnl_percent': 0,
                'current_drawdown': 0,
                'max_drawdown': 0,
                'peak_balance': 0,
                'daily_trades': 0,
                'daily_wins': 0,
                'daily_losses': 0,
                'day_trades_today': 0,
                'open_positions': 0,
            }

    @staticmethod
    def resolve_risk_limits(info) -> Dict[str, Any]:
        """
        Resolve current risk limits.
        """
        try:
            from risk.risk_manager import RiskManager
            from risk.models import RiskLimits

            risk_manager = None
            try:
                from api.v1.routes import get_risk_manager_instance
                risk_manager = get_risk_manager_instance()
            except ImportError:
                pass

            if risk_manager is None:
                limits = RiskLimits()
            else:
                limits = risk_manager.limits

            return {
                'max_position_size_pct': limits.max_position_size_pct,
                'max_daily_loss_pct': limits.max_daily_loss_pct,
                'max_drawdown_pct': limits.max_drawdown_pct,
                'max_daily_trades': limits.max_daily_trades,
                'max_open_positions': limits.max_open_positions,
                'min_risk_reward': limits.min_risk_reward,
            }

        except ImportError:
            return {
                'max_position_size_pct': 10.0,
                'max_daily_loss_pct': 2.0,
                'max_drawdown_pct': 10.0,
                'max_daily_trades': 10,
                'max_open_positions': 5,
                'min_risk_reward': 2.0,
            }


# =============================================================================
# ML Query Resolvers
# =============================================================================

class MLResolver:
    """Resolver for ML-related queries"""

    @staticmethod
    def resolve_market_regime(info) -> Dict[str, Any]:
        """
        Resolve current market regime.

        Returns regime prediction with confidence scores.
        """
        try:
            from ml.regime_detector import MarketRegimeDetector
            from pathlib import Path

            # Try to get global detector instance
            detector = None
            try:
                from api.v1.routes import _regime_detector
                detector = _regime_detector
            except ImportError:
                pass

            if detector is None:
                # Try to load from default path
                model_path = Path(__file__).parent.parent.parent / "models" / "regime_detector.pkl"
                if model_path.exists():
                    detector = MarketRegimeDetector()
                    detector.load_model(str(model_path))
                else:
                    return {
                        'regime': 'unknown',
                        'confidence_scores': {},
                        'strategy_allocation': {},
                        'model_loaded': False,
                    }

            regime, info_dict = detector.predict(return_confidence=True)

            return {
                'regime': regime,
                'confidence_scores': info_dict.get('confidence_scores', {}),
                'strategy_allocation': info_dict.get('strategy_allocation', {}),
                'timestamp': info_dict.get('timestamp', datetime.utcnow().isoformat()),
                'model_loaded': True,
            }

        except ImportError:
            logger.warning("ML components not available")
            return {
                'regime': 'unknown',
                'confidence_scores': {},
                'strategy_allocation': {},
                'model_loaded': False,
            }
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return {
                'regime': 'error',
                'confidence_scores': {},
                'strategy_allocation': {},
                'error': str(e),
                'model_loaded': False,
            }

    @staticmethod
    def resolve_ml_models_status(info) -> Dict[str, Any]:
        """
        Resolve ML models status.

        Returns status of all ML models.
        """
        from pathlib import Path

        model_dir = Path(__file__).parent.parent.parent / "models"

        models = {
            'regime_detector': {
                'file': model_dir / 'regime_detector.pkl',
                'type': 'HMM',
            },
            'xgboost': {
                'file': model_dir / 'xgboost_trade_classifier.pkl',
                'type': 'XGBoost',
            },
            'random_forest': {
                'file': model_dir / 'random_forest_classifier.pkl',
                'type': 'RandomForest',
            },
            'ensemble': {
                'file': model_dir / 'ensemble',
                'type': 'StackedEnsemble',
            },
        }

        status = {}
        for name, config in models.items():
            path = config['file']
            status[name] = {
                'type': config['type'],
                'file_exists': path.exists(),
            }
            if path.exists():
                if path.is_file():
                    status[name]['file_size_kb'] = path.stat().st_size / 1024
                    status[name]['modified'] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                elif path.is_dir():
                    status[name]['is_directory'] = True

        return {
            'models': status,
            'ml_available': True,
            'timestamp': datetime.utcnow().isoformat(),
        }
