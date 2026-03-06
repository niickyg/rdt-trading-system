"""
Dashboard data layer and session-auth API blueprint.

Provides helper functions that query the trading database via SQLAlchemy ORM,
plus a Flask Blueprint (/dashboard/api/*) for AJAX refresh from the browser
using session cookies (@login_required) instead of API keys.
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal

from flask import Blueprint, jsonify, request
from flask_login import login_required
from loguru import logger
from sqlalchemy import func, case, desc
from sqlalchemy.sql import literal_column

from data.database.connection import get_db_manager
from data.database.models import (
    Position, Trade, Signal, OptionsPosition, TradeStatus,
    RejectedSignal, DailyStats, EquitySnapshot, ParameterChange,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_python(val):
    """Convert SQLAlchemy types (Decimal, Enum, datetime) to JSON-safe Python types."""
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, datetime):
        return val.isoformat()
    if hasattr(val, 'value'):  # Enum
        return val.value
    return val


def _row_to_dict(obj, columns):
    """Convert an ORM model instance to a dict with JSON-safe values."""
    return {col: _to_python(getattr(obj, col, None)) for col in columns}


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------

def get_open_stock_positions(strategy=None):
    """Return open stock positions as list of dicts, optionally filtered by strategy."""
    columns = [
        'id', 'symbol', 'direction', 'entry_price', 'shares', 'entry_time',
        'stop_loss', 'take_profit', 'current_price', 'unrealized_pnl',
        'rrs_at_entry', 'strategy_name', 'updated_at',
    ]
    try:
        with get_db_manager().get_session() as session:
            q = session.query(Position)
            if strategy and strategy != 'all':
                q = q.filter(Position.strategy_name == strategy)
            rows = q.all()
            return [_row_to_dict(r, columns) for r in rows]
    except Exception as e:
        logger.warning(f"Error fetching stock positions: {e}")
        return []


def get_open_options_positions():
    """Return open options positions as list of dicts, with legs parsed."""
    columns = [
        'id', 'symbol', 'strategy_name', 'direction', 'contracts',
        'entry_time', 'entry_premium', 'total_premium', 'entry_iv',
        'entry_delta', 'legs_json', 'fill_details_json',
        'current_premium', 'unrealized_pnl', 'updated_at',
    ]
    try:
        with get_db_manager().get_session() as session:
            rows = session.query(OptionsPosition).all()
            result = []
            for r in rows:
                d = _row_to_dict(r, columns)
                # Parse legs_json -> legs
                raw_legs = d.pop('legs_json', '[]') or '[]'
                try:
                    d['legs'] = json.loads(raw_legs)
                except (json.JSONDecodeError, TypeError):
                    d['legs'] = []
                # Parse fill_details_json -> fill_details
                raw_fill = d.pop('fill_details_json', '[]') or '[]'
                try:
                    d['fill_details'] = json.loads(raw_fill)
                except (json.JSONDecodeError, TypeError):
                    d['fill_details'] = []
                result.append(d)
            return result
    except Exception as e:
        logger.warning(f"Error fetching options positions: {e}")
        return []


def get_closed_trades(days=30, direction=None, result=None, strategy=None):
    """Return closed trades within *days*, with optional direction/result/strategy filter."""
    columns = [
        'id', 'symbol', 'direction', 'entry_price', 'exit_price', 'shares',
        'entry_time', 'exit_time', 'pnl', 'pnl_percent', 'rrs_at_entry',
        'stop_loss', 'take_profit', 'exit_reason', 'vix_regime', 'market_regime',
        'strategy_name',
    ]
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with get_db_manager().get_session() as session:
            q = session.query(Trade).filter(
                Trade.status.in_([TradeStatus.CLOSED, 'closed', 'CLOSED']),
                Trade.exit_time >= cutoff,
            )
            if direction and direction != 'all':
                q = q.filter(func.upper(Trade.direction) == direction.upper())
            if result == 'win':
                q = q.filter(Trade.pnl > 0)
            elif result == 'loss':
                q = q.filter(Trade.pnl <= 0)
            if strategy and strategy != 'all':
                q = q.filter(Trade.strategy_name == strategy)
            q = q.order_by(desc(Trade.exit_time))
            rows = q.all()
            return [_row_to_dict(r, columns) for r in rows]
    except Exception as e:
        logger.warning(f"Error fetching closed trades: {e}")
        return []


def get_trade_stats():
    """Return aggregate trade stats: win_rate, total_pnl, profit_factor, total_trades."""
    try:
        with get_db_manager().get_session() as session:
            row = session.query(
                func.count().label('total'),
                func.sum(case((Trade.pnl > 0, 1), else_=0)).label('wins'),
                func.sum(Trade.pnl).label('total_pnl'),
                func.sum(case((Trade.pnl > 0, Trade.pnl), else_=0)).label('gross_profit'),
                func.sum(case((Trade.pnl <= 0, func.abs(Trade.pnl)), else_=0)).label('gross_loss'),
            ).filter(
                Trade.status.in_([TradeStatus.CLOSED, 'closed', 'CLOSED'])
            ).one()

            total = row.total or 0
            if total == 0:
                return {
                    'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                    'profit_factor': 0, 'avg_pnl': 0,
                }
            wins = row.wins or 0
            total_pnl = float(row.total_pnl or 0)
            gross_profit = float(row.gross_profit or 0)
            gross_loss = float(row.gross_loss or 0)
            return {
                'total_trades': total,
                'win_rate': round((wins / total) * 100, 1),
                'total_pnl': round(total_pnl, 2),
                'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
                'avg_pnl': round(total_pnl / total, 2),
            }
    except Exception as e:
        logger.warning(f"Error fetching trade stats: {e}")
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'profit_factor': 0, 'avg_pnl': 0}


def get_recent_signals(limit=50, strategy=None):
    """Return recent signals from the signals table, optionally filtered by strategy."""
    columns = [
        'id', 'symbol', 'timestamp', 'rrs', 'status', 'direction', 'price',
        'atr', 'daily_strong', 'daily_weak', 'volume', 'market_regime', 'strategy_name',
    ]
    try:
        with get_db_manager().get_session() as session:
            q = session.query(Signal)
            if strategy and strategy != 'all':
                q = q.filter(Signal.strategy_name == strategy)
            q = q.order_by(desc(Signal.timestamp)).limit(limit)
            rows = q.all()
            return [_row_to_dict(r, columns) for r in rows]
    except Exception as e:
        logger.warning(f"Error fetching signals: {e}")
        return []


def get_trade_stats_by_strategy():
    """Return per-strategy trade stats."""
    try:
        with get_db_manager().get_session() as session:
            rows = session.query(
                func.coalesce(Trade.strategy_name, 'rrs_momentum').label('strategy_name'),
                func.count().label('total'),
                func.sum(case((Trade.pnl > 0, 1), else_=0)).label('wins'),
                func.sum(Trade.pnl).label('total_pnl'),
                func.sum(case((Trade.pnl > 0, Trade.pnl), else_=0)).label('gross_profit'),
                func.sum(case((Trade.pnl <= 0, func.abs(Trade.pnl)), else_=0)).label('gross_loss'),
            ).filter(
                Trade.status.in_([TradeStatus.CLOSED, 'closed', 'CLOSED'])
            ).group_by(
                func.coalesce(Trade.strategy_name, 'rrs_momentum')
            ).all()

            result = {}
            for row in rows:
                total = row.total or 0
                wins = row.wins or 0
                total_pnl = float(row.total_pnl or 0)
                gross_profit = float(row.gross_profit or 0)
                gross_loss = float(row.gross_loss or 0)
                result[row.strategy_name] = {
                    'total_trades': total,
                    'win_rate': round((wins / total) * 100, 1) if total else 0,
                    'total_pnl': round(total_pnl, 2),
                    'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
                    'avg_pnl': round(total_pnl / total, 2) if total else 0,
                }
            return result
    except Exception as e:
        logger.warning(f"Error fetching per-strategy stats: {e}")
        return {}


def get_market_status():
    """Return market status based on current Eastern time."""
    try:
        from datetime import timezone
        import zoneinfo
        eastern = zoneinfo.ZoneInfo("America/New_York")
        now_et = datetime.now(eastern)
        weekday = now_et.weekday()  # 0=Mon, 6=Sun
        hour = now_et.hour
        minute = now_et.minute
        time_minutes = hour * 60 + minute

        # Market hours: Mon-Fri 9:30 AM - 4:00 PM ET
        if weekday < 5 and 570 <= time_minutes < 960:  # 9:30=570, 16:00=960
            return 'open'
        return 'closed'
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------
# Blueprint for AJAX refresh (/dashboard/api/*)
# ---------------------------------------------------------------------------

dashboard_data_bp = Blueprint('dashboard_data', __name__)


@dashboard_data_bp.route('/dashboard/api/positions')
@login_required
def api_positions():
    """JSON positions (stock + options) for AJAX refresh."""
    try:
        strategy = request.args.get('strategy', '')
        stock = get_open_stock_positions(strategy=strategy)
        options = get_open_options_positions()
        return jsonify({
            'stock_positions': stock,
            'options_positions': options,
            'total': len(stock) + len(options),
            'market_status': get_market_status(),
        })
    except Exception as e:
        logger.error(f"Error in api_positions: {e}")
        return jsonify({'error': 'Failed to fetch positions'}), 500


@dashboard_data_bp.route('/dashboard/api/trades')
@login_required
def api_trades():
    """JSON closed trades for AJAX refresh."""
    try:
        days = request.args.get('days', 30, type=int)
        direction = request.args.get('direction', '')
        result = request.args.get('result', '')
        strategy = request.args.get('strategy', '')
        trades = get_closed_trades(days=days, direction=direction, result=result, strategy=strategy)
        stats = get_trade_stats()
        strategy_stats = get_trade_stats_by_strategy()
        return jsonify({
            'trades': trades,
            'stats': stats,
            'strategy_stats': strategy_stats,
        })
    except Exception as e:
        logger.error(f"Error in api_trades: {e}")
        return jsonify({'error': 'Failed to fetch trades'}), 500


@dashboard_data_bp.route('/dashboard/api/signals')
@login_required
def api_signals():
    """JSON signals for AJAX refresh."""
    try:
        limit = request.args.get('limit', 50, type=int)
        strategy = request.args.get('strategy', '')
        signals = get_recent_signals(limit=min(limit, 200), strategy=strategy)
        return jsonify({
            'signals': signals,
            'total': len(signals),
        })
    except Exception as e:
        logger.error(f"Error in api_signals: {e}")
        return jsonify({'error': 'Failed to fetch signals'}), 500


@dashboard_data_bp.route('/dashboard/api/options')
@login_required
def api_options():
    """JSON options positions for AJAX refresh."""
    try:
        positions = get_open_options_positions()
        return jsonify({
            'positions': positions,
            'total': len(positions),
        })
    except Exception as e:
        logger.error(f"Error in api_options: {e}")
        return jsonify({'error': 'Failed to fetch options'}), 500


@dashboard_data_bp.route('/dashboard/api/strategies')
@login_required
def api_strategies():
    """Get status of all registered strategies."""
    try:
        from strategies.registry import StrategyRegistry
        strategies = []
        for name, strategy in StrategyRegistry.get_all().items():
            strategies.append({
                'name': name,
                'is_active': strategy.is_active,
                'capital_allocation': strategy.capital_allocation,
                'max_positions': strategy.max_positions,
                'open_positions': len(strategy.positions),
                'risk_per_trade': strategy.risk_per_trade,
            })
        return jsonify({'strategies': strategies})
    except Exception as e:
        logger.error(f"Error in api_strategies: {e}")
        return jsonify({'error': 'Failed to fetch strategies'}), 500


@dashboard_data_bp.route('/dashboard/api/agents')
@login_required
def api_agents():
    """Per-agent state, metrics, uptime, and errors from the orchestrator."""
    try:
        from agents.orchestrator import get_running_orchestrator
        orch = get_running_orchestrator()
        if orch is None:
            return jsonify({'agents': [], 'orchestrator_running': False})

        agents_info = []
        agent_refs = [
            ('Scanner', orch.scanner),
            ('Analyzer', orch.analyzer),
            ('Executor', orch.executor),
            ('AdaptiveLearner', orch.adaptive_learner),
            ('DailyStats', orch.daily_stats),
            ('OutcomeTracker', orch.outcome_tracker),
            ('DataCollection', orch.data_collection),
        ]
        for name, agent in agent_refs:
            if agent is None:
                agents_info.append({'name': name, 'state': 'not_started', 'events_processed': 0,
                                    'events_published': 0, 'errors': 0, 'last_activity': None, 'uptime': 0})
                continue
            m = agent.metrics
            info = {
                'name': name,
                'state': agent.state.value,
                'events_processed': m.events_processed,
                'events_published': m.events_published,
                'errors': m.errors,
                'last_activity': m.last_activity.isoformat() if m.last_activity else None,
                'uptime': m.uptime_seconds,
            }
            # Extra details for AdaptiveLearner
            if name == 'AdaptiveLearner' and hasattr(agent, 'get_current_parameters'):
                info['adaptive_params'] = agent.get_current_parameters()
            agents_info.append(info)

        return jsonify({'agents': agents_info, 'orchestrator_running': True})
    except Exception as e:
        logger.error(f"Error in api_agents: {e}")
        return jsonify({'error': 'Failed to fetch agents'}), 500


@dashboard_data_bp.route('/dashboard/api/strategies/detail')
@login_required
def api_strategies_detail():
    """Strategy config + DB trade stats + adaptive params."""
    try:
        from strategies.registry import StrategyRegistry
        from agents.orchestrator import get_running_orchestrator

        strategies = []
        all_strats = StrategyRegistry.get_all()
        trade_stats = get_trade_stats_by_strategy()

        # Get adaptive learner params + training phase if available
        adaptive_params = {}
        training_phase = {}
        orch = get_running_orchestrator()
        if orch and orch.adaptive_learner:
            if hasattr(orch.adaptive_learner, 'get_strategy_parameters'):
                for name in all_strats:
                    try:
                        adaptive_params[name] = orch.adaptive_learner.get_strategy_parameters(name)
                    except Exception:
                        pass
            if hasattr(orch.adaptive_learner, 'get_training_phase_status'):
                for name in all_strats:
                    try:
                        training_phase[name] = orch.adaptive_learner.get_training_phase_status(name)
                    except Exception:
                        pass

        for name, strategy in all_strats.items():
            stats = trade_stats.get(name, {})
            strat_info = {
                'name': name,
                'is_active': strategy.is_active,
                'capital_allocation': strategy.capital_allocation,
                'max_positions': strategy.max_positions,
                'open_positions': len(strategy.positions),
                'risk_per_trade': strategy.risk_per_trade,
                'stats': stats,
                'adaptive_params': adaptive_params.get(name, {}),
                'training_phase': training_phase.get(name),
            }
            strategies.append(strat_info)

        return jsonify({'strategies': strategies})
    except Exception as e:
        logger.error(f"Error in api_strategies_detail: {e}")
        return jsonify({'error': 'Failed to fetch strategy details'}), 500


@dashboard_data_bp.route('/dashboard/api/system/status')
@login_required
def api_system_status():
    """Thin wrapper on Orchestrator.get_system_status()."""
    try:
        from agents.orchestrator import get_running_orchestrator
        orch = get_running_orchestrator()
        if orch is None:
            return jsonify({'running': False, 'status': {}})
        return jsonify({'running': True, 'status': orch.get_system_status()})
    except Exception as e:
        logger.error(f"Error in api_system_status: {e}")
        return jsonify({'error': 'Failed to fetch system status'}), 500


@dashboard_data_bp.route('/dashboard/api/overview')
@login_required
def api_overview():
    """Aggregated overview: stats + strategy breakdown + agent health."""
    try:
        from strategies.registry import StrategyRegistry
        from agents.orchestrator import get_running_orchestrator

        stats = get_trade_stats()
        strategy_stats = get_trade_stats_by_strategy()

        # Strategy breakdown
        strategy_breakdown = []
        for name, strategy in StrategyRegistry.get_all().items():
            strategy_breakdown.append({
                'name': name,
                'is_active': strategy.is_active,
                'capital_allocation': strategy.capital_allocation,
                'max_positions': strategy.max_positions,
                'open_positions': len(strategy.positions),
                'stats': strategy_stats.get(name, {}),
            })

        # Agent health
        orch = get_running_orchestrator()
        agent_health = []
        orchestrator_running = orch is not None
        if orch:
            for aname, agent in [('Scanner', orch.scanner), ('Analyzer', orch.analyzer),
                                  ('Executor', orch.executor), ('AdaptiveLearner', orch.adaptive_learner),
                                  ('DailyStats', orch.daily_stats), ('OutcomeTracker', orch.outcome_tracker),
                                  ('DataCollection', orch.data_collection)]:
                agent_health.append({
                    'name': aname,
                    'state': agent.state.value if agent else 'not_started',
                })

        return jsonify({
            'stats': stats,
            'strategy_breakdown': strategy_breakdown,
            'agent_health': agent_health,
            'orchestrator_running': orchestrator_running,
            'market_status': get_market_status(),
        })
    except Exception as e:
        logger.error(f"Error in api_overview: {e}")
        return jsonify({'error': 'Failed to fetch overview'}), 500


@dashboard_data_bp.route('/dashboard/api/strategies/<strategy_name>/toggle', methods=['POST'])
@login_required
def api_toggle_strategy(strategy_name):
    """Enable or disable a strategy at runtime."""
    try:
        from strategies.registry import StrategyRegistry
        from agents.orchestrator import get_running_orchestrator

        strategy = StrategyRegistry.get(strategy_name)
        if not strategy:
            return jsonify({'error': f"Strategy '{strategy_name}' not found"}), 404

        orch = get_running_orchestrator()
        if orch is None:
            return jsonify({'error': 'Orchestrator is not running. Changes would not persist.'}), 503

        data = request.get_json(silent=True) or {}
        # If 'active' is provided, set it; otherwise toggle
        if 'active' in data:
            strategy.is_active = bool(data['active'])
        else:
            strategy.is_active = not strategy.is_active

        logger.info(f"Strategy '{strategy_name}' {'enabled' if strategy.is_active else 'disabled'} via dashboard")
        return jsonify({
            'name': strategy_name,
            'is_active': strategy.is_active,
        })
    except Exception as e:
        logger.error(f"Error in api_toggle_strategy: {e}")
        return jsonify({'error': 'Failed to toggle strategy'}), 500


# ---------------------------------------------------------------------------
# AI Signal Confidence endpoints
# ---------------------------------------------------------------------------

@dashboard_data_bp.route('/dashboard/api/confidence/overview')
@login_required
def api_confidence_overview():
    """Aggregate ML confidence stats and rejection summary."""
    try:
        with get_db_manager().get_session() as session:
            # --- Confidence buckets from closed trades ---
            closed_filter = Trade.status.in_([TradeStatus.CLOSED, 'closed', 'CLOSED'])
            has_conf = Trade.ml_confidence.isnot(None)

            trades_with_conf = session.query(func.count()).filter(closed_filter, has_conf).scalar() or 0
            avg_conf = session.query(func.avg(Trade.ml_confidence)).filter(closed_filter, has_conf).scalar()
            avg_conf = round(float(avg_conf), 2) if avg_conf else 0

            buckets = []
            for lo, hi, label in [(60, 70, '60-70'), (70, 80, '70-80'), (80, 90, '80-90'), (90, 101, '90+')]:
                row = session.query(
                    func.count().label('cnt'),
                    func.sum(case((Trade.pnl > 0, 1), else_=0)).label('wins'),
                ).filter(
                    closed_filter, has_conf,
                    Trade.ml_confidence >= lo,
                    Trade.ml_confidence < hi,
                ).one()
                cnt = row.cnt or 0
                wins = row.wins or 0
                buckets.append({
                    'range': label, 'count': cnt, 'wins': wins,
                    'win_rate': round((wins / cnt) * 100, 1) if cnt else 0,
                })

            # --- Rejection summary ---
            total_rejected = session.query(func.count()).select_from(RejectedSignal).scalar() or 0
            avg_whpnl = session.query(func.avg(RejectedSignal.would_have_pnl_1d)).scalar()
            avg_whpnl = round(float(avg_whpnl), 4) if avg_whpnl else None

            # Top rejection reasons — parse JSON arrays and count
            reason_rows = session.query(RejectedSignal.rejection_reasons).all()
            reason_counts: dict[str, int] = {}
            for (raw,) in reason_rows:
                try:
                    reasons = json.loads(raw) if raw else []
                except (json.JSONDecodeError, TypeError):
                    reasons = [raw] if raw else []
                for r in reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
            top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return jsonify({
            'trades_with_confidence': trades_with_conf,
            'avg_confidence': avg_conf,
            'confidence_buckets': buckets,
            'rejection_summary': {
                'total': total_rejected,
                'top_reasons': [{'reason': r, 'count': c} for r, c in top_reasons],
                'avg_would_have_pnl_1d': avg_whpnl,
            },
        })
    except Exception as e:
        logger.error(f"Error in api_confidence_overview: {e}")
        return jsonify({'error': 'Failed to fetch confidence overview'}), 500


@dashboard_data_bp.route('/dashboard/api/confidence/rejected')
@login_required
def api_confidence_rejected():
    """Rejected signals detail list."""
    try:
        days = request.args.get('days', 30, type=int)
        strategy = request.args.get('strategy', '')
        cutoff = datetime.utcnow() - timedelta(days=days)

        columns = [
            'id', 'symbol', 'direction', 'rrs', 'price', 'timestamp',
            'rejection_reasons', 'ml_probability', 'ml_confidence',
            'strategy_name', 'would_have_pnl_1h', 'would_have_pnl_4h', 'would_have_pnl_1d',
        ]
        with get_db_manager().get_session() as session:
            q = session.query(RejectedSignal).filter(RejectedSignal.timestamp >= cutoff)
            if strategy and strategy != 'all':
                q = q.filter(RejectedSignal.strategy_name == strategy)
            q = q.order_by(desc(RejectedSignal.timestamp)).limit(100)
            rows = q.all()
            signals = []
            for r in rows:
                d = _row_to_dict(r, columns)
                # Parse rejection_reasons JSON
                raw = d.get('rejection_reasons', '[]') or '[]'
                try:
                    d['rejection_reasons'] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d['rejection_reasons'] = [raw] if raw else []
                signals.append(d)

        return jsonify({'signals': signals})
    except Exception as e:
        logger.error(f"Error in api_confidence_rejected: {e}")
        return jsonify({'error': 'Failed to fetch rejected signals'}), 500


@dashboard_data_bp.route('/dashboard/api/confidence/params')
@login_required
def api_confidence_params():
    """Parameter change history from adaptive learner."""
    try:
        days = request.args.get('days', 90, type=int)
        cutoff = datetime.utcnow() - timedelta(days=days)
        columns = ['id', 'timestamp', 'parameter_name', 'old_value', 'new_value', 'reason']
        with get_db_manager().get_session() as session:
            rows = session.query(ParameterChange).filter(
                ParameterChange.timestamp >= cutoff,
            ).order_by(desc(ParameterChange.timestamp)).limit(200).all()
            return jsonify({'changes': [_row_to_dict(r, columns) for r in rows]})
    except Exception as e:
        logger.error(f"Error in api_confidence_params: {e}")
        return jsonify({'error': 'Failed to fetch parameter changes'}), 500


# ---------------------------------------------------------------------------
# Trading Journal endpoints
# ---------------------------------------------------------------------------

@dashboard_data_bp.route('/dashboard/api/journal/equity')
@login_required
def api_journal_equity():
    """Equity curve snapshots."""
    try:
        days = request.args.get('days', 90, type=int)
        cutoff = datetime.utcnow() - timedelta(days=days)
        columns = ['id', 'timestamp', 'equity_value', 'drawdown_pct', 'high_water_mark',
                    'cash', 'positions_value', 'open_positions_count']
        with get_db_manager().get_session() as session:
            rows = session.query(EquitySnapshot).filter(
                EquitySnapshot.timestamp >= cutoff,
            ).order_by(EquitySnapshot.timestamp).all()
            return jsonify({'snapshots': [_row_to_dict(r, columns) for r in rows]})
    except Exception as e:
        logger.error(f"Error in api_journal_equity: {e}")
        return jsonify({'error': 'Failed to fetch equity data'}), 500


@dashboard_data_bp.route('/dashboard/api/journal/daily')
@login_required
def api_journal_daily():
    """Daily P&L statistics."""
    try:
        days = request.args.get('days', 90, type=int)
        cutoff = datetime.utcnow() - timedelta(days=days)
        columns = ['id', 'date', 'pnl', 'pnl_percent', 'num_trades', 'winners', 'losers',
                    'win_rate', 'largest_win', 'largest_loss', 'market_regime']
        with get_db_manager().get_session() as session:
            rows = session.query(DailyStats).filter(
                DailyStats.date >= cutoff.date(),
            ).order_by(DailyStats.date).all()
            result = []
            for r in rows:
                d = _row_to_dict(r, columns)
                # date objects need isoformat
                if hasattr(d.get('date'), 'isoformat'):
                    d['date'] = d['date'].isoformat()
                result.append(d)
            return jsonify({'days': result})
    except Exception as e:
        logger.error(f"Error in api_journal_daily: {e}")
        return jsonify({'error': 'Failed to fetch daily stats'}), 500


@dashboard_data_bp.route('/dashboard/api/journal/trades')
@login_required
def api_journal_trades():
    """Extended closed trade data with all columns for journal analytics."""
    try:
        days = request.args.get('days', 90, type=int)
        strategy = request.args.get('strategy', '')
        cutoff = datetime.utcnow() - timedelta(days=days)
        columns = [
            'id', 'symbol', 'direction', 'entry_price', 'exit_price', 'shares',
            'entry_time', 'exit_time', 'pnl', 'pnl_percent', 'rrs_at_entry',
            'stop_loss', 'take_profit', 'exit_reason', 'vix_regime', 'market_regime',
            'strategy_name', 'notes', 'ml_confidence', 'bars_held',
            'peak_mfe', 'peak_mae', 'peak_mfe_pct', 'peak_mae_pct',
            'peak_mfe_r', 'peak_mae_r', 'bars_to_mfe',
            'sector_name', 'spy_trend',
        ]
        with get_db_manager().get_session() as session:
            q = session.query(Trade).filter(
                Trade.status.in_([TradeStatus.CLOSED, 'closed', 'CLOSED']),
                Trade.exit_time >= cutoff,
            )
            if strategy and strategy != 'all':
                q = q.filter(Trade.strategy_name == strategy)
            q = q.order_by(desc(Trade.exit_time))
            rows = q.all()
            return jsonify({'trades': [_row_to_dict(r, columns) for r in rows]})
    except Exception as e:
        logger.error(f"Error in api_journal_trades: {e}")
        return jsonify({'error': 'Failed to fetch journal trades'}), 500


@dashboard_data_bp.route('/dashboard/api/journal/trades/<int:trade_id>/notes', methods=['PUT'])
@login_required
def api_journal_trade_notes(trade_id):
    """Update trade notes."""
    try:
        data = request.get_json(silent=True) or {}
        notes = data.get('notes', '')
        with get_db_manager().get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                return jsonify({'error': 'Trade not found'}), 404
            trade.notes = notes
            session.commit()
        return jsonify({'success': True, 'trade_id': trade_id, 'notes': notes})
    except Exception as e:
        logger.error(f"Error in api_journal_trade_notes: {e}")
        return jsonify({'error': 'Failed to update trade notes'}), 500
