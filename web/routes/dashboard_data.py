"""
Dashboard data layer and session-auth API blueprint.

Provides helper functions that query the trading database via SQLAlchemy ORM,
plus a Flask Blueprint (/dashboard/api/*) for AJAX refresh from the browser
using session cookies (@login_required) instead of API keys.
"""

import json
import os
import fcntl
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

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
            positions = [_row_to_dict(r, columns) for r in rows]

            # Enrich with previous close from daily_bars for today's P&L calc
            if positions:
                from sqlalchemy import text
                # Build lookup with both IBKR format (spaces) and dash format
                # since positions may use "BF B" but daily_bars has "BF-B"
                all_symbols = set()
                for p in positions:
                    sym = p['symbol']
                    all_symbols.add(sym)
                    all_symbols.add(sym.replace(' ', '-'))  # IBKR→dash
                    all_symbols.add(sym.replace('-', ' '))  # dash→IBKR
                all_symbols = list(all_symbols)
                placeholders = ','.join([f':s{i}' for i in range(len(all_symbols))])
                prev_close_query = text(f"""
                    SELECT DISTINCT ON (symbol) symbol, close
                    FROM daily_bars
                    WHERE symbol IN ({placeholders})
                    ORDER BY symbol, bar_date DESC
                """)
                params = {f's{i}': sym for i, sym in enumerate(all_symbols)}
                try:
                    prev_rows = session.execute(prev_close_query, params).fetchall()
                    # Map both space and dash variants to the close price
                    prev_map = {}
                    for r in prev_rows:
                        val = float(r[1])
                        prev_map[r[0]] = val
                        prev_map[r[0].replace('-', ' ')] = val
                        prev_map[r[0].replace(' ', '-')] = val
                    for p in positions:
                        p['previous_close'] = prev_map.get(p['symbol'])
                except Exception:
                    pass

            return positions
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
        all_strats = StrategyRegistry.get_all()

        if all_strats:
            for name, strategy in all_strats.items():
                strategies.append({
                    'name': name,
                    'is_active': strategy.is_active,
                    'capital_allocation': strategy.capital_allocation,
                    'max_positions': strategy.max_positions,
                    'open_positions': len(strategy.positions),
                    'risk_per_trade': strategy.risk_per_trade,
                })
        else:
            # Bot not running — return known strategies with DB stats
            trade_stats = get_trade_stats_by_strategy()
            KNOWN_STRATEGIES = {
                'rrs_momentum': {'capital_allocation': 0.40, 'max_positions': 6, 'risk_per_trade': 0.01},
                'rsi2_mean_reversion': {'capital_allocation': 0.20, 'max_positions': 4, 'risk_per_trade': 0.01},
                'trend_breakout': {'capital_allocation': 0.20, 'max_positions': 4, 'risk_per_trade': 0.015},
                'pead': {'capital_allocation': 0.10, 'max_positions': 3, 'risk_per_trade': 0.01},
                'gap_fill': {'capital_allocation': 0.10, 'max_positions': 3, 'risk_per_trade': 0.01},
            }
            for name, defaults in KNOWN_STRATEGIES.items():
                strategies.append({
                    'name': name,
                    'is_active': False,
                    'capital_allocation': defaults['capital_allocation'],
                    'max_positions': defaults['max_positions'],
                    'open_positions': 0,
                    'risk_per_trade': defaults['risk_per_trade'],
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

        if all_strats:
            # Bot is running — use live strategy objects
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
        elif trade_stats:
            # Bot not running but we have DB trade history — show stats-only cards
            KNOWN_STRATEGIES = {
                'rrs_momentum': {'capital_allocation': 0.40, 'max_positions': 6, 'risk_per_trade': 0.01},
                'rsi2_mean_reversion': {'capital_allocation': 0.20, 'max_positions': 4, 'risk_per_trade': 0.01},
                'trend_breakout': {'capital_allocation': 0.20, 'max_positions': 4, 'risk_per_trade': 0.015},
                'pead': {'capital_allocation': 0.10, 'max_positions': 3, 'risk_per_trade': 0.01},
                'gap_fill': {'capital_allocation': 0.10, 'max_positions': 3, 'risk_per_trade': 0.01},
            }
            for name, stats in trade_stats.items():
                defaults = KNOWN_STRATEGIES.get(name, {'capital_allocation': 0, 'max_positions': 0, 'risk_per_trade': 0})
                strategies.append({
                    'name': name,
                    'is_active': False,
                    'capital_allocation': defaults['capital_allocation'],
                    'max_positions': defaults['max_positions'],
                    'open_positions': 0,
                    'risk_per_trade': defaults['risk_per_trade'],
                    'stats': stats,
                    'adaptive_params': {},
                    'training_phase': None,
                })

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
        all_strats = StrategyRegistry.get_all()
        if all_strats:
            for name, strategy in all_strats.items():
                strategy_breakdown.append({
                    'name': name,
                    'is_active': strategy.is_active,
                    'capital_allocation': strategy.capital_allocation,
                    'max_positions': strategy.max_positions,
                    'open_positions': len(strategy.positions),
                    'stats': strategy_stats.get(name, {}),
                })
        elif strategy_stats:
            # Bot not running — show DB stats
            KNOWN_STRATEGIES = {
                'rrs_momentum': {'capital_allocation': 0.40, 'max_positions': 6},
                'rsi2_mean_reversion': {'capital_allocation': 0.20, 'max_positions': 4},
                'trend_breakout': {'capital_allocation': 0.20, 'max_positions': 4},
                'pead': {'capital_allocation': 0.10, 'max_positions': 3},
                'gap_fill': {'capital_allocation': 0.10, 'max_positions': 3},
            }
            for name, s_stats in strategy_stats.items():
                defaults = KNOWN_STRATEGIES.get(name, {'capital_allocation': 0, 'max_positions': 0})
                strategy_breakdown.append({
                    'name': name,
                    'is_active': False,
                    'capital_allocation': defaults['capital_allocation'],
                    'max_positions': defaults['max_positions'],
                    'open_positions': 0,
                    'stats': s_stats,
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


# ---------------------------------------------------------------------------
# User Price Alerts (file-backed JSON store)
# ---------------------------------------------------------------------------

_ALERTS_FILE = Path(os.environ.get('DATA_DIR', 'data')) / 'user_alerts.json'


def _read_alerts() -> list:
    """Read alerts from JSON file with file locking."""
    if not _ALERTS_FILE.exists():
        return []
    try:
        with open(_ALERTS_FILE, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data
    except (json.JSONDecodeError, IOError):
        return []


def _write_alerts(alerts: list):
    """Write alerts to JSON file with file locking."""
    _ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_ALERTS_FILE, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(alerts, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)


@dashboard_data_bp.route('/dashboard/api/alerts')
@login_required
def api_alerts_list():
    """List all user price alerts."""
    try:
        alerts = _read_alerts()
        return jsonify({'alerts': alerts})
    except Exception as e:
        logger.error(f"Error listing alerts: {e}")
        return jsonify({'error': 'Failed to list alerts'}), 500


@dashboard_data_bp.route('/dashboard/api/alerts', methods=['POST'])
@login_required
def api_alerts_create():
    """Create a new price alert."""
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or '').strip().upper()
        condition = data.get('condition', '')
        value = data.get('value')
        if not symbol or not condition or value is None:
            return jsonify({'error': 'symbol, condition, and value are required'}), 400

        alerts = _read_alerts()
        new_alert = {
            'id': int(datetime.utcnow().timestamp() * 1000),
            'symbol': symbol,
            'condition': condition,
            'value': float(value),
            'notification_method': data.get('notification_method', 'email'),
            'note': data.get('note', ''),
            'status': 'active',
            'created_at': datetime.utcnow().isoformat() + 'Z',
        }
        alerts.insert(0, new_alert)
        _write_alerts(alerts)
        return jsonify({'alert': new_alert}), 201
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        return jsonify({'error': 'Failed to create alert'}), 500


@dashboard_data_bp.route('/dashboard/api/alerts/<int:alert_id>', methods=['DELETE'])
@login_required
def api_alerts_delete(alert_id):
    """Delete a price alert."""
    try:
        alerts = _read_alerts()
        alerts = [a for a in alerts if a.get('id') != alert_id]
        _write_alerts(alerts)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting alert: {e}")
        return jsonify({'error': 'Failed to delete alert'}), 500


@dashboard_data_bp.route('/dashboard/api/alerts/<int:alert_id>/toggle', methods=['POST'])
@login_required
def api_alerts_toggle(alert_id):
    """Toggle alert active/paused status."""
    try:
        alerts = _read_alerts()
        for a in alerts:
            if a.get('id') == alert_id:
                a['status'] = 'paused' if a.get('status') == 'active' else 'active'
                _write_alerts(alerts)
                return jsonify({'alert': a})
        return jsonify({'error': 'Alert not found'}), 404
    except Exception as e:
        logger.error(f"Error toggling alert: {e}")
        return jsonify({'error': 'Failed to toggle alert'}), 500


# ---------------------------------------------------------------------------
# Settings (file-backed JSON store)
# ---------------------------------------------------------------------------

_SETTINGS_FILE = Path(os.environ.get('DATA_DIR', 'data')) / 'user_settings.json'


@dashboard_data_bp.route('/dashboard/api/settings')
@login_required
def api_settings_get():
    """Get current user settings."""
    try:
        if _SETTINGS_FILE.exists():
            with open(_SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
        return jsonify({'settings': settings})
    except Exception as e:
        logger.error(f"Error reading settings: {e}")
        return jsonify({'error': 'Failed to read settings'}), 500


@dashboard_data_bp.route('/dashboard/api/settings', methods=['PUT'])
@login_required
def api_settings_put():
    """Save user settings."""
    try:
        data = request.get_json(silent=True) or {}
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({'success': True, 'settings': data})
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return jsonify({'error': 'Failed to save settings'}), 500


# ---------------------------------------------------------------------------
# Position Close
# ---------------------------------------------------------------------------

@dashboard_data_bp.route('/dashboard/api/positions/<symbol>/update', methods=['PUT'])
@login_required
def api_position_update(symbol):
    """Update stop/target for a stock position."""
    try:
        data = request.get_json() or {}
        new_stop = data.get('stop_price')
        new_target = data.get('target_price')

        if new_stop is None and new_target is None:
            return jsonify({'error': 'Provide stop_price and/or target_price'}), 400

        from trading.position_tracker import PositionTracker
        tracker = None

        # Try to get tracker from orchestrator first
        try:
            from agents.orchestrator import get_running_orchestrator
            orch = get_running_orchestrator()
            if orch and hasattr(orch, 'position_tracker'):
                tracker = orch.position_tracker
        except Exception:
            pass

        # Fallback to singleton
        if tracker is None:
            try:
                from risk.position_tracker import get_position_tracker
                tracker = get_position_tracker()
            except Exception:
                pass

        if tracker is None:
            return jsonify({'error': 'Position tracker not available'}), 503

        result = {}
        if new_stop is not None:
            res = tracker.update_stop(symbol, float(new_stop))
            if 'error' in res:
                return jsonify({'error': res['error']}), 400
            result = res

        if new_target is not None:
            res = tracker.update_target(symbol, float(new_target))
            if 'error' in res:
                return jsonify({'error': res['error']}), 400
            result = res

        logger.info(f"Position {symbol} updated via dashboard: stop={new_stop}, target={new_target}")
        return jsonify({'success': True, 'position': result})
    except Exception as e:
        logger.error(f"Error updating position {symbol}: {e}")
        return jsonify({'error': f'Failed to update position: {e}'}), 500


@dashboard_data_bp.route('/dashboard/api/positions/<symbol>/close', methods=['POST'])
@login_required
def api_position_close(symbol):
    """Close a stock position via the broker/position tracker."""
    try:
        from agents.orchestrator import get_running_orchestrator

        orch = get_running_orchestrator()
        if orch is None:
            return jsonify({'error': 'Trading system is not running'}), 503

        # Try to close via executor agent or position tracker
        closed = False
        if hasattr(orch, 'executor') and orch.executor:
            try:
                if hasattr(orch.executor, 'close_position'):
                    orch.executor.close_position(symbol)
                    closed = True
            except Exception as ex:
                logger.warning(f"Executor close failed for {symbol}: {ex}")

        if not closed:
            # Fallback: close via position tracker
            try:
                from risk.position_tracker import get_position_tracker
                tracker = get_position_tracker()
                if tracker and hasattr(tracker, 'close_position'):
                    tracker.close_position(symbol, reason='manual_dashboard_close')
                    closed = True
            except Exception as ex:
                logger.warning(f"Position tracker close failed for {symbol}: {ex}")

        if closed:
            logger.info(f"Position {symbol} closed via dashboard")
            return jsonify({'success': True, 'symbol': symbol})
        else:
            return jsonify({'error': f'Could not close position for {symbol}. No handler available.'}), 500
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        return jsonify({'error': f'Failed to close position: {e}'}), 500


# ---------------------------------------------------------------------------
# ML Status (AJAX endpoint)
# ---------------------------------------------------------------------------

@dashboard_data_bp.route('/dashboard/api/ml/status')
@login_required
def api_ml_status():
    """Return ML monitoring data for AJAX refresh."""
    try:
        ml_data = {'error': None}

        try:
            from ml.model_monitor import get_model_monitors
            monitors = get_model_monitors()
            if not monitors:
                ml_data['error'] = 'no_monitors'
            else:
                monitor = monitors[0]
                ml_data['model_status'] = monitor.get_status() if hasattr(monitor, 'get_status') else None
                ml_data['drift_status'] = monitor.get_drift_report() if hasattr(monitor, 'get_drift_report') else None
                ml_data['performance_status'] = monitor.get_performance_status() if hasattr(monitor, 'get_performance_status') else None
                ml_data['feature_drift'] = monitor.get_feature_drift() if hasattr(monitor, 'get_feature_drift') else None
                ml_data['recent_predictions'] = monitor.get_recent_predictions(20) if hasattr(monitor, 'get_recent_predictions') else None
        except ImportError:
            ml_data['error'] = 'ml_module_not_available'
        except Exception as ex:
            ml_data['error'] = str(ex)

        return jsonify(ml_data)
    except Exception as e:
        logger.error(f"Error in api_ml_status: {e}")
        return jsonify({'error': 'Failed to fetch ML status'}), 500


# ---------------------------------------------------------------------------
# Backtest (session-auth proxy — avoids API-key requirement from dashboard)
# ---------------------------------------------------------------------------
@dashboard_data_bp.route('/dashboard/api/backtest', methods=['POST'])
@login_required
def api_backtest_proxy():
    """Run a backtest via session auth (proxies to the shared backtest engine)."""
    try:
        from api.v1.routes import run_backtest_with_params
    except ImportError:
        return jsonify({'error': 'Backtest engine not available'}), 503

    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    # Basic validation
    days = data.get('days', 365)
    if not isinstance(days, (int, float)) or days < 30 or days > 730:
        return jsonify({'error': 'days must be between 30 and 730'}), 400

    try:
        result = run_backtest_with_params(data)
        return jsonify({
            'parameters': data,
            'result': result,
        })
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500


# ---------------------------------------------------------------------------
# Scanner (session-auth proxy — avoids API-key requirement from dashboard)
# ---------------------------------------------------------------------------
@dashboard_data_bp.route('/dashboard/api/scanner/scan', methods=['GET'])
@login_required
def api_scanner_scan():
    """Run RRS scan via session auth."""
    try:
        from api.v1.routes import run_full_rrs_scan, format_timestamp
        results = run_full_rrs_scan()
        return jsonify({
            'timestamp': format_timestamp(),
            'count': len(results),
            'strongest': results[:10],
            'weakest': results[-10:] if len(results) >= 10 else [],
            'all_results': results,
        })
    except ImportError:
        return jsonify({'error': 'Scanner not available'}), 503
    except Exception as e:
        logger.error(f"Scanner scan error: {e}")
        return jsonify({'error': str(e)}), 500


@dashboard_data_bp.route('/dashboard/api/scanner/rrs/<symbol>', methods=['GET'])
@login_required
def api_scanner_rrs(symbol):
    """Get RRS for a specific symbol via session auth."""
    try:
        from api.v1.routes import calculate_rrs_for_symbol, format_timestamp
        import re
        # Basic symbol validation
        symbol = symbol.upper().strip()
        if not re.match(r'^[A-Z]{1,5}$', symbol):
            return jsonify({'error': 'Invalid symbol'}), 400
        rrs_data = calculate_rrs_for_symbol(symbol)
        if rrs_data is None:
            return jsonify({'error': f'Symbol {symbol} not found'}), 404
        return jsonify({'timestamp': format_timestamp(), 'symbol': symbol, **rrs_data})
    except ImportError:
        return jsonify({'error': 'RRS calculator not available'}), 503
    except Exception as e:
        logger.error(f"RRS lookup error: {e}")
        return jsonify({'error': str(e)}), 500


@dashboard_data_bp.route('/dashboard/api/onboarding/complete', methods=['POST'])
@login_required
def complete_onboarding():
    """Mark onboarding as completed for current user."""
    from flask_login import current_user
    from data.database.models import User
    try:
        db = get_db_manager()
        with db.session() as session:
            user = session.query(User).filter_by(id=current_user.id).first()
            if user:
                user.onboarding_completed = True
                session.commit()
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Onboarding complete error: {e}")
        return jsonify({'error': str(e)}), 500
