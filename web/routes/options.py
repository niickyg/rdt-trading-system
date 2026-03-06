"""
Options Trading API Routes for the RDT Trading System.

Provides endpoints for option chain data, IV analysis, strategy selection,
position management, execution, and portfolio risk summary.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
from typing import Dict, Optional
from flask import Blueprint, jsonify, request
from flask_login import login_required
from loguru import logger


options_bp = Blueprint('options', __name__, url_prefix='/api/options')


def _get_options_components() -> Optional[Dict]:
    """Get options components from trading_init."""
    try:
        from web.app import _trading_components
        return _trading_components.get('options')
    except Exception:
        return None


def _options_not_enabled():
    return jsonify({
        'status': 'error',
        'message': 'Options trading is not enabled. Set OPTIONS_ENABLED=true in .env'
    }), 400


def _serialize_greeks(greeks):
    """Serialize OptionGreeks to dict."""
    if greeks is None:
        return None
    return {
        'delta': round(greeks.delta, 4),
        'gamma': round(greeks.gamma, 6),
        'theta': round(greeks.theta, 4),
        'vega': round(greeks.vega, 4),
        'implied_vol': round(greeks.implied_vol, 4),
        'underlying_price': round(greeks.underlying_price, 2),
        'option_price': round(greeks.option_price, 2),
        'bid': round(greeks.bid, 2),
        'ask': round(greeks.ask, 2),
        'mid_price': round(greeks.mid_price, 2),
        'spread': round(greeks.spread, 2),
        'spread_pct': round(greeks.spread_pct, 2),
        'volume': greeks.volume,
        'open_interest': greeks.open_interest,
    }


def _serialize_strategy(strategy):
    """Serialize OptionsStrategy to dict."""
    legs = []
    for leg in strategy.legs:
        legs.append({
            'contract': {
                'symbol': leg.contract.symbol,
                'expiry': leg.contract.expiry,
                'strike': leg.contract.strike,
                'right': leg.contract.right.value,
            },
            'action': leg.action.value,
            'quantity': leg.quantity,
            'greeks': _serialize_greeks(leg.greeks),
        })
    return {
        'name': strategy.name,
        'underlying': strategy.underlying,
        'direction': strategy.direction.value,
        'legs': legs,
        'max_loss': round(strategy.max_loss, 2),
        'max_profit': round(strategy.max_profit, 2),
        'breakeven': [round(b, 2) for b in strategy.breakeven],
        'net_premium': round(strategy.net_premium, 2),
        'net_delta': round(strategy.net_delta, 4),
        'net_gamma': round(strategy.net_gamma, 6),
        'net_theta': round(strategy.net_theta, 4),
        'net_vega': round(strategy.net_vega, 4),
        'is_debit': strategy.is_debit,
        'is_credit': strategy.is_credit,
        'is_defined_risk': strategy.is_defined_risk,
        'risk_reward_ratio': round(strategy.risk_reward_ratio, 2),
        'expiry': strategy.expiry,
    }


def _serialize_position(symbol, position):
    """Serialize an options position dict to JSON-safe format."""
    strategy = position.get('strategy')
    entry_premium = position.get('total_premium') or position.get('entry_premium', 0)
    unrealized_pnl = position.get('unrealized_pnl', 0)
    # Compute P&L % since executors don't provide it
    if abs(entry_premium) > 0.01:
        unrealized_pnl_pct = (unrealized_pnl / abs(entry_premium)) * 100
    else:
        unrealized_pnl_pct = 0.0

    result = {
        'symbol': symbol,
        'strategy': _serialize_strategy(strategy) if strategy else None,
        'contracts': position.get('contracts', 0),
        'entry_time': position.get('entry_time', ''),
        'entry_premium': round(entry_premium, 2),
        'entry_iv': round(position.get('entry_iv', 0), 4),
        'entry_delta': round(position.get('entry_delta', 0), 4),
        'current_value': round(position.get('current_value', 0), 2),
        'unrealized_pnl': round(unrealized_pnl, 2),
        'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
        'status': position.get('status', 'open'),
    }
    if isinstance(result['entry_time'], datetime):
        result['entry_time'] = result['entry_time'].isoformat()
    return result


def _get_underlying_price(opts, symbol):
    """Get the underlying stock price from the chain provider."""
    try:
        chain_manager = opts['chain_manager']
        return chain_manager._provider.get_underlying_price(symbol)
    except Exception:
        return 0.0


@options_bp.route('/chain/<symbol>', methods=['GET'])
@login_required
def get_option_chain(symbol):
    """Fetch option chain expirations and strikes for a symbol."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        chain_manager = opts['chain_manager']
        expirations = chain_manager.get_expirations(symbol.upper())

        if not expirations:
            return jsonify({
                'status': 'error',
                'message': f'No option chain data available for {symbol.upper()}'
            }), 404

        # Get strikes from chain params
        chain_data = chain_manager._get_chain_params(symbol.upper())
        strikes = chain_data.get('strikes', []) if chain_data else []

        return jsonify({
            'status': 'success',
            'data': {
                'symbol': symbol.upper(),
                'expirations': expirations,
                'strikes': strikes,
                'multiplier': chain_data.get('multiplier', 100) if chain_data else 100,
            }
        })
    except Exception as e:
        logger.error(f"Error fetching chain for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/chain/<symbol>/<expiry>', methods=['GET'])
@login_required
def get_option_chain_expiry(symbol, expiry):
    """Fetch option chain for a specific expiry with Greeks grid."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        from options.models import OptionRight
        chain_manager = opts['chain_manager']

        # Get contracts for this expiry
        calls = chain_manager.get_chain(symbol.upper(), expiry, OptionRight.CALL)
        puts = chain_manager.get_chain(symbol.upper(), expiry, OptionRight.PUT)

        # Fetch greeks via batch method on provider
        all_contracts = calls + puts
        provider = chain_manager._provider
        greeks_map = provider.get_greeks_batch(all_contracts)

        # Build grid
        call_data = []
        for c in calls:
            g = greeks_map.get(c.display_name)
            call_data.append({
                'strike': c.strike,
                'greeks': _serialize_greeks(g),
            })

        put_data = []
        for p in puts:
            g = greeks_map.get(p.display_name)
            put_data.append({
                'strike': p.strike,
                'greeks': _serialize_greeks(g),
            })

        # Determine underlying price
        underlying_price = 0
        if greeks_map:
            try:
                first_greeks = next(iter(greeks_map.values()))
                underlying_price = first_greeks.underlying_price
            except StopIteration:
                underlying_price = _get_underlying_price(opts, symbol.upper())

        return jsonify({
            'status': 'success',
            'data': {
                'symbol': symbol.upper(),
                'expiry': expiry,
                'underlying_price': round(underlying_price, 2),
                'calls': call_data,
                'puts': put_data,
            }
        })
    except Exception as e:
        logger.error(f"Error fetching chain for {symbol}/{expiry}: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/iv/<symbol>', methods=['GET'])
@login_required
def get_iv_analysis(symbol):
    """Get IV analysis (rank, percentile, HV, regime) for a symbol."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        iv_analyzer = opts['iv_analyzer']
        analysis = iv_analyzer.analyze(symbol.upper())

        return jsonify({
            'status': 'success',
            'data': {
                'symbol': symbol.upper(),
                'current_iv': round(analysis.current_iv, 4),
                'iv_rank': round(analysis.iv_rank, 1),
                'iv_percentile': round(analysis.iv_percentile, 1),
                'hv_20': round(analysis.hv_20, 4),
                'iv_high_52w': round(analysis.iv_high_52w, 4),
                'iv_low_52w': round(analysis.iv_low_52w, 4),
                'iv_hv_ratio': round(analysis.iv_hv_ratio, 2),
                'regime': analysis.regime.value,
                'has_iv_premium': analysis.has_iv_premium,
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing IV for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/strategies/<symbol>', methods=['GET'])
@login_required
def get_strategies(symbol):
    """Get recommended options strategies for a symbol given current IV regime."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        strategy_selector = opts['strategy_selector']
        iv_analyzer = opts['iv_analyzer']
        account_size = float(os.environ.get('ACCOUNT_SIZE', 25000))

        # Get IV analysis first
        iv_analysis = iv_analyzer.analyze(symbol.upper())

        # Get actual underlying price for strategy construction
        underlying_price = _get_underlying_price(opts, symbol.upper())

        # Get strategies for both directions
        strategies = {}
        for direction in ['long', 'short']:
            signal = {
                'symbol': symbol.upper(),
                'direction': direction,
                'entry_price': underlying_price,
                'atr': 0,
            }
            strategy = strategy_selector.select_strategy(signal, account_size)
            if strategy:
                strategies[direction] = _serialize_strategy(strategy)

        return jsonify({
            'status': 'success',
            'data': {
                'symbol': symbol.upper(),
                'underlying_price': round(underlying_price, 2),
                'iv_regime': iv_analysis.regime.value,
                'iv_rank': round(iv_analysis.iv_rank, 1),
                'strategies': strategies,
            }
        })
    except Exception as e:
        logger.error(f"Error getting strategies for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/positions', methods=['GET'])
@login_required
def get_positions():
    """Get all open options positions with live Greeks/P&L."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        executor = opts['executor']
        positions = executor.get_all_positions()

        serialized = []
        for symbol, pos in positions.items():
            serialized.append(_serialize_position(symbol, pos))

        return jsonify({
            'status': 'success',
            'data': {
                'positions': serialized,
                'count': len(serialized),
            }
        })
    except Exception as e:
        logger.error(f"Error fetching options positions: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/positions/<symbol>/close', methods=['POST'])
@login_required
def close_position(symbol):
    """Close an options position by underlying symbol."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        executor = opts['executor']
        result = executor.close_position(symbol.upper())

        if result is None:
            return jsonify({
                'status': 'error',
                'message': f'Position for {symbol.upper()} not found or could not be closed'
            }), 404

        return jsonify({
            'status': 'success',
            'data': {
                'symbol': symbol.upper(),
                'close_result': result,
            }
        })
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/execute', methods=['POST'])
@login_required
def execute_strategy():
    """Execute a specific options strategy."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'JSON body required'}), 400

        symbol = data.get('symbol', '').upper()
        direction = data.get('direction', 'long')

        if not symbol:
            return jsonify({'status': 'error', 'message': 'symbol is required'}), 400

        strategy_selector = opts['strategy_selector']
        position_sizer = opts['position_sizer']
        risk_manager = opts['risk_manager']
        executor = opts['executor']
        account_size = float(os.environ.get('ACCOUNT_SIZE', 25000))

        # Get actual underlying price if not provided
        entry_price = data.get('entry_price')
        if not entry_price:
            entry_price = _get_underlying_price(opts, symbol)

        # Build signal
        signal = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'atr': data.get('atr', 0),
        }

        # Select strategy
        strategy = strategy_selector.select_strategy(signal, account_size)
        if not strategy:
            return jsonify({
                'status': 'error',
                'message': f'No suitable strategy found for {symbol}'
            }), 400

        # Size the position
        max_risk = float(os.environ.get('MAX_RISK_PER_TRADE', 0.015))
        size_result = position_sizer.calculate(strategy, account_size, max_risk)
        if size_result.contracts <= 0:
            return jsonify({
                'status': 'error',
                'message': f'Position sizing resulted in 0 contracts: {size_result.reason}'
            }), 400

        # Risk check
        existing_positions = executor.get_all_positions()
        risk_check = risk_manager.validate_new_trade(
            strategy, size_result, existing_positions, account_size
        )
        if not risk_check:
            return jsonify({
                'status': 'error',
                'message': f'Risk check failed: {risk_check.reason}'
            }), 400

        # Execute
        result = executor.execute_strategy(strategy, size_result)
        if result is None:
            return jsonify({
                'status': 'error',
                'message': 'Execution failed'
            }), 500

        return jsonify({
            'status': 'success',
            'data': {
                'strategy': _serialize_strategy(strategy),
                'contracts': size_result.contracts,
                'max_risk': round(size_result.max_risk, 2),
                'execution': result,
                'warnings': risk_check.warnings if hasattr(risk_check, 'warnings') else [],
            }
        })
    except Exception as e:
        logger.error(f"Error executing options strategy: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/risk', methods=['GET'])
@login_required
def get_risk_summary():
    """Get portfolio-level options risk summary."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        risk_manager = opts['risk_manager']
        executor = opts['executor']
        account_size = float(os.environ.get('ACCOUNT_SIZE', 25000))

        positions = executor.get_all_positions()
        risk_data = risk_manager.get_portfolio_risk(positions, account_size)

        return jsonify({
            'status': 'success',
            'data': risk_data,
        })
    except Exception as e:
        logger.error(f"Error fetching options risk summary: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


@options_bp.route('/config', methods=['GET'])
@login_required
def get_options_config():
    """Get current options configuration (read-only)."""
    opts = _get_options_components()
    if not opts:
        return _options_not_enabled()

    try:
        config = opts['config']
        return jsonify({
            'status': 'success',
            'data': {
                'enabled': config.enabled,
                'mode': config.mode.value,
                'dte_target': config.dte_target,
                'dte_min': config.dte_min,
                'dte_max': config.dte_max,
                'profit_target_pct': config.profit_target_pct,
                'stop_loss_pct': config.stop_loss_pct,
                'iv_rank_low': config.iv_rank_low,
                'iv_rank_high': config.iv_rank_high,
                'iv_rank_very_high': config.iv_rank_very_high,
                'max_premium_risk_pct': config.max_premium_at_risk_pct,
                'max_portfolio_delta': config.max_portfolio_delta,
                'max_per_underlying': config.max_positions_per_underlying,
                'is_options_enabled': config.is_options_enabled,
                'is_stocks_enabled': config.is_stocks_enabled,
            }
        })
    except Exception as e:
        logger.error(f"Error fetching options config: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500
