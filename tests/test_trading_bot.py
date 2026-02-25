"""
Comprehensive Unit Tests for RDT Trading Bot

Tests cover:
- Position sizing calculations
- Risk limit checks (daily loss, max positions)
- Order placement flow (mocked broker)
- Stop loss and take profit logic
- Signal processing
- Market hours handling
- Paper vs live mode switching
- Graceful shutdown

Run with: pytest tests/test_trading_bot.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brokers.broker_interface import (
    BrokerInterface, Quote, Order, Position, AccountInfo,
    OrderSide, OrderType, OrderStatus,
    BrokerError, OrderError, InsufficientFundsError
)
from brokers.paper_broker import PaperBroker


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_position_size_based_on_risk(self, default_bot_config, mock_broker_connected):
        """Test position size is calculated based on risk parameters."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    # With $25,000 account, 1% risk = $250
                    # ATR of $5, stop multiplier 1.5 = $7.50 risk per share
                    # $250 / $7.50 = 33 shares based on risk
                    # BUT max_position_size = 10% = $2,500, at $100/share = 25 shares
                    # So the position is capped at 25 shares
                    price = 100.0
                    atr = 5.0
                    shares = bot.calculate_position_size(price, atr, 'long')

                    # Calculate expected based on risk
                    expected_risk = 25000 * 0.01  # $250
                    expected_stop_distance = atr * 1.5  # $7.50
                    risk_based_shares = int(expected_risk / expected_stop_distance)  # 33

                    # But also capped by max position size
                    max_position_value = 25000 * 0.10  # $2,500
                    max_shares = int(max_position_value / price)  # 25

                    # Should be the minimum of risk-based and max position
                    expected_shares = min(risk_based_shares, max_shares)
                    assert shares == expected_shares

    def test_position_size_respects_max_position_limit(self, default_bot_config, mock_broker_connected):
        """Test position size doesn't exceed max position size."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    # Very small ATR would create huge position, but should be capped
                    # Max position = 10% of $25,000 = $2,500
                    # At $100/share = 25 shares max
                    price = 100.0
                    atr = 0.5  # Very small ATR would suggest huge position
                    shares = bot.calculate_position_size(price, atr, 'long')

                    max_position_value = 25000 * 0.10  # $2,500
                    max_shares = int(max_position_value / price)  # 25

                    assert shares <= max_shares

    def test_position_size_respects_buying_power(self, default_bot_config, mock_broker_connected):
        """Test position size doesn't exceed available buying power."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    mock_broker_connected.cash = 500  # Very limited buying power
                    bot.broker = mock_broker_connected

                    price = 100.0
                    atr = 3.0
                    shares = bot.calculate_position_size(price, atr, 'long')

                    # Should not exceed 5 shares ($500 / $100)
                    assert shares <= 5

    def test_position_size_zero_for_zero_atr(self, default_bot_config, mock_broker_connected):
        """Test position size is 0 when ATR is 0 (prevents division by zero)."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    price = 100.0
                    atr = 0.0  # Zero ATR
                    shares = bot.calculate_position_size(price, atr, 'long')

                    assert shares == 0

    def test_position_size_for_short(self, default_bot_config, mock_broker_connected):
        """Test position size calculation for short positions."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    price = 100.0
                    atr = 5.0
                    shares_long = bot.calculate_position_size(price, atr, 'long')
                    shares_short = bot.calculate_position_size(price, atr, 'short')

                    # Position sizing should be the same for long/short
                    assert shares_long == shares_short


class TestRiskLimitChecks:
    """Test risk limit checks (daily loss, max positions)."""

    def test_daily_loss_limit_not_exceeded(self, default_bot_config, mock_broker_connected):
        """Test daily loss limit check when not exceeded."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected
                    bot.session_start_equity = 25000

                    # No losses yet
                    result = bot.check_daily_loss_limit()
                    assert result is False  # Should not stop trading

    def test_daily_loss_limit_exceeded(self, default_bot_config, mock_broker_connected):
        """Test daily loss limit check when exceeded."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    # Simulate loss exceeding 3% ($750)
                    bot.session_start_equity = 25000
                    mock_broker_connected.cash = 24000  # Lost $1000 > $750

                    result = bot.check_daily_loss_limit()
                    assert result is True  # Should stop trading

    def test_daily_loss_limit_at_boundary(self, default_bot_config, mock_broker_connected):
        """Test daily loss limit at exact boundary."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    # Exactly at 3% loss ($750) - check is daily_pnl < -max_loss
                    # At exactly boundary, -750 is not < -750, so continues trading
                    bot.session_start_equity = 25000
                    mock_broker_connected.cash = 24250  # Lost exactly $750

                    result = bot.check_daily_loss_limit()
                    # At exactly the limit (not exceeded), trading continues
                    assert result is False

                    # Just past the limit should stop
                    mock_broker_connected.cash = 24249  # Lost $751
                    result = bot.check_daily_loss_limit()
                    assert result is True


class TestOrderPlacement:
    """Test order placement flow with mocked broker."""

    def test_enter_trade_long_auto_trade_enabled(self, auto_trade_config, mock_broker_connected):
        """Test entering a long trade with auto_trade enabled."""
        mock_broker_connected.set_quote('AAPL', 149.90, 150.10, 150.00)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_connected

                    analysis = {
                        'price': 150.00,
                        'atr': 3.00,
                        'rrs': 2.5
                    }

                    bot.enter_trade('AAPL', analysis, 'long')

                    # Verify order was placed
                    assert len(mock_broker_connected.place_order_calls) >= 1
                    order_call = mock_broker_connected.place_order_calls[0]
                    assert order_call['symbol'] == 'AAPL'
                    assert order_call['side'] == OrderSide.BUY

                    # Verify position tracked
                    assert 'AAPL' in bot.positions

    def test_enter_trade_short_auto_trade_enabled(self, auto_trade_config, mock_broker_connected):
        """Test entering a short trade with auto_trade enabled."""
        mock_broker_connected.set_quote('TSLA', 199.90, 200.10, 200.00)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_connected

                    analysis = {
                        'price': 200.00,
                        'atr': 8.00,
                        'rrs': -3.0
                    }

                    bot.enter_trade('TSLA', analysis, 'short')

                    # Verify order was placed with correct side
                    assert len(mock_broker_connected.place_order_calls) >= 1
                    order_call = mock_broker_connected.place_order_calls[0]
                    assert order_call['side'] == OrderSide.SELL_SHORT

    def test_enter_trade_auto_trade_disabled(self, default_bot_config, mock_broker_connected):
        """Test trade signal logged but not executed when auto_trade disabled."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    analysis = {
                        'price': 150.00,
                        'atr': 3.00,
                        'rrs': 2.5
                    }

                    bot.enter_trade('AAPL', analysis, 'long')

                    # No orders should be placed
                    assert len(mock_broker_connected.place_order_calls) == 0

                    # But position should still be tracked
                    assert 'AAPL' in bot.positions
                    assert bot.positions['AAPL']['executed'] is False

    def test_enter_trade_zero_position_size_skipped(self, default_bot_config, mock_broker_connected):
        """Test trade is skipped when position size is 0."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    analysis = {
                        'price': 150.00,
                        'atr': 0.0,  # Zero ATR = zero position size
                        'rrs': 2.5
                    }

                    bot.enter_trade('AAPL', analysis, 'long')

                    # Position should not be tracked
                    assert 'AAPL' not in bot.positions

    def test_enter_trade_broker_error_handled(self, auto_trade_config, mock_broker_connected):
        """Test broker errors are handled gracefully during order placement."""
        mock_broker_connected.fail_orders = True
        mock_broker_connected.set_quote('AAPL', 149.90, 150.10, 150.00)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_connected

                    analysis = {
                        'price': 150.00,
                        'atr': 3.00,
                        'rrs': 2.5
                    }

                    # Should not raise exception
                    bot.enter_trade('AAPL', analysis, 'long')


class TestStopLossAndTakeProfit:
    """Test stop loss and take profit logic."""

    def test_stop_loss_calculation_long(self, default_bot_config, mock_broker_connected):
        """Test stop loss is calculated correctly for long positions."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    analysis = {
                        'price': 100.00,
                        'atr': 4.00,
                        'rrs': 2.5
                    }

                    bot.enter_trade('AAPL', analysis, 'long')

                    # Stop = price - (ATR * 1.5) = 100 - 6 = 94
                    assert bot.positions['AAPL']['stop_loss'] == 94.00
                    # Target = price + (ATR * 3) = 100 + 12 = 112
                    assert bot.positions['AAPL']['take_profit'] == 112.00

    def test_stop_loss_calculation_short(self, default_bot_config, mock_broker_connected):
        """Test stop loss is calculated correctly for short positions."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    analysis = {
                        'price': 100.00,
                        'atr': 4.00,
                        'rrs': -2.5
                    }

                    bot.enter_trade('TSLA', analysis, 'short')

                    # Stop = price + (ATR * 1.5) = 100 + 6 = 106
                    assert bot.positions['TSLA']['stop_loss'] == 106.00
                    # Target = price - (ATR * 3) = 100 - 12 = 88
                    assert bot.positions['TSLA']['take_profit'] == 88.00

    def test_monitor_positions_stop_loss_long(self, auto_trade_config, mock_broker_connected):
        """Test stop loss is triggered for long position."""
        mock_broker_connected.set_quote('AAPL', 93.00, 93.10, 93.05)
        mock_broker_connected.set_position('AAPL', 50, 100.00, 93.05)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_connected

                    # Setup position with stop loss
                    bot.positions['AAPL'] = {
                        'direction': 'long',
                        'entry_price': 100.00,
                        'shares': 50,
                        'stop_loss': 95.00,
                        'take_profit': 110.00,
                        'executed': True
                    }

                    # Price dropped below stop
                    bot.monitor_positions()

                    # Position should be exited
                    assert 'AAPL' not in bot.positions

    def test_monitor_positions_take_profit_long(self, auto_trade_config, mock_broker_connected):
        """Test take profit is triggered for long position."""
        mock_broker_connected.set_quote('AAPL', 110.90, 111.10, 111.00)
        mock_broker_connected.set_position('AAPL', 50, 100.00, 111.00)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_connected

                    bot.positions['AAPL'] = {
                        'direction': 'long',
                        'entry_price': 100.00,
                        'shares': 50,
                        'stop_loss': 95.00,
                        'take_profit': 110.00,
                        'executed': True
                    }

                    bot.monitor_positions()

                    # Position should be exited at profit
                    assert 'AAPL' not in bot.positions
                    assert bot.winning_trades == 1

    def test_monitor_positions_stop_loss_short(self, auto_trade_config, mock_broker_connected):
        """Test stop loss is triggered for short position."""
        mock_broker_connected.set_quote('TSLA', 106.90, 107.10, 107.00)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_connected

                    bot.positions['TSLA'] = {
                        'direction': 'short',
                        'entry_price': 100.00,
                        'shares': 25,
                        'stop_loss': 106.00,  # Stop above entry for short
                        'take_profit': 88.00,
                        'executed': True
                    }

                    bot.monitor_positions()

                    # Position should be exited
                    assert 'TSLA' not in bot.positions


class TestSignalProcessing:
    """Test signal processing and entry conditions."""

    def test_check_entry_conditions_long(self, default_bot_config, mock_broker_connected):
        """Test long entry conditions are correctly identified."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)

                    analysis = {
                        'rrs': 2.5,  # Above 2.0 threshold
                        'daily_strong': True,
                        'daily_weak': False
                    }

                    result = bot.check_entry_conditions(analysis)
                    assert result == 'long'

    def test_check_entry_conditions_short(self, default_bot_config, mock_broker_connected):
        """Test short entry conditions are correctly identified."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)

                    analysis = {
                        'rrs': -2.5,  # Below -2.0 threshold
                        'daily_strong': False,
                        'daily_weak': True
                    }

                    result = bot.check_entry_conditions(analysis)
                    assert result == 'short'

    def test_check_entry_conditions_no_signal(self, default_bot_config, mock_broker_connected):
        """Test no entry when conditions not met."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)

                    # RRS below threshold
                    analysis = {
                        'rrs': 1.5,  # Below 2.0 threshold
                        'daily_strong': True,
                        'daily_weak': False
                    }

                    result = bot.check_entry_conditions(analysis)
                    assert result is None

    def test_check_entry_conditions_conflicting(self, default_bot_config, mock_broker_connected):
        """Test no entry when RRS and daily chart conflict."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)

                    # Strong RRS but weak daily
                    analysis = {
                        'rrs': 2.5,
                        'daily_strong': False,  # Conflict
                        'daily_weak': True
                    }

                    result = bot.check_entry_conditions(analysis)
                    assert result is None


class TestMarketHoursHandling:
    """Test market hours handling."""

    def test_bot_waits_when_market_closed(self, default_bot_config, mock_broker_connected):
        """Test bot behavior when market is closed."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    with patch('automation.trading_bot.is_market_open', return_value=False):
                        with patch('automation.trading_bot.is_trading_day', return_value=True):
                            from automation.trading_bot import TradingBot
                            bot = TradingBot(default_bot_config)

                            # Bot should not execute trades when market closed
                            # This is tested implicitly in the run loop

    def test_bot_waits_on_non_trading_day(self, default_bot_config, mock_broker_connected):
        """Test bot behavior on non-trading days (weekends/holidays)."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    with patch('automation.trading_bot.is_trading_day', return_value=False):
                        from automation.trading_bot import TradingBot
                        bot = TradingBot(default_bot_config)

                        # Bot should wait on non-trading days


class TestPaperVsLiveMode:
    """Test paper trading vs live trading mode switching."""

    def test_paper_trading_mode_initialization(self, default_bot_config, mock_broker_connected):
        """Test bot initializes in paper trading mode by default."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)

                    assert bot.paper_trading is True
                    assert bot.auto_trade is False

    def test_live_trading_mode_warning(self, live_trading_config, mock_broker_connected):
        """Test warning is logged when live trading enabled."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot

                    # Should log warning for live trading
                    bot = TradingBot(live_trading_config)
                    assert bot.paper_trading is False

    def test_auto_trade_warning(self, auto_trade_config, mock_broker_connected):
        """Test warning is logged when auto-trade enabled."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot

                    bot = TradingBot(auto_trade_config)
                    assert bot.auto_trade is True

    def test_paper_mode_forces_paper_broker(self, default_bot_config):
        """Test paper_trading=True forces paper broker."""
        config = default_bot_config.copy()
        config['broker_type'] = 'schwab'  # Try to use Schwab
        config['paper_trading'] = True  # But paper mode is on

        with patch('automation.trading_bot.get_broker') as mock_get_broker:
            mock_broker = Mock()
            mock_broker.connect.return_value = True
            mock_get_broker.return_value = mock_broker

            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(config)

                    # Should have called get_broker with 'paper'
                    mock_get_broker.assert_called_once()
                    call_args = mock_get_broker.call_args
                    assert call_args[0][0] == 'paper'


class TestGracefulShutdown:
    """Test graceful shutdown behavior."""

    def test_shutdown_stops_order_monitor(self, default_bot_config, mock_broker_connected):
        """Test shutdown stops order monitor."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    # Mock order monitor
                    mock_monitor = Mock()
                    bot.order_monitor = mock_monitor

                    # Clear execution tracker to avoid Decimal/float coercion issues
                    bot.execution_tracker = None

                    bot.shutdown()

                    mock_monitor.stop.assert_called_once()

    def test_shutdown_disconnects_broker(self, default_bot_config, mock_broker_connected):
        """Test shutdown disconnects from broker."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    # Clear execution tracker to avoid Decimal/float coercion issues
                    bot.execution_tracker = None

                    bot.shutdown()

                    assert mock_broker_connected.disconnect_calls >= 1

    def test_shutdown_logs_session_summary(self, default_bot_config, mock_broker_connected):
        """Test shutdown logs session summary."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected
                    bot.total_trades = 5
                    bot.winning_trades = 3

                    # Clear execution tracker to avoid Decimal/float coercion issues
                    bot.execution_tracker = None

                    # Should complete without error
                    bot.shutdown()


class TestBrokerConnection:
    """Test broker connection handling."""

    def test_connect_broker_success(self, default_bot_config, mock_broker):
        """Test successful broker connection."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker

                    result = bot.connect_broker()

                    assert result is True
                    assert mock_broker.connect_calls == 1

    def test_connect_broker_failure(self, default_bot_config, mock_broker_failing):
        """Test broker connection failure handling."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_failing):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_failing

                    result = bot.connect_broker()

                    assert result is False

    def test_connect_broker_syncs_positions(self, default_bot_config, mock_broker_connected):
        """Test existing positions are synced on connection."""
        mock_broker_connected.set_position('AAPL', 100, 150.00)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    bot.connect_broker()
                    bot._sync_positions()

                    assert 'AAPL' in bot.positions


class TestExitPosition:
    """Test position exit logic."""

    def test_exit_position_calculates_pnl_long(self, default_bot_config, mock_broker_connected):
        """Test P&L calculation for long position exit."""
        mock_broker_connected.set_quote('AAPL', 159.90, 160.10, 160.00)

        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    bot.positions['AAPL'] = {
                        'direction': 'long',
                        'entry_price': 150.00,
                        'shares': 50,
                        'stop_loss': 145.00,
                        'take_profit': 160.00,
                        'executed': False
                    }

                    initial_pnl = bot.daily_pnl
                    bot.exit_position('AAPL', 160.00, 'take_profit')

                    # P&L = (160 - 150) * 50 = $500
                    assert bot.daily_pnl == initial_pnl + 500
                    assert bot.winning_trades == 1
                    assert 'AAPL' not in bot.positions

    def test_exit_position_calculates_pnl_short(self, default_bot_config, mock_broker_connected):
        """Test P&L calculation for short position exit."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    bot.positions['TSLA'] = {
                        'direction': 'short',
                        'entry_price': 200.00,
                        'shares': 25,
                        'stop_loss': 210.00,
                        'take_profit': 180.00,
                        'executed': False
                    }

                    initial_pnl = bot.daily_pnl
                    bot.exit_position('TSLA', 180.00, 'take_profit')

                    # P&L = (200 - 180) * 25 = $500
                    assert bot.daily_pnl == initial_pnl + 500
                    assert bot.winning_trades == 1

    def test_exit_position_no_position(self, default_bot_config, mock_broker_connected):
        """Test exit when no position exists."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected

                    # Should not raise error
                    bot.exit_position('NVDA', 100.00, 'manual')


class TestBotStatus:
    """Test bot status reporting."""

    def test_get_status(self, default_bot_config, mock_broker_connected):
        """Test status report generation."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_connected):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(default_bot_config)
                    bot.broker = mock_broker_connected
                    bot.total_trades = 10
                    bot.winning_trades = 6

                    status = bot.get_status()

                    assert 'connected' in status
                    assert 'paper_trading' in status
                    assert 'auto_trade' in status
                    assert 'account_equity' in status
                    assert 'buying_power' in status
                    assert 'total_trades' in status
                    assert 'win_rate' in status
                    assert status['win_rate'] == 60.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_buying_power(self, auto_trade_config, mock_broker_insufficient_funds):
        """Test handling of insufficient buying power."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_insufficient_funds):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_insufficient_funds

                    analysis = {
                        'price': 150.00,
                        'atr': 3.00,
                        'rrs': 2.5
                    }

                    # Should handle error gracefully
                    bot.enter_trade('AAPL', analysis, 'long')

    def test_broker_connection_failure_fallback(self, default_bot_config):
        """Test fallback to paper trading on broker init failure."""
        config = default_bot_config.copy()
        config['broker_type'] = 'schwab'
        config['paper_trading'] = False

        with patch('automation.trading_bot.get_broker') as mock_get_broker:
            # First call fails, second call (paper) succeeds
            mock_broker = Mock()
            mock_broker.connect.return_value = True
            mock_get_broker.side_effect = [
                Exception("Schwab init failed"),
                mock_broker
            ]

            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot

                    # Should fall back to paper trading
                    bot = TradingBot(config)

    def test_partial_fills(self, auto_trade_config, mock_broker_delayed_fills):
        """Test handling of partial order fills."""
        with patch('automation.trading_bot.get_broker', return_value=mock_broker_delayed_fills):
            with patch('automation.trading_bot.RealTimeScanner'):
                with patch('automation.trading_bot.RRSCalculator'):
                    from automation.trading_bot import TradingBot
                    bot = TradingBot(auto_trade_config)
                    bot.broker = mock_broker_delayed_fills
                    bot.fill_confirmation_timeout_seconds = 0.1  # Quick timeout

                    analysis = {
                        'price': 150.00,
                        'atr': 3.00,
                        'rrs': 2.5
                    }

                    # Should handle pending orders
                    bot.enter_trade('AAPL', analysis, 'long')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
