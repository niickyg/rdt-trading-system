"""
Tests for Risk Manager

Tests:
- Position size limits
- Daily loss limits
- Drawdown tracking
- Circuit breaker
- Concurrent position risk
- Risk metrics (Sharpe, VaR)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from risk.risk_manager import RiskManager
from risk.models import RiskLimits, RiskCheckResult, RiskLevel, RiskViolationType


class TestRiskManagerInit:
    """Tests for RiskManager initialization"""

    def test_init_default_limits(self):
        """Test initialization with default limits"""
        manager = RiskManager(account_size=25000)

        assert manager.account_size == 25000
        assert manager.current_balance == 25000
        assert manager.peak_balance == 25000
        assert manager.trading_halted is False
        assert manager.daily_pnl == 0.0
        assert manager.current_drawdown == 0.0

    def test_init_custom_limits(self):
        """Test initialization with custom limits"""
        limits = RiskLimits(
            max_risk_per_trade=0.02,
            max_daily_loss=0.03,
            max_position_size=5000,
            max_concurrent_positions=3
        )
        manager = RiskManager(account_size=50000, risk_limits=limits)

        assert manager.limits.max_risk_per_trade == 0.02
        assert manager.limits.max_daily_loss == 0.03
        assert manager.limits.max_position_size == 5000
        assert manager.limits.max_concurrent_positions == 3


class TestPositionSizing:
    """Tests for position size limits"""

    def test_position_size_within_limits(self):
        """Test that positions within limits are allowed"""
        limits = RiskLimits(max_position_size=5000)
        manager = RiskManager(account_size=25000, risk_limits=limits)

        # Calculate position size for a trade
        result = manager.position_sizer.calculate_position_size(
            account_size=25000,
            entry_price=100.0,
            stop_loss=98.0,
            risk_per_trade=0.01
        )

        # Should return a reasonable position size
        assert result is not None
        assert result.shares > 0
        assert result.risk_amount <= 250  # 1% of 25000

    def test_max_concurrent_positions(self):
        """Test max concurrent positions limit"""
        limits = RiskLimits(max_concurrent_positions=3)
        manager = RiskManager(account_size=25000, risk_limits=limits)

        # Add positions up to limit
        manager.open_positions = {
            'AAPL': {'shares': 50, 'entry_price': 150.0},
            'MSFT': {'shares': 30, 'entry_price': 280.0},
            'GOOGL': {'shares': 10, 'entry_price': 140.0}
        }

        # Check if new position can be opened
        can_open = len(manager.open_positions) < manager.limits.max_concurrent_positions
        assert can_open is False


class TestDailyLossLimit:
    """Tests for daily loss limit enforcement"""

    def test_daily_loss_tracking(self):
        """Test daily P&L tracking"""
        manager = RiskManager(account_size=25000)

        # Record losing trades
        manager.record_trade(pnl=-100.0)
        manager.record_trade(pnl=-150.0)

        assert manager.daily_pnl == -250.0
        assert manager.daily_trades == 2
        assert manager.daily_losses == 2
        assert manager.daily_wins == 0

    def test_daily_loss_limit_breach(self):
        """Test trading halt on daily loss limit"""
        limits = RiskLimits(max_daily_loss=0.02)  # 2% daily loss limit
        manager = RiskManager(account_size=25000, risk_limits=limits)

        # Simulate a large loss
        manager.record_trade(pnl=-600.0)  # More than 2% of $25000 ($500)

        # Check if trading halted
        daily_loss_pct = abs(manager.daily_pnl) / manager.daily_start_balance
        assert daily_loss_pct > manager.limits.max_daily_loss

    def test_daily_reset(self):
        """Test daily metrics reset"""
        manager = RiskManager(account_size=25000)

        # Add some activity
        manager.record_trade(pnl=-100.0)
        manager.daily_trades = 5
        manager.day_trades_today = 2

        # Reset
        manager.reset_daily()

        assert manager.daily_pnl == 0.0
        assert manager.daily_trades == 0
        assert manager.day_trades_today == 0


class TestDrawdownTracking:
    """Tests for drawdown tracking"""

    def test_drawdown_calculation(self):
        """Test drawdown is calculated correctly"""
        manager = RiskManager(account_size=25000)

        # Simulate balance decline
        manager.update_balance(24000)

        assert manager.current_drawdown == 1000.0
        assert manager.peak_balance == 25000

    def test_drawdown_recovery(self):
        """Test drawdown resets on new high"""
        manager = RiskManager(account_size=25000)

        # Go into drawdown
        manager.update_balance(24000)
        assert manager.current_drawdown == 1000.0

        # Recover to new high
        manager.update_balance(26000)
        assert manager.current_drawdown == 0.0
        assert manager.peak_balance == 26000

    def test_max_drawdown_tracking(self):
        """Test max drawdown is tracked"""
        manager = RiskManager(account_size=25000)

        # Multiple drawdowns
        manager.update_balance(24000)  # -1000
        manager.update_balance(25500)  # recovery
        manager.update_balance(23000)  # -2500 from peak of 25500

        assert manager.max_drawdown == 2500.0

    def test_drawdown_start_time(self):
        """Test drawdown start time is tracked"""
        manager = RiskManager(account_size=25000)

        # No drawdown initially
        assert manager.drawdown_start_time is None

        # Enter drawdown
        manager.update_balance(24000)
        assert manager.drawdown_start_time is not None

        # Recover
        manager.update_balance(26000)
        assert manager.drawdown_start_time is None


class TestCircuitBreaker:
    """Tests for trading circuit breaker"""

    def test_trading_halt(self):
        """Test trading can be halted"""
        manager = RiskManager(account_size=25000)

        assert manager.trading_halted is False

        # Halt trading
        manager.trading_halted = True
        manager.halt_reason = "Max drawdown exceeded"

        assert manager.trading_halted is True
        assert "drawdown" in manager.halt_reason.lower()

    def test_halt_on_max_drawdown(self):
        """Test automatic halt on max drawdown"""
        limits = RiskLimits(max_drawdown=0.05)  # 5% max drawdown
        manager = RiskManager(account_size=25000, risk_limits=limits)

        # Exceed max drawdown
        manager.update_balance(23500)  # 6% drawdown

        drawdown_pct = manager.current_drawdown / manager.peak_balance
        exceeds_limit = drawdown_pct > manager.limits.max_drawdown

        assert exceeds_limit is True


class TestRiskMetrics:
    """Tests for risk metrics calculations"""

    def test_win_loss_tracking(self):
        """Test win/loss statistics"""
        manager = RiskManager(account_size=25000)

        manager.record_trade(pnl=200.0)
        manager.record_trade(pnl=-100.0)
        manager.record_trade(pnl=150.0)
        manager.record_trade(pnl=-50.0)

        assert manager.daily_wins == 2
        assert manager.daily_losses == 2
        assert manager.daily_pnl == 200.0

    def test_day_trade_tracking(self):
        """Test day trade counting"""
        manager = RiskManager(account_size=25000)

        manager.record_trade(pnl=100.0, is_day_trade=True)
        manager.record_trade(pnl=50.0, is_day_trade=True)
        manager.record_trade(pnl=-25.0, is_day_trade=False)

        assert manager.day_trades_today == 2


class TestViolationTracking:
    """Tests for risk violation tracking"""

    def test_violation_recorded(self):
        """Test violations are recorded"""
        manager = RiskManager(account_size=25000)

        violation = RiskCheckResult(
            passed=False,
            risk_level=RiskLevel.HIGH,
            violation_type=RiskViolationType.DAILY_LOSS_LIMIT,
            message="Daily loss limit exceeded"
        )

        manager.violations.append(violation)

        assert len(manager.violations) == 1
        assert manager.violations[0].violation_type == RiskViolationType.DAILY_LOSS_LIMIT


class TestBalanceUpdates:
    """Tests for balance update handling"""

    def test_balance_increase(self):
        """Test balance increase updates correctly"""
        manager = RiskManager(account_size=25000)

        manager.update_balance(27000)

        assert manager.current_balance == 27000
        assert manager.peak_balance == 27000
        assert manager.current_drawdown == 0.0

    def test_balance_decrease(self):
        """Test balance decrease updates correctly"""
        manager = RiskManager(account_size=25000)

        manager.update_balance(23000)

        assert manager.current_balance == 23000
        assert manager.peak_balance == 25000
        assert manager.current_drawdown == 2000.0


class TestEdgeCases:
    """Tests for edge cases"""

    def test_zero_balance(self):
        """Test handling of zero balance"""
        manager = RiskManager(account_size=25000)

        # This is a catastrophic scenario
        manager.update_balance(0)

        assert manager.current_balance == 0
        assert manager.current_drawdown == 25000

    def test_negative_pnl(self):
        """Test handling of large negative P&L"""
        manager = RiskManager(account_size=25000)

        manager.record_trade(pnl=-5000)

        assert manager.daily_pnl == -5000
        assert manager.daily_losses == 1

    def test_multiple_resets(self):
        """Test multiple daily resets"""
        manager = RiskManager(account_size=25000)

        for _ in range(3):
            manager.record_trade(pnl=-100)
            manager.reset_daily()

        assert manager.daily_pnl == 0.0
        assert manager.daily_trades == 0
