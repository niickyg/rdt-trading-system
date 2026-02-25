"""
Tests for OptionsRiskManager (options/risk.py).

Covers portfolio-level risk limit validation:
- Max positions per underlying
- Premium at risk limits
- Net portfolio delta limits
- Daily theta limits
- Expiration clustering warnings
- Risk/reward ratio warnings
- get_portfolio_risk metrics
- RiskCheckResult truthiness
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from options.risk import OptionsRiskManager, RiskCheckResult
from options.models import (
    OptionsStrategy,
    OptionsPositionSizeResult,
    OptionContract,
    OptionGreeks,
    OptionLeg,
    OptionAction,
    OptionRight,
    StrategyDirection,
)
from options.config import OptionsConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contract(symbol: str = "AAPL", expiry: str = "20260320", strike: float = 150.0,
                   right: OptionRight = OptionRight.CALL, multiplier: int = 100) -> OptionContract:
    return OptionContract(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        right=right,
        multiplier=multiplier,
    )


def _make_leg(symbol: str = "AAPL", expiry: str = "20260320", strike: float = 150.0,
              right: OptionRight = OptionRight.CALL, action: OptionAction = OptionAction.BUY,
              delta: float = 0.5, theta: float = -0.05, multiplier: int = 100) -> OptionLeg:
    contract = _make_contract(symbol=symbol, expiry=expiry, strike=strike,
                              right=right, multiplier=multiplier)
    greeks = OptionGreeks(delta=delta, theta=theta)
    return OptionLeg(contract=contract, action=action, greeks=greeks)


def _make_strategy(
    underlying: str = "AAPL",
    max_loss: float = 500.0,
    max_profit: float = 500.0,
    legs: list = None,
    expiry: str = "20260320",
    direction: StrategyDirection = StrategyDirection.LONG,
) -> OptionsStrategy:
    """Build an OptionsStrategy with a single leg so that net_delta, net_theta,
    and expiry are derived from that leg.  The leg's greeks are set so that
    signed_delta == net_delta for the strategy (BUY leg, quantity 1).
    """
    if legs is None:
        legs = [_make_leg(symbol=underlying, expiry=expiry)]
    return OptionsStrategy(
        name="test_strategy",
        underlying=underlying,
        direction=direction,
        legs=legs,
        max_loss=max_loss,
        max_profit=max_profit,
    )


def _make_size_result(contracts: int = 1, max_risk: float = 500.0) -> OptionsPositionSizeResult:
    return OptionsPositionSizeResult(
        strategy_name="test_strategy",
        contracts=contracts,
        max_risk=max_risk,
        premium_cost=max_risk,
    )


def _make_position(strategy: OptionsStrategy, contracts: int = 1) -> dict:
    return {"strategy": strategy, "contracts": contracts}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chain_manager():
    """OptionsChainManager is required by the constructor but not used
    during validate_new_trade or get_portfolio_risk."""
    return MagicMock()


@pytest.fixture
def config():
    """Default OptionsConfig – uses env-free defaults."""
    return OptionsConfig()


@pytest.fixture
def risk_mgr(mock_chain_manager, config):
    return OptionsRiskManager(chain_manager=mock_chain_manager, config=config)


@pytest.fixture
def account_size():
    return 25_000.0


# ---------------------------------------------------------------------------
# 1. New trade passes all checks with empty portfolio
# ---------------------------------------------------------------------------

class TestNewTradeEmptyPortfolio:
    def test_passes_all_checks(self, risk_mgr, account_size):
        strategy = _make_strategy(max_loss=500.0)
        size_result = _make_size_result(contracts=1, max_risk=500.0)

        result = risk_mgr.validate_new_trade(strategy, size_result, {}, account_size)

        assert result.passed is True
        assert bool(result) is True
        assert result.reason == ""
        assert result.warnings == []


# ---------------------------------------------------------------------------
# 2. Max positions per underlying
# ---------------------------------------------------------------------------

class TestMaxPositionsPerUnderlying:
    def test_third_position_blocked(self, risk_mgr, account_size):
        """2 existing AAPL positions -> 3rd AAPL is rejected."""
        existing = {
            "pos1": _make_position(_make_strategy(underlying="AAPL")),
            "pos2": _make_position(_make_strategy(underlying="AAPL")),
        }
        new_strategy = _make_strategy(underlying="AAPL")
        size_result = _make_size_result()

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is False
        assert "Max 2 positions per underlying" in result.reason
        assert "AAPL" in result.reason

    def test_different_underlying_allowed(self, risk_mgr, account_size):
        """2 existing AAPL positions should not block a MSFT trade."""
        existing = {
            "pos1": _make_position(_make_strategy(underlying="AAPL")),
            "pos2": _make_position(_make_strategy(underlying="AAPL")),
        }
        new_strategy = _make_strategy(underlying="MSFT")
        size_result = _make_size_result(max_risk=500.0)

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is True


# ---------------------------------------------------------------------------
# 3 & 4. Premium at risk limit
# ---------------------------------------------------------------------------

class TestPremiumAtRisk:
    def test_within_limit_passes(self, risk_mgr, account_size):
        """Existing $2000 risk + new $500 = $2500 = 10% of $25K -> exactly at limit, passes."""
        # max_loss * contracts for each existing position
        existing = {
            "pos1": _make_position(_make_strategy(underlying="MSFT", max_loss=1000.0), contracts=2),
        }
        # existing risk = 1000 * 2 = 2000
        new_strategy = _make_strategy(underlying="AAPL", max_loss=200.0)
        size_result = _make_size_result(contracts=1, max_risk=500.0)
        # total_after = 2000 + 500 = 2500
        # max_allowed = 25000 * 0.10 = 2500
        # 2500 > 2500 is False, so it passes

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is True

    def test_boundary_just_over_limit_blocked(self, risk_mgr, account_size):
        """Existing $2000 risk + new $501 = $2501 > $2500 limit -> blocked."""
        existing = {
            "pos1": _make_position(_make_strategy(underlying="MSFT", max_loss=1000.0), contracts=2),
        }
        new_strategy = _make_strategy(underlying="AAPL", max_loss=200.0)
        size_result = _make_size_result(contracts=1, max_risk=501.0)
        # total_after = 2000 + 501 = 2501 > 2500

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is False
        assert "premium at risk" in result.reason.lower()

    def test_exceeded_with_reason_message(self, risk_mgr, account_size):
        """Premium at risk exceeded gives informative reason."""
        existing = {
            "pos1": _make_position(_make_strategy(underlying="MSFT", max_loss=2000.0), contracts=1),
        }
        new_strategy = _make_strategy(underlying="AAPL", max_loss=200.0)
        size_result = _make_size_result(contracts=1, max_risk=600.0)
        # total = 2000 + 600 = 2600 > 2500

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is False
        assert "$2600" in result.reason
        assert "$2500" in result.reason
        assert "10%" in result.reason


# ---------------------------------------------------------------------------
# 5. Net portfolio delta exceeded
# ---------------------------------------------------------------------------

class TestNetDeltaExceeded:
    def test_delta_exceeded_blocked(self, risk_mgr, account_size):
        """Net delta exceeding 200 is rejected.

        Keep max_loss tiny so premium-at-risk check passes first.
        strategy.net_delta = 0.5 (BUY leg with delta=0.5)
        existing: 0.5 * 300 = 150 portfolio delta
        new: 0.5 * 110 = 55
        total = 205 > 200 -> blocked
        """
        high_delta_leg = _make_leg(symbol="MSFT", delta=0.5)
        existing = {
            "pos1": _make_position(
                _make_strategy(underlying="MSFT", max_loss=1.0, legs=[high_delta_leg]),
                contracts=300,
            ),
        }
        new_leg = _make_leg(symbol="AAPL", delta=0.5)
        new_strategy = _make_strategy(underlying="AAPL", max_loss=1.0, legs=[new_leg])
        size_result = _make_size_result(contracts=110, max_risk=1.0)

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is False
        assert "delta" in result.reason.lower()
        assert "205" in result.reason

    def test_negative_delta_exceeded_blocked(self, risk_mgr, account_size):
        """Negative delta magnitude exceeding 200 is also rejected."""
        sell_leg = _make_leg(symbol="MSFT", delta=0.5, action=OptionAction.SELL)
        existing = {
            "pos1": _make_position(
                _make_strategy(underlying="MSFT", max_loss=1.0, legs=[sell_leg]),
                contracts=300,
            ),
        }
        # existing delta = -0.5 * 300 = -150
        new_sell_leg = _make_leg(symbol="AAPL", delta=0.5, action=OptionAction.SELL)
        new_strategy = _make_strategy(underlying="AAPL", max_loss=1.0, legs=[new_sell_leg])
        size_result = _make_size_result(contracts=110, max_risk=1.0)
        # new_delta = -0.5 * 110 = -55  =>  total = -205, abs = 205 > 200

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is False
        assert "delta" in result.reason.lower()


# ---------------------------------------------------------------------------
# 6. Daily theta exceeded
# ---------------------------------------------------------------------------

class TestDailyThetaExceeded:
    def test_theta_exceeded_blocked(self, risk_mgr, account_size):
        """Daily theta exceeding 0.5% of account ($125) is rejected.

        max_theta = 25000 * 0.005 = 125
        _total_daily_theta for existing = net_theta * contracts * multiplier
        For new trade: new_theta = strategy.net_theta * size_result.contracts (no multiplier)

        existing: -0.05 * 24 * 100 = -120
        new: -0.05 * 120 = -6
        total = -126, abs=126 > 125 -> blocked

        max_loss kept at 1.0 so premium check passes (24*1 + 1 = 25 << 2500).
        """
        leg = _make_leg(symbol="MSFT", theta=-0.05, multiplier=100)
        existing = {
            "pos1": _make_position(
                _make_strategy(underlying="MSFT", max_loss=1.0, legs=[leg]),
                contracts=24,
            ),
        }

        new_leg = _make_leg(symbol="AAPL", theta=-0.05, multiplier=100)
        new_strategy = _make_strategy(underlying="AAPL", max_loss=1.0, legs=[new_leg])
        size_result = _make_size_result(contracts=120, max_risk=1.0)

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is False
        assert "theta" in result.reason.lower()

    def test_theta_within_limit_passes(self, risk_mgr, account_size):
        """Daily theta within 0.5% of account passes."""
        leg = _make_leg(symbol="MSFT", theta=-0.01, multiplier=100)
        existing = {
            "pos1": _make_position(
                _make_strategy(underlying="MSFT", max_loss=100.0, legs=[leg]),
                contracts=5,
            ),
        }
        # existing theta = -0.01 * 5 * 100 = -5
        # existing premium = 100 * 5 = 500 < 2500

        new_leg = _make_leg(symbol="AAPL", theta=-0.01, multiplier=100)
        new_strategy = _make_strategy(underlying="AAPL", max_loss=100.0, legs=[new_leg])
        size_result = _make_size_result(contracts=1, max_risk=100.0)
        # new_theta = -0.01 * 1 = -0.01
        # total = -5.01, abs = 5.01 < 125
        # total premium = 500 + 100 = 600 < 2500

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is True


# ---------------------------------------------------------------------------
# 7. Expiration clustering: warning but passes
# ---------------------------------------------------------------------------

class TestExpirationClustering:
    def test_three_same_expiry_warns(self, risk_mgr, account_size):
        """2 existing + 1 new with same expiry -> warning but passes."""
        expiry = "20260417"
        existing = {
            "pos1": _make_position(
                _make_strategy(underlying="MSFT", expiry=expiry, max_loss=100.0)
            ),
            "pos2": _make_position(
                _make_strategy(underlying="GOOG", expiry=expiry, max_loss=100.0)
            ),
        }
        new_strategy = _make_strategy(underlying="AAPL", expiry=expiry, max_loss=100.0)
        size_result = _make_size_result(max_risk=100.0)

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is True
        assert len(result.warnings) >= 1
        clustering_warnings = [w for w in result.warnings if "clustering" in w.lower()]
        assert len(clustering_warnings) == 1
        assert "3 positions" in clustering_warnings[0]

    # -------------------------------------------------------------------
    # 8. No clustering -> no warnings
    # -------------------------------------------------------------------
    def test_no_clustering_no_warnings(self, risk_mgr, account_size):
        """Different expiries -> no expiration clustering warning."""
        existing = {
            "pos1": _make_position(
                _make_strategy(underlying="MSFT", expiry="20260320", max_loss=100.0)
            ),
        }
        new_strategy = _make_strategy(underlying="AAPL", expiry="20260417", max_loss=100.0)
        size_result = _make_size_result(max_risk=100.0)

        result = risk_mgr.validate_new_trade(new_strategy, size_result, existing, account_size)

        assert result.passed is True
        clustering_warnings = [w for w in result.warnings if "clustering" in w.lower()]
        assert len(clustering_warnings) == 0


# ---------------------------------------------------------------------------
# 9. Low risk/reward ratio warning
# ---------------------------------------------------------------------------

class TestRiskRewardWarning:
    def test_low_rr_ratio_warns(self, risk_mgr, account_size):
        """risk_reward_ratio < 0.5 on a defined-risk strategy triggers warning."""
        # max_profit / max_loss < 0.5  =>  max_profit=200, max_loss=500 => 0.4
        strategy = _make_strategy(
            underlying="AAPL",
            max_loss=500.0,
            max_profit=200.0,
        )
        assert strategy.is_defined_risk
        assert strategy.risk_reward_ratio < 0.5

        size_result = _make_size_result(max_risk=500.0)
        result = risk_mgr.validate_new_trade(strategy, size_result, {}, account_size)

        assert result.passed is True
        rr_warnings = [w for w in result.warnings if "risk/reward" in w.lower()]
        assert len(rr_warnings) == 1
        assert "0.40" in rr_warnings[0]

    def test_adequate_rr_ratio_no_warning(self, risk_mgr, account_size):
        """risk_reward_ratio >= 0.5 does not trigger warning."""
        strategy = _make_strategy(
            underlying="AAPL",
            max_loss=500.0,
            max_profit=300.0,
        )
        assert strategy.risk_reward_ratio == 0.6

        size_result = _make_size_result(max_risk=500.0)
        result = risk_mgr.validate_new_trade(strategy, size_result, {}, account_size)

        assert result.passed is True
        rr_warnings = [w for w in result.warnings if "risk/reward" in w.lower()]
        assert len(rr_warnings) == 0


# ---------------------------------------------------------------------------
# 10. get_portfolio_risk returns correct metrics
# ---------------------------------------------------------------------------

class TestGetPortfolioRisk:
    def test_returns_correct_metrics(self, risk_mgr, account_size):
        leg_msft = _make_leg(symbol="MSFT", expiry="20260320", delta=0.6, theta=-0.03)
        leg_aapl = _make_leg(symbol="AAPL", expiry="20260417", delta=0.4, theta=-0.02)

        positions = {
            "pos1": _make_position(
                _make_strategy(underlying="MSFT", max_loss=1000.0, legs=[leg_msft]),
                contracts=2,
            ),
            "pos2": _make_position(
                _make_strategy(underlying="AAPL", max_loss=300.0, legs=[leg_aapl]),
                contracts=3,
            ),
        }

        metrics = risk_mgr.get_portfolio_risk(positions, account_size)

        # total_premium_at_risk = 1000*2 + 300*3 = 2900
        assert metrics["total_premium_at_risk"] == pytest.approx(2900.0)
        assert metrics["premium_risk_pct"] == pytest.approx(2900.0 / 25000.0)

        # net_portfolio_delta:
        # pos1: net_delta=0.6 (BUY leg, delta=0.6) * 2 contracts = 1.2
        # pos2: net_delta=0.4 (BUY leg, delta=0.4) * 3 contracts = 1.2
        assert metrics["net_portfolio_delta"] == pytest.approx(1.2 + 1.2)

        # daily_theta:
        # pos1: net_theta=-0.03, contracts=2, multiplier=100 => -0.03*2*100 = -6
        # pos2: net_theta=-0.02, contracts=3, multiplier=100 => -0.02*3*100 = -6
        assert metrics["daily_theta"] == pytest.approx(-12.0)
        assert metrics["theta_pct"] == pytest.approx(12.0 / 25000.0)

        assert metrics["position_count"] == 2

        assert metrics["expiration_distribution"] == {"20260320": 1, "20260417": 1}

        # Limits sub-dict
        assert metrics["limits"]["max_premium_risk_pct"] == 0.10
        assert metrics["limits"]["max_portfolio_delta"] == 200.0
        assert metrics["limits"]["max_daily_theta_pct"] == 0.005
        assert metrics["limits"]["max_per_underlying"] == 2

    def test_empty_portfolio(self, risk_mgr, account_size):
        metrics = risk_mgr.get_portfolio_risk({}, account_size)

        assert metrics["total_premium_at_risk"] == 0.0
        assert metrics["premium_risk_pct"] == 0.0
        assert metrics["net_portfolio_delta"] == 0.0
        assert metrics["daily_theta"] == 0.0
        assert metrics["position_count"] == 0
        assert metrics["expiration_distribution"] == {}

    def test_zero_account_size(self, risk_mgr):
        """Zero account size should not cause division by zero."""
        metrics = risk_mgr.get_portfolio_risk({}, 0.0)

        assert metrics["premium_risk_pct"] == 0.0
        assert metrics["theta_pct"] == 0.0


# ---------------------------------------------------------------------------
# 11. RiskCheckResult truthiness
# ---------------------------------------------------------------------------

class TestRiskCheckResultTruthiness:
    def test_passed_true_is_truthy(self):
        result = RiskCheckResult(passed=True)
        assert result
        assert bool(result) is True

    def test_passed_false_is_falsy(self):
        result = RiskCheckResult(passed=False, reason="test")
        assert not result
        assert bool(result) is False

    def test_repr_pass(self):
        result = RiskCheckResult(passed=True)
        assert "PASS" in repr(result)

    def test_repr_fail(self):
        result = RiskCheckResult(passed=False, reason="too risky")
        assert "FAIL" in repr(result)
        assert "too risky" in repr(result)

    def test_default_warnings_empty(self):
        result = RiskCheckResult(passed=True)
        assert result.warnings == []

    def test_warnings_preserved(self):
        result = RiskCheckResult(passed=True, warnings=["w1", "w2"])
        assert result.warnings == ["w1", "w2"]
