"""
Tests for the OptionsPositionSizer module.

Covers contract calculation, debit/credit handling, risk percent,
buying power, edge cases (zero max_loss, can't afford), and custom risk values.
"""

import pytest

from options.models import OptionsStrategy, StrategyDirection, OptionsPositionSizeResult
from options.position_sizer import OptionsPositionSizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(
    name: str = "bull_call_spread",
    underlying: str = "AAPL",
    direction: StrategyDirection = StrategyDirection.LONG,
    max_loss: float = 200.0,
    max_profit: float = 300.0,
    net_premium: float = -200.0,
) -> OptionsStrategy:
    """Create a minimal OptionsStrategy with known values."""
    return OptionsStrategy(
        name=name,
        underlying=underlying,
        direction=direction,
        max_loss=max_loss,
        max_profit=max_profit,
        net_premium=net_premium,
    )


@pytest.fixture
def sizer() -> OptionsPositionSizer:
    return OptionsPositionSizer()


# ---------------------------------------------------------------------------
# 1. Basic sizing: $25K account, 1.5% risk ($375), $200 max_loss -> 1 contract
# ---------------------------------------------------------------------------

class TestBasicSizing:
    def test_one_contract(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=200.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        # floor(375 / 200) = 1
        assert result.contracts == 1
        assert result.max_risk == 200.0
        assert result.reason != ""

    def test_result_is_correct_type(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy()
        result = sizer.calculate(strategy, account_size=25_000)
        assert isinstance(result, OptionsPositionSizeResult)

    def test_strategy_name_propagated(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(name="iron_condor")
        result = sizer.calculate(strategy, account_size=50_000)
        assert result.strategy_name == "iron_condor"


# ---------------------------------------------------------------------------
# 2. Multiple contracts: $375 risk budget / $100 max_loss -> 3 contracts
# ---------------------------------------------------------------------------

class TestMultipleContracts:
    def test_three_contracts(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=100.0, net_premium=-100.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        # floor(375 / 100) = 3
        assert result.contracts == 3
        assert result.max_risk == 300.0

    def test_floor_not_round(self, sizer: OptionsPositionSizer):
        """Verify we floor, not round — e.g. 375/120 = 3.125 -> 3."""
        strategy = _make_strategy(max_loss=120.0, net_premium=-120.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        assert result.contracts == 3  # floor(3.125)
        assert result.max_risk == 360.0


# ---------------------------------------------------------------------------
# 3. Can't afford even 1 contract: max_loss > risk budget -> 0 + reason
# ---------------------------------------------------------------------------

class TestCannotAfford:
    def test_zero_contracts_when_too_expensive(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=500.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        # 375 < 500 -> 0 contracts
        assert result.contracts == 0
        assert result.max_risk == 0
        assert result.premium_cost == 0
        assert "exceeds risk budget" in result.reason

    def test_reason_contains_dollar_amounts(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=500.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        assert "$500.00" in result.reason
        assert "$375.00" in result.reason


# ---------------------------------------------------------------------------
# 4. Exactly 1 contract when risk_budget == max_loss
# ---------------------------------------------------------------------------

class TestExactBoundary:
    def test_exact_match_gives_one_contract(self, sizer: OptionsPositionSizer):
        # risk_budget = 25_000 * 0.02 = 500.0, max_loss = 500.0
        strategy = _make_strategy(max_loss=500.0, net_premium=-500.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.02)

        # floor(500/500) = 1, but also the fallback path checks >=
        assert result.contracts == 1
        assert result.max_risk == 500.0

    def test_one_penny_short_gives_zero(self, sizer: OptionsPositionSizer):
        """If max_loss is just barely above the budget, we get 0."""
        strategy = _make_strategy(max_loss=500.01)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.02)

        assert result.contracts == 0


# ---------------------------------------------------------------------------
# 5. Zero max_loss -> 0 contracts with "Invalid max loss"
# ---------------------------------------------------------------------------

class TestZeroMaxLoss:
    def test_zero_max_loss(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=0.0)
        result = sizer.calculate(strategy, account_size=25_000)

        assert result.contracts == 0
        assert result.reason == "Invalid max loss"

    def test_negative_max_loss(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=-100.0)
        result = sizer.calculate(strategy, account_size=25_000)

        assert result.contracts == 0
        assert result.reason == "Invalid max loss"


# ---------------------------------------------------------------------------
# 6. Debit strategy: premium_cost set correctly (net_premium < 0)
# ---------------------------------------------------------------------------

class TestDebitStrategy:
    def test_premium_cost_for_debit(self, sizer: OptionsPositionSizer):
        # Debit spread: pay $150 per contract, max_loss $200
        strategy = _make_strategy(
            max_loss=200.0,
            net_premium=-150.0,  # negative = debit
        )
        result = sizer.calculate(strategy, account_size=50_000, max_risk_per_trade=0.02)

        # risk_budget = 1000, floor(1000/200) = 5
        assert result.contracts == 5
        assert result.premium_cost == 750.0  # abs(-150) * 5
        assert result.premium_received == 0.0

    def test_is_debit_property(self):
        strategy = _make_strategy(net_premium=-250.0)
        assert strategy.is_debit is True
        assert strategy.is_credit is False


# ---------------------------------------------------------------------------
# 7. Credit strategy: premium_received set correctly (net_premium > 0)
# ---------------------------------------------------------------------------

class TestCreditStrategy:
    def test_premium_received_for_credit(self, sizer: OptionsPositionSizer):
        # Credit spread: receive $80 per contract, max_loss $420
        strategy = _make_strategy(
            name="bull_put_spread",
            max_loss=420.0,
            max_profit=80.0,
            net_premium=80.0,  # positive = credit
        )
        result = sizer.calculate(strategy, account_size=100_000, max_risk_per_trade=0.015)

        # risk_budget = 1500, floor(1500/420) = 3
        assert result.contracts == 3
        assert result.premium_received == 240.0  # 80 * 3
        assert result.premium_cost == 0.0

    def test_is_credit_property(self):
        strategy = _make_strategy(net_premium=80.0)
        assert strategy.is_credit is True
        assert strategy.is_debit is False


# ---------------------------------------------------------------------------
# 8. Verify risk_percent calculation (max_risk / account_size)
# ---------------------------------------------------------------------------

class TestRiskPercent:
    def test_risk_percent(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=100.0, net_premium=-100.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        # 3 contracts * $100 = $300 risk, 300/25000 = 0.012
        assert result.contracts == 3
        assert result.risk_percent == pytest.approx(0.012, abs=1e-9)

    def test_risk_percent_single_contract(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=200.0, net_premium=-200.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        # 1 contract * $200 = $200 risk, 200/25000 = 0.008
        assert result.risk_percent == pytest.approx(0.008, abs=1e-9)

    def test_risk_percent_zero_when_no_contracts(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=500.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        assert result.contracts == 0
        assert result.risk_percent == 0.0

    def test_risk_percent_zero_account(self, sizer: OptionsPositionSizer):
        """Zero account_size should not cause division by zero."""
        strategy = _make_strategy(max_loss=0.0)
        result = sizer.calculate(strategy, account_size=0)

        assert result.risk_percent == 0.0


# ---------------------------------------------------------------------------
# 9. Buying power: defined risk = max_risk * contracts, undefined = premium_cost
# ---------------------------------------------------------------------------

class TestBuyingPower:
    def test_defined_risk_buying_power(self, sizer: OptionsPositionSizer):
        """Defined risk strategy: buying_power = max_loss * contracts."""
        strategy = _make_strategy(
            max_loss=200.0,
            max_profit=300.0,
            net_premium=-200.0,
        )
        assert strategy.is_defined_risk is True

        result = sizer.calculate(strategy, account_size=50_000, max_risk_per_trade=0.02)
        # risk_budget=1000, floor(1000/200)=5
        assert result.contracts == 5
        assert result.buying_power_required == 1000.0  # 200 * 5

    def test_undefined_risk_buying_power(self, sizer: OptionsPositionSizer):
        """Undefined risk strategy (max_loss=inf): buying_power = premium_cost."""
        strategy = _make_strategy(
            name="long_call",
            max_loss=300.0,
            max_profit=float('inf'),
            net_premium=-300.0,
        )
        # max_loss=300 and 300 < inf, so is_defined_risk is True
        # To test the undefined-risk branch, we need max_loss == 0 or inf
        # But max_loss=0 triggers the early return. Use inf to trigger undefined.
        strategy_undef = OptionsStrategy(
            name="naked_put",
            underlying="SPY",
            direction=StrategyDirection.SHORT,
            max_loss=float('inf'),
            net_premium=-500.0,
        )
        # is_defined_risk: max_loss > 0 and max_loss < inf => False
        assert strategy_undef.is_defined_risk is False

        # However, max_loss=inf means floor(budget/inf) = 0 contracts
        # So we can't actually get to the buying_power branch with inf max_loss
        # through the normal path. Let's test that it returns 0 contracts.
        result = sizer.calculate(strategy_undef, account_size=100_000, max_risk_per_trade=0.02)
        assert result.contracts == 0

    def test_is_defined_risk_property(self):
        defined = _make_strategy(max_loss=200.0)
        assert defined.is_defined_risk is True

        infinite = OptionsStrategy(
            name="naked",
            underlying="SPY",
            direction=StrategyDirection.SHORT,
            max_loss=float('inf'),
        )
        assert infinite.is_defined_risk is False

        zero = _make_strategy(max_loss=0.0)
        assert zero.is_defined_risk is False


# ---------------------------------------------------------------------------
# 10. Custom max_risk_per_trade values
# ---------------------------------------------------------------------------

class TestCustomRisk:
    def test_half_percent_risk(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=100.0, net_premium=-100.0)
        result = sizer.calculate(strategy, account_size=50_000, max_risk_per_trade=0.005)

        # risk_budget = 250, floor(250/100) = 2
        assert result.contracts == 2
        assert result.max_risk == 200.0

    def test_three_percent_risk(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=100.0, net_premium=-100.0)
        result = sizer.calculate(strategy, account_size=50_000, max_risk_per_trade=0.03)

        # risk_budget = 1500, floor(1500/100) = 15
        assert result.contracts == 15
        assert result.max_risk == 1500.0

    def test_default_risk_is_one_point_five_percent(self, sizer: OptionsPositionSizer):
        """Verify the default max_risk_per_trade is 1.5%."""
        strategy = _make_strategy(max_loss=100.0, net_premium=-100.0)
        result = sizer.calculate(strategy, account_size=100_000)

        # Default 1.5% of 100K = 1500, floor(1500/100) = 15
        assert result.contracts == 15

    def test_very_small_risk(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=50.0, net_premium=-50.0)
        result = sizer.calculate(strategy, account_size=10_000, max_risk_per_trade=0.001)

        # risk_budget = 10, floor(10/50) = 0 -> can't afford
        assert result.contracts == 0
        assert "exceeds risk budget" in result.reason

    def test_large_account_many_contracts(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=100.0, net_premium=-100.0)
        result = sizer.calculate(strategy, account_size=1_000_000, max_risk_per_trade=0.02)

        # risk_budget = 20_000, floor(20_000/100) = 200
        assert result.contracts == 200
        assert result.max_risk == 20_000.0


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_reason_message_includes_pct(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=100.0, net_premium=-100.0)
        result = sizer.calculate(strategy, account_size=50_000, max_risk_per_trade=0.015)

        assert "1.5% risk" in result.reason

    def test_net_cost_property(self, sizer: OptionsPositionSizer):
        """Verify the OptionsPositionSizeResult.net_cost property."""
        strategy = _make_strategy(max_loss=200.0, net_premium=-200.0)
        result = sizer.calculate(strategy, account_size=50_000, max_risk_per_trade=0.02)

        # Debit: premium_cost = 200*5 = 1000, premium_received = 0
        assert result.net_cost == 1000.0

    def test_credit_net_cost(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(
            name="credit_spread",
            max_loss=400.0,
            net_premium=100.0,
        )
        result = sizer.calculate(strategy, account_size=100_000, max_risk_per_trade=0.02)

        # risk_budget=2000, floor(2000/400)=5
        # premium_cost=0, premium_received=100*5=500
        assert result.net_cost == -500.0  # negative = net credit

    def test_max_risk_equals_max_loss_times_contracts(self, sizer: OptionsPositionSizer):
        strategy = _make_strategy(max_loss=175.0, net_premium=-175.0)
        result = sizer.calculate(strategy, account_size=25_000, max_risk_per_trade=0.015)

        # floor(375/175) = 2
        assert result.contracts == 2
        assert result.max_risk == 350.0  # 175 * 2
