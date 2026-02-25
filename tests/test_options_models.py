"""
Tests for options/models.py — pure dataclass and property tests, no mocking needed.
"""

import math
import pytest

from options.models import (
    IVAnalysis,
    IVRegime,
    OptionAction,
    OptionContract,
    OptionGreeks,
    OptionLeg,
    OptionRight,
    OptionsPositionSizeResult,
    OptionsStrategy,
    StrategyDirection,
)


# ---------------------------------------------------------------------------
# Helpers — reusable factory functions
# ---------------------------------------------------------------------------

def make_contract(symbol="AAPL", expiry="20260320", strike=200.0,
                  right=OptionRight.CALL) -> OptionContract:
    return OptionContract(symbol=symbol, expiry=expiry, strike=strike, right=right)


def make_greeks(delta=0.50, gamma=0.03, theta=-0.05, vega=0.12,
                bid=3.00, ask=3.20, option_price=3.10, **kwargs) -> OptionGreeks:
    return OptionGreeks(
        delta=delta, gamma=gamma, theta=theta, vega=vega,
        bid=bid, ask=ask, option_price=option_price, **kwargs,
    )


def make_leg(strike=200.0, right=OptionRight.CALL, action=OptionAction.BUY,
             quantity=1, delta=0.50, gamma=0.03, theta=-0.05, vega=0.12,
             bid=3.00, ask=3.20, option_price=3.10) -> OptionLeg:
    contract = make_contract(strike=strike, right=right)
    greeks = make_greeks(
        delta=delta, gamma=gamma, theta=theta, vega=vega,
        bid=bid, ask=ask, option_price=option_price,
    )
    return OptionLeg(contract=contract, action=action, quantity=quantity, greeks=greeks)


# ===========================================================================
# 1-2: OptionContract
# ===========================================================================

class TestOptionContract:
    def test_is_call(self):
        c = make_contract(right=OptionRight.CALL)
        assert c.is_call is True
        assert c.is_put is False

    def test_is_put(self):
        c = make_contract(right=OptionRight.PUT)
        assert c.is_put is True
        assert c.is_call is False

    def test_display_name(self):
        c = make_contract(symbol="AAPL", expiry="20260320", strike=200.0,
                          right=OptionRight.CALL)
        assert c.display_name == "AAPL 20260320 200.0C"

    def test_display_name_put(self):
        c = make_contract(symbol="TSLA", expiry="20260619", strike=150.5,
                          right=OptionRight.PUT)
        assert c.display_name == "TSLA 20260619 150.5P"


# ===========================================================================
# 3-6: OptionGreeks
# ===========================================================================

class TestOptionGreeks:
    def test_mid_price_from_bid_ask(self):
        g = make_greeks(bid=3.00, ask=3.20, option_price=5.00)
        assert g.mid_price == pytest.approx(3.10)

    def test_mid_price_fallback_to_option_price(self):
        g = make_greeks(bid=0.0, ask=0.0, option_price=2.50)
        assert g.mid_price == pytest.approx(2.50)

    def test_mid_price_fallback_when_bid_zero(self):
        g = make_greeks(bid=0.0, ask=3.20, option_price=1.80)
        assert g.mid_price == pytest.approx(1.80)

    def test_mid_price_fallback_when_ask_zero(self):
        g = make_greeks(bid=3.00, ask=0.0, option_price=1.80)
        assert g.mid_price == pytest.approx(1.80)

    def test_spread(self):
        g = make_greeks(bid=3.00, ask=3.20)
        assert g.spread == pytest.approx(0.20)

    def test_spread_zero_when_no_bid_ask(self):
        g = make_greeks(bid=0.0, ask=0.0)
        assert g.spread == pytest.approx(0.0)

    def test_spread_pct(self):
        g = make_greeks(bid=3.00, ask=3.20)
        # spread=0.20, mid=3.10, pct = 0.20/3.10*100
        expected = (0.20 / 3.10) * 100
        assert g.spread_pct == pytest.approx(expected)

    def test_spread_pct_zero_when_mid_zero(self):
        g = make_greeks(bid=0.0, ask=0.0, option_price=0.0)
        assert g.spread_pct == pytest.approx(0.0)


# ===========================================================================
# 7-9: OptionLeg
# ===========================================================================

class TestOptionLeg:
    def test_is_long(self):
        leg = make_leg(action=OptionAction.BUY)
        assert leg.is_long is True
        assert leg.is_short is False

    def test_is_short(self):
        leg = make_leg(action=OptionAction.SELL)
        assert leg.is_short is True
        assert leg.is_long is False

    def test_signed_delta_long(self):
        leg = make_leg(action=OptionAction.BUY, delta=0.50, quantity=1)
        assert leg.signed_delta == pytest.approx(0.50)

    def test_signed_delta_short(self):
        leg = make_leg(action=OptionAction.SELL, delta=0.50, quantity=1)
        assert leg.signed_delta == pytest.approx(-0.50)

    def test_signed_delta_with_quantity(self):
        leg = make_leg(action=OptionAction.BUY, delta=0.40, quantity=3)
        assert leg.signed_delta == pytest.approx(1.20)

    def test_signed_delta_no_greeks(self):
        contract = make_contract()
        leg = OptionLeg(contract=contract, action=OptionAction.BUY, greeks=None)
        assert leg.signed_delta == pytest.approx(0.0)

    def test_premium_buy_is_negative(self):
        # BUY: we pay, so premium should be negative
        leg = make_leg(action=OptionAction.BUY, bid=3.00, ask=3.20, quantity=1)
        # mid_price=3.10, multiplier=100 => -3.10 * 100 = -310
        assert leg.premium == pytest.approx(-310.0)

    def test_premium_sell_is_positive(self):
        # SELL: we receive, so premium should be positive
        leg = make_leg(action=OptionAction.SELL, bid=3.00, ask=3.20, quantity=1)
        assert leg.premium == pytest.approx(310.0)

    def test_premium_no_greeks(self):
        contract = make_contract()
        leg = OptionLeg(contract=contract, action=OptionAction.BUY, greeks=None)
        assert leg.premium == pytest.approx(0.0)

    def test_premium_scales_with_quantity(self):
        leg = make_leg(action=OptionAction.BUY, bid=2.00, ask=2.20, quantity=5)
        # mid=2.10, cost = -1 * 2.10 * 100 * 5 = -1050
        assert leg.premium == pytest.approx(-1050.0)


# ===========================================================================
# 10-16: OptionsStrategy (multi-leg)
# ===========================================================================

class TestOptionsStrategy:

    # --- Net greeks aggregation (10-11) ---

    def test_net_delta_sums_signed_deltas(self):
        leg1 = make_leg(action=OptionAction.BUY, delta=0.60)
        leg2 = make_leg(action=OptionAction.SELL, delta=0.30)
        strat = OptionsStrategy(name="test", underlying="AAPL",
                                direction=StrategyDirection.LONG, legs=[leg1, leg2])
        # 0.60 - 0.30 = 0.30
        assert strat.net_delta == pytest.approx(0.30)

    def test_net_gamma(self):
        leg1 = make_leg(action=OptionAction.BUY, gamma=0.05)
        leg2 = make_leg(action=OptionAction.SELL, gamma=0.03)
        strat = OptionsStrategy(name="test", underlying="X",
                                direction=StrategyDirection.NEUTRAL, legs=[leg1, leg2])
        assert strat.net_gamma == pytest.approx(0.02)

    def test_net_theta(self):
        leg1 = make_leg(action=OptionAction.BUY, theta=-0.08)
        leg2 = make_leg(action=OptionAction.SELL, theta=-0.04)
        strat = OptionsStrategy(name="test", underlying="X",
                                direction=StrategyDirection.NEUTRAL, legs=[leg1, leg2])
        # long: 1*(-0.08) = -0.08, short: -1*(-0.04) = 0.04 => net -0.04
        assert strat.net_theta == pytest.approx(-0.04)

    def test_net_vega(self):
        leg1 = make_leg(action=OptionAction.BUY, vega=0.15)
        leg2 = make_leg(action=OptionAction.SELL, vega=0.10)
        strat = OptionsStrategy(name="test", underlying="X",
                                direction=StrategyDirection.NEUTRAL, legs=[leg1, leg2])
        assert strat.net_vega == pytest.approx(0.05)

    # --- Debit / credit (12) ---

    def test_is_debit(self):
        strat = OptionsStrategy(name="long_call", underlying="AAPL",
                                direction=StrategyDirection.LONG, net_premium=-500.0)
        assert strat.is_debit is True
        assert strat.is_credit is False

    def test_is_credit(self):
        strat = OptionsStrategy(name="short_put", underlying="AAPL",
                                direction=StrategyDirection.SHORT, net_premium=250.0)
        assert strat.is_credit is True
        assert strat.is_debit is False

    def test_neither_debit_nor_credit_at_zero(self):
        strat = OptionsStrategy(name="zero", underlying="X",
                                direction=StrategyDirection.NEUTRAL, net_premium=0.0)
        assert strat.is_debit is False
        assert strat.is_credit is False

    # --- Defined risk (13) ---

    def test_is_defined_risk(self):
        strat = OptionsStrategy(name="spread", underlying="X",
                                direction=StrategyDirection.LONG,
                                max_loss=500.0, max_profit=500.0)
        assert strat.is_defined_risk is True

    def test_is_not_defined_risk_when_max_loss_zero(self):
        strat = OptionsStrategy(name="test", underlying="X",
                                direction=StrategyDirection.LONG, max_loss=0.0)
        assert strat.is_defined_risk is False

    def test_is_not_defined_risk_when_max_loss_infinite(self):
        strat = OptionsStrategy(name="test", underlying="X",
                                direction=StrategyDirection.LONG,
                                max_loss=float('inf'))
        assert strat.is_defined_risk is False

    # --- Risk/reward ratio (14) ---

    def test_risk_reward_ratio(self):
        strat = OptionsStrategy(name="test", underlying="X",
                                direction=StrategyDirection.LONG,
                                max_profit=1500.0, max_loss=500.0)
        assert strat.risk_reward_ratio == pytest.approx(3.0)

    def test_risk_reward_ratio_zero_loss(self):
        strat = OptionsStrategy(name="test", underlying="X",
                                direction=StrategyDirection.LONG,
                                max_profit=1000.0, max_loss=0.0)
        assert strat.risk_reward_ratio == pytest.approx(0.0)

    # --- Bull call spread (15) ---

    def test_bull_call_spread(self):
        """Buy 200C, sell 210C — net long delta, debit spread."""
        buy_leg = make_leg(
            strike=200.0, right=OptionRight.CALL, action=OptionAction.BUY,
            delta=0.55, gamma=0.04, theta=-0.06, vega=0.14,
            bid=8.00, ask=8.40,
        )
        sell_leg = make_leg(
            strike=210.0, right=OptionRight.CALL, action=OptionAction.SELL,
            delta=0.35, gamma=0.03, theta=-0.04, vega=0.10,
            bid=3.00, ask=3.40,
        )

        # net_premium: buy costs -820 (mid 8.20*100), sell receives +320 (mid 3.20*100)
        net_prem = buy_leg.premium + sell_leg.premium  # -820 + 320 = -500

        strat = OptionsStrategy(
            name="bull_call_spread", underlying="AAPL",
            direction=StrategyDirection.LONG,
            legs=[buy_leg, sell_leg],
            net_premium=net_prem,
            max_loss=abs(net_prem),   # 500
            max_profit=1000.0 - abs(net_prem),  # width*100 - debit = 500
            breakeven=[205.0],
        )

        # Net delta: 0.55 - 0.35 = 0.20
        assert strat.net_delta == pytest.approx(0.20)
        # Net gamma: 0.04 - 0.03 = 0.01
        assert strat.net_gamma == pytest.approx(0.01)
        # Net theta: -0.06 - (-0.04) = -0.02
        assert strat.net_theta == pytest.approx(-0.02)
        # Net vega: 0.14 - 0.10 = 0.04
        assert strat.net_vega == pytest.approx(0.04)

        assert strat.is_debit is True
        assert strat.is_credit is False
        assert strat.is_defined_risk is True
        assert strat.num_legs == 2
        assert strat.risk_reward_ratio == pytest.approx(1.0)

    # --- Iron condor (16) ---

    def test_iron_condor(self):
        """
        Iron condor on SPY at 500:
            Sell 490P, Buy 480P  (bull put spread)
            Sell 510C, Buy 520C  (bear call spread)
        All credit received, 4 legs, defined risk.
        """
        # Put spread legs
        sell_put = make_leg(
            strike=490.0, right=OptionRight.PUT, action=OptionAction.SELL,
            delta=-0.25, gamma=0.02, theta=-0.03, vega=0.08,
            bid=4.00, ask=4.40,
        )
        buy_put = make_leg(
            strike=480.0, right=OptionRight.PUT, action=OptionAction.BUY,
            delta=-0.15, gamma=0.01, theta=-0.02, vega=0.05,
            bid=2.00, ask=2.40,
        )
        # Call spread legs
        sell_call = make_leg(
            strike=510.0, right=OptionRight.CALL, action=OptionAction.SELL,
            delta=0.25, gamma=0.02, theta=-0.03, vega=0.08,
            bid=4.00, ask=4.40,
        )
        buy_call = make_leg(
            strike=520.0, right=OptionRight.CALL, action=OptionAction.BUY,
            delta=0.15, gamma=0.01, theta=-0.02, vega=0.05,
            bid=2.00, ask=2.40,
        )

        legs = [sell_put, buy_put, sell_call, buy_call]

        # Net premium: sell_put(+420) + buy_put(-220) + sell_call(+420) + buy_call(-220) = +400
        net_prem = sum(l.premium for l in legs)
        assert net_prem == pytest.approx(400.0)

        strat = OptionsStrategy(
            name="iron_condor", underlying="SPY",
            direction=StrategyDirection.NEUTRAL,
            legs=legs,
            net_premium=net_prem,
            max_loss=600.0,   # width(1000) - credit(400) = 600
            max_profit=net_prem,
            breakeven=[494.0, 514.0],
        )

        assert strat.num_legs == 4

        # Net delta:
        #   sell_put: -1*(-0.25) = 0.25
        #   buy_put:  1*(-0.15) = -0.15
        #   sell_call: -1*(0.25) = -0.25
        #   buy_call:  1*(0.15) = 0.15
        #   sum = 0.0
        assert strat.net_delta == pytest.approx(0.0)

        # Net gamma:
        #   sell_put: -0.02, buy_put: +0.01, sell_call: -0.02, buy_call: +0.01 => -0.02
        assert strat.net_gamma == pytest.approx(-0.02)

        # Net theta:
        #   sell_put: -1*(-0.03)=0.03, buy_put: 1*(-0.02)=-0.02
        #   sell_call: -1*(-0.03)=0.03, buy_call: 1*(-0.02)=-0.02
        #   sum = 0.02
        assert strat.net_theta == pytest.approx(0.02)

        # Net vega:
        #   sell_put: -0.08, buy_put: +0.05, sell_call: -0.08, buy_call: +0.05 => -0.06
        assert strat.net_vega == pytest.approx(-0.06)

        assert strat.is_credit is True
        assert strat.is_debit is False
        assert strat.is_defined_risk is True
        assert strat.risk_reward_ratio == pytest.approx(400.0 / 600.0)


# ===========================================================================
# 17-18: IVAnalysis
# ===========================================================================

class TestIVAnalysis:

    @pytest.mark.parametrize("iv_rank, expected_regime", [
        (10.0, IVRegime.LOW),
        (29.9, IVRegime.LOW),
        (30.0, IVRegime.LOW),       # boundary: <= 30 is LOW
        (30.1, IVRegime.NORMAL),    # > 30
        (50.0, IVRegime.NORMAL),    # boundary: <= 50 is NORMAL
        (50.1, IVRegime.HIGH),      # > 50
        (80.0, IVRegime.HIGH),      # boundary: <= 80 is HIGH
        (80.1, IVRegime.VERY_HIGH), # > 80
        (95.0, IVRegime.VERY_HIGH),
    ])
    def test_regime(self, iv_rank, expected_regime):
        iv = IVAnalysis(symbol="AAPL", iv_rank=iv_rank)
        assert iv.regime == expected_regime

    def test_has_iv_premium_true(self):
        iv = IVAnalysis(symbol="AAPL", iv_hv_ratio=1.25)
        assert iv.has_iv_premium is True

    def test_has_iv_premium_false(self):
        iv = IVAnalysis(symbol="AAPL", iv_hv_ratio=0.85)
        assert iv.has_iv_premium is False

    def test_has_iv_premium_exactly_one(self):
        iv = IVAnalysis(symbol="AAPL", iv_hv_ratio=1.0)
        assert iv.has_iv_premium is False


# ===========================================================================
# 19: OptionsPositionSizeResult
# ===========================================================================

class TestOptionsPositionSizeResult:

    def test_net_cost_debit(self):
        r = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=500.0,
            premium_cost=500.0, premium_received=0.0,
        )
        assert r.net_cost == pytest.approx(500.0)

    def test_net_cost_credit(self):
        r = OptionsPositionSizeResult(
            strategy_name="short_put", contracts=1, max_risk=1000.0,
            premium_cost=0.0, premium_received=300.0,
        )
        assert r.net_cost == pytest.approx(-300.0)

    def test_net_cost_mixed(self):
        r = OptionsPositionSizeResult(
            strategy_name="iron_condor", contracts=2, max_risk=1200.0,
            premium_cost=440.0, premium_received=840.0,
        )
        assert r.net_cost == pytest.approx(-400.0)
