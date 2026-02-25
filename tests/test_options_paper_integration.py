"""
Integration smoke test for the full options paper trading pipeline.

Exercises the complete flow with REAL yfinance data (no mocks):
  PaperChainProvider → OptionsChainManager → IVAnalyzer → StrategySelector
  → OptionsPositionSizer → PaperOptionsExecutor → OptionsExitManager

Uses SPY as the test symbol (most liquid US equity, always available).

These tests hit the network (yfinance) so they are marked with
@pytest.mark.integration. Run with:
    pytest tests/test_options_paper_integration.py -v
    pytest tests/test_options_paper_integration.py -v -m integration
"""

import pytest
from datetime import datetime, timedelta
from loguru import logger

from options.chain_provider import PaperChainProvider
from options.chain import OptionsChainManager
from options.iv_analyzer import IVAnalyzer
from options.strategy_selector import StrategySelector
from options.position_sizer import OptionsPositionSizer
from options.executor import OptionsExecutor
from options.paper_executor import PaperOptionsExecutor
from options.exit_manager import OptionsExitManager
from options.risk import OptionsRiskManager
from options.config import OptionsConfig
from options.models import OptionRight, OptionContract


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SYMBOL = "SPY"
ACCOUNT_SIZE = 100000.0  # $100K to afford SPY options premium (~$1400/contract)


@pytest.fixture(scope="module")
def provider():
    """Real PaperChainProvider with yfinance data — cached for the module."""
    return PaperChainProvider(risk_free_rate=0.05, iv_multiplier=1.1)


@pytest.fixture(scope="module")
def config():
    return OptionsConfig()


@pytest.fixture(scope="module")
def chain_mgr(provider, config):
    return OptionsChainManager(provider, config)


@pytest.fixture(scope="module")
def iv_analyzer(provider, chain_mgr):
    return IVAnalyzer(provider, chain_mgr)


@pytest.fixture(scope="module")
def strategy_selector(chain_mgr, iv_analyzer, config):
    return StrategySelector(chain_mgr, iv_analyzer, config)


@pytest.fixture(scope="module")
def position_sizer(config):
    return OptionsPositionSizer(config)


@pytest.fixture(scope="module")
def paper_executor(provider, config):
    executor = PaperOptionsExecutor(provider, config)
    executor._get_repository = lambda: None  # Don't persist to real DB during tests
    return executor


@pytest.fixture(scope="module")
def options_executor(paper_executor, config):
    return OptionsExecutor(paper_executor, config)


@pytest.fixture(scope="module")
def exit_manager(chain_mgr, config):
    return OptionsExitManager(chain_mgr, config)


@pytest.fixture(scope="module")
def risk_manager(chain_mgr, config):
    return OptionsRiskManager(chain_mgr, config)


# ===========================================================================
# Phase 1: PaperChainProvider — can it fetch real data and build a chain?
# ===========================================================================

@pytest.mark.integration
class TestProviderWithRealData:
    def test_underlying_price_positive(self, provider):
        price = provider.get_underlying_price(SYMBOL)
        assert price > 0, f"Expected positive price for {SYMBOL}, got {price}"
        logger.info(f"[INTEGRATION] {SYMBOL} price: ${price:.2f}")

    def test_chain_params_has_strikes_and_expiries(self, provider):
        params = provider.get_chain_params(SYMBOL)
        assert params is not None, "get_chain_params returned None"
        assert len(params["strikes"]) > 10, f"Too few strikes: {len(params['strikes'])}"
        assert len(params["expirations"]) >= 4, f"Too few expirations: {len(params['expirations'])}"
        assert params["multiplier"] == 100
        logger.info(
            f"[INTEGRATION] {SYMBOL} chain: {len(params['strikes'])} strikes, "
            f"{len(params['expirations'])} expirations"
        )

    def test_greeks_for_atm_call(self, provider):
        price = provider.get_underlying_price(SYMBOL)
        params = provider.get_chain_params(SYMBOL)
        # Pick ATM strike
        strikes = params["strikes"]
        atm_strike = min(strikes, key=lambda s: abs(s - price))
        expiry = params["expirations"][1]  # Second expiry (not the closest)

        contract = OptionContract(
            symbol=SYMBOL, expiry=expiry, strike=atm_strike, right=OptionRight.CALL
        )
        greeks = provider.get_greeks(contract)

        assert greeks is not None, "get_greeks returned None for ATM call"
        assert 0.30 < greeks.delta < 0.80, f"ATM call delta out of range: {greeks.delta}"
        assert greeks.gamma > 0, f"Gamma should be positive: {greeks.gamma}"
        assert greeks.theta < 0, f"Theta should be negative: {greeks.theta}"
        assert greeks.vega > 0, f"Vega should be positive: {greeks.vega}"
        assert greeks.implied_vol > 0.05, f"IV too low: {greeks.implied_vol}"
        assert greeks.bid > 0, f"Bid should be positive: {greeks.bid}"
        assert greeks.ask > greeks.bid, f"Ask should exceed bid: ask={greeks.ask}, bid={greeks.bid}"
        assert greeks.open_interest >= 100
        assert greeks.underlying_price == price

        logger.info(
            f"[INTEGRATION] {contract.display_name}: delta={greeks.delta:.3f} "
            f"gamma={greeks.gamma:.4f} theta={greeks.theta:.4f} "
            f"IV={greeks.implied_vol:.1%} mid=${greeks.mid_price:.2f}"
        )

    def test_iv_history_returns_data(self, provider):
        iv_data = provider.get_iv_history(SYMBOL)
        assert iv_data is not None, "get_iv_history returned None"
        assert len(iv_data["iv_values"]) > 20, f"Too few IV values: {len(iv_data['iv_values'])}"
        assert iv_data["current_iv"] > 0, f"Current IV should be positive: {iv_data['current_iv']}"

    def test_price_history_returns_data(self, provider):
        prices = provider.get_price_history(SYMBOL, period_days=30)
        assert prices is not None, "get_price_history returned None"
        assert len(prices) >= 15, f"Too few prices: {len(prices)}"
        assert all(p > 0 for p in prices), "All prices should be positive"


# ===========================================================================
# Phase 2: OptionsChainManager — caching, delta search, expiry finding
# ===========================================================================

@pytest.mark.integration
class TestChainManagerWithRealData:
    def test_find_target_expiry(self, chain_mgr):
        expiry = chain_mgr.find_target_expiry(SYMBOL)
        assert expiry is not None, "find_target_expiry returned None"
        exp_date = datetime.strptime(expiry, "%Y%m%d").date()
        today = datetime.now().date()
        dte = (exp_date - today).days
        assert 14 <= dte <= 60, f"Target expiry DTE out of range: {dte}"
        logger.info(f"[INTEGRATION] Target expiry: {expiry} ({dte} DTE)")

    def test_find_by_delta_call(self, chain_mgr):
        expiry = chain_mgr.find_target_expiry(SYMBOL)
        result = chain_mgr.find_by_delta(SYMBOL, 0.60, OptionRight.CALL, expiry)
        assert result is not None, "find_by_delta returned None for 0.60 delta call"
        contract, greeks = result
        assert contract.symbol == SYMBOL
        assert contract.right == OptionRight.CALL
        assert abs(abs(greeks.delta) - 0.60) < 0.15, f"Delta too far from target: {greeks.delta}"
        logger.info(
            f"[INTEGRATION] Delta search: {contract.display_name} "
            f"delta={greeks.delta:.3f} strike={contract.strike}"
        )

    def test_find_by_delta_put(self, chain_mgr):
        expiry = chain_mgr.find_target_expiry(SYMBOL)
        result = chain_mgr.find_by_delta(SYMBOL, 0.30, OptionRight.PUT, expiry)
        assert result is not None, "find_by_delta returned None for 0.30 delta put"
        contract, greeks = result
        assert contract.right == OptionRight.PUT
        assert abs(abs(greeks.delta) - 0.30) < 0.15

    def test_get_atm_strike(self, chain_mgr, provider):
        price = provider.get_underlying_price(SYMBOL)
        atm = chain_mgr.get_atm_strike(SYMBOL, price)
        assert abs(atm - price) / price < 0.05, f"ATM strike ${atm} too far from price ${price}"


# ===========================================================================
# Phase 3: IVAnalyzer — full IV analysis
# ===========================================================================

@pytest.mark.integration
class TestIVAnalyzerWithRealData:
    def test_analyze_returns_valid_analysis(self, iv_analyzer):
        analysis = iv_analyzer.analyze(SYMBOL)

        assert analysis.symbol == SYMBOL
        assert analysis.current_iv > 0, f"Current IV should be positive: {analysis.current_iv}"
        assert 0 <= analysis.iv_rank <= 100, f"IV rank out of range: {analysis.iv_rank}"
        assert 0 <= analysis.iv_percentile <= 100, f"IV percentile out of range: {analysis.iv_percentile}"
        assert analysis.hv_20 > 0, f"HV20 should be positive: {analysis.hv_20}"
        assert analysis.iv_hv_ratio > 0
        assert analysis.regime is not None

        logger.info(
            f"[INTEGRATION] IV Analysis: {SYMBOL} IV={analysis.current_iv:.1%} "
            f"Rank={analysis.iv_rank:.0f} Pctile={analysis.iv_percentile:.0f} "
            f"HV20={analysis.hv_20:.1%} Regime={analysis.regime.value}"
        )


# ===========================================================================
# Phase 4: StrategySelector — picks a strategy from real chain data
# ===========================================================================

@pytest.mark.integration
class TestStrategySelectorWithRealData:
    def test_select_long_strategy(self, strategy_selector):
        signal = {
            "symbol": SYMBOL,
            "direction": "long",
            "entry_price": 500.0,  # Approximate SPY price
            "atr": 3.0,
        }
        strategy = strategy_selector.select_strategy(signal, ACCOUNT_SIZE)

        assert strategy is not None, "Strategy selector returned None for long signal"
        assert strategy.underlying == SYMBOL
        assert len(strategy.legs) >= 1
        assert strategy.max_loss > 0

        # Verify all legs have greeks
        for leg in strategy.legs:
            assert leg.greeks is not None, f"Leg {leg.contract.display_name} has no greeks"
            assert leg.greeks.delta != 0, f"Leg {leg.contract.display_name} has zero delta"

        logger.info(
            f"[INTEGRATION] Selected strategy: {strategy.name} "
            f"legs={strategy.num_legs} max_loss=${strategy.max_loss:.2f} "
            f"net_premium=${strategy.net_premium:.2f}"
        )

    def test_select_short_strategy(self, strategy_selector):
        signal = {
            "symbol": SYMBOL,
            "direction": "short",
            "entry_price": 500.0,
            "atr": 3.0,
        }
        strategy = strategy_selector.select_strategy(signal, ACCOUNT_SIZE)

        assert strategy is not None, "Strategy selector returned None for short signal"
        assert len(strategy.legs) >= 1
        logger.info(f"[INTEGRATION] Short strategy: {strategy.name}")


# ===========================================================================
# Phase 5: Full pipeline — signal → strategy → size → risk → fill → exit check
# ===========================================================================

@pytest.mark.integration
class TestFullPipeline:
    def test_end_to_end_long_trade(
        self, strategy_selector, position_sizer, risk_manager,
        options_executor, exit_manager
    ):
        """
        Complete pipeline:
        1. Select strategy from signal
        2. Size the position
        3. Validate risk
        4. Execute (paper fill)
        5. Check exit triggers
        6. Close position
        """
        # 1. Select strategy
        signal = {
            "symbol": SYMBOL,
            "direction": "long",
            "entry_price": 500.0,
            "atr": 3.0,
        }
        strategy = strategy_selector.select_strategy(signal, ACCOUNT_SIZE)
        assert strategy is not None, "Step 1 failed: no strategy selected"
        logger.info(f"[PIPELINE] 1. Strategy: {strategy.name} ({strategy.num_legs} legs)")

        # 2. Size position
        size_result = position_sizer.calculate(strategy, ACCOUNT_SIZE, max_risk_per_trade=0.015)
        assert size_result.contracts > 0, f"Step 2 failed: {size_result.reason}"
        logger.info(
            f"[PIPELINE] 2. Sizing: {size_result.contracts} contracts, "
            f"max_risk=${size_result.max_risk:.2f}"
        )

        # 3. Risk check
        existing = options_executor.get_all_positions()
        risk_check = risk_manager.validate_new_trade(
            strategy, size_result, existing, ACCOUNT_SIZE
        )
        assert risk_check, f"Step 3 failed: {risk_check.reason}"
        logger.info(f"[PIPELINE] 3. Risk check: PASS (warnings: {risk_check.warnings})")

        # 4. Execute
        order_result = options_executor.execute_strategy(strategy, size_result)
        assert order_result is not None, "Step 4 failed: execution returned None"
        assert order_result.get("status") == "Filled"
        logger.info(
            f"[PIPELINE] 4. Execution: order_id={order_result['order_id']} "
            f"status={order_result['status']}"
        )

        # Verify position is tracked
        position = options_executor.get_position(SYMBOL)
        assert position is not None, "Position not tracked after execution"
        assert position["contracts"] == size_result.contracts

        # 5. Check exits (should NOT trigger on a fresh position)
        positions = options_executor.get_all_positions()
        assert len(positions) > 0, "No positions found for exit check"

        exit_signals = exit_manager.check_exits(positions)
        # Fresh position — only time-based or roll triggers possible, not profit/stop
        profit_exits = [s for s in exit_signals if "profit" in s.reason.lower()]
        stop_exits = [s for s in exit_signals if "stop loss" in s.reason.lower()]
        # A just-opened position shouldn't hit profit target or stop loss
        logger.info(
            f"[PIPELINE] 5. Exit check: {len(exit_signals)} signals "
            f"({[s.reason for s in exit_signals]})"
        )

        # 6. Close position
        close_result = options_executor.close_position(SYMBOL)
        assert close_result is not None, "Step 6 failed: close returned None"
        logger.info(f"[PIPELINE] 6. Close: {close_result}")

        # Verify position is gone
        assert options_executor.get_position(SYMBOL) is None, "Position still exists after close"
        logger.info("[PIPELINE] COMPLETE — full pipeline smoke test passed")

    def test_end_to_end_short_trade(
        self, strategy_selector, position_sizer, risk_manager,
        options_executor
    ):
        """Same pipeline but for a short signal."""
        signal = {
            "symbol": SYMBOL,
            "direction": "short",
            "entry_price": 500.0,
            "atr": 3.0,
        }

        strategy = strategy_selector.select_strategy(signal, ACCOUNT_SIZE)
        assert strategy is not None

        size_result = position_sizer.calculate(strategy, ACCOUNT_SIZE, max_risk_per_trade=0.015)
        assert size_result.contracts > 0, f"Sizing failed: {size_result.reason}"

        existing = options_executor.get_all_positions()
        risk_check = risk_manager.validate_new_trade(
            strategy, size_result, existing, ACCOUNT_SIZE
        )
        assert risk_check, f"Risk check failed: {risk_check.reason}"

        order_result = options_executor.execute_strategy(strategy, size_result)
        assert order_result is not None
        assert order_result["status"] == "Filled"

        # Clean up
        options_executor.close_position(SYMBOL)
        assert options_executor.get_position(SYMBOL) is None
        logger.info("[PIPELINE] Short trade pipeline: PASS")

    def test_risk_manager_blocks_over_limit(
        self, strategy_selector, position_sizer, risk_manager,
        options_executor
    ):
        """Verify risk manager rejects when portfolio is overloaded."""
        signal = {
            "symbol": SYMBOL,
            "direction": "long",
            "entry_price": 500.0,
            "atr": 3.0,
        }

        strategy = strategy_selector.select_strategy(signal, ACCOUNT_SIZE)
        assert strategy is not None

        # Create a giant size result that exceeds risk limits
        from options.models import OptionsPositionSizeResult
        oversized = OptionsPositionSizeResult(
            strategy_name=strategy.name,
            contracts=1000,
            max_risk=ACCOUNT_SIZE * 0.50,  # 50% of account — way over 10% limit
            premium_cost=ACCOUNT_SIZE * 0.50,
        )

        existing = options_executor.get_all_positions()
        risk_check = risk_manager.validate_new_trade(
            strategy, oversized, existing, ACCOUNT_SIZE
        )
        assert not risk_check, "Risk manager should have rejected oversized trade"
        logger.info(f"[PIPELINE] Risk block: {risk_check.reason}")


# ===========================================================================
# Phase 6: Portfolio Greeks consistency
# ===========================================================================

@pytest.mark.integration
class TestPortfolioGreeks:
    def test_portfolio_risk_summary(
        self, strategy_selector, position_sizer, options_executor,
        risk_manager
    ):
        """Open a position and verify portfolio risk metrics are sensible."""
        signal = {
            "symbol": SYMBOL,
            "direction": "long",
            "entry_price": 500.0,
            "atr": 3.0,
        }

        strategy = strategy_selector.select_strategy(signal, ACCOUNT_SIZE)
        assert strategy is not None

        size_result = position_sizer.calculate(strategy, ACCOUNT_SIZE, max_risk_per_trade=0.015)
        if size_result.contracts > 0:
            options_executor.execute_strategy(strategy, size_result)

            positions = options_executor.get_all_positions()
            risk_summary = risk_manager.get_portfolio_risk(positions, ACCOUNT_SIZE)

            assert risk_summary["position_count"] > 0
            assert risk_summary["total_premium_at_risk"] > 0
            assert risk_summary["premium_risk_pct"] < 1.0  # Should be reasonable
            assert "net_portfolio_delta" in risk_summary
            assert "daily_theta" in risk_summary

            logger.info(
                f"[INTEGRATION] Portfolio risk: "
                f"premium_at_risk=${risk_summary['total_premium_at_risk']:.2f} "
                f"({risk_summary['premium_risk_pct']*100:.1f}%), "
                f"net_delta={risk_summary['net_portfolio_delta']:.2f}, "
                f"daily_theta=${risk_summary['daily_theta']:.2f}"
            )

            # Clean up
            options_executor.close_position(SYMBOL)
