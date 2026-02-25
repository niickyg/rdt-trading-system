"""
Options Strategy Selector for the RDT Trading System.

Converts directional stock signals into optimal options strategies
based on IV regime, signal direction, and risk parameters.

Decision tree:
| Direction | IV Regime    | Strategy                |
|-----------|-------------|-------------------------|
| LONG      | LOW (<30)   | Long Call               |
| LONG      | NORMAL      | Bull Call Spread         |
| LONG      | HIGH (>50)  | Bull Put Spread (credit) |
| SHORT     | LOW         | Long Put                |
| SHORT     | NORMAL      | Bear Put Spread          |
| SHORT     | HIGH        | Bear Call Spread (credit) |
| Any       | VERY HIGH   | Iron Condor             |
"""

from typing import Dict, Optional
from loguru import logger

from options.models import (
    OptionContract, OptionGreeks, OptionLeg, OptionsStrategy,
    OptionRight, OptionAction, StrategyDirection, IVRegime,
)
from options.config import OptionsConfig
from options.chain import OptionsChainManager
from options.iv_analyzer import IVAnalyzer


class StrategySelector:
    """
    Selects the optimal options strategy for a given signal.

    Usage:
        selector = StrategySelector(chain_mgr, iv_analyzer, config)
        strategy = selector.select_strategy(signal, account_size=25000)
    """

    def __init__(
        self,
        chain_manager: OptionsChainManager,
        iv_analyzer: IVAnalyzer,
        config: Optional[OptionsConfig] = None,
    ):
        self._chain = chain_manager
        self._iv = iv_analyzer
        self._config = config or OptionsConfig()

    def select_strategy(
        self,
        signal: Dict,
        account_size: float,
    ) -> Optional[OptionsStrategy]:
        """
        Select the optimal options strategy for a signal.

        Args:
            signal: Trade signal dict with keys:
                - symbol (str)
                - direction (str): "long" or "short"
                - entry_price (float)
                - atr (float)
            account_size: Current account size for position sizing context

        Returns:
            OptionsStrategy or None if no suitable strategy found
        """
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "long").lower()

        if not symbol:
            logger.error("Signal missing symbol")
            return None

        # Check for forced strategy
        if self._config.force_strategy:
            return self._build_forced_strategy(signal)

        # Get IV analysis
        iv_analysis = self._iv.analyze(symbol)
        regime = iv_analysis.regime

        logger.info(
            f"Strategy selection: {symbol} direction={direction} "
            f"IV_rank={iv_analysis.iv_rank:.0f} regime={regime.value}"
        )

        # Find target expiry
        expiry = self._chain.find_target_expiry(symbol)
        if not expiry:
            logger.error(f"No suitable expiry found for {symbol}")
            return None

        # Very high IV -> Iron Condor regardless of direction
        if regime == IVRegime.VERY_HIGH:
            return self._build_iron_condor(signal, expiry, iv_analysis)

        # Direction-based selection
        if direction == "long":
            if regime == IVRegime.LOW:
                return self._build_long_call(signal, expiry, iv_analysis)
            elif regime == IVRegime.NORMAL:
                return self._build_bull_call_spread(signal, expiry, iv_analysis)
            else:  # HIGH
                return self._build_bull_put_spread(signal, expiry, iv_analysis)
        else:  # short
            if regime == IVRegime.LOW:
                return self._build_long_put(signal, expiry, iv_analysis)
            elif regime == IVRegime.NORMAL:
                return self._build_bear_put_spread(signal, expiry, iv_analysis)
            else:  # HIGH
                return self._build_bear_call_spread(signal, expiry, iv_analysis)

    def _build_long_call(self, signal: Dict, expiry: str, iv) -> Optional[OptionsStrategy]:
        """Long call: buy call at target delta (0.55-0.65)."""
        symbol = signal["symbol"]
        target_delta = self._config.long_delta_target

        result = self._chain.find_by_delta(symbol, target_delta, OptionRight.CALL, expiry)
        if not result:
            return None

        contract, greeks = result
        underlying_price = greeks.underlying_price or signal.get("entry_price", 0)

        leg = OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1, greeks=greeks)
        premium = greeks.mid_price * contract.multiplier

        strategy = OptionsStrategy(
            name="long_call",
            underlying=symbol,
            direction=StrategyDirection.LONG,
            legs=[leg],
            max_loss=premium,
            max_profit=float('inf'),
            breakeven=[contract.strike + greeks.mid_price],
            net_premium=-premium,
        )

        logger.info(
            f"Strategy: LONG CALL {contract.display_name} "
            f"delta={greeks.delta:.2f} premium=${premium:.2f}"
        )
        return strategy

    def _build_long_put(self, signal: Dict, expiry: str, iv) -> Optional[OptionsStrategy]:
        """Long put: buy put at target delta (-0.55 to -0.65)."""
        symbol = signal["symbol"]
        target_delta = self._config.long_delta_target  # Will match abs delta

        result = self._chain.find_by_delta(symbol, target_delta, OptionRight.PUT, expiry)
        if not result:
            return None

        contract, greeks = result
        leg = OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1, greeks=greeks)
        premium = greeks.mid_price * contract.multiplier

        strategy = OptionsStrategy(
            name="long_put",
            underlying=symbol,
            direction=StrategyDirection.SHORT,
            legs=[leg],
            max_loss=premium,
            max_profit=(contract.strike - greeks.mid_price) * contract.multiplier,
            breakeven=[contract.strike - greeks.mid_price],
            net_premium=-premium,
        )

        logger.info(
            f"Strategy: LONG PUT {contract.display_name} "
            f"delta={greeks.delta:.2f} premium=${premium:.2f}"
        )
        return strategy

    def _build_bull_call_spread(self, signal: Dict, expiry: str, iv) -> Optional[OptionsStrategy]:
        """Bull call spread: buy ITM call, sell OTM call."""
        symbol = signal["symbol"]
        underlying_price = signal.get("entry_price", 0)

        # Buy call at long delta target (0.55-0.65)
        long_result = self._chain.find_by_delta(
            symbol, self._config.long_delta_target, OptionRight.CALL, expiry
        )
        if not long_result:
            return None

        # Sell call at short leg delta (0.25-0.35)
        short_result = self._chain.find_by_delta(
            symbol, self._config.short_leg_delta_target, OptionRight.CALL, expiry
        )
        if not short_result:
            return None

        long_contract, long_greeks = long_result
        short_contract, short_greeks = short_result

        # Validate spread width
        spread_width = short_contract.strike - long_contract.strike
        max_width = self._config.get_max_spread_width(underlying_price)
        if spread_width <= 0 or spread_width > max_width:
            logger.warning(f"Invalid spread width ${spread_width} for {symbol}")
            return None

        long_leg = OptionLeg(
            contract=long_contract, action=OptionAction.BUY, quantity=1, greeks=long_greeks
        )
        short_leg = OptionLeg(
            contract=short_contract, action=OptionAction.SELL, quantity=1, greeks=short_greeks
        )

        net_debit = (long_greeks.mid_price - short_greeks.mid_price) * long_contract.multiplier
        max_profit = (spread_width * long_contract.multiplier) - net_debit

        strategy = OptionsStrategy(
            name="bull_call_spread",
            underlying=symbol,
            direction=StrategyDirection.LONG,
            legs=[long_leg, short_leg],
            max_loss=net_debit,
            max_profit=max_profit,
            breakeven=[long_contract.strike + (net_debit / long_contract.multiplier)],
            net_premium=-net_debit,
        )

        logger.info(
            f"Strategy: BULL CALL SPREAD {symbol} "
            f"{long_contract.strike}/{short_contract.strike} "
            f"debit=${net_debit:.2f} max_profit=${max_profit:.2f}"
        )
        return strategy

    def _build_bear_put_spread(self, signal: Dict, expiry: str, iv) -> Optional[OptionsStrategy]:
        """Bear put spread: buy ITM put, sell OTM put."""
        symbol = signal["symbol"]
        underlying_price = signal.get("entry_price", 0)

        # Buy put at long delta target
        long_result = self._chain.find_by_delta(
            symbol, self._config.long_delta_target, OptionRight.PUT, expiry
        )
        if not long_result:
            return None

        # Sell put at short leg delta
        short_result = self._chain.find_by_delta(
            symbol, self._config.short_leg_delta_target, OptionRight.PUT, expiry
        )
        if not short_result:
            return None

        long_contract, long_greeks = long_result
        short_contract, short_greeks = short_result

        # For puts: long strike should be higher than short strike
        spread_width = long_contract.strike - short_contract.strike
        max_width = self._config.get_max_spread_width(underlying_price)
        if spread_width <= 0 or spread_width > max_width:
            logger.warning(f"Invalid put spread width ${spread_width} for {symbol}")
            return None

        long_leg = OptionLeg(
            contract=long_contract, action=OptionAction.BUY, quantity=1, greeks=long_greeks
        )
        short_leg = OptionLeg(
            contract=short_contract, action=OptionAction.SELL, quantity=1, greeks=short_greeks
        )

        net_debit = (long_greeks.mid_price - short_greeks.mid_price) * long_contract.multiplier
        max_profit = (spread_width * long_contract.multiplier) - net_debit

        strategy = OptionsStrategy(
            name="bear_put_spread",
            underlying=symbol,
            direction=StrategyDirection.SHORT,
            legs=[long_leg, short_leg],
            max_loss=net_debit,
            max_profit=max_profit,
            breakeven=[long_contract.strike - (net_debit / long_contract.multiplier)],
            net_premium=-net_debit,
        )

        logger.info(
            f"Strategy: BEAR PUT SPREAD {symbol} "
            f"{long_contract.strike}/{short_contract.strike} "
            f"debit=${net_debit:.2f} max_profit=${max_profit:.2f}"
        )
        return strategy

    def _build_bull_put_spread(self, signal: Dict, expiry: str, iv) -> Optional[OptionsStrategy]:
        """Bull put spread (credit): sell higher put, buy lower put."""
        symbol = signal["symbol"]
        underlying_price = signal.get("entry_price", 0)

        # Sell put at 0.30-0.40 delta
        sell_result = self._chain.find_by_delta(
            symbol, 0.35, OptionRight.PUT, expiry
        )
        if not sell_result:
            return None

        # Buy put at 0.10-0.20 delta
        buy_result = self._chain.find_by_delta(
            symbol, 0.15, OptionRight.PUT, expiry
        )
        if not buy_result:
            return None

        sell_contract, sell_greeks = sell_result
        buy_contract, buy_greeks = buy_result

        spread_width = sell_contract.strike - buy_contract.strike
        max_width = self._config.get_max_spread_width(underlying_price)
        if spread_width <= 0 or spread_width > max_width:
            logger.warning(f"Invalid credit spread width ${spread_width} for {symbol}")
            return None

        sell_leg = OptionLeg(
            contract=sell_contract, action=OptionAction.SELL, quantity=1, greeks=sell_greeks
        )
        buy_leg = OptionLeg(
            contract=buy_contract, action=OptionAction.BUY, quantity=1, greeks=buy_greeks
        )

        net_credit = (sell_greeks.mid_price - buy_greeks.mid_price) * sell_contract.multiplier
        max_loss = (spread_width * sell_contract.multiplier) - net_credit

        strategy = OptionsStrategy(
            name="bull_put_spread",
            underlying=symbol,
            direction=StrategyDirection.LONG,
            legs=[sell_leg, buy_leg],
            max_loss=max_loss,
            max_profit=net_credit,
            breakeven=[sell_contract.strike - (net_credit / sell_contract.multiplier)],
            net_premium=net_credit,
        )

        logger.info(
            f"Strategy: BULL PUT SPREAD (credit) {symbol} "
            f"{sell_contract.strike}/{buy_contract.strike} "
            f"credit=${net_credit:.2f} max_loss=${max_loss:.2f}"
        )
        return strategy

    def _build_bear_call_spread(self, signal: Dict, expiry: str, iv) -> Optional[OptionsStrategy]:
        """Bear call spread (credit): sell lower call, buy higher call."""
        symbol = signal["symbol"]
        underlying_price = signal.get("entry_price", 0)

        # Sell call at 0.30-0.40 delta
        sell_result = self._chain.find_by_delta(
            symbol, 0.35, OptionRight.CALL, expiry
        )
        if not sell_result:
            return None

        # Buy call at 0.10-0.20 delta
        buy_result = self._chain.find_by_delta(
            symbol, 0.15, OptionRight.CALL, expiry
        )
        if not buy_result:
            return None

        sell_contract, sell_greeks = sell_result
        buy_contract, buy_greeks = buy_result

        spread_width = buy_contract.strike - sell_contract.strike
        max_width = self._config.get_max_spread_width(underlying_price)
        if spread_width <= 0 or spread_width > max_width:
            logger.warning(f"Invalid credit spread width ${spread_width} for {symbol}")
            return None

        sell_leg = OptionLeg(
            contract=sell_contract, action=OptionAction.SELL, quantity=1, greeks=sell_greeks
        )
        buy_leg = OptionLeg(
            contract=buy_contract, action=OptionAction.BUY, quantity=1, greeks=buy_greeks
        )

        net_credit = (sell_greeks.mid_price - buy_greeks.mid_price) * sell_contract.multiplier
        max_loss = (spread_width * sell_contract.multiplier) - net_credit

        strategy = OptionsStrategy(
            name="bear_call_spread",
            underlying=symbol,
            direction=StrategyDirection.SHORT,
            legs=[sell_leg, buy_leg],
            max_loss=max_loss,
            max_profit=net_credit,
            breakeven=[sell_contract.strike + (net_credit / sell_contract.multiplier)],
            net_premium=net_credit,
        )

        logger.info(
            f"Strategy: BEAR CALL SPREAD (credit) {symbol} "
            f"{sell_contract.strike}/{buy_contract.strike} "
            f"credit=${net_credit:.2f} max_loss=${max_loss:.2f}"
        )
        return strategy

    def _build_iron_condor(self, signal: Dict, expiry: str, iv) -> Optional[OptionsStrategy]:
        """Iron condor: sell OTM put spread + sell OTM call spread."""
        symbol = signal["symbol"]
        underlying_price = signal.get("entry_price", 0)
        max_width = self._config.get_max_spread_width(underlying_price)

        # Sell put at 0.20 delta, buy put at 0.10 delta
        sell_put_result = self._chain.find_by_delta(symbol, 0.20, OptionRight.PUT, expiry)
        buy_put_result = self._chain.find_by_delta(symbol, 0.10, OptionRight.PUT, expiry)

        # Sell call at 0.20 delta, buy call at 0.10 delta
        sell_call_result = self._chain.find_by_delta(symbol, 0.20, OptionRight.CALL, expiry)
        buy_call_result = self._chain.find_by_delta(symbol, 0.10, OptionRight.CALL, expiry)

        if not all([sell_put_result, buy_put_result, sell_call_result, buy_call_result]):
            logger.warning(f"Could not build iron condor for {symbol} — missing legs")
            return None

        sp_contract, sp_greeks = sell_put_result
        bp_contract, bp_greeks = buy_put_result
        sc_contract, sc_greeks = sell_call_result
        bc_contract, bc_greeks = buy_call_result

        # Validate wing widths
        put_width = sp_contract.strike - bp_contract.strike
        call_width = bc_contract.strike - sc_contract.strike
        if put_width <= 0 or call_width <= 0:
            logger.warning(f"Invalid iron condor wing widths for {symbol}")
            return None

        legs = [
            OptionLeg(contract=sp_contract, action=OptionAction.SELL, quantity=1, greeks=sp_greeks),
            OptionLeg(contract=bp_contract, action=OptionAction.BUY, quantity=1, greeks=bp_greeks),
            OptionLeg(contract=sc_contract, action=OptionAction.SELL, quantity=1, greeks=sc_greeks),
            OptionLeg(contract=bc_contract, action=OptionAction.BUY, quantity=1, greeks=bc_greeks),
        ]

        multiplier = sp_contract.multiplier
        net_credit = (
            (sp_greeks.mid_price - bp_greeks.mid_price) +
            (sc_greeks.mid_price - bc_greeks.mid_price)
        ) * multiplier

        wider_wing = max(put_width, call_width)
        max_loss = (wider_wing * multiplier) - net_credit

        strategy = OptionsStrategy(
            name="iron_condor",
            underlying=symbol,
            direction=StrategyDirection.NEUTRAL,
            legs=legs,
            max_loss=max_loss,
            max_profit=net_credit,
            breakeven=[
                sp_contract.strike - (net_credit / multiplier),
                sc_contract.strike + (net_credit / multiplier),
            ],
            net_premium=net_credit,
        )

        logger.info(
            f"Strategy: IRON CONDOR {symbol} "
            f"puts={bp_contract.strike}/{sp_contract.strike} "
            f"calls={sc_contract.strike}/{bc_contract.strike} "
            f"credit=${net_credit:.2f} max_loss=${max_loss:.2f}"
        )
        return strategy

    def _build_forced_strategy(self, signal: Dict) -> Optional[OptionsStrategy]:
        """Build a specific forced strategy from config override."""
        forced = self._config.force_strategy.lower()
        symbol = signal["symbol"]
        expiry = self._chain.find_target_expiry(symbol)
        if not expiry:
            return None

        iv_analysis = self._iv.analyze(symbol)

        strategy_map = {
            "long_call": self._build_long_call,
            "long_put": self._build_long_put,
            "bull_call_spread": self._build_bull_call_spread,
            "bear_put_spread": self._build_bear_put_spread,
            "bull_put_spread": self._build_bull_put_spread,
            "bear_call_spread": self._build_bear_call_spread,
            "iron_condor": self._build_iron_condor,
        }

        builder = strategy_map.get(forced)
        if builder is None:
            logger.error(f"Unknown forced strategy: {forced}")
            return None

        logger.info(f"Using forced strategy: {forced} for {symbol}")
        return builder(signal, expiry, iv_analysis)
