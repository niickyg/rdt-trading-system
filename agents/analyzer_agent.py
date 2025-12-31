"""
Analyzer Agent
Analyzes trading signals and validates setups
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from agents.base import BaseAgent
from agents.events import Event, EventType
from risk import RiskManager, PositionSizer, RiskLimits


class AnalyzerAgent(BaseAgent):
    """
    Trade setup analyzer agent

    Responsibilities:
    - Analyze signals from scanner
    - Validate against entry criteria
    - Calculate position sizing
    - Run risk checks
    - Publish valid setups for execution
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        position_sizer: Optional[PositionSizer] = None,
        **kwargs
    ):
        super().__init__(name="AnalyzerAgent", **kwargs)

        self.risk_manager = risk_manager
        self.position_sizer = position_sizer or PositionSizer()

        # Analysis configuration
        self.min_rrs = 2.0
        self.min_rr_ratio = 2.0
        self.require_daily_alignment = True
        self.max_atr_percent = 5.0  # Max ATR as % of price

        # Tracking
        self.signals_analyzed = 0
        self.setups_approved = 0
        self.setups_rejected = 0

    async def initialize(self):
        """Initialize analyzer"""
        logger.info("Analyzer agent initialized")
        self.metrics.custom_metrics["signals_analyzed"] = 0
        self.metrics.custom_metrics["approval_rate"] = 0

    async def cleanup(self):
        """Cleanup analyzer"""
        pass

    def get_subscribed_events(self) -> List[EventType]:
        """Events this agent listens to"""
        return [
            EventType.SIGNAL_FOUND,
            EventType.ANALYSIS_REQUESTED
        ]

    async def handle_event(self, event: Event):
        """Handle incoming events"""
        if event.event_type == EventType.SIGNAL_FOUND:
            await self.analyze_signal(event.data)

        elif event.event_type == EventType.ANALYSIS_REQUESTED:
            await self.analyze_signal(event.data)

    async def analyze_signal(self, signal: Dict):
        """
        Analyze a trading signal

        Args:
            signal: Signal data from scanner
        """
        self.signals_analyzed += 1
        self.metrics.custom_metrics["signals_analyzed"] = self.signals_analyzed

        symbol = signal.get("symbol")
        direction = signal.get("direction")
        price = signal.get("price")
        atr = signal.get("atr", 0)
        rrs = signal.get("rrs", 0)

        logger.info(f"Analyzing: {symbol} {direction} RRS={rrs:.2f}")

        # Validation checks
        rejection_reasons = []

        # Check 1: RRS strength
        if abs(rrs) < self.min_rrs:
            rejection_reasons.append(f"RRS too weak: {rrs:.2f} < {self.min_rrs}")

        # Check 2: Daily chart alignment
        if self.require_daily_alignment:
            if direction == "long" and not signal.get("daily_strong"):
                rejection_reasons.append("Daily chart not strong for long")
            elif direction == "short" and not signal.get("daily_weak"):
                rejection_reasons.append("Daily chart not weak for short")

        # Check 3: ATR reasonableness
        if price > 0 and atr > 0:
            atr_percent = (atr / price) * 100
            if atr_percent > self.max_atr_percent:
                rejection_reasons.append(f"ATR too high: {atr_percent:.1f}%")

        # Calculate position sizing
        position_result = self.position_sizer.calculate_position_size(
            account_size=self.risk_manager.current_balance,
            entry_price=price,
            atr=atr,
            direction=direction
        )

        # Check 4: Position viability
        if position_result.shares == 0:
            rejection_reasons.append("Position size calculated as 0")

        # Check 5: Risk/Reward
        if position_result.risk_reward_ratio < self.min_rr_ratio:
            rejection_reasons.append(
                f"R/R too low: {position_result.risk_reward_ratio:.2f}"
            )

        # Check 6: Risk management validation
        trade_risk = self.risk_manager.validate_trade(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            shares=position_result.shares,
            stop_price=position_result.stop_price,
            target_price=position_result.target_price,
            atr=atr
        )

        for failed_check in trade_risk.checks_failed:
            rejection_reasons.append(failed_check.message)

        # Decision
        if rejection_reasons:
            await self._reject_setup(symbol, direction, rejection_reasons)
        else:
            await self._approve_setup(signal, position_result, trade_risk)

    async def _approve_setup(
        self,
        signal: Dict,
        position_result,
        trade_risk
    ):
        """Approve a valid setup"""
        self.setups_approved += 1
        self._update_approval_rate()

        setup = {
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "entry_price": signal["price"],
            "stop_price": position_result.stop_price,
            "target_price": position_result.target_price,
            "shares": position_result.shares,
            "position_value": position_result.position_value,
            "risk_amount": position_result.risk_amount,
            "risk_reward_ratio": position_result.risk_reward_ratio,
            "rrs": signal["rrs"],
            "atr": signal.get("atr"),
            "signal": signal,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            f"APPROVED: {setup['symbol']} {setup['direction'].upper()} "
            f"{setup['shares']} shares @ ${setup['entry_price']:.2f} "
            f"Stop: ${setup['stop_price']:.2f} Target: ${setup['target_price']:.2f}"
        )

        await self.publish(EventType.SETUP_VALID, setup)

    async def _reject_setup(
        self,
        symbol: str,
        direction: str,
        reasons: List[str]
    ):
        """Reject an invalid setup"""
        self.setups_rejected += 1
        self._update_approval_rate()

        logger.info(f"REJECTED: {symbol} {direction} - {'; '.join(reasons)}")

        await self.publish(EventType.SETUP_INVALID, {
            "symbol": symbol,
            "direction": direction,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat()
        })

    def _update_approval_rate(self):
        """Update approval rate metric"""
        if self.signals_analyzed > 0:
            rate = (self.setups_approved / self.signals_analyzed) * 100
            self.metrics.custom_metrics["approval_rate"] = round(rate, 1)
            self.metrics.custom_metrics["setups_approved"] = self.setups_approved
            self.metrics.custom_metrics["setups_rejected"] = self.setups_rejected

    def set_min_rrs(self, value: float):
        """Update minimum RRS threshold"""
        self.min_rrs = value
        logger.info(f"Min RRS updated to {value}")

    def set_min_rr_ratio(self, value: float):
        """Update minimum risk/reward ratio"""
        self.min_rr_ratio = value
        logger.info(f"Min R/R ratio updated to {value}")
