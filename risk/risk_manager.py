"""
Risk Management Engine
Enforces trading rules and risk limits
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from loguru import logger

from risk.models import (
    RiskCheckResult, RiskMetrics, RiskLimits, TradeRisk,
    DailyRiskReport, RiskLevel, RiskViolationType
)
from risk.position_sizer import PositionSizer

# VIX-based regime filter (optional)
try:
    from scanner.vix_filter import VIXFilter
    VIX_FILTER_AVAILABLE = True
except ImportError:
    VIX_FILTER_AVAILABLE = False


# Sector mapping for position concentration limits
# Maps symbols to sectors for sector-level risk checks
SECTOR_MAP: Dict[str, str] = {
    # Tech
    'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'GOOG': 'tech',
    'AMZN': 'tech', 'NVDA': 'tech', 'META': 'tech', 'TSLA': 'tech',
    'AMD': 'tech', 'AVGO': 'tech', 'QCOM': 'tech', 'INTC': 'tech',
    'MU': 'tech', 'CRM': 'tech', 'ADBE': 'tech', 'ORCL': 'tech',
    'NFLX': 'tech',
    # Finance
    'JPM': 'finance', 'BAC': 'finance', 'GS': 'finance', 'MS': 'finance',
    'WFC': 'finance', 'C': 'finance', 'V': 'finance', 'MA': 'finance',
    'AXP': 'finance', 'SCHW': 'finance', 'BLK': 'finance',
    # Healthcare
    'UNH': 'healthcare', 'JNJ': 'healthcare', 'PFE': 'healthcare',
    'ABBV': 'healthcare', 'MRK': 'healthcare', 'LLY': 'healthcare',
    'TMO': 'healthcare', 'ABT': 'healthcare', 'BMY': 'healthcare',
    # Consumer
    'WMT': 'consumer', 'HD': 'consumer', 'COST': 'consumer',
    'MCD': 'consumer', 'SBUX': 'consumer', 'NKE': 'consumer',
    'TGT': 'consumer', 'LOW': 'consumer',
    # Consumer Goods
    'PG': 'consumer', 'KO': 'consumer', 'PEP': 'consumer',
    'GIS': 'consumer',
    # Industrial
    'BA': 'industrial', 'CAT': 'industrial', 'GE': 'industrial',
    'HON': 'industrial', 'UPS': 'industrial', 'MMM': 'industrial',
    'DE': 'industrial',
    # Energy
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy',
    'FCX': 'energy', 'NEM': 'energy',
}

# Maximum positions allowed in the same sector
MAX_SECTOR_POSITIONS = 3


class RiskManager:
    """
    Central risk management engine

    Responsibilities:
    - Validate trades against risk rules
    - Track daily P&L and exposure
    - Monitor drawdown
    - Enforce PDT rules
    - Enforce sector concentration limits
    - Generate risk reports
    """

    def __init__(
        self,
        account_size: float,
        risk_limits: Optional[RiskLimits] = None
    ):
        """
        Initialize risk manager

        Args:
            account_size: Initial account size
            risk_limits: Risk limit configuration
        """
        self.account_size = account_size
        self.starting_balance = account_size
        self.current_balance = account_size
        self.limits = risk_limits or RiskLimits()
        self.position_sizer = PositionSizer(self.limits)

        # Daily tracking - use fixed start-of-day balance for loss limit (Fix 6)
        self.daily_start_balance = account_size
        self._start_of_day_balance = account_size
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.day_trades_today = 0

        # Drawdown tracking
        self.peak_balance = account_size
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_start_time: Optional[datetime] = None

        # Position tracking
        self.open_positions: Dict[str, Dict] = {}

        # Violation history
        self.violations: List[RiskCheckResult] = []

        # Trading state
        self.trading_halted = False
        self.halt_reason = ""

        # VIX regime filter (optional)
        self._vix_filter: Optional[VIXFilter] = None
        if VIX_FILTER_AVAILABLE:
            try:
                self._vix_filter = VIXFilter()
                logger.info("VIX regime filter enabled in risk manager")
            except Exception as e:
                logger.warning(f"Failed to initialize VIX filter in risk manager: {e}")

        logger.info(f"RiskManager initialized: ${account_size:,.2f}")

    def reset_daily(self):
        """Reset daily tracking (call at market open)"""
        self.daily_start_balance = self.current_balance
        self._start_of_day_balance = self.current_balance  # Fix 6: frozen SOD balance
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.day_trades_today = 0
        logger.info("Daily risk metrics reset")

    def update_balance(self, new_balance: float):
        """Update current account balance"""
        self.current_balance = new_balance

        # Update peak and drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            self.current_drawdown = 0.0
            self.drawdown_start_time = None
        else:
            self.current_drawdown = self.peak_balance - new_balance
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            if self.drawdown_start_time is None:
                self.drawdown_start_time = datetime.now()

    def record_trade(self, pnl: float, is_day_trade: bool = False):
        """Record a completed trade"""
        self.daily_pnl += pnl
        self.daily_trades += 1

        if pnl >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        if is_day_trade:
            self.day_trades_today += 1

        self.update_balance(self.current_balance + pnl)

        # Check if we hit daily loss limit
        if self.check_daily_loss_limit():
            self.halt_trading("Daily loss limit exceeded")

    # ==================== Risk Checks ====================

    def validate_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        shares: int,
        stop_price: float,
        target_price: float,
        atr: float
    ) -> TradeRisk:
        """
        Validate a potential trade against all risk rules

        Returns:
            TradeRisk with all check results
        """
        # Guard against zero or negative balance
        if self.current_balance <= 0:
            trade_risk = TradeRisk(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                shares=shares,
                position_value=entry_price * shares,
                risk_amount=abs(entry_price - stop_price) * shares,
                reward_amount=abs(target_price - entry_price) * shares,
                risk_reward_ratio=0,
                risk_percent_of_account=0,
                position_percent_of_account=0,
                atr=atr,
                atr_percent=(atr / entry_price) * 100 if entry_price > 0 else 0
            )
            failed_check = RiskCheckResult(
                passed=False,
                violation_type=RiskViolationType.INSUFFICIENT_BUYING_POWER,
                message="Zero or negative account balance",
                current_value=self.current_balance,
                limit_value=0,
                risk_level=RiskLevel.CRITICAL
            )
            trade_risk.checks_failed.append(failed_check)
            self.violations.append(failed_check)
            return trade_risk

        position_value = entry_price * shares
        risk_amount = abs(entry_price - stop_price) * shares
        reward_amount = abs(target_price - entry_price) * shares
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        trade_risk = TradeRisk(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            shares=shares,
            position_value=position_value,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=rr_ratio,
            risk_percent_of_account=(risk_amount / self.current_balance) * 100 if self.current_balance > 0 else 0,
            position_percent_of_account=(position_value / self.current_balance) * 100 if self.current_balance > 0 else 0,
            atr=atr,
            atr_percent=(atr / entry_price) * 100 if entry_price > 0 else 0
        )

        # Apply VIX regime adjustment to position size
        vix_adjusted_position_value = position_value
        if self._vix_filter is not None:
            try:
                vix_regime = self._vix_filter.get_vix_regime()
                vix_check = self._check_vix_regime(direction, vix_regime)
                if not vix_check.passed:
                    # VIX regime blocks this trade entirely
                    trade_risk.checks_failed.append(vix_check)
                    self.violations.append(vix_check)
                    return trade_risk
                # Adjust position value by VIX multiplier for size checks
                vix_mult = vix_regime.get('position_size_multiplier', 1.0)
                if vix_mult < 1.0:
                    vix_adjusted_position_value = position_value * vix_mult
                    logger.debug(
                        f"VIX regime '{vix_regime['level']}': position value "
                        f"adjusted ${position_value:,.0f} -> ${vix_adjusted_position_value:,.0f}"
                    )
            except Exception as e:
                logger.warning(f"VIX regime check failed in risk manager: {e}")

        # Run all risk checks
        checks = [
            self._check_position_size(vix_adjusted_position_value),
            self._check_risk_per_trade(risk_amount),
            self._check_risk_reward(rr_ratio),
            self._check_daily_loss(),
            self._check_max_positions(),
            self._check_sector_concentration(symbol),
            self._check_buying_power(position_value),
            self._check_drawdown(),
            self._check_trading_halted(),
        ]

        for check in checks:
            if check.passed:
                trade_risk.checks_passed.append(check)
            else:
                trade_risk.checks_failed.append(check)
                self.violations.append(check)

        return trade_risk

    def _check_position_size(self, position_value: float) -> RiskCheckResult:
        """Check if position size is within limits"""
        max_value = self.current_balance * self.limits.max_position_size
        passed = position_value <= max_value

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_POSITION_SIZE if not passed else None,
            message=f"Position size: ${position_value:,.0f} / ${max_value:,.0f} max",
            current_value=position_value,
            limit_value=max_value,
            risk_level=RiskLevel.MEDIUM if not passed else RiskLevel.LOW
        )

    def _check_risk_per_trade(self, risk_amount: float) -> RiskCheckResult:
        """Check if risk per trade is within limits"""
        max_risk = self.current_balance * self.limits.max_risk_per_trade
        passed = risk_amount <= max_risk

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_POSITION_SIZE if not passed else None,
            message=f"Risk: ${risk_amount:,.0f} / ${max_risk:,.0f} max",
            current_value=risk_amount,
            limit_value=max_risk,
            risk_level=RiskLevel.HIGH if not passed else RiskLevel.LOW
        )

    def _check_risk_reward(self, rr_ratio: float) -> RiskCheckResult:
        """Check if risk/reward ratio meets minimum"""
        passed = rr_ratio >= self.limits.min_risk_reward

        return RiskCheckResult(
            passed=passed,
            message=f"R/R ratio: {rr_ratio:.2f} (min {self.limits.min_risk_reward})",
            current_value=rr_ratio,
            limit_value=self.limits.min_risk_reward,
            risk_level=RiskLevel.MEDIUM if not passed else RiskLevel.LOW
        )

    def _check_daily_loss(self) -> RiskCheckResult:
        """Check if daily loss limit has been hit"""
        # Fix 6: Use frozen start-of-day balance, not current balance
        max_loss = self._start_of_day_balance * self.limits.max_daily_loss
        current_loss = abs(min(0, self.daily_pnl))
        passed = current_loss < max_loss

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_DAILY_LOSS if not passed else None,
            message=f"Daily loss: ${current_loss:,.0f} / ${max_loss:,.0f} limit",
            current_value=current_loss,
            limit_value=max_loss,
            risk_level=RiskLevel.CRITICAL if not passed else RiskLevel.LOW
        )

    def _check_max_positions(self) -> RiskCheckResult:
        """Check if maximum open positions limit reached"""
        current = len(self.open_positions)
        max_pos = self.limits.max_open_positions
        passed = current < max_pos

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_OPEN_POSITIONS if not passed else None,
            message=f"Open positions: {current} / {max_pos} max",
            current_value=current,
            limit_value=max_pos,
            risk_level=RiskLevel.MEDIUM if not passed else RiskLevel.LOW
        )

    def _check_sector_concentration(self, symbol: str) -> RiskCheckResult:
        """Check if adding this position would exceed sector concentration limit"""
        sector = SECTOR_MAP.get(symbol, 'other')
        # Count existing positions in the same sector
        same_sector_count = sum(
            1 for sym in self.open_positions
            if SECTOR_MAP.get(sym, 'other') == sector
        )
        passed = same_sector_count < MAX_SECTOR_POSITIONS

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_SECTOR_EXPOSURE if not passed else None,
            message=f"Sector '{sector}' positions: {same_sector_count} / {MAX_SECTOR_POSITIONS} max",
            current_value=same_sector_count,
            limit_value=MAX_SECTOR_POSITIONS,
            risk_level=RiskLevel.MEDIUM if not passed else RiskLevel.LOW
        )

    def _check_buying_power(self, position_value: float) -> RiskCheckResult:
        """Check if sufficient buying power available"""
        # Simplified check - in reality would query broker
        available = self.current_balance - sum(
            p.get("position_value", 0) for p in self.open_positions.values()
        )
        passed = position_value <= available

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.INSUFFICIENT_BUYING_POWER if not passed else None,
            message=f"Buying power: ${available:,.0f} available",
            current_value=position_value,
            limit_value=available,
            risk_level=RiskLevel.HIGH if not passed else RiskLevel.LOW
        )

    def _check_drawdown(self) -> RiskCheckResult:
        """Check if maximum drawdown limit reached"""
        max_dd = self.peak_balance * self.limits.max_drawdown
        passed = self.current_drawdown < max_dd

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.MAX_DRAWDOWN if not passed else None,
            message=f"Drawdown: ${self.current_drawdown:,.0f} / ${max_dd:,.0f} max",
            current_value=self.current_drawdown,
            limit_value=max_dd,
            risk_level=RiskLevel.CRITICAL if not passed else RiskLevel.LOW
        )

    def _check_trading_halted(self) -> RiskCheckResult:
        """Check if trading is halted"""
        return RiskCheckResult(
            passed=not self.trading_halted,
            message=self.halt_reason if self.trading_halted else "Trading active",
            risk_level=RiskLevel.CRITICAL if self.trading_halted else RiskLevel.LOW
        )

    def _check_vix_regime(self, direction: str, vix_regime: Dict) -> RiskCheckResult:
        """Check if the VIX regime allows this trade direction."""
        level = vix_regime.get('level', 'normal')
        vix_value = vix_regime.get('vix_value')
        allow_longs = vix_regime.get('allow_longs', True)
        allow_shorts = vix_regime.get('allow_shorts', True)

        # Extreme VIX: block longs entirely
        if level == 'extreme' and direction.lower() == 'long':
            return RiskCheckResult(
                passed=False,
                violation_type=RiskViolationType.MAX_DAILY_LOSS,  # closest available type
                message=f"VIX EXTREME ({vix_value:.1f}): long trades blocked",
                current_value=vix_value if vix_value else 0,
                limit_value=35.0,
                risk_level=RiskLevel.CRITICAL
            )

        if direction.lower() == 'long' and not allow_longs:
            return RiskCheckResult(
                passed=False,
                violation_type=RiskViolationType.MAX_DAILY_LOSS,
                message=f"VIX {level.upper()} ({vix_value}): longs not allowed",
                current_value=vix_value if vix_value else 0,
                limit_value=35.0,
                risk_level=RiskLevel.HIGH
            )

        if direction.lower() == 'short' and not allow_shorts:
            return RiskCheckResult(
                passed=False,
                violation_type=RiskViolationType.MAX_DAILY_LOSS,
                message=f"VIX {level.upper()} ({vix_value}): shorts not allowed",
                current_value=vix_value if vix_value else 0,
                limit_value=35.0,
                risk_level=RiskLevel.HIGH
            )

        return RiskCheckResult(
            passed=True,
            message=f"VIX regime: {level} (VIX={vix_value})",
            current_value=vix_value if vix_value else 0,
            limit_value=35.0,
            risk_level=RiskLevel.LOW
        )

    def check_daily_loss_limit(self) -> bool:
        """Quick check if daily loss limit exceeded"""
        # Fix 6: Use frozen start-of-day balance, not current balance
        max_loss = self._start_of_day_balance * self.limits.max_daily_loss
        return self.daily_pnl < -max_loss

    def check_pdt_rule(self, is_day_trade: bool) -> RiskCheckResult:
        """Check Pattern Day Trader rule"""
        if self.current_balance >= self.limits.pdt_account_minimum:
            # PDT rule doesn't apply
            return RiskCheckResult(passed=True, message="Account above PDT minimum")

        if not is_day_trade:
            return RiskCheckResult(passed=True, message="Not a day trade")

        remaining = self.limits.day_trade_limit - self.day_trades_today
        passed = remaining > 0

        return RiskCheckResult(
            passed=passed,
            violation_type=RiskViolationType.PATTERN_DAY_TRADER if not passed else None,
            message=f"Day trades: {self.day_trades_today}/{self.limits.day_trade_limit}",
            current_value=self.day_trades_today,
            limit_value=self.limits.day_trade_limit,
            risk_level=RiskLevel.HIGH if not passed else RiskLevel.LOW
        )

    # ==================== Position Management ====================

    def add_position(self, symbol: str, position_data: Dict):
        """Track a new open position"""
        self.open_positions[symbol] = position_data
        logger.debug(f"Added position: {symbol}")

    def remove_position(self, symbol: str):
        """Remove a closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            logger.debug(f"Removed position: {symbol}")

    def update_position(self, symbol: str, updates: Dict):
        """Update position data"""
        if symbol in self.open_positions:
            self.open_positions[symbol].update(updates)

    # ==================== Control Methods ====================

    def halt_trading(self, reason: str):
        """Halt all trading"""
        self.trading_halted = True
        self.halt_reason = reason
        logger.warning(f"TRADING HALTED: {reason}")

    def resume_trading(self):
        """Resume trading after halt"""
        self.trading_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")

    # ==================== Metrics & Reports ====================

    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        total_exposure = sum(
            p.get("position_value", 0) for p in self.open_positions.values()
        )

        largest_position = max(
            (p.get("position_value", 0) for p in self.open_positions.values()),
            default=0
        )

        return RiskMetrics(
            daily_pnl=self.daily_pnl,
            daily_pnl_percent=(self.daily_pnl / self.daily_start_balance) * 100 if self.daily_start_balance else 0,
            daily_trades=self.daily_trades,
            daily_wins=self.daily_wins,
            daily_losses=self.daily_losses,
            open_positions=len(self.open_positions),
            total_exposure=total_exposure,
            exposure_percent=(total_exposure / self.current_balance) * 100 if self.current_balance else 0,
            largest_position_percent=(largest_position / self.current_balance) * 100 if self.current_balance else 0,
            current_drawdown=self.current_drawdown,
            current_drawdown_percent=(self.current_drawdown / self.peak_balance) * 100 if self.peak_balance else 0,
            max_drawdown=self.max_drawdown,
            max_drawdown_percent=(self.max_drawdown / self.peak_balance) * 100 if self.peak_balance else 0,
            timestamp=datetime.now()
        )

    def generate_daily_report(self) -> DailyRiskReport:
        """Generate end of day risk report"""
        win_rate = self.daily_wins / self.daily_trades if self.daily_trades > 0 else 0

        return DailyRiskReport(
            date=date.today(),
            starting_balance=self.daily_start_balance,
            ending_balance=self.current_balance,
            daily_pnl=self.daily_pnl,
            daily_pnl_percent=(self.daily_pnl / self.daily_start_balance) * 100,
            total_trades=self.daily_trades,
            winning_trades=self.daily_wins,
            losing_trades=self.daily_losses,
            win_rate=win_rate,
            avg_win=0,  # Would calculate from trade history
            avg_loss=0,
            profit_factor=0,  # Would calculate from trade history
            max_drawdown=self.max_drawdown,
            max_drawdown_percent=(self.max_drawdown / self.peak_balance) * 100,
            risk_violations=[v for v in self.violations if not v.passed]
        )
