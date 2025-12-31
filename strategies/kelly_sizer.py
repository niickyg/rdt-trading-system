"""
Kelly Criterion Position Sizing

Implements optimal position sizing based on the Kelly Criterion
for maximum geometric growth of capital.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from loguru import logger


@dataclass
class KellyResult:
    """Result from Kelly calculation"""
    full_kelly: float           # Optimal fraction (can be > 1)
    half_kelly: float           # Half Kelly (safer)
    quarter_kelly: float        # Quarter Kelly (conservative)
    recommended: float          # Final recommendation
    edge: float                 # Expected edge per trade
    reason: str                 # Explanation


class KellyCriterionSizer:
    """
    Kelly Criterion Position Sizing Calculator

    The Kelly Criterion maximizes the long-term growth rate of capital.
    Formula: f* = (bp - q) / b

    Where:
        f* = optimal fraction of capital to bet
        b = win/loss ratio (average win / average loss)
        p = probability of winning
        q = probability of losing (1 - p)

    Important Notes:
    - Full Kelly is mathematically optimal but causes large drawdowns
    - Half Kelly is recommended for most traders
    - Quarter Kelly is conservative but still captures most edge
    - Negative Kelly means strategy has negative expected value
    """

    def __init__(
        self,
        max_position_pct: float = 0.25,  # Never risk more than 25%
        use_fraction: str = 'half',       # 'full', 'half', 'quarter'
        min_trades_required: int = 30,    # Min trades for reliable estimate
        recent_window_days: int = 90,     # Look at recent performance
    ):
        self.max_position_pct = max_position_pct
        self.use_fraction = use_fraction
        self.min_trades_required = min_trades_required
        self.recent_window_days = recent_window_days

        # Trade history for adaptive sizing
        self.trade_history: List[Dict] = []

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> KellyResult:
        """
        Calculate Kelly fraction from trading statistics

        Args:
            win_rate: Historical win percentage (0.0 to 1.0)
            avg_win: Average winning trade amount (positive)
            avg_loss: Average losing trade amount (positive, will be treated as loss)

        Returns:
            KellyResult with recommended position size
        """
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            return KellyResult(
                full_kelly=0,
                half_kelly=0,
                quarter_kelly=0,
                recommended=0,
                edge=0,
                reason=f"Invalid win rate: {win_rate}"
            )

        if avg_win <= 0 or avg_loss <= 0:
            return KellyResult(
                full_kelly=0,
                half_kelly=0,
                quarter_kelly=0,
                recommended=0,
                edge=0,
                reason=f"Invalid win/loss amounts"
            )

        # Calculate Kelly components
        p = win_rate          # Probability of win
        q = 1 - win_rate      # Probability of loss
        b = avg_win / avg_loss  # Win/loss ratio

        # Kelly formula: f* = (bp - q) / b
        kelly = (b * p - q) / b

        # Calculate edge (expected value per dollar risked)
        edge = (p * avg_win) - (q * avg_loss)

        # Determine fractions
        full_kelly = kelly
        half_kelly = kelly / 2
        quarter_kelly = kelly / 4

        # Select recommended based on preference
        fraction_map = {
            'full': full_kelly,
            'half': half_kelly,
            'quarter': quarter_kelly
        }
        recommended = fraction_map.get(self.use_fraction, half_kelly)

        # Apply constraints
        if kelly < 0:
            recommended = 0
            reason = f"Negative Kelly ({kelly:.3f}) - strategy has negative edge"
        elif kelly > 1:
            recommended = min(recommended, self.max_position_pct)
            reason = f"Kelly > 100% capped at {self.max_position_pct*100}%"
        else:
            recommended = min(max(recommended, 0), self.max_position_pct)
            reason = f"Using {self.use_fraction} Kelly"

        return KellyResult(
            full_kelly=round(full_kelly, 4),
            half_kelly=round(half_kelly, 4),
            quarter_kelly=round(quarter_kelly, 4),
            recommended=round(recommended, 4),
            edge=round(edge, 4),
            reason=reason
        )

    def calculate_from_trades(
        self,
        trades: List[Dict],
        use_recent_only: bool = True
    ) -> KellyResult:
        """
        Calculate Kelly from actual trade history

        Args:
            trades: List of trade dicts with 'pnl' key
            use_recent_only: Only use trades from recent_window_days

        Returns:
            KellyResult based on trade history
        """
        if use_recent_only:
            cutoff = datetime.now() - timedelta(days=self.recent_window_days)
            recent_trades = [
                t for t in trades
                if t.get('exit_date', datetime.now()) >= cutoff
            ]
        else:
            recent_trades = trades

        if len(recent_trades) < self.min_trades_required:
            return KellyResult(
                full_kelly=0,
                half_kelly=0,
                quarter_kelly=0,
                recommended=0.01,  # Default to 1%
                edge=0,
                reason=f"Insufficient trades ({len(recent_trades)} < {self.min_trades_required})"
            )

        # Calculate statistics
        winners = [t['pnl'] for t in recent_trades if t.get('pnl', 0) > 0]
        losers = [abs(t['pnl']) for t in recent_trades if t.get('pnl', 0) <= 0]

        if not winners or not losers:
            return KellyResult(
                full_kelly=0,
                half_kelly=0,
                quarter_kelly=0,
                recommended=0.01,
                edge=0,
                reason="Need both winners and losers to calculate"
            )

        win_rate = len(winners) / len(recent_trades)
        avg_win = sum(winners) / len(winners)
        avg_loss = sum(losers) / len(losers)

        result = self.calculate_kelly(win_rate, avg_win, avg_loss)

        # Log calculation
        logger.info(
            f"Kelly calculation: Win Rate={win_rate:.1%}, "
            f"Avg Win=${avg_win:.2f}, Avg Loss=${avg_loss:.2f}, "
            f"Recommended={result.recommended:.1%}"
        )

        return result

    def add_trade(self, trade: Dict):
        """Add a trade to history for adaptive sizing"""
        self.trade_history.append(trade)

    def get_adaptive_size(self) -> float:
        """Get current recommended position size based on recent performance"""
        if len(self.trade_history) < self.min_trades_required:
            return 0.01  # Default 1%

        result = self.calculate_from_trades(self.trade_history)
        return result.recommended

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> Tuple[int, Dict]:
        """
        Calculate actual position size in shares

        Args:
            capital: Total trading capital
            entry_price: Entry price for trade
            stop_price: Stop loss price
            win_rate: Optional override win rate
            avg_win: Optional override average win
            avg_loss: Optional override average loss

        Returns:
            Tuple of (shares, metadata dict)
        """
        # Get Kelly fraction
        if win_rate and avg_win and avg_loss:
            kelly_result = self.calculate_kelly(win_rate, avg_win, avg_loss)
        else:
            kelly_result = self.calculate_from_trades(self.trade_history)

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0, {'error': 'Stop equals entry'}

        # Calculate position value based on Kelly
        kelly_fraction = kelly_result.recommended
        position_value = capital * kelly_fraction

        # Calculate shares
        shares = int(position_value / entry_price)

        # Verify risk doesn't exceed Kelly allocation
        total_risk = shares * risk_per_share
        max_risk = capital * kelly_fraction
        if total_risk > max_risk:
            shares = int(max_risk / risk_per_share)

        metadata = {
            'kelly_fraction': kelly_fraction,
            'full_kelly': kelly_result.full_kelly,
            'edge': kelly_result.edge,
            'reason': kelly_result.reason,
            'position_value': shares * entry_price,
            'total_risk': shares * risk_per_share,
            'risk_percent': (shares * risk_per_share) / capital if capital > 0 else 0
        }

        return shares, metadata


class VolatilityAdjustedSizer:
    """
    Adjust position size based on market volatility (VIX)

    In high volatility: Reduce position size
    In low volatility: Increase position size
    """

    def __init__(
        self,
        base_risk: float = 0.02,  # 2% base risk
        vix_baseline: float = 20.0,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5
    ):
        self.base_risk = base_risk
        self.vix_baseline = vix_baseline
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def adjust_for_volatility(self, current_vix: float) -> float:
        """
        Adjust risk based on current VIX

        Args:
            current_vix: Current VIX level

        Returns:
            Adjusted risk percentage
        """
        if current_vix <= 0:
            return self.base_risk

        # Multiplier = baseline / current
        # VIX at 30 -> multiplier = 20/30 = 0.67 -> reduce risk
        # VIX at 15 -> multiplier = 20/15 = 1.33 -> increase risk
        multiplier = self.vix_baseline / current_vix

        # Apply bounds
        multiplier = max(self.min_multiplier, min(multiplier, self.max_multiplier))

        adjusted_risk = self.base_risk * multiplier

        logger.debug(
            f"VIX adjustment: VIX={current_vix}, "
            f"Multiplier={multiplier:.2f}, "
            f"Adjusted Risk={adjusted_risk:.2%}"
        )

        return adjusted_risk


class CorrelationAdjustedSizer:
    """
    Adjust position limits based on portfolio correlation

    Allows more positions when they're uncorrelated,
    fewer positions when highly correlated.
    """

    # Approximate sector correlations (can be made dynamic)
    SECTOR_CORRELATIONS = {
        ('technology', 'technology'): 1.0,
        ('technology', 'financials'): 0.6,
        ('technology', 'healthcare'): 0.5,
        ('technology', 'energy'): 0.3,
        ('technology', 'utilities'): 0.2,
        ('financials', 'financials'): 1.0,
        ('financials', 'energy'): 0.5,
        ('healthcare', 'healthcare'): 1.0,
        ('energy', 'energy'): 1.0,
    }

    def __init__(
        self,
        max_portfolio_heat: float = 0.15,  # 15% total risk
        max_sector_exposure: float = 0.40   # 40% in one sector
    ):
        self.max_portfolio_heat = max_portfolio_heat
        self.max_sector_exposure = max_sector_exposure

    def get_correlation(self, sector1: str, sector2: str) -> float:
        """Get correlation between two sectors"""
        key = tuple(sorted([sector1.lower(), sector2.lower()]))
        return self.SECTOR_CORRELATIONS.get(key, 0.5)  # Default 0.5

    def calculate_portfolio_heat(
        self,
        positions: List[Dict]
    ) -> float:
        """
        Calculate total portfolio risk considering correlations

        Args:
            positions: List of position dicts with 'risk' and 'sector' keys

        Returns:
            Total portfolio heat (correlation-adjusted risk)
        """
        if not positions:
            return 0

        total_heat = 0

        # Sum individual risks
        for pos in positions:
            total_heat += pos.get('risk', 0) ** 2

        # Add correlated risk
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                corr = self.get_correlation(
                    pos1.get('sector', 'unknown'),
                    pos2.get('sector', 'unknown')
                )
                combined = 2 * pos1.get('risk', 0) * pos2.get('risk', 0) * corr
                total_heat += combined

        return math.sqrt(total_heat)

    def can_add_position(
        self,
        new_position: Dict,
        existing_positions: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Check if new position can be added within risk limits

        Args:
            new_position: Dict with 'risk' and 'sector'
            existing_positions: Current positions

        Returns:
            Tuple of (can_add, reason)
        """
        # Check sector concentration
        new_sector = new_position.get('sector', 'unknown')
        sector_risk = sum(
            p.get('risk', 0) for p in existing_positions
            if p.get('sector', '') == new_sector
        )
        sector_risk += new_position.get('risk', 0)

        if sector_risk > self.max_sector_exposure:
            return False, f"Sector {new_sector} would exceed {self.max_sector_exposure*100}% limit"

        # Check total portfolio heat
        all_positions = existing_positions + [new_position]
        heat = self.calculate_portfolio_heat(all_positions)

        if heat > self.max_portfolio_heat:
            return False, f"Portfolio heat {heat:.2%} would exceed {self.max_portfolio_heat*100}% limit"

        return True, "Position approved"


def optimize_kelly_for_drawdown(
    win_rate: float,
    win_loss_ratio: float,
    max_acceptable_drawdown: float = 0.20
) -> float:
    """
    Find Kelly fraction that keeps expected drawdown below threshold

    Full Kelly has expected drawdown of ~50%.
    Reducing Kelly fraction reduces expected drawdown.

    Args:
        win_rate: Historical win rate
        win_loss_ratio: Average win / Average loss
        max_acceptable_drawdown: Maximum acceptable drawdown (0.20 = 20%)

    Returns:
        Adjusted Kelly fraction
    """
    # Full Kelly
    p = win_rate
    q = 1 - p
    b = win_loss_ratio
    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0

    # Approximate drawdown at different Kelly fractions
    # At full Kelly: ~50% expected drawdown
    # At half Kelly: ~25% expected drawdown
    # At quarter Kelly: ~12.5% expected drawdown

    # Find fraction that gives target drawdown
    # drawdown ~ kelly_fraction * 0.5
    target_fraction = (max_acceptable_drawdown / 0.5)

    adjusted_kelly = min(full_kelly * target_fraction, full_kelly)

    return max(0, adjusted_kelly)
