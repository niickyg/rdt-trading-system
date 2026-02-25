"""
Dynamic Position Sizing Module

Adjusts risk per trade based on signal quality, market conditions,
and recent performance. Uses a rule-based approach with a Kelly
Criterion overlay for trades with sufficient historical data.

Output: risk multiplier (0.5x to 2.0x of base risk), giving an
effective risk range of 0.75% to 3.0% per trade when base risk = 1.5%.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Sector map lives in risk_manager; import it for concentration checks
try:
    from risk.risk_manager import SECTOR_MAP
except ImportError:
    SECTOR_MAP: Dict[str, str] = {}


@dataclass
class SizingInput:
    """All inputs the dynamic sizer considers."""
    # ML & signal quality
    ml_confidence: float = 50.0        # 0-100 scale (from ensemble)
    rrs_strength: float = 0.0          # absolute RRS value
    quality_warnings: List[str] = field(default_factory=list)

    # Market context
    market_regime: str = "unknown"     # bull_trending, bear_trending, high_volatility, etc.
    direction: str = "long"            # long or short

    # Recent performance
    recent_win_rate: float = 0.5       # last N trades win rate (0-1)
    recent_avg_win: float = 1.0        # average win in R-multiples
    recent_avg_loss: float = 1.0       # average loss in R-multiples (positive)
    recent_trade_count: int = 0        # number of recent trades available

    # Portfolio state
    portfolio_heat: float = 0.0        # total risk deployed as fraction of account (0-1)
    sector_concentration: int = 0      # existing positions in same sector
    symbol: str = ""                   # for sector lookup

    # Time of day (Eastern Time hour, 0-23)
    time_of_day_hour: int = 10         # default = morning


@dataclass
class SizingResult:
    """Output from the dynamic sizer."""
    multiplier: float = 1.0            # final risk multiplier (0.5 - 2.0)
    effective_risk_pct: float = 0.015  # base_risk * multiplier
    kelly_fraction: Optional[float] = None  # raw Kelly fraction if computed
    kelly_risk: Optional[float] = None      # quarter-Kelly risk if computed
    adjustments: Dict[str, float] = field(default_factory=dict)  # breakdown
    reason: str = ""


class DynamicPositionSizer:
    """
    Adjusts per-trade risk based on signal conviction, market regime,
    recent performance, portfolio state, and time of day.

    Rule-based (deterministic) -- an ML-based version can replace the
    scoring logic later while keeping the same interface.
    """

    # --- Tunable thresholds ---
    ML_CONFIDENCE_HIGH: float = 80.0     # confidence above this = bonus
    ML_CONFIDENCE_LOW: float = 50.0      # confidence below this = penalty
    RRS_STRONG: float = 3.0             # RRS above this = bonus
    RRS_WEAK: float = 2.0               # RRS below this = penalty
    PORTFOLIO_HEAT_HIGH: float = 0.06    # >6% total risk = reduce sizing
    AFTERNOON_HOUR: int = 14             # 14:00 ET and later = reduce
    SECTOR_CONC_LIMIT: int = 2           # >=2 same-sector positions = reduce
    MIN_KELLY_TRADES: int = 10           # minimum trades for Kelly overlay

    # Hard clamp
    MIN_MULTIPLIER: float = 0.5
    MAX_MULTIPLIER: float = 2.0

    def __init__(
        self,
        base_risk: float = 0.015,
        enabled: bool = True,
    ):
        """
        Args:
            base_risk: Default risk per trade as fraction (0.015 = 1.5%).
            enabled: Whether dynamic sizing is active.
        """
        self.base_risk = base_risk
        self.enabled = enabled

        # Rolling trade history for Kelly & win-rate calculations
        self._trade_results: List[Dict] = []  # [{pnl_r: float, win: bool, ...}]
        self._max_history: int = 100

        logger.info(
            f"DynamicPositionSizer initialized (enabled={enabled}, "
            f"base_risk={base_risk*100:.1f}%)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_multiplier(self, inputs: SizingInput) -> SizingResult:
        """
        Calculate the risk multiplier for a trade.

        Args:
            inputs: SizingInput dataclass with all signal/context features.

        Returns:
            SizingResult with the final multiplier and breakdown.
        """
        if not self.enabled:
            return SizingResult(
                multiplier=1.0,
                effective_risk_pct=self.base_risk,
                reason="Dynamic sizing disabled",
            )

        adjustments: Dict[str, float] = {}
        multiplier = 1.0

        # ----- 1. ML Confidence -----
        if inputs.ml_confidence > self.ML_CONFIDENCE_HIGH:
            adj = 0.3
            adjustments["ml_confidence_high"] = adj
            multiplier += adj
        elif inputs.ml_confidence < self.ML_CONFIDENCE_LOW:
            adj = -0.2
            adjustments["ml_confidence_low"] = adj
            multiplier += adj

        # ----- 2. RRS Strength -----
        abs_rrs = abs(inputs.rrs_strength)
        if abs_rrs > self.RRS_STRONG:
            adj = 0.2
            adjustments["rrs_strong"] = adj
            multiplier += adj
        elif abs_rrs < self.RRS_WEAK:
            adj = -0.1
            adjustments["rrs_weak"] = adj
            multiplier += adj

        # ----- 3. Market Regime -----
        regime_adj = self._regime_adjustment(inputs.market_regime, inputs.direction)
        if regime_adj != 0.0:
            adjustments["market_regime"] = regime_adj
            multiplier += regime_adj

        # ----- 4. Recent Win Rate (only if enough data) -----
        if inputs.recent_trade_count >= 5:
            if inputs.recent_win_rate > 0.60:
                adj = 0.1
                adjustments["win_rate_high"] = adj
                multiplier += adj
            elif inputs.recent_win_rate < 0.35:
                adj = -0.2
                adjustments["win_rate_low"] = adj
                multiplier += adj

        # ----- 5. Quality Warnings -----
        if inputs.quality_warnings:
            # Scale penalty by number of warnings
            n_warnings = len(inputs.quality_warnings)
            adj = -0.15 * min(n_warnings, 3)  # cap at -0.45
            adjustments["quality_warnings"] = adj
            multiplier += adj

        # ----- 6. Portfolio Heat -----
        if inputs.portfolio_heat > self.PORTFOLIO_HEAT_HIGH:
            adj = -0.2
            adjustments["portfolio_heat"] = adj
            multiplier += adj

        # ----- 7. Sector Concentration -----
        sector_count = inputs.sector_concentration
        if sector_count == 0 and inputs.symbol:
            # Auto-detect from SECTOR_MAP
            sector_count = self._count_sector_positions(inputs.symbol)
        if sector_count >= self.SECTOR_CONC_LIMIT:
            adj = -0.2
            adjustments["sector_concentration"] = adj
            multiplier += adj

        # ----- 8. Time of Day -----
        if inputs.time_of_day_hour >= self.AFTERNOON_HOUR:
            adj = -0.2
            adjustments["afternoon_fade"] = adj
            multiplier += adj

        # ----- Clamp -----
        multiplier = max(self.MIN_MULTIPLIER, min(self.MAX_MULTIPLIER, multiplier))

        # ----- Kelly Criterion Overlay -----
        kelly_fraction = None
        kelly_risk = None
        if inputs.recent_trade_count >= self.MIN_KELLY_TRADES:
            kelly_fraction, kelly_risk = self._kelly_overlay(
                inputs.recent_win_rate,
                inputs.recent_avg_win,
                inputs.recent_avg_loss,
            )
            if kelly_risk is not None and kelly_risk > 0:
                rule_risk = self.base_risk * multiplier
                if kelly_risk < rule_risk:
                    # Kelly says size down -- respect it
                    multiplier = kelly_risk / self.base_risk
                    multiplier = max(self.MIN_MULTIPLIER, multiplier)
                    adjustments["kelly_cap"] = kelly_risk
                    logger.debug(
                        f"Kelly cap applied: kelly_risk={kelly_risk*100:.2f}% "
                        f"< rule_risk={rule_risk*100:.2f}%"
                    )

        effective_risk = self.base_risk * multiplier

        result = SizingResult(
            multiplier=round(multiplier, 3),
            effective_risk_pct=round(effective_risk, 5),
            kelly_fraction=kelly_fraction,
            kelly_risk=kelly_risk,
            adjustments=adjustments,
            reason=self._build_reason(adjustments),
        )

        logger.info(
            f"Dynamic sizing: multiplier={result.multiplier:.2f}x "
            f"risk={result.effective_risk_pct*100:.2f}% "
            f"({result.reason})"
        )
        return result

    def record_trade_result(
        self,
        pnl_r: float,
        win: bool,
        symbol: str = "",
        direction: str = "",
    ):
        """
        Record a completed trade for rolling statistics.

        Args:
            pnl_r: P&L in R-multiples (e.g. +2.0 for a 2R win, -1.0 for 1R loss).
            win: Whether the trade was a winner.
            symbol: Trading symbol.
            direction: Trade direction.
        """
        self._trade_results.append({
            "pnl_r": pnl_r,
            "win": win,
            "symbol": symbol,
            "direction": direction,
            "timestamp": datetime.now().isoformat(),
        })
        # Trim to max history
        if len(self._trade_results) > self._max_history:
            self._trade_results = self._trade_results[-self._max_history:]

    def get_recent_stats(self, n: int = 10) -> Dict:
        """
        Get statistics for the last N trades.

        Returns:
            Dict with win_rate, avg_win_r, avg_loss_r, trade_count.
        """
        recent = self._trade_results[-n:] if self._trade_results else []
        if not recent:
            return {
                "win_rate": 0.5,
                "avg_win_r": 1.0,
                "avg_loss_r": 1.0,
                "trade_count": 0,
            }

        wins = [t for t in recent if t["win"]]
        losses = [t for t in recent if not t["win"]]
        win_rate = len(wins) / len(recent) if recent else 0.5
        avg_win_r = (
            sum(t["pnl_r"] for t in wins) / len(wins) if wins else 1.0
        )
        avg_loss_r = (
            abs(sum(t["pnl_r"] for t in losses) / len(losses))
            if losses
            else 1.0
        )

        return {
            "win_rate": win_rate,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "trade_count": len(recent),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _regime_adjustment(self, regime: str, direction: str) -> float:
        """Return regime-based adjustment to multiplier."""
        regime_lower = regime.lower() if regime else ""

        if direction == "long":
            if regime_lower == "bull_trending":
                return 0.2
            elif regime_lower == "bear_trending":
                return -0.3
            elif regime_lower in ("high_volatility", "choppy"):
                return -0.1
        elif direction == "short":
            if regime_lower == "bear_trending":
                return 0.2
            elif regime_lower == "bull_trending":
                return -0.3
            elif regime_lower in ("high_volatility", "choppy"):
                return -0.1

        return 0.0

    def _kelly_overlay(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Kelly Criterion fraction and quarter-Kelly risk.

        Args:
            win_rate: Win rate (0-1).
            avg_win_r: Average win in R-multiples.
            avg_loss_r: Average loss in R-multiples (positive value).

        Returns:
            Tuple of (kelly_fraction, quarter_kelly_risk) or (None, None)
            if insufficient data or negative Kelly.
        """
        if avg_loss_r <= 0 or avg_win_r <= 0:
            return None, None

        # Kelly formula: f* = W - (1-W) / (W/L ratio)
        win_loss_ratio = avg_win_r / avg_loss_r
        kelly_fraction = win_rate - (1.0 - win_rate) / win_loss_ratio

        if kelly_fraction <= 0:
            # Negative Kelly means edge is negative -- use minimum sizing
            logger.debug(f"Negative Kelly fraction: {kelly_fraction:.4f}")
            return kelly_fraction, None

        # Quarter-Kelly for safety
        kelly_risk = kelly_fraction * 0.25

        # Never let Kelly exceed the hard max
        max_risk = self.base_risk * self.MAX_MULTIPLIER
        kelly_risk = min(kelly_risk, max_risk)

        return kelly_fraction, kelly_risk

    def _count_sector_positions(self, symbol: str) -> int:
        """
        Count existing open positions in the same sector.
        Placeholder -- in production this would query the risk manager.
        """
        # This is a simplified version; the real count is passed in
        # via SizingInput.sector_concentration from the risk manager.
        return 0

    def _build_reason(self, adjustments: Dict[str, float]) -> str:
        """Build a human-readable reason string from adjustments."""
        if not adjustments:
            return "base sizing (no adjustments)"

        parts = []
        for key, val in adjustments.items():
            sign = "+" if val > 0 else ""
            parts.append(f"{key}={sign}{val:.2f}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_sizer: Optional[DynamicPositionSizer] = None


def get_dynamic_sizer(
    base_risk: float = 0.015,
    enabled: bool = True,
) -> DynamicPositionSizer:
    """Get or create the module-level DynamicPositionSizer singleton."""
    global _default_sizer
    if _default_sizer is None:
        _default_sizer = DynamicPositionSizer(
            base_risk=base_risk, enabled=enabled
        )
    return _default_sizer
