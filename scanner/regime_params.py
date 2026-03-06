"""
Regime-Adaptive Parameter System

Dynamically adjusts scanner and trading parameters based on the detected
market regime from ml/regime_detector.py.

Regimes (from MarketRegimeDetector):
    - bull_trending: Strong upward momentum
    - bear_trending: Strong downward momentum
    - high_volatility: Elevated market uncertainty
    - low_volatility: Calm market conditions
"""

from typing import Dict, List, Optional, Tuple
from loguru import logger


# Default parameter values that match current config if regime detection fails
_DEFAULTS = {
    'rrs_threshold': 1.75,
    'min_confidence': 62.0,
    'stop_multiplier': 1.5,
    'target_multiplier': 2.5,
    'max_positions': 10,
    'risk_per_trade': 0.03,
    'prefer_mean_reversion': False,
    'prefer_momentum': True,
    'sector_concentration_limit': 3,
}


class RegimeAdaptiveParams:
    """
    Provides regime-specific parameter profiles for scanning and trading.

    Each market regime maps to a tuned set of thresholds, multipliers, and
    position-sizing rules.  When the regime is unknown or detection fails,
    sensible defaults (matching the existing static config) are returned.
    """

    # Parameter profiles keyed by regime label.
    # Values were chosen to express the following philosophy:
    #   bull_trending   -> aggressive: lower entry bar, wider targets, more positions
    #   bear_trending   -> conservative: higher entry bar, tighter targets, fewer positions
    #   high_volatility -> defensive: wider stops, reduced size, balanced strategy
    #   low_volatility  -> moderate: standard thresholds, slight mean-reversion tilt
    PROFILES: Dict[str, Dict] = {
        'bull_trending': {
            'rrs_threshold': 1.5,           # Lower bar — trend is your friend
            'min_confidence': 55.0,         # Relax confidence in strong trend
            'stop_multiplier': 1.5,         # Standard stops
            'target_multiplier': 3.0,       # Strong trends allow wider targets
            'max_positions': 10,            # More room in bull market
            'risk_per_trade': 0.03,         # Aggressive risk in bull
            'prefer_mean_reversion': False,
            'prefer_momentum': True,
            'sector_concentration_limit': 4, # Allow more same-sector exposure
        },
        'bear_trending': {
            'rrs_threshold': 2.0,           # Higher bar — only strong setups
            'min_confidence': 70.0,         # Require high conviction
            'stop_multiplier': 1.5,         # Standard stops
            'target_multiplier': 2.0,       # Tighter targets — reversals faster
            'max_positions': 5,             # Fewer positions
            'risk_per_trade': 0.02,         # Moderate risk
            'prefer_mean_reversion': False,
            'prefer_momentum': True,        # Momentum (short-side) still works
            'sector_concentration_limit': 2, # Strict sector limits
        },
        'high_volatility': {
            'rrs_threshold': 2.0,           # Slightly higher bar
            'min_confidence': 65.0,         # Moderate confidence required
            'stop_multiplier': 2.0,         # Wider stops to avoid whipsaws
            'target_multiplier': 2.5,       # Wide swings but not extreme
            'max_positions': 6,             # Fewer positions, managed risk
            'risk_per_trade': 0.02,         # Moderate per-trade risk
            'prefer_mean_reversion': True,  # Mean reversion works in chop
            'prefer_momentum': False,
            'sector_concentration_limit': 2,
        },
        'low_volatility': {
            'rrs_threshold': 1.75,          # Can be a bit more lenient
            'min_confidence': 60.0,         # Slightly relaxed
            'stop_multiplier': 1.5,         # Standard stops
            'target_multiplier': 2.0,       # Tighter ranges in calm markets
            'max_positions': 10,            # Full capacity
            'risk_per_trade': 0.03,         # Full risk in calm markets
            'prefer_mean_reversion': True,  # Mean reversion thrives in calm
            'prefer_momentum': True,        # Momentum also fine
            'sector_concentration_limit': 3,
        },
    }

    def __init__(self, overrides: Optional[Dict[str, Dict]] = None):
        """
        Initialize with optional per-regime overrides.

        Args:
            overrides: Dict mapping regime label -> partial param dict.
                       Values provided here take precedence over built-in profiles.
        """
        self._profiles = {}
        for regime, params in self.PROFILES.items():
            merged = dict(params)
            if overrides and regime in overrides:
                merged.update(overrides[regime])
            self._profiles[regime] = merged

        logger.info(
            f"RegimeAdaptiveParams initialized with {len(self._profiles)} regime profiles"
        )

    def get_params(self, regime_label: Optional[str] = None) -> Dict:
        """
        Return the parameter dict for a given regime.

        Args:
            regime_label: One of 'bull_trending', 'bear_trending',
                          'high_volatility', 'low_volatility'.
                          If None or unrecognised, returns safe defaults.

        Returns:
            Dict with all parameter keys populated.
        """
        if regime_label and regime_label in self._profiles:
            params = dict(self._profiles[regime_label])
            params['regime'] = regime_label
            return params

        # Fallback to defaults
        if regime_label:
            logger.warning(
                f"Unknown regime '{regime_label}', returning default parameters"
            )
        params = dict(_DEFAULTS)
        params['regime'] = regime_label or 'unknown'
        return params

    def blend_params(
        self, regimes_with_weights: List[Tuple[str, float]]
    ) -> Dict:
        """
        Compute a weighted average of parameters when multiple regime signals
        exist (e.g. transition periods or confidence-weighted posteriors).

        Boolean parameters use majority vote (weight > 0.5 threshold).
        Numeric parameters are linearly blended.

        Args:
            regimes_with_weights: List of (regime_label, weight) tuples.
                                  Weights do not need to sum to 1 (they will
                                  be normalised internally).

        Returns:
            Blended parameter dict.
        """
        if not regimes_with_weights:
            logger.warning("blend_params called with empty list, returning defaults")
            return self.get_params(None)

        # Normalise weights
        total_weight = sum(w for _, w in regimes_with_weights)
        if total_weight <= 0:
            logger.warning("blend_params: total weight <= 0, returning defaults")
            return self.get_params(None)

        normalised = [
            (regime, w / total_weight)
            for regime, w in regimes_with_weights
        ]

        # Collect per-regime param dicts
        param_sets = []
        for regime, weight in normalised:
            p = self.get_params(regime)
            param_sets.append((p, weight))

        # Numeric keys to blend
        numeric_keys = [
            'rrs_threshold', 'min_confidence', 'stop_multiplier',
            'target_multiplier', 'max_positions', 'risk_per_trade',
            'sector_concentration_limit',
        ]
        # Boolean keys — majority vote
        bool_keys = ['prefer_mean_reversion', 'prefer_momentum']

        blended: Dict = {}

        for key in numeric_keys:
            blended[key] = sum(
                p[key] * w for p, w in param_sets
            )
            # Round integers back to int
            if key in ('max_positions', 'sector_concentration_limit'):
                blended[key] = max(1, round(blended[key]))

        for key in bool_keys:
            weighted_true = sum(
                w for p, w in param_sets if p.get(key, False)
            )
            blended[key] = weighted_true > 0.5

        # Determine dominant regime label for logging
        dominant_regime = max(normalised, key=lambda x: x[1])[0]
        blended['regime'] = f"blended({dominant_regime})"

        logger.debug(
            f"Blended params from {len(normalised)} regimes "
            f"(dominant={dominant_regime}): "
            f"rrs_threshold={blended['rrs_threshold']:.2f}, "
            f"max_positions={blended['max_positions']}"
        )

        return blended

    def get_available_regimes(self) -> List[str]:
        """Return list of known regime labels."""
        return list(self._profiles.keys())

    def __repr__(self) -> str:
        return f"RegimeAdaptiveParams(regimes={list(self._profiles.keys())})"
