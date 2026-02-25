"""
Analyzer Agent
Analyzes trading signals and validates setups
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from agents.base import BaseAgent
from agents.events import Event, EventType
from risk import RiskManager, PositionSizer, RiskLimits
from ml import StackedEnsemble, RegimeDetector, FeatureEngineer

# News sentiment integration
try:
    from shared.news_sentiment import get_news_analyzer, NewsSentimentAnalyzer
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    logger.warning("News sentiment module not available")

# Adaptive learning integration
try:
    from agents.adaptive_learner import get_adaptive_learner
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    logger.warning("Adaptive learner module not available")

# Multi-timeframe support
try:
    from strategies.timeframes import (
        Timeframe, TimeframeConfig, TimeframeManager,
        get_timeframe_manager, TIMEFRAME_CONFIGS
    )
    TIMEFRAME_AVAILABLE = True
except ImportError:
    TIMEFRAME_AVAILABLE = False
    logger.warning("Multi-timeframe module not available")

# A/B Testing support
try:
    from ml.ab_testing import (
        get_experiment_manager,
        ExperimentManager,
        Experiment,
        ModelVariant,
        OutcomeType,
    )
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False
    logger.warning("A/B testing module not available")

# Regime-adaptive parameter system
try:
    from scanner.regime_params import RegimeAdaptiveParams
    REGIME_PARAMS_AVAILABLE = True
except ImportError:
    REGIME_PARAMS_AVAILABLE = False
    logger.debug("Regime-adaptive parameters not available for analyzer")


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

        # ML Components - use sophisticated StackedEnsemble with XGBoost and Random Forest
        self.ensemble = StackedEnsemble(use_xgboost=True, use_random_forest=True, use_lstm=False)
        self.regime_detector = RegimeDetector()
        self.feature_engineer = FeatureEngineer()

        # ML Thresholds - RELAXED for more signals (previous: 72.0 / 75.0)
        self.min_ml_probability = 62.0  # Was 72.0 — allow more setups through
        self.min_overall_confidence = 65.0  # Was 75.0 — allow more setups through

        # Analysis configuration - matches 10-year optimized params
        self.min_rrs = 1.75  # Optimized: 1.75 CAGR-optimal threshold
        self.min_rr_ratio = 1.5  # Lower R:R allows tighter targets (higher win rate)
        self.require_daily_alignment = False  # Relaxed criteria handled by scanner
        self.max_atr_percent = 6.0  # Was 4.0 — allow higher-volatility stocks

        # Regime filtering - allow signals in all non-extreme regimes
        self.favorable_regimes_long = ['bull_trending', 'low_volatility', 'high_volatility']
        self.favorable_regimes_short = ['bear_trending', 'high_volatility']
        self.use_regime_filter = True

        # News sentiment filtering
        self.use_news_filter = NEWS_AVAILABLE
        self.news_analyzer = get_news_analyzer() if NEWS_AVAILABLE else None
        self.avoid_earnings_days = 2  # Days before/after earnings to avoid

        # Tracking
        self.signals_analyzed = 0
        self.setups_approved = 0
        self.setups_rejected = 0

        # Adaptive learning integration
        self.use_adaptive_params = ADAPTIVE_AVAILABLE
        self.adaptive_learner = get_adaptive_learner() if ADAPTIVE_AVAILABLE else None

        # Multi-timeframe support
        self.use_timeframes = TIMEFRAME_AVAILABLE
        self.timeframe_manager = get_timeframe_manager() if TIMEFRAME_AVAILABLE else None
        self.default_timeframe = Timeframe.MEDIUM if TIMEFRAME_AVAILABLE else None

        # A/B Testing support
        self.use_ab_testing = AB_TESTING_AVAILABLE
        self.experiment_manager: Optional[ExperimentManager] = None
        if AB_TESTING_AVAILABLE:
            try:
                self.experiment_manager = get_experiment_manager()
                logger.info("A/B testing experiment manager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize experiment manager: {e}")
                self.use_ab_testing = False

        # Track A/B test request IDs for outcome recording
        self._ab_request_mapping: Dict[str, str] = {}  # symbol -> request_id

        # Regime-adaptive parameter system
        self._regime_adaptive_params: Optional[RegimeAdaptiveParams] = None
        self._use_regime_adaptive = REGIME_PARAMS_AVAILABLE
        if self._use_regime_adaptive:
            try:
                self._regime_adaptive_params = RegimeAdaptiveParams()
                logger.info("Regime-adaptive parameter system enabled in analyzer")
            except Exception as e:
                logger.warning(f"Could not initialize regime-adaptive params: {e}")
                self._use_regime_adaptive = False

    def _get_adaptive_params(self) -> dict:
        """Get parameters from adaptive learner if available"""
        if self.use_adaptive_params and self.adaptive_learner:
            params = self.adaptive_learner.get_current_parameters()
            return {
                'rrs_threshold': params.get('rrs_threshold', self.min_rrs),
                'ml_confidence_threshold': params.get('ml_confidence_threshold', self.min_ml_probability),
                'is_in_drawdown': params.get('is_in_drawdown', False),
                'max_positions': params.get('max_positions', 5),
                'stop_multiplier': params.get('stop_multiplier', 1.0),
                'target_multiplier': params.get('target_multiplier', 1.0)
            }
        return {
            'rrs_threshold': self.min_rrs,
            'ml_confidence_threshold': self.min_ml_probability,
            'is_in_drawdown': False,
            'max_positions': 8,  # Was 5
            'stop_multiplier': 1.5,  # Was 1.0 — wider stops
            'target_multiplier': 2.5  # Was 1.0 — wider targets
        }

    async def initialize(self):
        """Initialize analyzer"""
        logger.info("Analyzer agent initialized with ML capabilities")
        self.metrics.custom_metrics["signals_analyzed"] = 0
        self.metrics.custom_metrics["approval_rate"] = 0
        self.metrics.custom_metrics["ml_enabled"] = True

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
        Analyze a trading signal with ML enhancement

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

        # Get adaptive parameters (may be modified by recent performance)
        adaptive_params = self._get_adaptive_params()
        effective_rrs_threshold = adaptive_params['rrs_threshold']
        effective_ml_threshold = adaptive_params['ml_confidence_threshold']
        is_in_drawdown = adaptive_params['is_in_drawdown']

        if is_in_drawdown:
            logger.info(f"System in drawdown mode - using conservative parameters")

        # Validation checks
        rejection_reasons = []

        # Check 1: RRS strength (using adaptive threshold)
        if abs(rrs) < effective_rrs_threshold:
            rejection_reasons.append(f"RRS too weak: {rrs:.2f} < {effective_rrs_threshold:.2f}")

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

        # Build signal features for dynamic position sizing
        signal_features = {
            "ml_confidence": 50.0,  # updated below if ML analysis succeeds
            "rrs": abs(rrs),
            "quality_warnings": signal.get("quality_warnings", []),
            "market_regime": "unknown",  # updated below if ML analysis succeeds
            "symbol": symbol,
            "portfolio_heat": self._calculate_portfolio_heat(),
            "sector_concentration": self._count_sector_positions(symbol),
        }

        # Calculate position sizing (using adaptive stop/target multipliers)
        position_result = self.position_sizer.calculate_position_size(
            account_size=self.risk_manager.current_balance,
            entry_price=price,
            atr=atr,
            direction=direction,
            stop_multiplier=adaptive_params['stop_multiplier'],
            target_multiplier=adaptive_params['target_multiplier'],
            signal_features=signal_features,
        )

        # Check 4: Position viability
        if position_result.shares == 0:
            rejection_reasons.append("Position size calculated as 0")

        # Note: R/R and risk manager checks are deferred until after regime
        # overrides may update stop/target multipliers (see re-calculation below)
        trade_risk = None

        # ML Enhancement - Extract features and get predictions
        try:
            # Extract features
            feature_result = self.feature_engineer.extract_features(signal)
            features = feature_result['features']
            top_features = feature_result['top_features']

            # Get ML prediction using StackedEnsemble
            # StackedEnsemble returns probability 0-1, convert to 0-100 for consistency
            if self.ensemble.is_trained:
                # Reshape features for batch prediction (1 sample)
                features_array = np.array(features).reshape(1, -1)
                ml_probability = self.ensemble.predict_success_probability(features_array)[0] * 100.0
            else:
                # Fallback: use feature-based heuristic when model not yet trained
                rrs_score = min(abs(signal.get('rrs', 0)) * 15, 50)
                trend_score = 25 if (direction == 'long' and signal.get('daily_strong')) or \
                                   (direction == 'short' and signal.get('daily_weak')) else 10
                ml_probability = min(rrs_score + trend_score + 25, 100.0)

            # Detect market regime
            market_regime, regime_confidence = self.regime_detector.detect_regime(signal)

            # Apply regime-adaptive parameter overrides
            regime_params = None
            if self._use_regime_adaptive and self._regime_adaptive_params:
                try:
                    regime_params = self._regime_adaptive_params.get_params(market_regime)

                    # Override thresholds with regime-adaptive values
                    effective_rrs_threshold = regime_params['rrs_threshold']
                    effective_ml_threshold = regime_params['min_confidence']

                    # Override stop/target multipliers in adaptive_params
                    adaptive_params['stop_multiplier'] = regime_params['stop_multiplier']
                    adaptive_params['target_multiplier'] = regime_params['target_multiplier']
                    adaptive_params['max_positions'] = regime_params['max_positions']
                    adaptive_params['risk_per_trade'] = regime_params['risk_per_trade']

                    logger.info(
                        f"Regime-adaptive overrides for {market_regime}: "
                        f"rrs_threshold={effective_rrs_threshold:.2f}, "
                        f"min_confidence={effective_ml_threshold:.1f}%, "
                        f"stop_mult={regime_params['stop_multiplier']:.2f}, "
                        f"target_mult={regime_params['target_multiplier']:.2f}, "
                        f"max_positions={regime_params['max_positions']}, "
                        f"prefer_momentum={regime_params['prefer_momentum']}, "
                        f"prefer_mr={regime_params['prefer_mean_reversion']}"
                    )

                    # Re-validate RRS with updated threshold (check 1 may have
                    # used the old threshold; remove old rejection and re-check)
                    old_rrs_rejection = [
                        r for r in rejection_reasons if r.startswith("RRS too weak")
                    ]
                    for r in old_rrs_rejection:
                        rejection_reasons.remove(r)
                    if abs(rrs) < effective_rrs_threshold:
                        rejection_reasons.append(
                            f"RRS too weak: {rrs:.2f} < {effective_rrs_threshold:.2f} "
                            f"(regime={market_regime})"
                        )

                except Exception as e:
                    logger.warning(f"Regime-adaptive param lookup failed: {e}")

            # Calculate overall confidence
            regime_multiplier = self.regime_detector.get_regime_multiplier(market_regime)
            overall_confidence = ml_probability * regime_multiplier

            logger.info(
                f"ML Analysis: {symbol} - Probability: {ml_probability:.1f}%, "
                f"Regime: {market_regime} ({regime_confidence:.2f}), "
                f"Overall Confidence: {overall_confidence:.1f}%"
            )

            # Check 7: ML Probability threshold (using adaptive/regime threshold)
            if ml_probability < effective_ml_threshold:
                rejection_reasons.append(
                    f"ML probability too low: {ml_probability:.1f}% < {effective_ml_threshold:.1f}%"
                )

            # Check 8: Overall confidence threshold
            if overall_confidence < self.min_overall_confidence:
                rejection_reasons.append(
                    f"Overall confidence too low: {overall_confidence:.1f}% < {self.min_overall_confidence:.1f}%"
                )

            # Check 9: Regime filter - only trade in favorable market conditions
            # With regime-adaptive params, we dynamically adjust favorable regimes
            if self.use_regime_filter:
                # In regime-adaptive mode, allow trades aligned with regime preference
                if regime_params and regime_params.get('prefer_momentum'):
                    # Momentum regimes: allow both trending directions
                    favorable_long = ['bull_trending', 'low_volatility', 'high_volatility']
                    favorable_short = ['bear_trending', 'high_volatility']
                elif regime_params and regime_params.get('prefer_mean_reversion'):
                    # Mean-reversion regimes: allow counter-trend in calm markets
                    favorable_long = ['bull_trending', 'low_volatility', 'high_volatility']
                    favorable_short = ['bear_trending', 'low_volatility', 'high_volatility']
                else:
                    # Fallback to static lists
                    favorable_long = self.favorable_regimes_long
                    favorable_short = self.favorable_regimes_short

                if direction == "long" and market_regime not in favorable_long:
                    rejection_reasons.append(
                        f"Unfavorable regime for long: {market_regime}"
                    )
                elif direction == "short" and market_regime not in favorable_short:
                    rejection_reasons.append(
                        f"Unfavorable regime for short: {market_regime}"
                    )

            # Check 10a: Sector concentration limit (regime-adaptive)
            if regime_params:
                sector_limit = regime_params.get('sector_concentration_limit', 3)
                sector_count = self._count_sector_positions(symbol)
                if sector_count >= sector_limit:
                    rejection_reasons.append(
                        f"Sector concentration limit reached: {sector_count} >= "
                        f"{sector_limit} (regime={market_regime})"
                    )

            # Check 10: News sentiment filter - avoid negative news and earnings
            if self.use_news_filter and self.news_analyzer:
                try:
                    should_trade, news_reason = self.news_analyzer.should_trade(symbol, direction)
                    if not should_trade:
                        rejection_reasons.append(f"News filter: {news_reason}")
                    else:
                        logger.debug(f"News check passed for {symbol}: {news_reason}")
                except Exception as e:
                    logger.warning(f"News check failed for {symbol}: {e}")

            # Re-calculate position sizing with regime-adjusted multipliers
            signal_features["ml_confidence"] = ml_probability
            signal_features["market_regime"] = market_regime
            position_result = self.position_sizer.calculate_position_size(
                account_size=self.risk_manager.current_balance,
                entry_price=price,
                atr=atr,
                direction=direction,
                stop_multiplier=adaptive_params['stop_multiplier'],
                target_multiplier=adaptive_params['target_multiplier'],
                signal_features=signal_features,
            )

            # Now check R/R and risk management with final position params
            if position_result.shares == 0:
                # Clear any earlier "Position size calculated as 0" and re-add
                rejection_reasons = [r for r in rejection_reasons if "Position size" not in r]
                rejection_reasons.append("Position size calculated as 0")
            else:
                rejection_reasons = [r for r in rejection_reasons if "Position size" not in r]

            if position_result.risk_reward_ratio < self.min_rr_ratio:
                rejection_reasons.append(
                    f"R/R too low: {position_result.risk_reward_ratio:.2f}"
                )

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

            # Create enhanced signal data
            ml_data = {
                'ml_probability': ml_probability,
                'confidence': overall_confidence,
                'market_regime': market_regime,
                'regime_confidence': regime_confidence,
                'top_features': top_features,
            }

            # Attach regime-adaptive metadata for downstream use
            if regime_params:
                ml_data['regime_params'] = {
                    'rrs_threshold': regime_params['rrs_threshold'],
                    'stop_multiplier': regime_params['stop_multiplier'],
                    'target_multiplier': regime_params['target_multiplier'],
                    'max_positions': regime_params['max_positions'],
                    'risk_per_trade': regime_params['risk_per_trade'],
                    'prefer_momentum': regime_params['prefer_momentum'],
                    'prefer_mean_reversion': regime_params['prefer_mean_reversion'],
                    'sector_concentration_limit': regime_params['sector_concentration_limit'],
                }

        except Exception as e:
            logger.error(f"ML analysis failed: {e}")
            rejection_reasons.append(f"ML analysis error: {str(e)}")
            ml_data = None

        # Multi-timeframe classification
        timeframe_data = None
        if self.use_timeframes and self.timeframe_manager:
            try:
                # Get weekly/monthly trend from signal if available
                weekly_trend = signal.get("weekly_trend")
                monthly_trend = signal.get("monthly_trend")

                # Add ML probability to signal for classification
                signal_with_ml = {**signal}
                if ml_data:
                    signal_with_ml["ml_probability"] = ml_data.get("ml_probability", 0)

                # Classify signal to best timeframe
                selected_tf, tf_config = self.timeframe_manager.classify_signal(
                    signal_with_ml, weekly_trend, monthly_trend
                )

                # Check if we can open position in this timeframe
                can_open, reason = self.timeframe_manager.can_open_position(selected_tf, symbol)
                if not can_open:
                    rejection_reasons.append(f"Timeframe limit: {reason}")
                else:
                    # Get timeframe-specific position parameters
                    tf_params = self.timeframe_manager.get_position_params(
                        selected_tf, price, atr, direction
                    )

                    timeframe_data = {
                        "timeframe": selected_tf.value,
                        "timeframe_name": tf_config.name,
                        "max_hold_days": tf_config.max_hold_days,
                        "trailing_enabled": tf_config.trailing_stop,
                        "partial_profit_enabled": tf_config.partial_profit_enabled,
                        **tf_params
                    }

                    logger.info(
                        f"Classified {symbol} as {selected_tf.value} trade "
                        f"(target: {tf_config.target_multiplier}x ATR, "
                        f"stop: {tf_config.stop_multiplier}x ATR)"
                    )

            except Exception as e:
                logger.warning(f"Timeframe classification failed for {symbol}: {e}")
                # Continue without timeframe data

        # Decision
        if rejection_reasons:
            await self._reject_setup(symbol, direction, rejection_reasons, signal=signal, ml_data=ml_data)
        else:
            await self._approve_setup(signal, position_result, trade_risk, ml_data, adaptive_params, timeframe_data)

    async def _approve_setup(
        self,
        signal: Dict,
        position_result,
        trade_risk,
        ml_data: Optional[Dict] = None,
        adaptive_params: Optional[Dict] = None,
        timeframe_data: Optional[Dict] = None
    ):
        """Approve a valid setup with ML enhancement and timeframe classification"""
        self.setups_approved += 1
        self._update_approval_rate()

        # Build enhanced setup with original signal data
        setup = {
            **signal,  # Include all original signal data
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
            "timestamp": datetime.now().isoformat()
        }

        # Add ML data if available
        if ml_data:
            setup.update({
                "ml_probability": ml_data["ml_probability"],
                "confidence": ml_data["confidence"],
                "market_regime": ml_data["market_regime"],
                "regime_confidence": ml_data["regime_confidence"],
                "top_features": ml_data["top_features"]
            })

        # Add adaptive parameters for learning feedback
        if adaptive_params:
            setup.update({
                "stop_multiplier": adaptive_params.get("stop_multiplier", 1.0),
                "target_multiplier": adaptive_params.get("target_multiplier", 1.0),
                "is_in_drawdown": adaptive_params.get("is_in_drawdown", False)
            })

        # Add timeframe data for multi-timeframe trading
        if timeframe_data:
            setup.update({
                "timeframe": timeframe_data.get("timeframe", "medium"),
                "timeframe_name": timeframe_data.get("timeframe_name", "Medium-Term Swing"),
                "max_hold_days": timeframe_data.get("max_hold_days", 28),
                "trailing_enabled": timeframe_data.get("trailing_enabled", False),
                "partial_profit_enabled": timeframe_data.get("partial_profit_enabled", False),
                "partial_profit_pct": timeframe_data.get("partial_profit_pct", 0)
            })
            # Use timeframe-specific stop/target if not overridden by adaptive params
            if "stop_price" in timeframe_data and not adaptive_params:
                setup["stop_price"] = timeframe_data["stop_price"]
                setup["target_price"] = timeframe_data["target_price"]

        # Record A/B test prediction if active experiment
        ab_request_id = None
        if self.use_ab_testing and ml_data:
            model_id, variant, experiment = self._get_ab_model_for_prediction(
                symbol=signal["symbol"],
                request_id=str(uuid.uuid4()),
            )
            if experiment and variant:
                ab_request_id = self._record_ab_prediction(
                    experiment=experiment,
                    symbol=signal["symbol"],
                    direction=signal["direction"],
                    variant=variant,
                    model_id=model_id,
                    ml_probability=ml_data["ml_probability"],
                    overall_confidence=ml_data["confidence"],
                    entry_price=signal["price"],
                    stop_price=position_result.stop_price,
                    target_price=position_result.target_price,
                    rrs_at_entry=signal.get("rrs"),
                    features=ml_data.get("top_features"),
                )
                if ab_request_id:
                    setup["ab_request_id"] = ab_request_id
                    setup["ab_variant"] = variant.value
                    setup["ab_experiment"] = experiment.name

        # Log the approved setup
        tf_label = f" [{setup.get('timeframe', 'swing').upper()}]" if timeframe_data else ""
        ab_label = f" [A/B:{setup.get('ab_variant', '')}]" if ab_request_id else ""
        if ml_data:
            logger.info(
                f"APPROVED{tf_label}{ab_label}: {setup['symbol']} {setup['direction'].upper()} "
                f"{setup['shares']} shares @ ${setup['entry_price']:.2f} "
                f"Stop: ${setup['stop_price']:.2f} Target: ${setup['target_price']:.2f} | "
                f"ML: {ml_data['ml_probability']:.1f}% Confidence: {ml_data['confidence']:.1f}% "
                f"Regime: {ml_data['market_regime']}"
            )
        else:
            logger.info(
                f"APPROVED{tf_label}: {setup['symbol']} {setup['direction'].upper()} "
                f"{setup['shares']} shares @ ${setup['entry_price']:.2f} "
                f"Stop: ${setup['stop_price']:.2f} Target: ${setup['target_price']:.2f}"
            )

        await self.publish(EventType.SETUP_VALID, setup)

    async def _reject_setup(
        self,
        symbol: str,
        direction: str,
        reasons: List[str],
        signal: Optional[Dict] = None,
        ml_data: Optional[Dict] = None,
    ):
        """Reject an invalid setup and persist to database for outcome tracking"""
        self.setups_rejected += 1
        self._update_approval_rate()

        logger.info(f"REJECTED: {symbol} {direction} - {'; '.join(reasons)}")

        await self.publish(EventType.SETUP_INVALID, {
            "symbol": symbol,
            "direction": direction,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat()
        })

        # Persist rejected signal for outcome analysis
        try:
            from data.database import get_trades_repository
            repo = get_trades_repository()
            rejected_data = {
                'symbol': symbol,
                'direction': direction,
                'rrs': signal.get('rrs', 0) if signal else 0,
                'price': signal.get('price', 0) if signal else 0,
                'timestamp': datetime.utcnow(),
                'rejection_reasons': reasons,
                'market_regime': ml_data.get('market_regime') if ml_data else None,
                'daily_strong': signal.get('daily_strong') if signal else None,
                'daily_weak': signal.get('daily_weak') if signal else None,
                'atr': signal.get('atr') if signal else None,
                'volume': signal.get('volume') if signal else None,
                'ml_probability': ml_data.get('ml_probability') if ml_data else None,
                'ml_confidence': ml_data.get('confidence') if ml_data else None,
            }
            repo.save_rejected_signal(rejected_data)
        except Exception as e:
            logger.warning(f"Failed to persist rejected signal for {symbol}: {e}")

    def _update_approval_rate(self):
        """Update approval rate metric"""
        if self.signals_analyzed > 0:
            rate = (self.setups_approved / self.signals_analyzed) * 100
            self.metrics.custom_metrics["approval_rate"] = round(rate, 1)
            self.metrics.custom_metrics["setups_approved"] = self.setups_approved
            self.metrics.custom_metrics["setups_rejected"] = self.setups_rejected

    def _calculate_portfolio_heat(self) -> float:
        """Calculate total risk deployed as fraction of account."""
        rm = self.risk_manager
        if rm.current_balance <= 0:
            return 1.0
        total_risk = sum(
            p.get("risk_amount", 0) for p in rm.open_positions.values()
        )
        return total_risk / rm.current_balance

    def _count_sector_positions(self, symbol: str) -> int:
        """Count existing open positions in the same sector as symbol."""
        try:
            from risk.risk_manager import SECTOR_MAP
        except ImportError:
            return 0
        sector = SECTOR_MAP.get(symbol, "other")
        count = sum(
            1 for sym in self.risk_manager.open_positions
            if SECTOR_MAP.get(sym, "other") == sector
        )
        return count

    def set_min_rrs(self, value: float):
        """Update minimum RRS threshold"""
        self.min_rrs = value
        logger.info(f"Min RRS updated to {value}")

    def set_min_rr_ratio(self, value: float):
        """Update minimum risk/reward ratio"""
        self.min_rr_ratio = value
        logger.info(f"Min R/R ratio updated to {value}")

    # =========================================================================
    # A/B Testing Methods
    # =========================================================================

    def _get_ab_model_for_prediction(
        self,
        symbol: str,
        request_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional["ModelVariant"], Optional["Experiment"]]:
        """
        Get the model to use for prediction from active A/B experiment.

        Args:
            symbol: Trading symbol
            request_id: Optional request ID for deterministic assignment

        Returns:
            Tuple of (model_id, variant, experiment) or (None, None, None) if no active experiment
        """
        if not self.use_ab_testing or not self.experiment_manager:
            return None, None, None

        try:
            experiment = self.experiment_manager.get_active_experiment()
            if experiment and experiment.is_active:
                model_id, variant = experiment.get_model_for_request(request_id)
                logger.debug(
                    f"A/B test '{experiment.name}': using model {model_id} "
                    f"(variant={variant.value}) for {symbol}"
                )
                return model_id, variant, experiment
        except Exception as e:
            logger.warning(f"A/B model selection failed: {e}")

        return None, None, None

    def _record_ab_prediction(
        self,
        experiment: "Experiment",
        symbol: str,
        direction: str,
        variant: "ModelVariant",
        model_id: str,
        ml_probability: float,
        overall_confidence: float,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        rrs_at_entry: Optional[float] = None,
        features: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Record a prediction in the active A/B experiment.

        Args:
            experiment: Active experiment
            symbol: Trading symbol
            direction: Trade direction
            variant: Model variant used
            model_id: Model identifier
            ml_probability: ML prediction probability (0-100)
            overall_confidence: Overall confidence (0-100)
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Target price
            rrs_at_entry: RRS value at entry
            features: Features used for prediction

        Returns:
            Request ID if successful
        """
        if not self.use_ab_testing or not self.experiment_manager:
            return None

        try:
            # Generate request ID
            request_id = str(uuid.uuid4())

            # Record prediction
            recorded_id = self.experiment_manager.record_prediction(
                experiment_name=experiment.name,
                symbol=symbol,
                direction=direction,
                variant=variant,
                model_id=model_id,
                prediction_probability=ml_probability / 100.0,  # Convert to 0-1
                prediction_class=1 if ml_probability >= 50.0 else 0,
                confidence=overall_confidence / 100.0,  # Convert to 0-1
                request_id=request_id,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                rrs_at_entry=rrs_at_entry,
                features=features,
            )

            if recorded_id:
                # Store mapping for later outcome recording
                self._ab_request_mapping[symbol] = recorded_id
                logger.debug(
                    f"Recorded A/B prediction for {symbol}: request_id={recorded_id}, "
                    f"variant={variant.value}, prob={ml_probability:.1f}%"
                )
                return recorded_id

        except Exception as e:
            logger.warning(f"Failed to record A/B prediction: {e}")

        return None

    def record_ab_outcome(
        self,
        symbol: str,
        outcome: str,  # 'win', 'loss', 'breakeven'
        pnl: Optional[float] = None,
        pnl_percent: Optional[float] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        holding_period_hours: Optional[float] = None,
    ) -> bool:
        """
        Record the outcome of a trade for A/B testing.

        Call this method when a trade is closed to record its outcome
        in the active A/B experiment.

        Args:
            symbol: Trading symbol
            outcome: Trade outcome ('win', 'loss', 'breakeven')
            pnl: Profit/loss in dollars
            pnl_percent: Profit/loss as percentage
            exit_price: Exit price
            exit_reason: Reason for exit
            holding_period_hours: How long the trade was held

        Returns:
            True if outcome was recorded successfully
        """
        if not self.use_ab_testing or not self.experiment_manager:
            return False

        # Check if we have a request ID for this symbol
        request_id = self._ab_request_mapping.get(symbol)
        if not request_id:
            logger.debug(f"No A/B test request ID found for {symbol}")
            return False

        try:
            # Get active experiment
            experiment = self.experiment_manager.get_active_experiment()
            if not experiment:
                logger.warning(f"No active experiment for recording outcome of {symbol}")
                return False

            # Map outcome string to OutcomeType
            outcome_type = OutcomeType.PENDING
            if outcome.lower() == 'win':
                outcome_type = OutcomeType.WIN
            elif outcome.lower() == 'loss':
                outcome_type = OutcomeType.LOSS
            elif outcome.lower() == 'breakeven':
                outcome_type = OutcomeType.BREAKEVEN

            # Record outcome
            success = self.experiment_manager.record_outcome(
                experiment_name=experiment.name,
                request_id=request_id,
                outcome=outcome_type,
                pnl=pnl,
                pnl_percent=pnl_percent,
                exit_price=exit_price,
                exit_reason=exit_reason,
                holding_period_hours=holding_period_hours,
            )

            if success:
                # Remove from mapping after recording
                del self._ab_request_mapping[symbol]
                logger.info(
                    f"Recorded A/B outcome for {symbol}: {outcome}, "
                    f"pnl=${pnl:.2f if pnl else 0:.2f}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to record A/B outcome: {e}")

        return False

    def get_ab_experiment_stats(self) -> Optional[Dict]:
        """
        Get statistics for the active A/B experiment.

        Returns:
            Experiment statistics dict or None if no active experiment
        """
        if not self.use_ab_testing or not self.experiment_manager:
            return None

        try:
            experiment = self.experiment_manager.get_active_experiment()
            if experiment:
                return experiment.get_stats()
        except Exception as e:
            logger.error(f"Failed to get A/B experiment stats: {e}")

        return None
