"""
Adaptive Learning Agent

Real-time parameter adjustment based on recent trade performance.
Aggressively adapts strategy parameters to current market conditions.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque
from loguru import logger

from agents.base import BaseAgent
from agents.events import Event, EventType


@dataclass
class TradeOutcome:
    """Record of a trade outcome"""
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    is_winner: bool
    exit_reason: str  # 'target', 'stop', 'time', 'manual'

    # Parameters used for this trade
    rrs_at_entry: float
    stop_multiplier: float
    target_multiplier: float
    market_regime: str = "unknown"


@dataclass
class StrategyParameters:
    """Current strategy parameters"""
    rrs_threshold: float = 2.0
    stop_multiplier: float = 1.0
    target_multiplier: float = 1.0
    max_positions: int = 5
    ml_confidence_threshold: float = 72.0

    # Adaptive bounds
    rrs_min: float = 1.5
    rrs_max: float = 3.0
    stop_min: float = 0.5
    stop_max: float = 2.0
    target_min: float = 0.5
    target_max: float = 3.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyParameters':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PerformanceMetrics:
    """Rolling performance metrics"""
    period_trades: int = 0
    period_wins: int = 0
    period_losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # Exit analysis
    stop_outs: int = 0
    target_hits: int = 0
    time_exits: int = 0


class AdaptiveLearner(BaseAgent):
    """
    Adaptive learning agent that adjusts parameters in real-time

    Features:
    - Rolling window performance tracking
    - Automatic parameter adjustment based on recent results
    - Drawdown protection (tighten parameters during losing streaks)
    - Momentum exploitation (loosen parameters during winning streaks)
    - Regime-specific parameter memory
    """

    def __init__(
        self,
        window_size: int = 20,  # Rolling window of trades
        adjustment_frequency: int = 5,  # Adjust every N trades
        learning_rate: float = 0.1,  # How aggressively to adjust
        config_path: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(name="AdaptiveLearner", **kwargs)

        self.window_size = window_size
        self.adjustment_frequency = adjustment_frequency
        self.learning_rate = learning_rate

        self.config_path = config_path or Path.home() / ".rdt-trading" / "adaptive_config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Current parameters
        self.params = StrategyParameters()

        # Trade history (rolling window)
        self.recent_trades: deque[TradeOutcome] = deque(maxlen=window_size)

        # Performance tracking
        self.perf_metrics = PerformanceMetrics()
        self.trades_since_adjustment = 0

        # Regime-specific parameters (learn what works in each regime)
        self.regime_params: Dict[str, StrategyParameters] = {}

        # State tracking
        self.current_regime = "unknown"
        self.is_in_drawdown = False
        self.peak_equity = 0.0
        self.current_equity = 0.0

    async def initialize(self):
        """Initialize adaptive learner"""
        logger.info("Initializing Adaptive Learner...")

        # Load saved configuration
        self._load_config()

        logger.info(f"Adaptive Learner initialized with parameters: {self.params.to_dict()}")

    async def cleanup(self):
        """Save state on cleanup"""
        self._save_config()
        logger.info("Adaptive Learner state saved")

    def get_subscribed_events(self) -> List[EventType]:
        """Events this agent listens to"""
        return [
            EventType.POSITION_CLOSED,  # Learn from trade outcomes
            EventType.REGIME_CHANGE,    # Adapt to regime changes
        ]

    async def handle_event(self, event: Event):
        """Handle incoming events"""
        try:
            if event.event_type == EventType.POSITION_CLOSED:
                await self._on_trade_closed(event.data)

            elif event.event_type == EventType.REGIME_CHANGE:
                await self._on_regime_change(event.data)

        except Exception as e:
            logger.error(f"Error in adaptive learner: {e}")

    async def _on_trade_closed(self, trade_data: Dict):
        """Process a closed trade and learn from it"""
        try:
            outcome = TradeOutcome(
                symbol=trade_data.get("symbol", ""),
                direction=trade_data.get("direction", "long"),
                entry_time=trade_data.get("entry_time", datetime.now()),
                exit_time=trade_data.get("exit_time", datetime.now()),
                entry_price=trade_data.get("entry_price", 0),
                exit_price=trade_data.get("exit_price", 0),
                pnl=trade_data.get("pnl", 0),
                pnl_percent=trade_data.get("pnl_percent", 0),
                is_winner=trade_data.get("pnl", 0) > 0,
                exit_reason=trade_data.get("exit_reason", "unknown"),
                rrs_at_entry=trade_data.get("rrs", 0),
                stop_multiplier=trade_data.get("stop_multiplier", self.params.stop_multiplier),
                target_multiplier=trade_data.get("target_multiplier", self.params.target_multiplier),
                market_regime=trade_data.get("market_regime", self.current_regime)
            )

            # Add to history
            self.recent_trades.append(outcome)
            self.trades_since_adjustment += 1

            # Update metrics
            self._update_metrics()

            # Check if adjustment needed
            if self.trades_since_adjustment >= self.adjustment_frequency:
                self._adjust_parameters()
                self.trades_since_adjustment = 0

            # Log outcome
            status = "WIN" if outcome.is_winner else "LOSS"
            logger.info(
                f"Trade {status}: {outcome.symbol} {outcome.pnl:+.2f} "
                f"(Win Rate: {self.perf_metrics.win_rate:.1%})"
            )

        except Exception as e:
            logger.error(f"Error processing trade outcome: {e}")

    async def _on_regime_change(self, regime_data: Dict):
        """Handle market regime change"""
        new_regime = regime_data.get("regime", "unknown")

        if new_regime != self.current_regime:
            # Save current params for old regime
            if self.current_regime != "unknown":
                self.regime_params[self.current_regime] = StrategyParameters(
                    rrs_threshold=self.params.rrs_threshold,
                    stop_multiplier=self.params.stop_multiplier,
                    target_multiplier=self.params.target_multiplier,
                    max_positions=self.params.max_positions,
                    ml_confidence_threshold=self.params.ml_confidence_threshold
                )

            # Load params for new regime if we have them
            if new_regime in self.regime_params:
                saved = self.regime_params[new_regime]
                self.params.rrs_threshold = saved.rrs_threshold
                self.params.stop_multiplier = saved.stop_multiplier
                self.params.target_multiplier = saved.target_multiplier
                logger.info(f"Loaded saved parameters for {new_regime} regime")

            self.current_regime = new_regime
            logger.info(f"Regime changed to: {new_regime}")

    def _update_metrics(self):
        """Update rolling performance metrics"""
        if not self.recent_trades:
            return

        trades = list(self.recent_trades)

        self.perf_metrics.period_trades = len(trades)
        self.perf_metrics.period_wins = sum(1 for t in trades if t.is_winner)
        self.perf_metrics.period_losses = len(trades) - self.perf_metrics.period_wins

        if self.perf_metrics.period_trades > 0:
            self.perf_metrics.win_rate = self.perf_metrics.period_wins / self.perf_metrics.period_trades

        wins = [t.pnl for t in trades if t.is_winner]
        losses = [abs(t.pnl) for t in trades if not t.is_winner]

        self.perf_metrics.avg_win = sum(wins) / len(wins) if wins else 0
        self.perf_metrics.avg_loss = sum(losses) / len(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = sum(losses)
        self.perf_metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        self.perf_metrics.total_pnl = sum(t.pnl for t in trades)

        # Count exit types
        self.perf_metrics.stop_outs = sum(1 for t in trades if t.exit_reason == 'stop')
        self.perf_metrics.target_hits = sum(1 for t in trades if t.exit_reason == 'target')
        self.perf_metrics.time_exits = sum(1 for t in trades if t.exit_reason == 'time')

        # Track consecutive wins/losses
        if trades:
            last_trade = trades[-1]
            if last_trade.is_winner:
                self.perf_metrics.consecutive_losses = 0
                self.perf_metrics.consecutive_wins += 1
            else:
                self.perf_metrics.consecutive_wins = 0
                self.perf_metrics.consecutive_losses += 1

    def _adjust_parameters(self):
        """Adjust strategy parameters based on recent performance"""
        if self.perf_metrics.period_trades < 5:
            return  # Not enough data

        logger.info(f"Adjusting parameters (Win Rate: {self.perf_metrics.win_rate:.1%})")

        # Track all parameter changes for DB persistence
        changes = []

        # === Win Rate Optimization ===
        if self.perf_metrics.win_rate < 0.45:
            # Low win rate - tighten targets, widen stops
            changes.extend(self._adjust_param('target_multiplier', -self.learning_rate))
            changes.extend(self._adjust_param('stop_multiplier', +self.learning_rate * 0.5))
            changes.extend(self._adjust_param('rrs_threshold', +self.learning_rate))
            logger.info("Low win rate - tightening targets")

        elif self.perf_metrics.win_rate > 0.60:
            # High win rate - can afford wider targets
            changes.extend(self._adjust_param('target_multiplier', +self.learning_rate * 0.5))
            changes.extend(self._adjust_param('rrs_threshold', -self.learning_rate * 0.5))
            logger.info("High win rate - widening targets")

        # === Stop Loss Optimization ===
        if self.perf_metrics.period_trades > 0:
            stop_rate = self.perf_metrics.stop_outs / self.perf_metrics.period_trades

            if stop_rate > 0.6:
                # Too many stop outs - widen stops
                changes.extend(self._adjust_param('stop_multiplier', +self.learning_rate))
                logger.info(f"High stop rate ({stop_rate:.1%}) - widening stops")

            elif stop_rate < 0.2 and self.perf_metrics.profit_factor < 1.5:
                # Low stop rate but poor profit factor - tighten stops
                changes.extend(self._adjust_param('stop_multiplier', -self.learning_rate * 0.5))
                logger.info("Low stops but poor PF - tightening stops")

        # === Drawdown Protection ===
        if self.perf_metrics.consecutive_losses >= 3:
            # Losing streak - become more conservative
            changes.extend(self._adjust_param('rrs_threshold', +self.learning_rate))
            changes.extend(self._adjust_param('max_positions', -1, min_val=2, max_val=10))
            self.is_in_drawdown = True
            logger.warning(f"Losing streak ({self.perf_metrics.consecutive_losses}) - becoming conservative")

        elif self.perf_metrics.consecutive_wins >= 3 and self.is_in_drawdown:
            # Recovery - return to normal
            self.is_in_drawdown = False
            changes.extend(self._adjust_param('max_positions', +1, min_val=2, max_val=10))
            logger.info("Recovery detected - normalizing parameters")

        # === Profit Factor Optimization ===
        if self.perf_metrics.profit_factor < 1.0:
            # Losing money - be more selective
            changes.extend(self._adjust_param('ml_confidence_threshold', +2))
            changes.extend(self._adjust_param('rrs_threshold', +self.learning_rate * 0.5))
            logger.warning("PF < 1.0 - increasing selectivity")

        # Save updated config (file-based)
        self._save_config()

        # Persist parameter changes to database
        self._persist_parameter_changes(changes)

        logger.info(
            f"New parameters: RRS={self.params.rrs_threshold:.2f}, "
            f"Stop={self.params.stop_multiplier:.2f}x, "
            f"Target={self.params.target_multiplier:.2f}x"
        )

    def _adjust_param(
        self,
        param_name: str,
        delta: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> List[Dict]:
        """Adjust a parameter within bounds. Returns list of change records."""
        current = getattr(self.params, param_name)

        # Get bounds
        if min_val is None:
            min_val = getattr(self.params, f"{param_name.split('_')[0]}_min", 0)
        if max_val is None:
            max_val = getattr(self.params, f"{param_name.split('_')[0]}_max", float('inf'))

        # Apply adjustment
        new_value = current + delta
        new_value = max(min_val, min(max_val, new_value))

        setattr(self.params, param_name, new_value)

        # Return change record if value actually changed
        if new_value != current:
            return [{
                'parameter_name': param_name,
                'old_value': current,
                'new_value': new_value,
            }]
        return []

    def _persist_parameter_changes(self, changes: List[Dict]):
        """Persist parameter change records to the database."""
        if not changes:
            return
        try:
            from data.database import get_trades_repository
            repo = get_trades_repository()
            for change in changes:
                change['reason'] = 'adaptive_adjustment'
                change['trade_count_basis'] = self.perf_metrics.period_trades
                change['win_rate_at_change'] = self.perf_metrics.win_rate
                change['regime'] = self.current_regime
                repo.save_parameter_change(change)
        except Exception as e:
            logger.warning(f"Failed to persist parameter changes to DB: {e}")

    def get_current_parameters(self) -> Dict:
        """Get current strategy parameters for use by other agents"""
        return {
            'rrs_threshold': self.params.rrs_threshold,
            'stop_multiplier': self.params.stop_multiplier,
            'target_multiplier': self.params.target_multiplier,
            'max_positions': self.params.max_positions,
            'ml_confidence_threshold': self.params.ml_confidence_threshold,
            'is_in_drawdown': self.is_in_drawdown,
            'current_regime': self.current_regime,
            'recent_win_rate': self.perf_metrics.win_rate,
            'recent_profit_factor': self.perf_metrics.profit_factor
        }

    def _load_config(self):
        """Load saved configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)

                self.params = StrategyParameters.from_dict(data.get('params', {}))

                # Load regime params
                for regime, params in data.get('regime_params', {}).items():
                    self.regime_params[regime] = StrategyParameters.from_dict(params)

                logger.info(f"Loaded adaptive config from {self.config_path}")

        except Exception as e:
            logger.warning(f"Could not load config: {e}")

    def _save_config(self):
        """Save current configuration to file (and DB snapshot via equity tracker)."""
        try:
            data = {
                'params': self.params.to_dict(),
                'regime_params': {k: v.to_dict() for k, v in self.regime_params.items()},
                'last_updated': datetime.now().isoformat(),
                'metrics': {
                    'win_rate': self.perf_metrics.win_rate,
                    'profit_factor': self.perf_metrics.profit_factor,
                    'total_trades': self.perf_metrics.period_trades
                }
            }

            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save config: {e}")


# Singleton instance
_learner: Optional[AdaptiveLearner] = None


def get_adaptive_learner() -> AdaptiveLearner:
    """Get global adaptive learner instance"""
    global _learner
    if _learner is None:
        _learner = AdaptiveLearner()
    return _learner
