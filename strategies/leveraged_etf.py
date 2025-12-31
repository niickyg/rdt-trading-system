"""
Leveraged ETF Trading Strategy

Trade 3x leveraged ETFs to amplify returns while controlling risk.
This provides synthetic leverage without using margin.
"""

from datetime import date
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from strategies.base_strategy import (
    BaseStrategy,
    StrategySignal,
    SignalDirection,
    SignalStrength
)


# Leveraged ETF pairs mapping
LEVERAGED_ETFS = {
    # Bull / Bear pairs for sector rotation
    'TQQQ': {'underlying': 'QQQ', 'leverage': 3, 'direction': 'bull'},
    'SQQQ': {'underlying': 'QQQ', 'leverage': 3, 'direction': 'bear'},
    'UPRO': {'underlying': 'SPY', 'leverage': 3, 'direction': 'bull'},
    'SPXU': {'underlying': 'SPY', 'leverage': 3, 'direction': 'bear'},
    'SOXL': {'underlying': 'SMH', 'leverage': 3, 'direction': 'bull'},
    'SOXS': {'underlying': 'SMH', 'leverage': 3, 'direction': 'bear'},
    'TNA': {'underlying': 'IWM', 'leverage': 3, 'direction': 'bull'},
    'TZA': {'underlying': 'IWM', 'leverage': 3, 'direction': 'bear'},
    'LABU': {'underlying': 'XBI', 'leverage': 3, 'direction': 'bull'},
    'LABD': {'underlying': 'XBI', 'leverage': 3, 'direction': 'bear'},
    'FAS': {'underlying': 'XLF', 'leverage': 3, 'direction': 'bull'},
    'FAZ': {'underlying': 'XLF', 'leverage': 3, 'direction': 'bear'},
    'ERX': {'underlying': 'XLE', 'leverage': 2, 'direction': 'bull'},
    'ERY': {'underlying': 'XLE', 'leverage': 2, 'direction': 'bear'},
}

# Underlying ETFs to monitor for signals
UNDERLYING_ETFS = ['QQQ', 'SPY', 'SMH', 'IWM', 'XBI', 'XLF', 'XLE']


class LeveragedETFStrategy(BaseStrategy):
    """
    Trade leveraged ETFs based on underlying index strength

    Strategy Logic:
    1. Calculate RRS of underlying ETF vs SPY
    2. If strong RS (RRS > threshold), go long bull ETF
    3. If weak RS (RRS < -threshold), go long bear ETF
    4. Use 1/3 of normal position size (due to 3x leverage)
    5. Tighter stops due to increased volatility

    Benefits:
    - 3x amplification of moves without margin
    - Capital efficient (control $30K of exposure with $10K)
    - Defined risk (can't lose more than invested)
    - No margin calls
    """

    def __init__(
        self,
        name: str = "Leveraged_ETF",
        capital_allocation: float = 0.25,  # 25% of total capital
        max_positions: int = 4,
        risk_per_trade: float = 0.02,  # 2% risk
        rrs_threshold: float = 1.5,
        stop_atr_mult: float = 0.5,  # Tighter stops for leverage
        target_atr_mult: float = 1.0,  # Smaller target, faster exits
        use_inverse: bool = True  # Use inverse ETFs for short exposure
    ):
        super().__init__(name, capital_allocation, max_positions, risk_per_trade)
        self.rrs_threshold = rrs_threshold
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.use_inverse = use_inverse

        from shared.indicators.rrs import RRSCalculator
        self.rrs_calc = RRSCalculator()

    def get_bull_etf(self, underlying: str) -> Optional[str]:
        """Get bull leveraged ETF for underlying"""
        for etf, info in LEVERAGED_ETFS.items():
            if info['underlying'] == underlying and info['direction'] == 'bull':
                return etf
        return None

    def get_bear_etf(self, underlying: str) -> Optional[str]:
        """Get bear leveraged ETF for underlying"""
        for etf, info in LEVERAGED_ETFS.items():
            if info['underlying'] == underlying and info['direction'] == 'bear':
                return etf
        return None

    def calculate_underlying_rrs(
        self,
        underlying_data: pd.DataFrame,
        spy_data: pd.DataFrame
    ) -> Optional[float]:
        """Calculate RRS for underlying ETF"""
        try:
            if len(underlying_data) < 20 or len(spy_data) < 20:
                return None

            # Normalize columns
            underlying = underlying_data.copy()
            underlying.columns = [c.lower() for c in underlying.columns]
            spy = spy_data.copy()
            spy.columns = [c.lower() for c in spy.columns]

            # Calculate ATR
            atr = self.rrs_calc.calculate_atr(underlying).iloc[-1]

            # Get prices
            underlying_close = underlying['close'].iloc[-1]
            underlying_prev = underlying['close'].iloc[-2]
            spy_close = spy['close'].iloc[-1]
            spy_prev = spy['close'].iloc[-2]

            # Calculate RRS
            result = self.rrs_calc.calculate_rrs_current(
                stock_data={'current_price': underlying_close, 'previous_close': underlying_prev},
                spy_data={'current_price': spy_close, 'previous_close': spy_prev},
                stock_atr=atr
            )

            return result['rrs']

        except Exception as e:
            logger.debug(f"Error calculating RRS: {e}")
            return None

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        current_date: Optional[date] = None
    ) -> List[StrategySignal]:
        """
        Scan for leveraged ETF opportunities

        Looks at underlying ETFs and generates signals for leveraged versions.
        """
        signals = []

        if current_date:
            spy_data = market_data[market_data.index.date <= current_date]
        else:
            spy_data = market_data

        for underlying in UNDERLYING_ETFS:
            # Skip if we already have a position in related ETF
            bull_etf = self.get_bull_etf(underlying)
            bear_etf = self.get_bear_etf(underlying)

            if bull_etf in self.positions or bear_etf in self.positions:
                continue

            # Get underlying data
            if underlying not in stock_data:
                continue

            underlying_data = stock_data[underlying]
            if current_date:
                underlying_data = underlying_data[underlying_data.index.date <= current_date]

            # Calculate RRS
            rrs = self.calculate_underlying_rrs(underlying_data, spy_data)
            if rrs is None:
                continue

            # Determine direction
            if rrs > self.rrs_threshold and bull_etf:
                # Go long bull ETF
                trade_etf = bull_etf
                direction = SignalDirection.LONG

            elif rrs < -self.rrs_threshold and self.use_inverse and bear_etf:
                # Go long bear ETF (inverse exposure)
                trade_etf = bear_etf
                direction = SignalDirection.LONG

            else:
                continue

            # Get leveraged ETF data for entry/exit prices
            if trade_etf not in stock_data:
                continue

            etf_data = stock_data[trade_etf]
            if current_date:
                etf_data = etf_data[etf_data.index.date <= current_date]

            if len(etf_data) < 20:
                continue

            # Normalize and calculate ATR for leveraged ETF
            etf_normalized = etf_data.copy()
            etf_normalized.columns = [c.lower() for c in etf_normalized.columns]

            etf_atr = self.rrs_calc.calculate_atr(etf_normalized).iloc[-1]
            etf_close = etf_normalized['close'].iloc[-1]

            # Set stops and targets (tighter due to leverage)
            stop_price = etf_close - (etf_atr * self.stop_atr_mult)
            target_price = etf_close + (etf_atr * self.target_atr_mult)

            # Determine signal strength
            if abs(rrs) > 3.0:
                strength = SignalStrength.VERY_STRONG
            elif abs(rrs) > 2.5:
                strength = SignalStrength.STRONG
            elif abs(rrs) > 2.0:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

            signal = StrategySignal(
                symbol=trade_etf,
                direction=direction,
                strength=strength,
                strategy_name=self.name,
                entry_price=etf_close,
                stop_price=stop_price,
                target_price=target_price,
                atr=etf_atr,
                risk_per_share=abs(etf_close - stop_price),
                rrs_value=rrs,
                suggested_position_pct=self.risk_per_trade / 3,  # 1/3 size due to leverage
                additional_data={
                    'underlying': underlying,
                    'leverage': LEVERAGED_ETFS[trade_etf]['leverage'],
                    'etf_type': LEVERAGED_ETFS[trade_etf]['direction']
                }
            )

            if signal.is_valid:
                signals.append(signal)
                logger.info(
                    f"Leveraged ETF signal: {trade_etf} ({direction.value}) "
                    f"based on {underlying} RRS={rrs:.2f}"
                )

        return signals

    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_data: pd.DataFrame,
        current_date: Optional[date] = None
    ) -> Optional[str]:
        """
        Check if leveraged ETF position should be exited

        Uses tighter time stops due to leverage decay.
        """
        try:
            if current_date:
                data = current_data[current_data.index.date == current_date]
            else:
                data = current_data.tail(1)

            if len(data) == 0:
                return None

            # Get OHLC
            cols = {c.lower(): c for c in data.columns}
            high = data[cols.get('high', 'High')].iloc[0]
            low = data[cols.get('low', 'Low')].iloc[0]

            stop_price = position['stop_price']
            target_price = position['target_price']

            # Check stop and target
            if low <= stop_price:
                return 'stop_loss'
            if high >= target_price:
                return 'take_profit'

            # Time-based exit (5 days max for leveraged ETFs)
            # Leveraged ETFs suffer from volatility decay over time
            entry_date = position.get('entry_date')
            if entry_date and current_date:
                holding_days = (current_date - entry_date).days
                if holding_days >= 5:
                    return 'time_stop_leverage_decay'

        except Exception:
            pass

        return None

    def calculate_position_size(
        self,
        signal: StrategySignal,
        allocated_capital: float
    ) -> int:
        """
        Calculate position size for leveraged ETF

        Uses 1/3 of normal size due to 3x leverage.
        """
        # Base calculation
        base_size = super().calculate_position_size(signal, allocated_capital)

        # Reduce by leverage factor
        leverage = signal.additional_data.get('leverage', 3)
        adjusted_size = int(base_size / leverage)

        return max(adjusted_size, 1) if base_size > 0 else 0


class SectorRotationStrategy(BaseStrategy):
    """
    Sector rotation using sector ETFs

    Identifies strongest and weakest sectors, goes long strongest
    and short weakest for market-neutral exposure.
    """

    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services',
    }

    def __init__(
        self,
        name: str = "Sector_Rotation",
        capital_allocation: float = 0.20,
        max_positions: int = 4,  # 2 long, 2 short
        risk_per_trade: float = 0.02,
        lookback_days: int = 20,
        top_n: int = 2,  # Number of sectors to long/short
    ):
        super().__init__(name, capital_allocation, max_positions, risk_per_trade)
        self.lookback_days = lookback_days
        self.top_n = top_n

        from shared.indicators.rrs import RRSCalculator
        self.rrs_calc = RRSCalculator()

    def calculate_sector_strength(
        self,
        stock_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        current_date: Optional[date] = None
    ) -> Dict[str, float]:
        """Calculate RRS for each sector ETF"""
        sector_rrs = {}

        for symbol in self.SECTOR_ETFS.keys():
            if symbol not in stock_data:
                continue

            sector_data = stock_data[symbol]
            if current_date:
                sector_data = sector_data[sector_data.index.date <= current_date]
                spy_filtered = spy_data[spy_data.index.date <= current_date]
            else:
                spy_filtered = spy_data

            if len(sector_data) < 20 or len(spy_filtered) < 20:
                continue

            try:
                # Normalize
                sector = sector_data.copy()
                sector.columns = [c.lower() for c in sector.columns]
                spy = spy_filtered.copy()
                spy.columns = [c.lower() for c in spy.columns]

                # Calculate RRS
                atr = self.rrs_calc.calculate_atr(sector).iloc[-1]
                sector_close = sector['close'].iloc[-1]
                sector_prev = sector['close'].iloc[-2]
                spy_close = spy['close'].iloc[-1]
                spy_prev = spy['close'].iloc[-2]

                result = self.rrs_calc.calculate_rrs_current(
                    stock_data={'current_price': sector_close, 'previous_close': sector_prev},
                    spy_data={'current_price': spy_close, 'previous_close': spy_prev},
                    stock_atr=atr
                )

                sector_rrs[symbol] = result['rrs']

            except Exception:
                continue

        return sector_rrs

    def scan(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        current_date: Optional[date] = None
    ) -> List[StrategySignal]:
        """Scan for sector rotation opportunities"""
        signals = []

        # Calculate all sector strengths
        sector_rrs = self.calculate_sector_strength(stock_data, market_data, current_date)

        if len(sector_rrs) < 5:  # Need minimum sectors
            return signals

        # Sort by RRS
        sorted_sectors = sorted(sector_rrs.items(), key=lambda x: x[1], reverse=True)

        # Get strongest and weakest
        strongest = sorted_sectors[:self.top_n]
        weakest = sorted_sectors[-self.top_n:]

        # Generate long signals for strongest
        for symbol, rrs in strongest:
            if symbol in self.positions or rrs < 0.5:
                continue

            signal = self._create_signal(
                symbol, stock_data, rrs, SignalDirection.LONG, current_date
            )
            if signal:
                signals.append(signal)

        # Generate short signals for weakest
        for symbol, rrs in weakest:
            if symbol in self.positions or rrs > -0.5:
                continue

            signal = self._create_signal(
                symbol, stock_data, rrs, SignalDirection.SHORT, current_date
            )
            if signal:
                signals.append(signal)

        return signals

    def _create_signal(
        self,
        symbol: str,
        stock_data: Dict[str, pd.DataFrame],
        rrs: float,
        direction: SignalDirection,
        current_date: Optional[date] = None
    ) -> Optional[StrategySignal]:
        """Create signal for sector ETF"""
        try:
            data = stock_data[symbol]
            if current_date:
                data = data[data.index.date <= current_date]

            data = data.copy()
            data.columns = [c.lower() for c in data.columns]

            atr = self.rrs_calc.calculate_atr(data).iloc[-1]
            close = data['close'].iloc[-1]

            if direction == SignalDirection.LONG:
                stop_price = close - (atr * 1.0)
                target_price = close + (atr * 2.0)
            else:
                stop_price = close + (atr * 1.0)
                target_price = close - (atr * 2.0)

            strength = SignalStrength.STRONG if abs(rrs) > 1.5 else SignalStrength.MODERATE

            return StrategySignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                strategy_name=self.name,
                entry_price=close,
                stop_price=stop_price,
                target_price=target_price,
                atr=atr,
                risk_per_share=abs(close - stop_price),
                rrs_value=rrs,
                suggested_position_pct=self.risk_per_trade,
                additional_data={'sector': self.SECTOR_ETFS.get(symbol)}
            )

        except Exception:
            return None

    def should_exit(
        self,
        symbol: str,
        position: Dict,
        current_data: pd.DataFrame,
        current_date: Optional[date] = None
    ) -> Optional[str]:
        """Check exit conditions for sector position"""
        try:
            if current_date:
                data = current_data[current_data.index.date == current_date]
            else:
                data = current_data.tail(1)

            if len(data) == 0:
                return None

            cols = {c.lower(): c for c in data.columns}
            high = data[cols.get('high', 'High')].iloc[0]
            low = data[cols.get('low', 'Low')].iloc[0]

            direction = position.get('direction', 'long')
            stop_price = position['stop_price']
            target_price = position['target_price']

            if direction == 'long':
                if low <= stop_price:
                    return 'stop_loss'
                if high >= target_price:
                    return 'take_profit'
            else:
                if high >= stop_price:
                    return 'stop_loss'
                if low <= target_price:
                    return 'take_profit'

            # Weekly rebalance for sector rotation
            entry_date = position.get('entry_date')
            if entry_date and current_date:
                holding_days = (current_date - entry_date).days
                if holding_days >= 7:
                    return 'weekly_rebalance'

        except Exception:
            pass

        return None
