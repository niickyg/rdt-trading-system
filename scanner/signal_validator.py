"""
Signal Quality Validator

Validates signal quality before persistence to catch bad data, halted stocks,
thin liquidity, and other anomalies that could lead to bad trades.

All checks annotate the signal with warning flags — signals are never rejected
outright, so the existing signal flow is preserved.
"""

from typing import Dict, Optional
from loguru import logger


def validate_signal_quality(signal: Dict, stock_data: Optional[Dict] = None) -> Dict:
    """
    Validate signal quality and annotate with warning flags.

    Checks performed:
    - Price gap detection: current price >10% from previous close
    - Volume anomaly: current volume <10% of 20-day average
    - Price sanity: entry_price, stop_price, target_price must all be > 0
    - Spread check: bid-ask spread >2% of price (if bid/ask available)
    - RRS outlier: absolute RRS value >10 suggests a data error

    Args:
        signal: The signal dict being built (must contain at minimum
                'entry_price', 'stop_price'/'stop_loss', 'target_price',
                'rrs', 'symbol').
        stock_data: Optional raw stock data dict from the scanner
                    (may contain '20d_avg_volume', 'volume', 'bid', 'ask',
                    'previous_close').  Pass None if unavailable.

    Returns:
        The same signal dict, annotated with quality flags.  Caller should
        replace its local reference with the returned dict.
    """
    symbol = signal.get('symbol', 'UNKNOWN')
    warnings_fired: list[str] = []

    # ------------------------------------------------------------------
    # 1. Price sanity — reject-equivalent: log as error + flag
    # ------------------------------------------------------------------
    entry_price = signal.get('entry_price', 0)
    # Support both 'stop_price' and 'stop_loss' key naming
    stop_price = signal.get('stop_price') or signal.get('stop_loss', 0)
    target_price = signal.get('target_price') or signal.get('target', 0)

    if not (isinstance(entry_price, (int, float)) and entry_price > 0):
        signal['price_sanity_error'] = True
        logger.error(
            f"[SIGNAL_VALIDATOR] {symbol}: entry_price={entry_price!r} is invalid (<= 0). "
            "Signal flagged with price_sanity_error."
        )
        warnings_fired.append('price_sanity_error')

    if not (isinstance(stop_price, (int, float)) and stop_price > 0):
        signal['price_sanity_error'] = True
        logger.error(
            f"[SIGNAL_VALIDATOR] {symbol}: stop_price={stop_price!r} is invalid (<= 0). "
            "Signal flagged with price_sanity_error."
        )
        if 'price_sanity_error' not in warnings_fired:
            warnings_fired.append('price_sanity_error')

    if not (isinstance(target_price, (int, float)) and target_price > 0):
        signal['price_sanity_error'] = True
        logger.error(
            f"[SIGNAL_VALIDATOR] {symbol}: target_price={target_price!r} is invalid (<= 0). "
            "Signal flagged with price_sanity_error."
        )
        if 'price_sanity_error' not in warnings_fired:
            warnings_fired.append('price_sanity_error')

    # ------------------------------------------------------------------
    # 2. RRS outlier — absolute value > 10 suggests a data error
    # ------------------------------------------------------------------
    rrs = signal.get('rrs')
    if isinstance(rrs, (int, float)) and abs(rrs) > 10:
        signal['rrs_outlier_warning'] = True
        logger.warning(
            f"[SIGNAL_VALIDATOR] {symbol}: RRS={rrs:.4f} is an extreme outlier "
            f"(|RRS| > 10). Possible data error. Signal flagged."
        )
        warnings_fired.append('rrs_outlier_warning')

    # ------------------------------------------------------------------
    # 3. Price gap detection — >10% move from previous close
    # ------------------------------------------------------------------
    # The scanner stores stock_change_pct which is already (price/prev_close - 1)*100
    stock_change_pct = signal.get('stock_change_pct')
    if isinstance(stock_change_pct, (int, float)) and abs(stock_change_pct) > 10:
        signal['gap_warning'] = True
        logger.warning(
            f"[SIGNAL_VALIDATOR] {symbol}: price gap detected — "
            f"stock_change_pct={stock_change_pct:.2f}% (threshold ±10%). "
            "Could be halted stock or earnings gap. Signal flagged."
        )
        warnings_fired.append('gap_warning')

    # ------------------------------------------------------------------
    # 4. Volume anomaly — current volume < 10% of 20-day average
    # ------------------------------------------------------------------
    # Try to get 20-day avg volume from stock_data if provided
    if stock_data is not None:
        avg_vol_20d = None

        # Check if daily OHLCV data is available for computing the avg
        daily_df = stock_data.get('daily')
        if daily_df is not None and not daily_df.empty:
            try:
                vol_col = 'volume' if 'volume' in daily_df.columns else daily_df.columns[4]
                # Use up to 20 most-recent complete days (exclude today's partial bar)
                vol_series = daily_df[vol_col].dropna()
                if len(vol_series) >= 2:
                    avg_vol_20d = vol_series.iloc[:-1].tail(20).mean()
            except Exception:
                avg_vol_20d = None

        # Fall back to a pre-computed key if present
        if avg_vol_20d is None:
            avg_vol_20d = stock_data.get('avg_volume_20d') or stock_data.get('20d_avg_volume')

        current_volume = stock_data.get('volume')

        if (
            avg_vol_20d is not None
            and isinstance(avg_vol_20d, (int, float))
            and avg_vol_20d > 0
            and isinstance(current_volume, (int, float))
            and current_volume < avg_vol_20d * 0.10
        ):
            signal['low_volume_warning'] = True
            logger.warning(
                f"[SIGNAL_VALIDATOR] {symbol}: low volume detected — "
                f"current_volume={current_volume:,.0f} is less than 10% of "
                f"20d_avg_volume={avg_vol_20d:,.0f}. "
                "Signal may be unreliable. Signal flagged."
            )
            warnings_fired.append('low_volume_warning')

    # ------------------------------------------------------------------
    # 5. Bid-ask spread — > 2% of price
    # ------------------------------------------------------------------
    bid = None
    ask = None

    # Check signal dict first (in case caller put them there)
    if 'bid' in signal and 'ask' in signal:
        bid = signal['bid']
        ask = signal['ask']
    elif stock_data is not None:
        bid = stock_data.get('bid')
        ask = stock_data.get('ask')

    if (
        isinstance(bid, (int, float))
        and isinstance(ask, (int, float))
        and bid > 0
        and ask > 0
        and entry_price > 0
    ):
        spread = ask - bid
        spread_pct = (spread / entry_price) * 100
        if spread_pct > 2.0:
            signal['wide_spread_warning'] = True
            logger.warning(
                f"[SIGNAL_VALIDATOR] {symbol}: wide bid-ask spread detected — "
                f"bid={bid:.4f}, ask={ask:.4f}, spread={spread_pct:.2f}% "
                f"(threshold 2%). Signal flagged."
            )
            warnings_fired.append('wide_spread_warning')

    # ------------------------------------------------------------------
    # Attach a summary list of all warnings for easy downstream inspection
    # ------------------------------------------------------------------
    if warnings_fired:
        signal['quality_warnings'] = warnings_fired
    else:
        # Explicitly mark clean signals so consumers can distinguish
        signal['quality_warnings'] = []

    return signal
