#!/usr/bin/env python3
"""
Backfill ML Training Data

Populates ML training tables with historical data using yfinance.
Can backfill specific tables or all tables at once.

Usage:
    python scripts/backfill_ml_data.py --days 60 --tables all
    python scripts/backfill_ml_data.py --days 30 --tables intraday_bars,technical_indicators
    python scripts/backfill_ml_data.py --days 5 --tables market_regime_daily,sector_data
    python scripts/backfill_ml_data.py --tables trade_snapshots  # retroactive MFE/MAE
    python scripts/backfill_ml_data.py --days 90 --tables earnings
"""

import argparse
import sys
import os
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loguru import logger


VALID_TABLES = [
    'intraday_bars',
    'technical_indicators',
    'market_regime_daily',
    'sector_data',
    'trade_snapshots',
    'earnings',
    'all',
]

SECTOR_ETFS = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Energy': 'XLE',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
    'Industrials': 'XLI',
    'Communication Services': 'XLC',
}


def get_watchlist_symbols():
    """Get symbols from the watchlist table."""
    try:
        from data.database import get_db_manager
        from data.database.models import WatchlistItem
        db = get_db_manager()
        with db.get_session() as session:
            items = session.query(WatchlistItem).filter(
                WatchlistItem.active == True
            ).all()
            symbols = [item.symbol for item in items]
            if symbols:
                return symbols
    except Exception as e:
        logger.warning(f"Could not load watchlist from DB: {e}")

    # Fallback default symbols
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'SPY']


def backfill_intraday_bars(symbols, days):
    """Backfill 5-minute intraday bars. yfinance supports max 60 days of 5m data."""
    import yfinance as yf
    from data.database.ml_repository import get_ml_repository

    repo = get_ml_repository()
    days = min(days, 60)  # yfinance limit for 5m data

    logger.info(f"Backfilling intraday bars for {len(symbols)} symbols, {days} days")
    total_saved = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval="5m")
            if df.empty:
                logger.debug(f"No 5m data for {symbol}")
                continue

            bars = []
            for ts, row in df.iterrows():
                bars.append({
                    'symbol': symbol,
                    'timestamp': ts.to_pydatetime(),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': int(row['Volume']),
                    'vwap': None,
                })

            saved = repo.save_intraday_bars(bars)
            total_saved += saved
            logger.info(f"  {symbol}: saved {saved} bars")

        except Exception as e:
            logger.error(f"  {symbol}: failed - {e}")

    logger.info(f"Intraday bars backfill complete: {total_saved} total bars saved")


def backfill_technical_indicators(symbols, days):
    """Backfill daily technical indicators from OHLCV data."""
    import yfinance as yf
    import pandas as pd
    from data.database.ml_repository import get_ml_repository

    repo = get_ml_repository()
    # Need extra history for indicator computation (200-day EMA needs 200+ days)
    fetch_days = days + 250

    logger.info(f"Backfilling technical indicators for {len(symbols)} symbols, {days} days")
    total_saved = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{fetch_days}d", interval="1d")
            if df.empty or len(df) < 50:
                logger.debug(f"Insufficient data for {symbol}")
                continue

            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            # RSI-14
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, float('nan'))
            rsi_14 = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - macd_signal

            # Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            bb_width = (bb_upper - bb_lower) / bb_middle

            # EMAs
            ema_9 = close.ewm(span=9, adjust=False).mean()
            ema_21 = close.ewm(span=21, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()
            ema_200 = close.ewm(span=200, adjust=False).mean()

            # ATR-14
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean()

            # ADX
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            atr_smooth = tr.ewm(span=14, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
            minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float('nan'))
            adx = dx.ewm(span=14, adjust=False).mean()

            # OBV
            obv = (volume * (~close.diff().le(0)).astype(int) - volume * close.diff().lt(0).astype(int)).cumsum()

            # Save only the last N days
            cutoff_date = date.today() - timedelta(days=days)
            saved_for_symbol = 0

            for i in range(len(df)):
                row_date = df.index[i].date() if hasattr(df.index[i], 'date') else df.index[i]
                if row_date < cutoff_date:
                    continue

                def safe_float(series, idx):
                    val = series.iloc[idx]
                    return float(val) if pd.notna(val) else None

                indicators = {
                    'symbol': symbol,
                    'date': row_date,
                    'rsi_14': safe_float(rsi_14, i),
                    'macd_line': safe_float(macd_line, i),
                    'macd_signal': safe_float(macd_signal, i),
                    'macd_histogram': safe_float(macd_histogram, i),
                    'bb_upper': safe_float(bb_upper, i),
                    'bb_middle': safe_float(bb_middle, i),
                    'bb_lower': safe_float(bb_lower, i),
                    'bb_width': safe_float(bb_width, i),
                    'ema_9': safe_float(ema_9, i),
                    'ema_21': safe_float(ema_21, i),
                    'ema_50': safe_float(ema_50, i),
                    'ema_200': safe_float(ema_200, i),
                    'adx': safe_float(adx, i),
                    'obv': int(obv.iloc[i]) if pd.notna(obv.iloc[i]) else None,
                    'atr_14': safe_float(atr_14, i),
                    'close_price': safe_float(close, i),
                }

                result = repo.save_technical_indicators(indicators)
                if result:
                    saved_for_symbol += 1

            total_saved += saved_for_symbol
            logger.info(f"  {symbol}: saved {saved_for_symbol} days of indicators")

        except Exception as e:
            logger.error(f"  {symbol}: failed - {e}")

    logger.info(f"Technical indicators backfill complete: {total_saved} records saved")


def backfill_market_regime(days):
    """Backfill daily market regime data using VIX and SPY historical."""
    import yfinance as yf
    from data.database.ml_repository import get_ml_repository

    repo = get_ml_repository()
    fetch_days = days + 250  # Extra for EMA computation

    logger.info(f"Backfilling market regime for {days} days")

    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period=f"{fetch_days}d")

        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period=f"{fetch_days}d")

        if vix_hist.empty or spy_hist.empty:
            logger.error("Could not fetch VIX or SPY historical data")
            return

        spy_ema_50 = spy_hist['Close'].ewm(span=50, adjust=False).mean()
        spy_ema_200 = spy_hist['Close'].ewm(span=200, adjust=False).mean()

        cutoff_date = date.today() - timedelta(days=days)
        saved = 0

        for i in range(len(spy_hist)):
            row_date = spy_hist.index[i].date() if hasattr(spy_hist.index[i], 'date') else spy_hist.index[i]
            if row_date < cutoff_date:
                continue

            spy_close_val = float(spy_hist['Close'].iloc[i])
            spy_50 = float(spy_ema_50.iloc[i])
            spy_200 = float(spy_ema_200.iloc[i])

            spy_above_50ema = spy_close_val > spy_50
            spy_above_200ema = spy_close_val > spy_200

            if spy_above_200ema and spy_above_50ema:
                spy_trend = "bullish"
            elif not spy_above_200ema and not spy_above_50ema:
                spy_trend = "bearish"
            else:
                spy_trend = "neutral"

            # Find matching VIX date
            vix_close = None
            vix_regime = None
            vix_dates = vix_hist.index
            matching_vix = vix_hist[vix_dates.date == row_date] if hasattr(vix_dates, 'date') else None
            if matching_vix is not None and not matching_vix.empty:
                vix_close = float(matching_vix['Close'].iloc[0])
                if vix_close < 15:
                    vix_regime = "low"
                elif vix_close < 25:
                    vix_regime = "normal"
                elif vix_close < 35:
                    vix_regime = "elevated"
                else:
                    vix_regime = "extreme"

            regime_label = "unknown"
            if vix_regime in ("low", "normal") and spy_trend == "bullish":
                regime_label = "risk_on"
            elif vix_regime in ("elevated", "extreme") and spy_trend == "bearish":
                regime_label = "risk_off"
            elif vix_regime in ("elevated", "extreme"):
                regime_label = "volatile"
            else:
                regime_label = "transitional"

            regime_data = {
                'date': row_date,
                'vix_close': vix_close,
                'vix_regime': vix_regime,
                'spy_close': spy_close_val,
                'spy_trend': spy_trend,
                'spy_above_200ema': spy_above_200ema,
                'spy_above_50ema': spy_above_50ema,
                'regime_label': regime_label,
            }

            result = repo.save_market_regime_daily(regime_data)
            if result:
                saved += 1

        logger.info(f"Market regime backfill complete: {saved} days saved")

    except Exception as e:
        logger.error(f"Market regime backfill failed: {e}")


def backfill_sector_data(days):
    """Backfill daily sector relative strength data."""
    import yfinance as yf
    from data.database.ml_repository import get_ml_repository

    repo = get_ml_repository()
    fetch_days = days + 90  # Extra for RS computation

    logger.info(f"Backfilling sector data for {days} days")

    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period=f"{fetch_days}d")
        if spy_hist.empty:
            logger.error("Could not fetch SPY data for sector RS")
            return

        # Fetch all sector ETFs
        sector_histories = {}
        for sector_name, etf_symbol in SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(etf_symbol)
                hist = ticker.history(period=f"{fetch_days}d")
                if not hist.empty:
                    sector_histories[sector_name] = (etf_symbol, hist)
            except Exception as e:
                logger.debug(f"Failed to fetch {etf_symbol}: {e}")

        cutoff_date = date.today() - timedelta(days=days)
        total_saved = 0

        # Get unique dates from SPY
        for i in range(len(spy_hist)):
            row_date = spy_hist.index[i].date() if hasattr(spy_hist.index[i], 'date') else spy_hist.index[i]
            if row_date < cutoff_date:
                continue

            # SPY returns for relative strength
            spy_ret_5d = None
            spy_ret_20d = None
            spy_ret_60d = None
            if i >= 5:
                spy_ret_5d = float(spy_hist['Close'].iloc[i] / spy_hist['Close'].iloc[i - 5] - 1)
            if i >= 20:
                spy_ret_20d = float(spy_hist['Close'].iloc[i] / spy_hist['Close'].iloc[i - 20] - 1)
            if i >= 60:
                spy_ret_60d = float(spy_hist['Close'].iloc[i] / spy_hist['Close'].iloc[i - 60] - 1)

            day_records = []
            for sector_name, (etf_symbol, hist) in sector_histories.items():
                # Find matching date in sector history
                matching = hist[hist.index.date == row_date] if hasattr(hist.index, 'date') else None
                if matching is None or matching.empty:
                    continue

                # Find the position index in sector history
                sector_idx = hist.index.get_loc(matching.index[0])
                close_price = float(hist['Close'].iloc[sector_idx])

                daily_return_pct = None
                if sector_idx >= 1:
                    daily_return_pct = float((hist['Close'].iloc[sector_idx] / hist['Close'].iloc[sector_idx - 1] - 1) * 100)

                rs_5d = None
                rs_20d = None
                rs_60d = None
                if sector_idx >= 5 and spy_ret_5d is not None:
                    etf_ret = float(hist['Close'].iloc[sector_idx] / hist['Close'].iloc[sector_idx - 5] - 1)
                    rs_5d = float((etf_ret - spy_ret_5d) * 100)
                if sector_idx >= 20 and spy_ret_20d is not None:
                    etf_ret = float(hist['Close'].iloc[sector_idx] / hist['Close'].iloc[sector_idx - 20] - 1)
                    rs_20d = float((etf_ret - spy_ret_20d) * 100)
                if sector_idx >= 60 and spy_ret_60d is not None:
                    etf_ret = float(hist['Close'].iloc[sector_idx] / hist['Close'].iloc[sector_idx - 60] - 1)
                    rs_60d = float((etf_ret - spy_ret_60d) * 100)

                day_records.append({
                    'date': row_date,
                    'sector': sector_name,
                    'etf_symbol': etf_symbol,
                    'close_price': close_price,
                    'daily_return_pct': daily_return_pct,
                    'relative_strength_5d': rs_5d,
                    'relative_strength_20d': rs_20d,
                    'relative_strength_60d': rs_60d,
                })

            # Rank by 20d relative strength
            day_records.sort(key=lambda x: x.get('relative_strength_20d') or -999, reverse=True)
            for rank, record in enumerate(day_records, 1):
                record['sector_rank'] = rank

            if day_records:
                saved = repo.save_sector_data_batch(day_records)
                total_saved += saved

        logger.info(f"Sector data backfill complete: {total_saved} records saved")

    except Exception as e:
        logger.error(f"Sector data backfill failed: {e}")


def backfill_trade_snapshots():
    """Retroactively compute MFE/MAE for existing closed trades."""
    import yfinance as yf
    from data.database import get_trades_repository
    from data.database.ml_repository import get_ml_repository

    trades_repo = get_trades_repository()
    ml_repo = get_ml_repository()

    logger.info("Backfilling MFE/MAE for existing closed trades")

    try:
        closed_trades = trades_repo.get_trades(status='closed')
        if not closed_trades:
            logger.info("No closed trades found")
            return

        updated = 0
        for trade in closed_trades:
            trade_id = trade['id']
            # Skip if already has MFE/MAE
            if trade.get('peak_mfe') is not None:
                continue

            symbol = trade['symbol']
            entry_price = float(trade['entry_price'])
            exit_price = float(trade['exit_price']) if trade.get('exit_price') else None
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            direction = trade.get('direction', 'long')
            stop_loss = float(trade['stop_loss']) if trade.get('stop_loss') else None
            shares = int(trade.get('shares', 1))

            if not entry_time or not exit_time:
                continue

            try:
                # Fetch intraday data for the trade period
                ticker = yf.Ticker(symbol)
                start_str = entry_time.strftime('%Y-%m-%d') if hasattr(entry_time, 'strftime') else str(entry_time)[:10]
                end_date = exit_time + timedelta(days=1) if hasattr(exit_time, 'strftime') else None
                end_str = end_date.strftime('%Y-%m-%d') if end_date else None

                if not end_str:
                    continue

                df = ticker.history(start=start_str, end=end_str, interval="5m")
                if df.empty:
                    # Try daily data as fallback
                    df = ticker.history(start=start_str, end=end_str, interval="1d")

                if df.empty:
                    continue

                # Compute MFE/MAE from price history
                peak_mfe = 0.0
                peak_mae = 0.0
                bars_to_mfe = 0

                for bar_idx, (ts, row) in enumerate(df.iterrows()):
                    price = float(row['High']) if direction == 'long' else float(row['Low'])
                    worst_price = float(row['Low']) if direction == 'long' else float(row['High'])

                    if direction == 'long':
                        favorable = (price - entry_price) * shares
                        adverse = (worst_price - entry_price) * shares
                    else:
                        favorable = (entry_price - price) * shares
                        adverse = (entry_price - worst_price) * shares

                    if favorable > peak_mfe:
                        peak_mfe = favorable
                        bars_to_mfe = bar_idx
                    if adverse < peak_mae:
                        peak_mae = adverse

                peak_mfe_pct = (peak_mfe / (entry_price * shares)) * 100 if entry_price else 0
                peak_mae_pct = (peak_mae / (entry_price * shares)) * 100 if entry_price else 0

                peak_mfe_r = None
                peak_mae_r = None
                if stop_loss:
                    total_risk = abs(entry_price - stop_loss) * shares
                    if total_risk > 0:
                        peak_mfe_r = peak_mfe / total_risk
                        peak_mae_r = peak_mae / total_risk

                mfe_mae_data = {
                    'peak_mfe': peak_mfe,
                    'peak_mae': peak_mae,
                    'peak_mfe_pct': peak_mfe_pct,
                    'peak_mae_pct': peak_mae_pct,
                    'peak_mfe_r': peak_mfe_r,
                    'peak_mae_r': peak_mae_r,
                    'bars_to_mfe': bars_to_mfe,
                    'bars_held': len(df),
                }

                if ml_repo.update_trade_mfe_mae(trade_id, mfe_mae_data):
                    updated += 1
                    logger.info(f"  Trade {trade_id} ({symbol}): MFE=${peak_mfe:.2f}, MAE=${peak_mae:.2f}")

            except Exception as e:
                logger.debug(f"  Trade {trade_id} ({symbol}): failed - {e}")

        logger.info(f"Trade snapshot backfill complete: {updated} trades updated")

    except Exception as e:
        logger.error(f"Trade snapshot backfill failed: {e}")


def backfill_earnings(symbols, days):
    """Backfill earnings calendar from yfinance."""
    import yfinance as yf
    from data.database.ml_repository import get_ml_repository

    repo = get_ml_repository()

    logger.info(f"Backfilling earnings for {len(symbols)} symbols")
    saved = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)

            # Get earnings dates
            earnings = ticker.earnings_dates
            if earnings is None or (hasattr(earnings, 'empty') and earnings.empty):
                continue

            for ts, row in earnings.iterrows():
                try:
                    earnings_date_val = ts.date() if hasattr(ts, 'date') else ts

                    earnings_data = {
                        'symbol': symbol,
                        'earnings_date': earnings_date_val,
                        'eps_estimate': float(row.get('EPS Estimate', 0)) if row.get('EPS Estimate') is not None else None,
                        'eps_actual': float(row.get('Reported EPS', 0)) if row.get('Reported EPS') is not None else None,
                    }

                    # Compute surprise
                    if earnings_data['eps_actual'] is not None and earnings_data['eps_estimate'] is not None:
                        if earnings_data['eps_estimate'] != 0:
                            earnings_data['eps_surprise_pct'] = (
                                (earnings_data['eps_actual'] - earnings_data['eps_estimate'])
                                / abs(earnings_data['eps_estimate'])
                            ) * 100

                    result = repo.save_earnings_event(earnings_data)
                    if result:
                        saved += 1
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"  {symbol}: failed - {e}")

    logger.info(f"Earnings backfill complete: {saved} events saved")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill ML training data tables with historical data"
    )
    parser.add_argument(
        '--days', type=int, default=60,
        help='Number of days to backfill (default: 60)'
    )
    parser.add_argument(
        '--tables', type=str, default='all',
        help='Comma-separated list of tables to backfill (default: all). '
             f'Options: {", ".join(VALID_TABLES)}'
    )
    parser.add_argument(
        '--symbols', type=str, default=None,
        help='Comma-separated list of symbols (default: watchlist from DB)'
    )

    args = parser.parse_args()

    # Parse tables
    tables = [t.strip() for t in args.tables.split(',')]
    for t in tables:
        if t not in VALID_TABLES:
            logger.error(f"Invalid table: {t}. Valid options: {', '.join(VALID_TABLES)}")
            sys.exit(1)

    if 'all' in tables:
        tables = [t for t in VALID_TABLES if t != 'all']

    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = get_watchlist_symbols()

    logger.info(f"Backfill configuration: days={args.days}, tables={tables}, symbols={len(symbols)}")

    # Initialize database
    from data.database import init_database
    init_database()

    # Run backfills
    if 'intraday_bars' in tables:
        backfill_intraday_bars(symbols, args.days)

    if 'technical_indicators' in tables:
        backfill_technical_indicators(symbols, args.days)

    if 'market_regime_daily' in tables:
        backfill_market_regime(args.days)

    if 'sector_data' in tables:
        backfill_sector_data(args.days)

    if 'trade_snapshots' in tables:
        backfill_trade_snapshots()

    if 'earnings' in tables:
        backfill_earnings(symbols, args.days)

    logger.info("Backfill complete!")


if __name__ == '__main__':
    main()
