"""
Real-Time RRS Scanner
Scans stocks for relative strength/weakness and sends alerts

This is the SEMI-AUTOMATED system - it scans and alerts, you execute manually
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger

from shared.indicators.rrs import RRSCalculator, check_daily_strength, check_daily_weakness
from alerts.notifier import send_alert


class RealTimeScanner:
    """Scan stocks in real-time for RRS signals"""

    def __init__(self, config: Dict):
        """
        Initialize scanner

        Args:
            config: Configuration dictionary with settings
        """
        self.config = config
        self.rrs_calc = RRSCalculator(atr_period=config.get('atr_period', 14))
        self.watchlist = self.load_watchlist()
        self.spy_data = None
        self.last_alerts = {}  # Track last alert time to avoid spam

        logger.info(f"Scanner initialized with {len(self.watchlist)} stocks")

    def load_watchlist(self) -> List[str]:
        """
        Load watchlist of stocks to scan

        Returns:
            List of ticker symbols
        """
        # You can customize this - load from file, API, or hardcode
        # For now, returning S&P 500 high-volume stocks
        watchlist = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'PEP', 'KO', 'AVGO', 'COST', 'PFE', 'TMO', 'WMT', 'MCD', 'CSCO',
            'ABT', 'DHR', 'ACN', 'VZ', 'ADBE', 'NKE', 'TXN', 'CRM', 'NEE',
            'LIN', 'ORCL', 'DIS', 'CMCSA', 'PM', 'WFC', 'BMY', 'RTX', 'HON',
            'UPS', 'QCOM', 'AMGN', 'SBUX', 'IBM', 'AMD', 'INTU', 'CAT', 'GE',
            'BA', 'AMAT', 'GILD', 'MDLZ', 'ADP', 'BKNG', 'LMT', 'MMM', 'ADI'
        ]
        return watchlist

    def fetch_spy_data(self) -> pd.DataFrame:
        """Fetch current SPY data"""
        try:
            spy = yf.Ticker('SPY')

            # Get 5-minute data for intraday
            spy_5m = spy.history(period='1d', interval='5m')

            # Get daily data for trend analysis
            spy_daily = spy.history(period='60d', interval='1d')

            self.spy_data = {
                '5m': spy_5m,
                'daily': spy_daily,
                'current_price': spy_5m['Close'].iloc[-1],
                'previous_close': spy_daily['Close'].iloc[-2]
            }

            return self.spy_data

        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return None

    def fetch_stock_data(self, symbol: str) -> Dict:
        """
        Fetch stock data for RRS calculation

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with stock data or None if error
        """
        try:
            ticker = yf.Ticker(symbol)

            # Get 5-minute data
            data_5m = ticker.history(period='1d', interval='5m')

            # Get daily data
            data_daily = ticker.history(period='60d', interval='1d')

            if data_5m.empty or data_daily.empty:
                return None

            # Normalize column names
            data_5m.columns = data_5m.columns.str.lower()
            data_daily.columns = data_daily.columns.str.lower()

            # Calculate ATR from daily data
            atr_series = self.rrs_calc.calculate_atr(data_daily)
            current_atr = atr_series.iloc[-1]

            return {
                '5m': data_5m,
                'daily': data_daily,
                'current_price': data_5m['close'].iloc[-1],
                'previous_close': data_daily['close'].iloc[-2],
                'atr': current_atr,
                'volume': data_daily['volume'].iloc[-1]
            }

        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None

    def calculate_stock_rrs(self, symbol: str, stock_data: Dict) -> Dict:
        """
        Calculate RRS for a stock

        Args:
            symbol: Stock ticker
            stock_data: Stock data dict

        Returns:
            Dict with RRS analysis
        """
        try:
            # Calculate current RRS
            rrs_data = self.rrs_calc.calculate_rrs_current(
                stock_data={
                    'current_price': stock_data['current_price'],
                    'previous_close': stock_data['previous_close']
                },
                spy_data={
                    'current_price': self.spy_data['current_price'],
                    'previous_close': self.spy_data['previous_close']
                },
                stock_atr=stock_data['atr']
            )

            # Check daily chart strength
            daily_strength = check_daily_strength(stock_data['daily'])
            daily_weakness = check_daily_weakness(stock_data['daily'])

            return {
                'symbol': symbol,
                'rrs': rrs_data['rrs'],
                'status': rrs_data['status'],
                'stock_pc': rrs_data['stock_pc'],
                'spy_pc': rrs_data['spy_pc'],
                'price': stock_data['current_price'],
                'volume': stock_data['volume'],
                'atr': stock_data['atr'],
                'daily_strong': daily_strength['is_strong'],
                'daily_weak': daily_weakness['is_weak'],
                'ema3': daily_strength['ema3'],
                'ema8': daily_strength['ema8']
            }

        except Exception as e:
            logger.error(f"Error calculating RRS for {symbol}: {e}")
            return None

    def should_alert(self, symbol: str, rrs: float) -> bool:
        """
        Check if we should send an alert for this stock

        Args:
            symbol: Stock ticker
            rrs: RRS value

        Returns:
            bool: True if should alert
        """
        # Check if we alerted for this stock recently (within 15 minutes)
        if symbol in self.last_alerts:
            time_since_last = (datetime.now() - self.last_alerts[symbol]).total_seconds()
            if time_since_last < 900:  # 15 minutes
                return False

        # Alert thresholds
        strong_threshold = self.config.get('rrs_strong_threshold', 2.0)
        weak_threshold = self.config.get('rrs_weak_threshold', -2.0)

        if abs(rrs) >= abs(strong_threshold):
            return True

        return False

    def format_alert_message(self, analysis: Dict) -> str:
        """Format alert message"""
        symbol = analysis['symbol']
        rrs = analysis['rrs']
        price = analysis['price']
        status = analysis['status']

        if rrs > 0:
            direction = "LONG"
            signal = "🟢 RELATIVE STRENGTH"
        else:
            direction = "SHORT"
            signal = "🔴 RELATIVE WEAKNESS"

        daily_context = ""
        if analysis['daily_strong']:
            daily_context = "\n✅ Daily chart: STRONG (3 green days, EMA bullish)"
        elif analysis['daily_weak']:
            daily_context = "\n❌ Daily chart: WEAK (3 red days, EMA bearish)"

        message = f"""
{signal} ALERT

{symbol} @ ${price:.2f}
RRS: {rrs:.2f} ({status})
Direction: {direction}

Stock Change: {analysis['stock_pc']:.2f}%
SPY Change: {analysis['spy_pc']:.2f}%
ATR: ${analysis['atr']:.2f}
{daily_context}

Time: {datetime.now().strftime('%I:%M:%S %p')}
        """.strip()

        return message

    def scan_once(self):
        """Run a single scan of the watchlist"""
        logger.info("Starting scan...")

        # Fetch SPY data first
        self.fetch_spy_data()

        if self.spy_data is None:
            logger.error("Failed to fetch SPY data, skipping scan")
            return

        spy_pc = self.spy_data['spy_pc'] = (
            (self.spy_data['current_price'] / self.spy_data['previous_close']) - 1
        ) * 100

        logger.info(f"SPY: ${self.spy_data['current_price']:.2f} ({spy_pc:+.2f}%)")

        # Results storage
        strong_rs = []
        strong_rw = []

        # Scan each stock
        for symbol in self.watchlist:
            try:
                # Fetch stock data
                stock_data = self.fetch_stock_data(symbol)
                if stock_data is None:
                    continue

                # Filter by volume and price
                if stock_data['volume'] < self.config.get('min_volume', 500000):
                    continue
                if stock_data['current_price'] < self.config.get('min_price', 5.0):
                    continue

                # Calculate RRS
                analysis = self.calculate_stock_rrs(symbol, stock_data)
                if analysis is None:
                    continue

                # Check for strong signals
                rrs = analysis['rrs']
                threshold = self.config.get('rrs_strong_threshold', 2.0)

                if rrs > threshold:
                    strong_rs.append(analysis)
                    if self.should_alert(symbol, rrs):
                        message = self.format_alert_message(analysis)
                        send_alert(message, self.config)
                        self.last_alerts[symbol] = datetime.now()

                elif rrs < -threshold:
                    strong_rw.append(analysis)
                    if self.should_alert(symbol, rrs):
                        message = self.format_alert_message(analysis)
                        send_alert(message, self.config)
                        self.last_alerts[symbol] = datetime.now()

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

        # Print summary
        logger.info(f"Scan complete: {len(strong_rs)} RS stocks, {len(strong_rw)} RW stocks")

        if strong_rs:
            logger.info("=== RELATIVE STRENGTH (Long Candidates) ===")
            for stock in sorted(strong_rs, key=lambda x: x['rrs'], reverse=True)[:10]:
                logger.info(f"  {stock['symbol']}: RRS={stock['rrs']:.2f}, Price=${stock['price']:.2f}")

        if strong_rw:
            logger.info("=== RELATIVE WEAKNESS (Short Candidates) ===")
            for stock in sorted(strong_rw, key=lambda x: x['rrs'])[:10]:
                logger.info(f"  {stock['symbol']}: RRS={stock['rrs']:.2f}, Price=${stock['price']:.2f}")

    def run_continuous(self):
        """Run scanner continuously"""
        scan_interval = self.config.get('scan_interval_seconds', 60)

        logger.info(f"Starting continuous scanner (interval: {scan_interval}s)")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                self.scan_once()
                logger.info(f"Waiting {scan_interval} seconds until next scan...")
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user")


if __name__ == "__main__":
    # Load configuration (in practice, load from .env file)
    config = {
        'atr_period': 14,
        'rrs_strong_threshold': 2.0,
        'rrs_weak_threshold': -2.0,
        'scan_interval_seconds': 300,  # 5 minutes
        'min_volume': 500000,
        'min_price': 5.0,
        'max_price': 500.0,
        'alert_method': 'desktop'
    }

    scanner = RealTimeScanner(config)
    scanner.run_continuous()
