# RDT Trading System

Autonomous trading bot implementing the r/RealDayTrading Real Relative Strength methodology. Scans stocks for institutional money flow and executes trades based on predefined criteria.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your settings:
```bash
PAPER_TRADING=true
ACCOUNT_SIZE=25000
MAX_RISK_PER_TRADE=0.01
RRS_STRONG_THRESHOLD=2.0
SCHWAB_APP_KEY=your_key
SCHWAB_APP_SECRET=your_secret
```

## Usage

**Scanner mode** (alerts only):
```bash
python scanner/realtime_scanner.py
```

**Automated trading** (executes trades):
```bash
python automation/trading_bot.py
```

## Strategy

Uses Real Relative Strength (RRS) to identify stocks with significant institutional flow relative to SPY.

**Formula:** `RRS = (Stock % Change - SPY % Change) / ATR`

**Entry criteria:**
- RRS > 2.0 for longs (or < -2.0 for shorts)
- Daily chart strength (3/8 EMA alignment)
- 5-minute compression break or VWAP test

**Risk management:**
- 1% risk per trade
- Stop loss at 1.5x ATR
- Take profit at 3x ATR (2:1 R/R)
- 3% max daily loss

## Structure

```
scanner/              # Scans and sends alerts
automation/           # Auto-trading bot
shared/indicators/    # RRS calculation
alerts/               # Desktop/SMS notifications
data/                 # Logs and trade history
```

## Schwab API

Required for automated trading:

1. Create developer account at https://developer.schwab.com/
2. Register app with callback URL `https://localhost:8080`
3. Add credentials to `.env`
4. Enable thinkorswim on your Schwab account
5. Run OAuth flow on first startup

## Configuration

**RRS threshold:**
```bash
RRS_STRONG_THRESHOLD=2.0  # Default
```

**Scan frequency:**
```bash
SCAN_INTERVAL_SECONDS=300  # 5 minutes
```

**Alerts:**
```bash
ALERT_METHOD=desktop  # Options: desktop, twilio, email
```

## Risk Warnings

- Start with paper trading
- 95%+ of day traders lose money
- Test extensively before going live
- $25k minimum to avoid PDT
- No guarantees of profit

Based on methodology from r/RealDayTrading (HariSeldon).

**Disclaimer:** Educational purposes only. Trading involves substantial risk. Use at your own risk.
