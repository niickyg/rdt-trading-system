# 🚀 QUICK START GUIDE

## Get Running in 5 Minutes

### 1. Install Python Packages
```bash
cd rdt-trading-system
pip install -r requirements.txt
```

**If TA-Lib fails:**
- macOS: `brew install ta-lib && pip install ta-lib`
- Linux: `sudo apt-get install ta-lib && pip install ta-lib`
- Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### 2. Create Your Config File
```bash
cp .env.example .env
```

**Minimum config to start:**
```bash
# In .env file:
PAPER_TRADING=true
AUTO_TRADE=false
ACCOUNT_SIZE=25000
ALERT_METHOD=desktop
RRS_STRONG_THRESHOLD=2.0
SCAN_INTERVAL_SECONDS=300
```

### 3. Run the Scanner
```bash
python scanner/realtime_scanner.py
```

**What you'll see:**
```
Scanner initialized with 60 stocks
Starting continuous scanner (interval: 300s)
SPY: $450.25 (+0.25%)
Scan complete: 3 RS stocks, 1 RW stocks

=== RELATIVE STRENGTH (Long Candidates) ===
  NVDA: RRS=2.45, Price=$485.50
  AAPL: RRS=2.12, Price=$178.25
  MSFT: RRS=1.85, Price=$375.80
```

**When alert fires, you'll get desktop notification:**
```
🟢 RELATIVE STRENGTH ALERT
NVDA @ $485.50
RRS: 2.45 (STRONG_RS)
Direction: LONG
```

### 4. Manually Execute Trade (in your broker)
- Open Schwab/TOS
- Look at 5-minute chart for entry
- Enter position with stop 1.5x ATR below entry
- Target 3x ATR above entry (2:1 R/R)

---

## Running Full Automation (After Testing)

### Prerequisites:
- [ ] 6+ months paper trading manually
- [ ] 75%+ win rate proven
- [ ] Schwab developer account created
- [ ] API credentials obtained

### 1. Add Schwab Credentials to .env
```bash
SCHWAB_APP_KEY=your_app_key_here
SCHWAB_APP_SECRET=your_secret_here
SCHWAB_CALLBACK_URL=https://localhost:8080

# Set these:
PAPER_TRADING=false  # ⚠️ REAL MONEY
AUTO_TRADE=true      # ⚠️ AUTO EXECUTE
```

### 2. Run Trading Bot
```bash
python automation/trading_bot.py
```

**First time:** Browser will open for OAuth. Login to Schwab and authorize.

**Bot will:**
- Scan every 5 minutes
- Calculate position sizes automatically
- Enter trades when RRS + daily criteria met
- Set stops and targets
- Monitor and exit positions
- Stop if daily loss limit hit

---

## Test Mode (Recommended)

**Want to see it work without real trades?**

Keep these in `.env`:
```bash
PAPER_TRADING=true   # No real orders
AUTO_TRADE=false     # Just scan and alert
```

Bot will log what it WOULD do without executing.

---

## Customize Watchlist

Edit `scanner/realtime_scanner.py`, line ~45:

```python
def load_watchlist(self) -> List[str]:
    return [
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
        # Add your stocks here
    ]
```

---

## Common Issues

### "No module named 'yfinance'"
```bash
pip install yfinance
```

### "No desktop notifications"
```bash
pip install plyer
```

### "Rate limited by Yahoo Finance"
- Increase `SCAN_INTERVAL_SECONDS` to 600 (10 min)
- Reduce number of stocks in watchlist

---

## What to Do While Running

1. **Watch the console output** - See which stocks have RRS > 2.0
2. **Check alerts** - When desktop notification appears, review chart
3. **Verify setup on TradingView** - Look at 5-min chart for entry
4. **Execute manually in broker** - Place trade with proper risk management
5. **Journal the trade** - Record entry, stop, target, reasoning

---

## Your First Trade Checklist

When you get an alert:

- [ ] Verify RRS > 2.0 (for longs) or < -2.0 (for shorts)
- [ ] Check daily chart (3 green days, EMA bullish)
- [ ] Look at 5-minute chart for clean entry
- [ ] Calculate position size (risk 1% of account)
- [ ] Set stop loss at 1.5x ATR
- [ ] Set target at 3x ATR (2:1 R/R)
- [ ] Enter trade in broker
- [ ] Log trade in journal

---

## Expected Performance

**Don't expect:**
- ❌ 100% win rate
- ❌ Daily profits
- ❌ Get rich quick

**DO expect:**
- ✅ 75%+ win rate (with experience)
- ✅ Some losing days/weeks
- ✅ Slow, steady growth over months/years
- ✅ Emotional challenges
- ✅ Continuous learning

---

## Next Steps

1. **Run scanner for 1 week** - Just observe, don't trade
2. **Paper trade for 1 month** - Execute trades in journal/simulator
3. **Review results** - Are you hitting 75%+ win rate?
4. **Read r/RealDayTrading wiki** - Understand the full methodology
5. **Join community** - Reddit, Twitter, learn from others
6. **Consider going live** - Only after consistent paper results

---

**You're ready!** Run `python scanner/realtime_scanner.py` and start learning.

Remember: This is a 2+ year journey. Patience and discipline win.
