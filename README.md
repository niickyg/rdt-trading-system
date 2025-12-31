# RDT Trading System
## Real Relative Strength (RRS) Trading Bot based on r/RealDayTrading Methodology

This system implements the r/RealDayTrading strategy with two modes:
1. **Semi-Automated**: Scanner finds setups, sends alerts, you trade manually
2. **Fully Automated**: Bot scans AND executes trades automatically (use with extreme caution)

---

## 🚀 QUICK START

### 1. Install Dependencies

```bash
cd rdt-trading-system
pip install -r requirements.txt
```

### 2. Configure Settings

```bash
cp .env.example .env
nano .env  # Edit with your settings
```

### 3. Run the Scanner (Semi-Automated)

```bash
python scanner/realtime_scanner.py
```

This will:
- Scan stocks every 5 minutes
- Calculate RRS vs SPY
- Send desktop alerts when strong RS/RW found
- YOU execute trades manually

### 4. Run the Trading Bot (Fully Automated)

⚠️ **ONLY after extensive paper trading and testing**

```bash
python automation/trading_bot.py
```

---

## 📋 WHAT YOU NEED

### Required:
- **Python 3.9+**
- **Internet connection**
- **Brokerage account** (Schwab recommended for API access)

### Optional (for full automation):
- **Schwab Developer Account** - https://developer.schwab.com/
- **$25,000 minimum** (to avoid PDT rule)
- **Extensive paper trading experience** (6+ months)

---

## 🎯 THE STRATEGY (RDT Methodology)

### Core Concept: Real Relative Strength (RRS)

**Formula**: `RRS = (Stock % Change - Expected % Change) / ATR`

**What it means**:
- RRS > 2.0 = Strong relative strength (institutional buying) → GO LONG
- RRS < -2.0 = Strong relative weakness (institutional selling) → GO SHORT

### Entry Criteria:
1. **RRS > 2.0** (or < -2.0 for shorts)
2. **Daily chart strong** (3 green days, 3 EMA > 8 EMA)
3. **Entry on 5-minute chart** (compression break, VWAP test)

### Risk Management:
- **1-2% risk per trade**
- **ATR-based stops** (1.5x ATR)
- **2:1 Risk/Reward minimum** (3x ATR target)
- **3% max daily loss**

---

## 📁 PROJECT STRUCTURE

```
rdt-trading-system/
├── scanner/
│   └── realtime_scanner.py      ← SEMI-AUTOMATED: Scans & alerts
├── automation/
│   └── trading_bot.py            ← FULLY AUTOMATED: Trades automatically
├── shared/
│   └── indicators/
│       └── rrs.py                ← RRS calculation engine
├── alerts/
│   └── notifier.py               ← Desktop/SMS/Email alerts
├── config/
│   └── (your config files)
├── data/
│   ├── logs/                     ← Trading logs
│   ├── trades/                   ← Trade history
│   └── historical/               ← Historical data cache
├── .env                          ← YOUR SETTINGS (copy from .env.example)
├── requirements.txt
└── README.md                     ← You are here
```

---

## ⚙️ CONFIGURATION (.env file)

### Essential Settings:

```bash
# Trading Mode
PAPER_TRADING=true         # ALWAYS start with this = true
AUTO_TRADE=false           # Set to true for full automation

# Account Settings
ACCOUNT_SIZE=25000         # Your account size
MAX_RISK_PER_TRADE=0.01    # 1% risk per trade
MAX_DAILY_LOSS=0.03        # 3% max daily loss

# RRS Settings
RRS_STRONG_THRESHOLD=2.0   # RRS above this = trade signal
ATR_PERIOD=14              # ATR calculation period

# Scanner Settings
SCAN_INTERVAL_SECONDS=300  # Scan every 5 minutes

# Alerts
ALERT_METHOD=desktop       # Options: desktop, twilio, email
```

### For Schwab API (Full Automation):

```bash
SCHWAB_APP_KEY=your_app_key
SCHWAB_APP_SECRET=your_app_secret
SCHWAB_CALLBACK_URL=https://localhost:8080
```

**Get these at**: https://developer.schwab.com/

---

## 🔧 USAGE

### Option 1: Semi-Automated Scanner (RECOMMENDED to start)

**Run the scanner:**
```bash
python scanner/realtime_scanner.py
```

**What happens:**
1. Scans your watchlist every 5 minutes
2. Calculates RRS for each stock
3. Finds stocks with RRS > 2.0 (strong RS) or < -2.0 (strong RW)
4. Sends desktop alert with:
   - Symbol & price
   - RRS value
   - Daily chart status
   - Suggested direction (long/short)
5. **YOU** review the alert and execute trade manually in your broker

**Pros:**
- ✅ Learn the strategy by executing manually
- ✅ No risk of bot errors
- ✅ Maintain discretion and judgment
- ✅ Can override signals
- ✅ No Schwab API needed

**Cons:**
- ❌ Requires you to be available during market hours
- ❌ Manual execution (slower entries)

### Option 2: Fully Automated Bot (ADVANCED - BE CAREFUL)

**⚠️ ONLY use after:**
- 6+ months paper trading manually
- 75%+ win rate proven
- Thoroughly tested in paper mode
- Schwab API set up correctly

**Run the bot:**
```bash
# First, test in paper mode
python automation/trading_bot.py
```

**What happens:**
1. Scans watchlist continuously
2. Calculates RRS and daily chart strength
3. **Automatically enters trades** when criteria met
4. Calculates position size based on ATR
5. Sets stop-loss orders (1.5x ATR)
6. Sets take-profit orders (3x ATR = 2:1 R/R)
7. Monitors positions and exits automatically
8. Stops trading if daily loss limit hit

**Pros:**
- ✅ Hands-off trading
- ✅ Consistent execution
- ✅ No emotional decisions
- ✅ Can run 24/7 (for futures/forex)

**Cons:**
- ❌ HIGH RISK if not tested properly
- ❌ Bugs can cost real money
- ❌ Market conditions change (bot may not adapt)
- ❌ Requires Schwab API setup
- ❌ Requires $25k+ to avoid PDT

---

## 🛠 SCHWAB API SETUP (for Full Automation)

### Step 1: Create Developer Account
1. Go to https://developer.schwab.com/
2. Sign up for free developer account
3. Verify email

### Step 2: Create App
1. Login to developer portal
2. Click "My Apps" → "Create App"
3. Fill in details:
   - **App Name**: RDT Trading Bot
   - **Callback URL**: https://localhost:8080
   - **Description**: Automated trading bot
4. Submit and wait for approval (usually instant)

### Step 3: Get Credentials
1. Once approved, you'll receive:
   - **App Key** (API Key)
   - **App Secret**
2. Copy these to your `.env` file

### Step 4: OAuth Authentication
The first time you run the bot, it will:
1. Open a browser window
2. Prompt you to login to Schwab
3. Ask you to authorize the app
4. Redirect to localhost with a code
5. Bot exchanges code for access token
6. Token saved to file for future use

### Step 5: Enable thinkorswim on Account
1. Login to Schwab.com
2. Go to "Trade" → "thinkorswim"
3. Enable thinkorswim on your account
4. API won't work without this

**Documentation**: https://developer.schwab.com/products/trader-api--individual

---

## 📊 HOW THE SCANNER WORKS

### Morning Routine (Before Market Open):
```python
# 1. Load watchlist (S&P 500 high-volume stocks by default)
# 2. Fetch SPY data for benchmark
# 3. Calculate SPY's daily price change
```

### During Market Hours (Every 5 minutes):
```python
for stock in watchlist:
    # 1. Fetch 5-minute and daily data
    # 2. Calculate ATR (14-period)
    # 3. Calculate RRS = (Stock %Δ - SPY %Δ) / ATR
    # 4. Check daily chart (3 EMA, 8 EMA)
    # 5. If RRS > 2.0 AND daily strong → ALERT
    # 6. If RRS < -2.0 AND daily weak → ALERT
```

### Alert Example:
```
🟢 RELATIVE STRENGTH ALERT

AAPL @ $178.50
RRS: 2.45 (STRONG_RS)
Direction: LONG

Stock Change: +1.2%
SPY Change: -0.3%
ATR: $2.15
✅ Daily chart: STRONG (3 green days, EMA bullish)

Time: 10:45:30 AM
```

---

## 🎓 LEARNING RESOURCES

### Official r/RealDayTrading Resources:
- **Reddit Wiki**: r/RealDayTrading/wiki
- **Podcast**: "Trading Lessons From Reddit" (Spotify, Apple)
- **Twitter**: @RealDayTrading

### Free TradingView Indicators:
- RDT's Real Relative Strength
- Daily Volume/RVol/RS Labels
- Volume Weighted RRS

### Timeline:
- **2+ years** to become profitable (official estimate)
- **6-12 months** paper trading minimum
- **75%+ win rate** target before live trading

---

## ⚠️ RISK WARNINGS

### CRITICAL:
- **95%+ of day traders FAIL**
- **You will likely lose money**
- **Start with paper trading**
- **Never risk more than 1-2% per trade**
- **Respect daily loss limits**
- **Bot errors can be expensive**

### Before Going Live:
- [ ] 6+ months paper trading
- [ ] 75%+ win rate for 3+ consecutive months
- [ ] Profit factor > 2.0
- [ ] Understand every line of code
- [ ] Tested extensively in paper mode
- [ ] Have $25k+ to avoid PDT
- [ ] Emotionally prepared for losses

### This is NOT:
- ❌ A get-rich-quick scheme
- ❌ Guaranteed profits
- ❌ Risk-free
- ❌ Suitable for everyone

### This IS:
- ✅ A legitimate methodology
- ✅ Based on institutional money flow
- ✅ Requires years of practice
- ✅ Needs exceptional discipline
- ✅ High risk, potentially high reward

---

## 🔍 CUSTOMIZATION

### Modify Watchlist:
Edit `scanner/realtime_scanner.py`, function `load_watchlist()`:

```python
def load_watchlist(self) -> List[str]:
    # Your custom watchlist
    return ['AAPL', 'TSLA', 'NVDA', 'SPY']
```

### Adjust RRS Threshold:
In `.env`:
```bash
RRS_STRONG_THRESHOLD=2.5  # More conservative (fewer signals)
# or
RRS_STRONG_THRESHOLD=1.5  # More aggressive (more signals)
```

### Change Scan Frequency:
```bash
SCAN_INTERVAL_SECONDS=60   # Every 1 minute (more frequent)
# or
SCAN_INTERVAL_SECONDS=600  # Every 10 minutes (less frequent)
```

---

## 🐛 TROUBLESHOOTING

### "Module not found" errors:
```bash
pip install -r requirements.txt
```

### TA-Lib installation fails:
```bash
# macOS
brew install ta-lib

# Linux
sudo apt-get install ta-lib

# Windows
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

### Yahoo Finance rate limiting:
- Add delays between requests
- Use fewer stocks in watchlist
- Consider paid data provider (Alpaca, Polygon)

### Schwab API not working:
1. Check credentials in `.env`
2. Verify callback URL matches exactly
3. Make sure thinkorswim enabled on account
4. Check token file permissions
5. Re-run OAuth flow

### No desktop alerts showing:
```bash
pip install plyer
# On Linux, you may need:
sudo apt-get install python3-gi
```

---

## 📈 NEXT STEPS

### Week 1: Setup & Learn
1. Install all dependencies
2. Read entire r/RealDayTrading wiki
3. Run scanner in observation mode
4. Watch how RRS changes throughout the day

### Week 2-4: Paper Trade Manually
1. Keep scanner running
2. When alert fires, manually check setup
3. If valid, "enter" trade in journal
4. Track results in spreadsheet
5. Goal: Understand what makes good vs bad setup

### Month 2-6: Refine Edge
1. Continue paper trading
2. Track statistics (win rate, profit factor)
3. Adjust thresholds based on results
4. Develop feel for market conditions

### Month 7-12: Small Live Trades
1. If paper results good (75%+ win rate, 2+ PF)
2. Start with TINY position sizes
3. Focus on execution, not profits
4. Scale up slowly

### Year 2+: Consider Automation
1. If consistently profitable manually
2. Codify your discretionary rules
3. Test bot in paper mode extensively
4. Slowly transition to automation

---

## 📝 TODO BEFORE LIVE TRADING

- [ ] Read entire r/RealDayTrading wiki
- [ ] Set up TradingView with RDT indicators
- [ ] Paper trade for 6+ months
- [ ] Achieve 75%+ win rate for 3+ months
- [ ] Test scanner with different thresholds
- [ ] Create trade journal and analyze results
- [ ] Fund account with $25k+
- [ ] Get Schwab API credentials (if automating)
- [ ] Test bot in paper mode for 100+ trades
- [ ] Verify all risk management working correctly
- [ ] Make peace with potential losses

---

## 📞 SUPPORT & COMMUNITY

- **Reddit**: r/RealDayTrading (116k+ members)
- **Twitter**: @RealDayTrading
- **Podcast**: "Trading Lessons From Reddit"

**For code issues**: Create GitHub issue (if this becomes a repo)

---

## 📄 LICENSE

MIT License - Use at your own risk. No warranty provided.

## ⚖️ DISCLAIMER

This software is for educational purposes only. Trading stocks, options, and futures involves substantial risk of loss and is not suitable for every investor. Past performance is not indicative of future results. The creators of this software are not responsible for any financial losses incurred through its use. Always consult with a licensed financial advisor before making investment decisions.

**BY USING THIS SOFTWARE, YOU ACKNOWLEDGE THAT YOU UNDERSTAND THE RISKS AND ACCEPT FULL RESPONSIBILITY FOR ANY TRADES EXECUTED.**

---

## 🙏 CREDITS

Based on the r/RealDayTrading methodology created by Vincent Bruzzese (HariSeldon).

Strategy documentation available at: reddit.com/r/RealDayTrading/wiki

---

**Good luck, trade safe, and remember: The market doesn't care about your feelings. Discipline wins.**
