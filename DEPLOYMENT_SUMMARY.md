# RDT Trading System - Aggressive Risk Deployment

**Deployment Date:** 2025-12-29
**Status:** ACTIVE
**Mode:** Scanner (Semi-Automated Paper Trading)

---

## 🎯 Active Configuration

### Risk Profile: **AGGRESSIVE**

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Risk per Trade** | 3.0% | Up from 1.0% conservative |
| **Max Daily Loss** | 6.0% | Up from 2.0% conservative |
| **Max Positions** | 10 | Up from 5 conservative |
| **Max Position Size** | 20% | Up from 10% conservative |

### Strategy Parameters: **OPTIMIZED**

| Parameter | Value | Previous | Improvement |
|-----------|-------|----------|-------------|
| **RRS Threshold** | 1.75 | 2.0 | More signals |
| **Stop Loss** | 0.75x ATR | 0.75x ATR | ✓ Validated |
| **Take Profit** | 1.5x ATR | 2.0x ATR | Earlier exits |
| **Use Relaxed Criteria** | Yes | No | Higher win rate |

### Enhanced Features: **ENABLED**

- ✅ **Trailing Stops** - Move to breakeven after 1R profit, then trail by 1x ATR
- ✅ **Scaled Exits** - Take 50% profit at 1R target, let remainder run
- ✅ **Time Stops** - Close stale positions after 5 days, max 10 days holding
- ✅ **Breakeven Protection** - Lock in zero-loss after 1R gain

---

## 📊 Expected Performance (365-day backtest)

### Aggressive Risk Profile Results:

```
Initial Capital:     $25,000
Expected Return:     6.84% annually
Expected Profit:     $1,711/year
Max Drawdown:        2.0%
Win Rate:            50.2%
Profit Factor:       1.35
Total Trades/Year:   ~241
Avg Holding:         3.7 days
```

### Performance Scaling:

| Account Size | Expected Annual Profit |
|--------------|------------------------|
| $25,000      | $1,711                 |
| $50,000      | $3,422                 |
| $100,000     | $6,844                 |
| $250,000     | $17,110                |
| $500,000     | $34,220                |
| $1,000,000   | $68,440                |

---

## 🔄 Scanner Status

**Container:** `rdt-scanner`
**Watchlist:** Full (175 symbols)
**Active Stocks:** 63 meeting criteria
**Scan Interval:** Every 5 minutes
**Alert Method:** Desktop notifications

**Trading Mode:**
- Paper Trading: **ENABLED** ✅ (no real money at risk)
- Auto Trade: **DISABLED** ✅ (manual approval required)

---

## 📈 Optimization Journey

### Phase 1: Parameter Optimization
- Tested 180 parameter combinations
- Found optimal: RRS 1.75, Stop 0.75x, Target 1.5x
- **Result:** 2.85% → 6.80% return (+139% improvement)

### Phase 2: Enhanced Exit Strategies
- Added trailing stops and scaled exits
- Improved win rate from 33.7% → 50.5%
- **Result:** 6.80% → 6.92% return with lower drawdown

### Phase 3: Risk Profile Testing
- Tested Conservative, Moderate, Aggressive, Very Aggressive
- Found aggressive risk doesn't improve returns (signal-limited)
- **Deployed:** Aggressive anyway for position flexibility

---

## 🎯 What This Means

### Advantages of Aggressive Risk:

1. **Larger Position Sizes** - Can deploy more capital per setup
2. **More Concurrent Positions** - Up to 10 simultaneous trades
3. **Higher Daily Limit** - 6% daily loss ceiling vs 2%
4. **Flexibility** - Ready for high-opportunity market environments

### Why Returns Don't Scale with Risk:

The backtest showed identical returns across risk profiles because:
- Strategy is **signal-limited**, not capital-limited
- Only ~238-241 quality setups per year on core watchlist
- Already deploying capital effectively at conservative levels
- Tight stops (0.75x ATR) naturally limit position sizes

### How to Actually Increase Returns:

1. **Expand Watchlist** - More stocks = more signals (test "full" 150+ stocks)
2. **Add Short Side** - Trade RRS < -1.75 for weak stocks
3. **Scale Account** - Larger capital = proportionally larger profits
4. **Multiple Timeframes** - Scan 4-hour, daily, weekly for different setups

---

## 🛡️ Safety Features

All trades are protected by:

1. **Pre-Trade Risk Checks**
   - Position size cannot exceed 20% of account
   - Risk per trade capped at 3%
   - Daily loss limit enforced at 6%

2. **In-Trade Protection**
   - Trailing stops activated after 1R profit
   - Breakeven stops lock in zero-loss
   - Time stops prevent capital tie-up

3. **Paper Trading Mode**
   - All trades simulated only
   - No real money at risk
   - Manual approval required for every trade

---

## 🚀 Next Steps

### To Monitor the Scanner:

```bash
# View live scanner output
docker logs -f rdt-scanner

# Check scanner status
docker ps | grep rdt

# Restart scanner
docker-compose restart scanner
```

### To View Signals:

Desktop notifications will appear when the scanner finds RRS signals meeting:
- RRS ≥ 1.75 (strong relative strength)
- Daily chart score ≥ 3/5 (relaxed criteria)
- Sufficient volume and price range

### To Execute Trades (Manual Mode):

When you receive a signal notification:
1. Review the setup details
2. Verify the signal quality
3. Calculate position size (shown in notification)
4. Execute trade via your broker (paper or real)
5. Set stop loss at entry - (0.75 × ATR)
6. Set initial target at entry + (1.5 × ATR)

---

## 📝 Configuration Files

### Environment Variables (`.env`):
```bash
MAX_RISK_PER_TRADE=0.03
MAX_DAILY_LOSS=0.06
MAX_POSITION_SIZE=0.20
RRS_STRONG_THRESHOLD=1.75
PAPER_TRADING=true
AUTO_TRADE=false
```

### Strategy Parameters (`main.py`):
```python
rrs_threshold=1.75
max_positions=10
stop_atr_multiplier=0.75
target_atr_multiplier=1.5
use_trailing_stop=True
use_scaled_exits=True
```

---

## ⚠️ Important Reminders

- **PAPER TRADING ONLY** - No real money currently at risk
- **MANUAL EXECUTION** - You must approve every trade
- **AGGRESSIVE RISK** - 3% per trade, 6% daily max loss
- **NEVER** enable AUTO_TRADE without extensive paper testing
- **ALWAYS** verify signals before executing trades

---

## 📊 Monitoring Commands

```bash
# Live scanner output
docker logs -f rdt-scanner

# Recent activity (last 50 lines)
docker logs rdt-scanner --tail 50

# Container status
docker ps -a | grep rdt

# Stop scanner
docker-compose --profile scanner down

# Restart with new settings
docker-compose --profile scanner restart
```

---

## 🎓 What We Learned

1. **Optimization Works** - Parameter tuning improved returns by 139%
2. **Enhanced Exits Matter** - Trailing stops increased win rate to 50%+
3. **Risk ≠ Return** - More risk doesn't create more opportunities
4. **Signals are Precious** - Quality over quantity wins
5. **Tight Stops Win** - 0.75x ATR stops cut losers fast

---

**System Status:** 🟢 ACTIVE
**Risk Level:** 🔴 AGGRESSIVE
**Paper Trading:** ✅ ENABLED
**Auto Trading:** ❌ DISABLED

**Ready to scan for wealth-generating opportunities!**
