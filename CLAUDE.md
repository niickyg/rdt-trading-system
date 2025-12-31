# RDT Trading System - Claude Code Context

## Project Overview

This is an autonomous trading system based on the r/RealDayTrading methodology. It implements Real Relative Strength (RRS) scanning and multi-agent autonomous trading.

## Tech Stack

- **Python 3.11** - Core runtime
- **SQLAlchemy 2.0** - Database ORM with async support
- **Pydantic** - Settings and validation
- **asyncio** - Async agent framework
- **yfinance** - Market data
- **loguru** - Logging
- **Docker** - Containerized deployment
- **Schwab API** - Live trading (optional)

## Project Structure

```
rdt-trading-system/
├── agents/                 # Multi-agent trading system
│   ├── base.py            # BaseAgent and ScheduledAgent classes
│   ├── events.py          # Event bus for agent communication
│   ├── scanner_agent.py   # RRS signal scanner
│   ├── analyzer_agent.py  # Trade setup analyzer
│   ├── executor_agent.py  # Order execution
│   ├── orchestrator.py    # System coordinator
│   └── research_agent.py  # RDT methodology research & optimization
├── brokers/               # Broker abstraction layer
│   ├── base.py           # AbstractBroker interface
│   ├── paper/            # Paper trading simulator
│   └── schwab/           # Schwab API integration
├── config/               # Configuration management
│   └── settings.py       # Pydantic settings
├── data/                 # Data layer
│   └── database/         # SQLAlchemy models
├── portfolio/            # Position management
│   └── position_manager.py
├── risk/                 # Risk management
│   ├── models.py         # Risk data models
│   ├── position_sizer.py # ATR-based sizing
│   └── risk_manager.py   # Risk engine
├── scanner/              # Original scanner code
├── shared/               # Shared utilities
│   ├── indicators/
│   │   └── rrs.py        # RRS calculator + daily chart checks
│   └── data_provider.py  # Market data abstraction
├── alerts/               # Notification system
├── backtesting/          # Backtesting framework
│   ├── engine.py         # Backtest engine
│   └── data_loader.py    # Historical data loader
├── monitoring/           # Dashboard and monitoring
├── main.py               # Main entry point
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-container setup
└── tests/                # Test suite
```

## Key Concepts

### Real Relative Strength (RRS)
Formula: `RRS = (Stock % Change - SPY % Change) / ATR`
- RRS > 2.0 = Strong RS → Long candidate
- RRS < -2.0 = Strong RW → Short candidate

### Validated Strategy Parameters (via 365-day backtest)

These parameters produced +2.85% return with 1.23 profit factor:

| Parameter | Value | Notes |
|-----------|-------|-------|
| RRS Threshold | 2.0 | Selective entries |
| Daily Chart | Relaxed | 3 of 5 conditions (not all) |
| Stop Loss | 0.75x ATR | Tight stops - cut losers fast |
| Take Profit | 1.5x ATR | Take profits early |
| Win Rate | ~34% | Low but compensated by R:R |
| Avg Win:Loss | 2.4:1 | Key to profitability |

### Daily Chart Criteria (Relaxed)

The `check_daily_strength_relaxed()` function uses a score-based approach:
- 3 EMA > 8 EMA (+1)
- 8 EMA > 21 EMA (+1)
- Close above 8 EMA (+1)
- Higher lows in last 5 days (+1)
- 2+ green days in last 3 (+1)

**Strong = 3+ points** (more lenient than requiring ALL conditions)

### Risk Management
- 1% max risk per trade
- 2% max daily loss
- Tight stops (0.75x ATR) - cut losers quickly
- Early profit taking (1.5x ATR target)
- Max 5 concurrent positions

### Agent Architecture
- **Orchestrator**: Coordinates all agents, handles lifecycle
- **ScannerAgent**: Scans watchlist for RRS signals
- **AnalyzerAgent**: Validates setups, calculates sizing
- **ExecutorAgent**: Executes trades through broker
- **ResearchAgent**: Compiles RDT methodology, optimizes parameters
- All agents communicate via EventBus (pub/sub)

## Common Commands

```bash
# Run with Docker (recommended)
docker-compose run --rm scanner     # Semi-automated scanner
docker-compose run --rm backtest    # 365-day backtest
docker-compose run --rm dashboard   # Monitoring dashboard

# Run locally
python main.py scanner              # Semi-automated scanner
python main.py bot                  # Bot (manual execution)
python main.py bot --auto           # Bot (auto execution)
python main.py backtest             # Backtest with optimized params
python main.py backtest --strict    # Backtest with original params
python main.py backtest --days 180  # Custom backtest period
python main.py backtest --enhanced  # Enhanced with trailing stops & scaled exits
python main.py optimize             # Parameter optimization (NEW)
python main.py optimize --days 730  # 2-year optimization
python main.py dashboard            # Monitoring dashboard

# Watchlist options (NEW)
python main.py scanner --watchlist core        # Top 50 liquid stocks
python main.py scanner --watchlist full        # 150+ stocks (default)
python main.py scanner --watchlist technology  # Tech sector only
python main.py scanner --watchlist aggressive  # High volatility

# Install dependencies
pip install -r requirements.txt
```

## Docker Commands

```bash
# Build all images
docker-compose build

# Run specific profiles
docker-compose --profile scanner up    # Scanner mode
docker-compose --profile production up # Full production stack

# Run one-off commands
docker-compose run --rm backtest
docker-compose run --rm dashboard
```

## Environment Variables

Key settings from `.env`:
```bash
PAPER_TRADING=true              # Use paper trading
AUTO_TRADE=false                # Manual execution mode
ACCOUNT_SIZE=25000
MAX_RISK_PER_TRADE=0.01         # 1%
MAX_DAILY_LOSS=0.02             # 2%
RRS_STRONG_THRESHOLD=2.0        # Entry threshold
```

## Code Conventions

- Use `loguru` for all logging
- Type hints on all function signatures
- Docstrings with Args/Returns sections
- Async functions for I/O-bound operations
- Events for inter-agent communication
- Risk checks before any trade execution
- Column names normalized to lowercase for indicator functions

## Backtest Results Summary

365-day backtest on $25,000 (2024-12-27 to 2025-12-26):
- **Total Return**: +$712 (+2.85%)
- **Trades**: 83
- **Win Rate**: 33.7%
- **Profit Factor**: 1.23
- **Max Drawdown**: 2.1%
- **Avg Holding Days**: 4.2

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## Important Notes

- **NEVER** enable `AUTO_TRADE=true` without extensive paper trading
- Risk management checks cannot be bypassed
- All trades must pass PositionSizer and RiskManager validation
- Daily loss limits are strictly enforced
- Use `--strict` flag for original (non-optimized) strategy in backtests
- Docker volumes use `:z` flag for SELinux compatibility on Fedora

## Files Modified for Optimization

1. `shared/indicators/rrs.py` - Added `check_daily_strength_relaxed()` and `check_daily_weakness_relaxed()`
2. `backtesting/engine.py` - Added `use_relaxed_criteria`, `stop_atr_multiplier`, `target_atr_multiplier` params
3. `agents/research_agent.py` - Contains validated strategy parameters from RDT methodology
4. `main.py` - Added `--strict` flag and optimized parameters for backtest mode

## Wealth Optimization Features (NEW)

### Parameter Optimization Engine
**File:** `backtesting/parameter_optimizer.py`
- Grid search across 100+ parameter combinations
- Composite scoring (return, profit factor, Sharpe, drawdown)
- Walk-forward validation to prevent overfitting
- Automated optimal parameter recommendation

### Enhanced Backtest Engine
**File:** `backtesting/engine_enhanced.py`
- Trailing stops (move to breakeven after 1R profit)
- Scaled exits (take 50% at 1R, trail the rest)
- Time stops (close stale positions after 5 days)
- MFE/MAE tracking for exit optimization

### Expanded Watchlists
**File:** `config/watchlists.py`
- `core`: 50 most liquid S&P 500 stocks
- `full`: 150+ diversified stocks
- `technology`, `financials`, `healthcare`, `energy`: Sector-specific
- `aggressive`: High volatility momentum stocks
- `etfs`: Sector and market ETFs

### Wealth Strategy Document
**File:** `WEALTH_OPTIMIZATION.md`
- Complete revenue projection model
- Implementation roadmap
- Risk management guidelines
- Revenue stream opportunities
