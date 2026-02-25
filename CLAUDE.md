# RDT Trading System - Project Guide

## Overview

Autonomous trading system implementing the r/RealDayTrading methodology with Real Relative Strength (RRS) scanning, ML-enhanced signal validation, and multi-broker execution.

**Stack:** Python 3.11, Flask, SQLAlchemy 2.0, PostgreSQL/TimescaleDB, Docker

## Project Structure

```
rdt-trading-system/
├── agents/                 # Agent-based architecture
│   ├── base.py            # BaseAgent, ScheduledAgent classes
│   ├── events.py          # EventBus pub-sub system (30+ event types)
│   ├── scanner_agent.py   # Scans watchlist for RRS signals
│   ├── analyzer_agent.py  # Validates signals with ML
│   ├── executor_agent.py  # Executes trades via broker
│   ├── risk_agent.py      # Real-time risk monitoring
│   ├── learning_agent.py  # Model monitoring & drift detection
│   ├── adaptive_learner.py # Parameter auto-tuning
│   └── orchestrator.py    # Coordinates all agents
│
├── api/
│   ├── v1/
│   │   ├── routes.py      # REST API endpoints (40+)
│   │   └── auth.py        # API key auth, rate limiting
│   └── graphql/
│       ├── views.py       # GraphQL blueprint (auth enforced)
│       └── auth.py        # GraphQL auth decorators
│
├── options/               # Options trading module
│   ├── models.py          # OptionContract, OptionGreeks, OptionsStrategy, etc.
│   ├── config.py          # OptionsConfig (Pydantic, env vars)
│   ├── chain.py           # OptionsChainManager (IBKR chain + Greeks + cache)
│   ├── iv_analyzer.py     # IVAnalyzer (IV rank/percentile/HV)
│   ├── strategy_selector.py # StrategySelector (signal → strategy via IV regime)
│   ├── position_sizer.py  # OptionsPositionSizer (contracts from risk budget)
│   ├── executor.py        # OptionsExecutor (single-leg + combo IBKR orders)
│   ├── exit_manager.py    # OptionsExitManager (profit/stop/time/delta/IV exits)
│   └── risk.py            # OptionsRiskManager (portfolio delta/theta/premium)
│
├── brokers/               # Trading execution
│   ├── broker_interface.py # Abstract interface (+ options methods)
│   ├── paper_broker.py    # Simulated trading
│   ├── schwab/            # Charles Schwab integration
│   ├── ibkr/              # Interactive Brokers integration
│   └── failover_manager.py # Multi-broker failover
│
├── ml/                    # Machine learning
│   ├── ensemble.py        # StackedEnsemble (XGBoost + RF + LSTM)
│   ├── safe_model_loader.py # Secure model loading with SHA-256 verification
│   ├── regime_detector.py # Market regime detection (HMM/heuristic)
│   ├── feature_engineering.py # 70+ technical features
│   ├── model_monitor.py   # Production model tracking
│   ├── drift_detector.py  # Model degradation detection
│   ├── model_version.py   # Model versioning with checksum validation
│   └── optimization/      # Optuna hyperparameter tuning
│
├── risk/                  # Risk management
│   ├── risk_manager.py    # Trade validation, P&L tracking
│   ├── models.py          # RiskLimits, RiskMetrics, etc.
│   └── position_sizer.py  # ATR-based position sizing
│
├── scanner/               # Signal scanning
│   ├── realtime_scanner.py # Main scanner with RDT-aligned filter gates
│   ├── trend_detector.py  # Trend identification (ADX, EMA alignment)
│   ├── timeframe_analyzer.py # Multi-timeframe analysis
│   ├── vix_filter.py      # VIX regime detection (5 levels, position sizing)
│   ├── sector_filter.py   # Sector RS vs SPY, SPY daily trend
│   ├── regime_params.py   # Regime-adaptive RRS thresholds & risk params
│   ├── news_filter.py     # News sentiment (advisory, doesn't block)
│   └── mean_reversion_scanner.py # DISABLED — contradicts RDT momentum philosophy
│
├── shared/
│   ├── data_provider.py   # DataProvider with bounded cache (max 200)
│   └── indicators/
│       └── rrs.py         # RRS calculation with NaN/infinity guards
│
├── trading/               # Trade management
│   ├── position_tracker.py # Real-time position management
│   ├── order_monitor.py   # Order lifecycle tracking (max 1000 completed)
│   ├── execution_tracker.py # Fill quality analysis
│   └── advanced_orders.py # Trailing stops, brackets
│
├── utils/
│   └── paths.py           # get_project_root() — NO hardcoded paths
│
├── data/
│   ├── database/
│   │   ├── models.py      # SQLAlchemy models (20+ tables)
│   │   ├── connection.py  # DatabaseManager with retry (thread-safe)
│   │   └── trades_repository.py # Trade CRUD operations (thread-safe)
│   ├── providers/
│   │   ├── yfinance_provider.py
│   │   └── provider_manager.py # Multi-provider with failover
│   └── signals/
│       └── active_signals.json # Live signals (file-locked with fcntl)
│
├── web/
│   ├── app.py             # Flask application entry point
│   ├── trading_init.py    # Component initialization (validates config)
│   ├── auth.py            # Authentication/sessions (brute force protection)
│   ├── middleware.py       # Security headers, CSP, rate limiting
│   ├── websocket.py       # SocketIO events (authenticated, header-only)
│   ├── session_manager.py # Session tracking
│   ├── static/
│   │   ├── sw.js          # Self-destruct service worker (clears caches)
│   │   └── js/
│   │       ├── pwa.js     # PWA module (unregisters SWs, no longer registers)
│   │       └── websocket.js # WebSocket client (XSS-safe, uses textContent)
│   └── templates/
│       ├── base.html      # Base template (inline SW killer script)
│       ├── dashboard_signals.html # Signals with Live Price + P&L columns
│       └── dashboard_backtest.html # Backtest with SW bypass
│
├── config/
│   ├── settings.py        # Pydantic configuration
│   └── watchlists.py      # Symbol lists
│
├── scripts/
│   ├── create_admin.py    # Create admin user (env vars, no hardcoded creds)
│   ├── create_api_user.py # Create API user
│   ├── init_db.py         # Initialize database
│   ├── generate_training_data.py
│   ├── train_ensemble.py
│   ├── train_from_history.py  # Train ML from historical yfinance data
│   ├── run_walkforward.py     # Walk-forward backtest V1 (1yr, baseline vs filtered)
│   └── run_walkforward_v2.py  # Walk-forward backtest V2 (2yr, 3-way comparison)
│
├── tests/
│   ├── integration/       # End-to-end tests
│   └── unit/              # Unit tests
│
├── migrations/            # Alembic database migrations
├── models/                # Trained ML model files
└── docker-compose.yml     # Container orchestration
```

## Core Concepts

### Real Relative Strength (RRS)
```
RRS = (Stock % Change - SPY % Change) / ATR
```
- RRS > 2.0 = Strong relative strength -> Long candidate
- RRS < -2.0 = Strong relative weakness -> Short candidate
- ATR normalized to percentage to avoid price-level dependence
- NaN/infinity guards: `rrs.replace([np.inf, -np.inf], np.nan).fillna(0)`

### Scanner Filter Gates (RDT "Market First" Methodology)

Signals pass through 4 sequential gates in `save_signals()` before being emitted. All gates are independently toggleable and fail-open (missing data = allow through).

```
Signal → SPY Hard Gate → 50/200 SMA Gate → VWAP Gate → MTF Gate → Output
```

| Gate | Config Key | Default | What It Does |
|------|-----------|---------|-------------|
| **SPY Hard Gate** | `spy_gate_enabled` | True | Blocks longs when SPY bearish (below 50+200 EMA), shorts when bullish |
| **50/200 SMA Gate** | `daily_sma_filter_enabled` | True | Blocks longs below 50 SMA, shorts above 50 SMA (per-stock daily chart) |
| **VWAP Gate** | `vwap_filter_enabled` | True | Longs must be above VWAP, shorts below (intraday) |
| **Lightweight MTF** | `mtf_lightweight_enabled` | True | Resamples 5m→15m/1h, blocks "weak" alignment (≤2/4 timeframes agree) |

Additional filters (applied after gates during signal processing):
- **VIX Regime** (`vix_filter.py`): 5 levels, adjusts position size (0.0x-1.1x) and RRS threshold
- **Sector RS** (`sector_filter.py`): Sector ETF strength vs SPY, boosts/penalizes RRS ±0.25-0.50
- **Regime-Adaptive** (`regime_params.py`): Bull/bear/high-vol regimes adjust thresholds dynamically
- **News Sentiment** (`news_filter.py`): Advisory only, warns but does NOT block (RDT: price action > news)
- **Intermarket** (`intermarket_analyzer.py`): Murphy framework — TLT/UUP/GLD/IWM vs SPY, adjusts RRS threshold ±0.25/+0.50 and position size 0.75x-1.10x by regime (risk_on/neutral/risk_off). Leading indicator layer, does NOT block signals.
- **Mean Reversion** (`mean_reversion_scanner.py`): DISABLED by default — contradicts RDT momentum philosophy

### Walk-Forward Backtest Results (2 Years, $25K Capital)

3-way comparison across 6 quarterly windows (Feb 2024 - Nov 2025):

| Metric | A) Baseline | B) Old Filters | C) RDT Filters |
|--------|------------|----------------|-----------------|
| Total Return | $815 (3.3%) | $1,267 (5.1%) | **$1,716 (6.9%)** |
| Win Rate | 47.7% | 47.9% | **49.5%** |
| Profit Factor | 1.12 | 1.12 | **1.24** |
| Total Trades | 235 | 457 | 279 |
| Worst Day | -$404 | -$378 | **-$257** |
| Annualized | 1.6% | 2.5% | **3.4%** |

RDT filters take fewer but higher-quality trades. 98% of raw signals are filtered out.

### Agent Architecture
Agents communicate via EventBus (pub-sub):
```
Scanner -> SIGNAL_FOUND -> Analyzer -> SETUP_VALID -> Executor -> ORDER_PLACED
                              |
                    RiskAgent monitors all events
```

### Risk Limits (Defaults)
- Max 2% risk per trade
- Max 5% daily loss (calculated from start-of-day balance)
- Max 10% drawdown
- Max 5 concurrent positions
- Min 2:1 risk/reward ratio

## Security Architecture

### Authentication
- **Dashboard**: Flask-Login with session cookies (`rdt_session_token`)
- **API**: `X-API-Key` header ONLY (no query param, no URL leaking)
- **WebSocket**: API key validated in connect handler via headers only
- **GraphQL**: `@graphql_bp.before_request` enforces API key or Flask-Login session
- **Brute force protection**: 5 failed attempts -> 15-minute lockout
- **Password policy**: 12+ chars, uppercase, lowercase, digit, special char

### API Key Security
- Keys are hashed with SHA-256 before storage in database
- Lookup by hash prefix (`api_secret_hash` stores first 8 chars of hash)
- Plaintext key returned only at creation time, never again

### Model Loading Security
- `ml/safe_model_loader.py` wraps all pickle/joblib loads
- SHA-256 checksum verification via `.manifest` files
- `allow_unverified=False` enforced at ALL 10+ call sites
- Checksum mismatch raises `ModelSecurityError` (not just a warning)

### Content Security Policy
- CSP headers set in `web/middleware.py`
- `unsafe-eval` removed from `script-src`
- `unsafe-inline` kept (needed for inline styles/scripts)
- HSTS header added for production

### Common Security Patterns
- Open redirect prevention: `urlparse(next_page)` + `not parsed.netloc`
- CSRF tokens on all forms and non-API POST requests
- Session cookies: `httponly=True`, `samesite='Lax'`, `secure` in production
- Error responses: generic messages to client, full details logged server-side
- File permissions: signal files use `fcntl.flock` for concurrent access

## Dashboard Features

### Signals Page (`/dashboard/signals`)
- Live Price column: fetches quotes every 15 seconds via `/api/v1/quotes`
- P&L column: calculated directionally (long vs short), colored green/red
- CSV export includes live price and P&L data
- Columns: Symbol, Direction, Entry, Stop, Target, RRS, Live Price, P&L, Confidence, Regime, Timeframe, Generated

### Backtest Page (`/dashboard/backtest`)
- POST `/api/v1/backtest` with configurable params (days, RRS threshold, ATR multipliers)
- `ensureNoServiceWorker()` runs before every fetch to prevent SW interception
- 3-minute timeout with AbortController
- Presets: Conservative, Balanced, Aggressive

### Service Worker Status
- **SW is DISABLED** - `sw.js` is a self-destruct that clears all caches and unregisters itself
- `pwa.js` unregisters any existing SWs instead of registering new ones
- `base.html` has inline script that detects active SW controller and force-reloads
- This was done because the old PWA service worker intercepted POST requests causing NetworkError on backtest

## Signal Persistence

Signals are stored in `data/signals/active_signals.json`:
- **Merge logic**: new scan results merge with existing signals
- Existing signals preserve original entry prices when direction unchanged
- Signals with deteriorating RRS (below 50% of threshold) are dropped
- File access uses `fcntl.flock` (shared lock for reads, exclusive for writes)
- Timezone-aware age calculation (Eastern time)

## Trading Logic Details

### Position Sizing (`risk/position_sizer.py`)
- ATR-based with minimum threshold: `max(0.10, entry_price * 0.05 / 100)`
- Zero entry price returns 0 shares immediately
- Shares capped by `max_position_pct / entry_price`

### P&L Calculation (`brokers/paper_broker.py`)
- Separated into realized and unrealized components
- `daily_pnl = realized_pnl_today + unrealized`
- Start-of-day balance tracked separately for daily loss limits

### Short Positions
- `unrealized_pnl` and `cost_basis` use `abs(shares)` for both long/short
- Short avg_cost recalculated on position adds (weighted average)
- BUY_TO_COVER validates position exists and is short before covering

### Column Name Normalization
- All data fetches normalize column names to lowercase immediately
- Prevents case inconsistency between batch and provider data paths
- Applies in `scanner/realtime_scanner.py` after every data fetch

## Initialization Flow

The system initializes components in this order (see `web/trading_init.py`):

1. **Configuration** - Load from environment, validate (account_size > 0, 0 < max_risk < 1, etc.)
2. **Infrastructure** - Database, EventBus, DataProvider
3. **Broker** - Paper/Schwab/IBKR with auto-connect
4. **Risk Management** - RiskManager with limits, start-of-day balance
5. **Trading Tracking** - PositionTracker, OrderMonitor, ExecutionTracker
6. **ML/Analysis** - EnsembleModel, RegimeDetector, FeaturePipeline
7. **Agents** (optional) - All agents via Orchestrator
8. **Monitoring** - AlertManager, Prometheus metrics

## API Endpoints

### Authentication
API calls require `X-API-Key` header OR Flask-Login session (dual auth).

### Core Endpoints
```
GET  /api/v1/signals/current   # List signals (paginated: ?page=1&per_page=50)
GET  /api/v1/quotes?symbols=AAPL,MSFT  # Live quotes (max 20 symbols)
GET  /api/v1/positions         # List positions (authenticated)
POST /api/v1/backtest          # Run backtest (authenticated)
GET  /api/v1/broker/account    # Account info
GET  /api/v1/risk/status       # Risk metrics
GET  /api/v1/ml/regime         # Market regime
POST /api/v1/ml/predict        # Trade prediction
POST /api/v1/risk/validate     # Validate trade
GET  /api/v1/health            # Basic health (public)
GET  /api/v1/health/detailed   # Detailed health (admin only)
GET  /metrics                  # Prometheus metrics (admin only)
```

## Environment Variables

```bash
# Core
PAPER_TRADING=true           # Use paper broker
AUTO_TRADE=false             # Automatic execution
ACCOUNT_SIZE=25000           # Starting capital
SECRET_KEY=                  # Flask secret (REQUIRED, no fallback)
RDT_CREDENTIAL_KEY=          # Credential encryption (REQUIRED, no fallback)

# Risk
MAX_RISK_PER_TRADE=0.015     # 1.5% per trade (Config C)
MAX_DAILY_LOSS=0.05
MAX_OPEN_POSITIONS=8

# Scanner
SCAN_INTERVAL=60             # Seconds between scans
RRS_THRESHOLD=2.0            # Signal threshold (regime-adaptive overrides this)
WATCHLIST=core               # Watchlist name

# Database
DATABASE_URL=postgresql://rdt:rdt_paper_2026@localhost:5432/rdt_trading

# Broker
BROKER_TYPE=ibkr             # paper, schwab, ibkr
IBKR_HOST=127.0.0.1
IBKR_PORT=4002               # 4001=live, 4002=paper
IBKR_CLIENT_ID=1
IBKR_PAPER_TRADING=true
IBKR_READONLY=false

# Stripe
STRIPE_WEBHOOK_SECRET=       # Required for webhook signature verification

# Alerts
ALERTS_ENABLED=false
ALERT_CHANNELS=desktop,discord
```

## Running the System

### Local Development
```bash
# Install dependencies
pip install -r requirements-web.txt

# Run web server (debug mode auto-enabled)
python web/app.py
# Note: werkzeug reloader creates parent+child processes
# Use WERKZEUG_RUN_MAIN env var to detect child vs parent

# Run scanner only
python -m scanner

# Run full trading system
python main.py bot --auto
```

### Testing
```bash
pytest tests/ -v
pytest tests/integration/ -v
pytest tests/ --cov=. --cov-report=html
```

## Known Issues & Gotchas

### Service Worker (CRITICAL for frontend work)
The PWA service worker was DISABLED because it intercepted POST requests, breaking backtest and other features. Three layers of defense:
1. `sw.js` is a self-destruct that clears caches and unregisters itself
2. `pwa.js` unregisters any existing SWs
3. `base.html` inline script detects active controller and reloads

**If you re-enable the SW, ALL POST endpoints will break.** Do not register a new service worker.

### SQLAlchemy Session in Auth Module
`web/auth.py` uses a global `_db_session` which causes stale session errors ("identity map is no longer valid"). The `session.close()` calls in `finally` blocks help but the root issue is the global singleton pattern. If refactoring, use scoped sessions or per-request sessions instead.

### Scanner Performance
- **OLD MTF** (`mtf_enabled`): Very slow (~6 min for 54 symbols) — makes per-symbol per-timeframe API calls. Disabled by default.
- **NEW Lightweight MTF** (`mtf_lightweight_enabled`, default True): Resamples existing 5m data to 15m/1h, reuses daily data. Zero extra API calls. Blocks signals with weak (<3/4 timeframe) alignment. Adds `mtf_alignment`, `mtf_score`, `mtf_details` fields to signals.
- Batch scan without MTF is ~30 seconds
- `get_spy_price()` does a live yfinance call (~12 seconds) - should be cached

### Missing Dependencies
- `hmmlearn` is optional - regime detector falls back to heuristic mode
- GPU dependencies (`tensorflow`, `torch`) are optional - ML works on CPU

### User Model Dual-Auth Compatibility
- Dashboard `User` model (Flask-Login) and `APIUserDTO` (API key auth) have different interfaces
- `User` now has compatibility properties: `user_id` (alias for `id`), `subscription_tier` (derived from subscriptions), `is_expired` (always False)
- Admin users get `ELITE` tier, others get `PRO` by default (or tier from active subscription)
- When adding new attributes to `APIUserDTO`, add matching properties to `User` in `data/database/models.py`

### yfinance Column Format
- Modern yfinance returns `MultiIndex` columns even for single symbols: `('Close', 'AAPL')`
- Always handle MultiIndex: `data[('Close', symbol)]` not `data['Close']`
- Column names may be 'Close' (uppercase) — check both cases

### Debug Mode
- `web/app.py` runs with `debug=True` - uses werkzeug reloader (parent+child processes)
- Signal file paths are relative to CWD - always run from project root

## Recent Audit & Remediation (Feb 2026)

Two comprehensive code audits were performed. Key changes:

### Security Fixes Applied
- Position management endpoints secured with `@require_api_key`
- API keys hashed before storage (SHA-256)
- API key query parameter fallback removed (header only)
- Hardcoded credential fallback keys removed (fail on missing env var)
- WebSocket authentication via headers only
- Stripe webhook signature verification added
- Unsafe pickle/joblib loads wrapped with `safe_model_loader.py`
- Model checksum mismatch now raises error (was just a warning)
- Exception details (`str(e)`) removed from client-facing error responses
- DOM XSS: `innerHTML` replaced with `textContent` for user data
- Open redirect prevention on login redirects
- GraphQL authentication enforced

### Trading Logic Fixes Applied
- RRS ATR normalization to percentage (price-level independent)
- P&L separated into realized + unrealized
- Short position avg_cost recalculated on adds
- BUY_TO_COVER crash on missing position fixed
- NaN/zero validation before ATR division
- Minimum ATR threshold for position sizing
- Daily loss limit uses start-of-day balance
- Column name normalization (lowercase) across all data paths
- Signal merge preserves original entry prices
- Stop/target price validation rejects zero/negative values
- Slippage direction adjusted for buy vs sell

### Infrastructure Fixes Applied
- Thread-safe singleton initialization (4 locations)
- Config validation at startup (account_size, risk params)
- Cache eviction in DataProvider (max 200 entries)
- File locking for signal file access (fcntl.flock)
- Completed orders list bounded (max 1000)
- Hardcoded paths replaced with `utils/paths.py`
- Bare `except:` replaced with `except Exception:`

## Broker: Interactive Brokers (IBKR)

### Setup
- IB Gateway running on port 4002 (paper trading)
- Paper account: DUP995654 ($1M equity)
- `ib_insync` library for API communication
- Python 3.14 compatibility: explicit `asyncio.set_event_loop()` before ib_insync import
- `TrailingStopOrder` doesn't exist in current ib_insync — use `IBOrder(orderType='TRAIL')` instead
- Paper trading requires `reqMarketDataType(3)` for delayed data

### Trade Metadata Columns (16 fields on `Trade` model)
```
vix_regime, vix_value, market_regime, sector_name, sector_rs, spy_trend,
ml_confidence, signal_strategy, news_sentiment, news_warning,
regime_rrs_threshold, regime_stop_multiplier, regime_target_multiplier,
vix_position_size_mult, sector_boost, first_hour_filtered
```

### ML Model Status
- **Exit Predictor**: 43.3% accuracy — SKIP (barely above 33% random)
- **Signal Decay**: MAE 1.1 min — DEPLOY
- **Dynamic Sizer**: Calibrated 150 trades, 55% WR — DEPLOY
- Rule-based filters provide all measurable improvement; ML is advisory-only

## Options Trading Module

### Architecture
```
Scanner → Analyzer → [StrategySelector] → Executor → [OptionsExecutor] → IBKR
```
Options integrates into `ExecutorAgent.execute_trade()` — if OPTIONS_ENABLED, signals are converted to options strategies before order placement. Falls through to stock execution if options path returns None.

### Strategy Selection (IV Regime-Based)
| Direction | IV Rank < 30 | IV Rank 30-50 | IV Rank > 50 | IV Rank > 80 |
|-----------|-------------|---------------|--------------|--------------|
| LONG | Long Call | Bull Call Spread | Bull Put Spread | Iron Condor |
| SHORT | Long Put | Bear Put Spread | Bear Call Spread | Iron Condor |

### Key Config (env vars with OPTIONS_ prefix)
- `OPTIONS_ENABLED=false` — master switch
- `OPTIONS_MODE=stocks` — stocks/options/both
- `OPTIONS_DTE_TARGET=35` — target days to expiry
- `OPTIONS_LONG_DELTA_TARGET=0.60` — delta for long legs
- `OPTIONS_IV_RANK_LOW=30` / `OPTIONS_IV_RANK_HIGH=50` — strategy selection thresholds
- `OPTIONS_PROFIT_TARGET_PCT=0.50` — close at 50% max profit (spreads)
- `OPTIONS_STOP_LOSS_PCT=0.50` — stop at 50% premium loss

### Exit Triggers (priority order)
1. Profit target (50% max profit for spreads, 100% gain for long options)
2. Stop loss (50% premium loss)
3. Time stop (DTE < 14)
4. Delta breach (|delta| > 0.80)
5. IV crush (IV drops > 20% from entry)
6. Roll recommendation (DTE < 21 + profitable)

### Portfolio Risk Limits
- Total premium at risk < 10% of account
- |Net portfolio delta| < 200
- Daily theta < 0.5% of account
- Max 2 options positions per underlying

### IBKR Integration
- `brokers/ibkr/client.py` has: `place_option_order()`, `place_combo_order()`, `get_option_chain_params()`, `get_option_greeks()`, `qualify_option_contract()`
- Combo orders use BAG contract with ComboLeg objects for atomic spread execution
- `get_positions()` recognizes `secType == 'OPT'` portfolio items
- Always LIMIT orders at mid price + configurable slippage ticks

## Murphy-Inspired Enhancements (Technical Analysis of the Financial Markets)

Planned enhancements based on John Murphy's framework, prioritized by impact/effort:

### Priority 1: Intermarket Analysis (IMPLEMENTED — `scanner/intermarket_analyzer.py`)
- Tracks TLT (bonds), UUP (dollar), GLD (gold), IWM (small caps) via yfinance (30-min cache)
- `bonds_stocks_divergence`: TLT trend vs SPY trend — bonds lead stocks at turns
- `dollar_trend`: rising dollar = headwind for equities
- `gold_signal`: flight-to-safety detection (rising gold + falling stocks)
- `risk_on_off_ratio`: IWM/SPY ratio confirms risk appetite
- `intermarket_composite`: weighted average (bonds 0.35, risk 0.30, dollar 0.20, gold 0.15)
- `intermarket_regime`: risk_on (>0.3) / neutral / risk_off (<-0.3)
- RRS adjustment: risk_on -0.25, risk_off +0.50; Position sizing: risk_on 1.10x, risk_off 0.75x
- Integrated into `realtime_scanner.py` — adjusts thresholds and sizing, does NOT block signals
- Config: `intermarket_enabled` (default True), `intermarket_cache_ttl_minutes` (default 30)

### Priority 2: OBV + Volume Divergence (IMPLEMENTED — `ml/feature_engineering.py`)
- `obv_trend`: OBV 10-bar linear regression slope direction (+1, 0, -1)
- `obv_price_divergence`: binary flag when price slope and OBV slope disagree
- `volume_climax`: binary flag when volume > 3x 20-day average

### Priority 3: ADX-Gated Indicator Selection (IMPLEMENTED — `ml/feature_engineering.py`)
- `adx`, `adx_rising`, `plus_di`, `minus_di` features added
- Rising ADX (>25): suppresses `reversal_probability` to 0.2 (prevents false signals in trends)
- ADX gating applied in `_calculate_derived_features()`

### Priority 4: Additional ML Features (IMPLEMENTED — `ml/feature_engineering.py`)
- `sma_200` + `price_above_sma200` + `golden_cross_state` (50/200 SMA crossover with 1% dead zone)
- `macd_histogram_slope`: 3-bar consecutive direction (+1, -1, 0)
- `macd_histogram_divergence`: compares price peaks vs histogram peaks across 40-bar window
- `rrs_slope` + `rrs_acceleration`: direction and momentum of RRS itself
- `bb_squeeze` + `bb_squeeze_duration`: BB width at/below 50-bar rolling min
- `pullback_depth_pct`: Fibonacci retracement % of last 20-bar range

Total features now: 87 (was 70). New `murphy_features` category with 17 features.

### Priority 5: Oscillator Divergence Detection (FUTURE)
- RSI bearish/bullish divergence (price vs RSI swing comparison)
- Weighted more heavily when RSI in overbought/oversold territory

## Patterns to Follow When Making Changes

1. **Always use `safe_model_loader.py`** for loading any pickle/joblib files
2. **Never expose `str(e)` in API responses** - log server-side, return generic message
3. **Use `@require_api_key` or `@login_required`** on all new endpoints
4. **Validate inputs at system boundaries** (API params, form data, file paths)
5. **Use `utils/paths.py:get_project_root()`** instead of hardcoded paths
6. **Normalize column names to lowercase** after any data fetch
7. **Use `fcntl.flock`** when reading/writing shared files
8. **Never re-enable the service worker** without a POST request bypass strategy
9. **Test with `python -c "import py_compile; py_compile.compile('file.py', doraise=True)"`** after edits
10. **Update this CLAUDE.md** when making significant architectural changes
11. **Scanner filter gates are sequential** — SPY → SMA → VWAP → MTF → output
12. **All gates fail-open** — missing data never blocks signals
13. **Mean reversion is DISABLED** — contradicts RDT momentum philosophy
