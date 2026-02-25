#!/bin/bash
# RDT Trading System - Docker Setup Script
# This script builds and initializes the trading system in Docker

set -e

echo "=============================================="
echo "  RDT Trading System - Docker Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project directory
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

echo -e "${YELLOW}Project directory: ${PROJECT_DIR}${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}Using: ${COMPOSE_CMD}${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p models logs data results ml/data/visualizations

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating template...${NC}"
    cat > .env << 'EOF'
# RDT Trading System Configuration
# Copy this file and fill in your credentials

# Schwab API (required for live trading)
SCHWAB_APP_KEY=your_app_key_here
SCHWAB_APP_SECRET=your_app_secret_here

# Account Settings
ACCOUNT_SIZE=25000
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.06
MAX_POSITION_SIZE=0.20

# Trading Mode
PAPER_TRADING=true
AUTO_TRADE=false

# RRS Settings (Aggressive)
RRS_STRONG_THRESHOLD=1.75

# ML Settings
ML_CONFIDENCE_THRESHOLD=0.65
REGIME_CONFIDENCE_THRESHOLD=0.70
EOF
    echo -e "${YELLOW}Please edit .env with your credentials before running${NC}"
fi

# Build the Docker image
echo ""
echo -e "${GREEN}Step 1: Building Docker image...${NC}"
echo "This may take a few minutes on first run..."
$COMPOSE_CMD build rdt-trading-bot

# Train the models
echo ""
echo -e "${GREEN}Step 2: Training ML models...${NC}"
echo "Training regime detector (this takes ~60 seconds)..."
$COMPOSE_CMD run --rm model-trainer

# Verify the build
echo ""
echo -e "${GREEN}Step 3: Verifying installation...${NC}"
$COMPOSE_CMD run --rm rdt-trading-bot python -c "
import sys
print('Python:', sys.version)

# Check ML libraries
try:
    import sklearn; print('✓ scikit-learn:', sklearn.__version__)
except: print('✗ scikit-learn NOT installed')

try:
    import xgboost; print('✓ xgboost:', xgboost.__version__)
except: print('✗ xgboost NOT installed')

try:
    import tensorflow; print('✓ tensorflow:', tensorflow.__version__)
except: print('✗ tensorflow NOT installed')

try:
    from hmmlearn import hmm; print('✓ hmmlearn: installed')
except: print('✗ hmmlearn NOT installed')

try:
    import lightgbm; print('✓ lightgbm:', lightgbm.__version__)
except: print('✗ lightgbm NOT installed')

# Check trading libraries
try:
    import yfinance; print('✓ yfinance:', yfinance.__version__)
except: print('✗ yfinance NOT installed')

try:
    import pandas; print('✓ pandas:', pandas.__version__)
except: print('✗ pandas NOT installed')

try:
    import numpy; print('✓ numpy:', numpy.__version__)
except: print('✗ numpy NOT installed')

# Check RDT modules
try:
    from ml.feature_engineering import FeatureEngineer
    print('✓ FeatureEngineer module')
except Exception as e:
    print(f'✗ FeatureEngineer: {e}')

try:
    from ml.ensemble import Ensemble
    print('✓ Ensemble module')
except Exception as e:
    print(f'✗ Ensemble: {e}')

try:
    from ml.regime_detector import MarketRegimeDetector
    print('✓ RegimeDetector module')
except Exception as e:
    print(f'✗ RegimeDetector: {e}')

try:
    from agents.learning_agent import LearningAgent
    print('✓ LearningAgent module')
except Exception as e:
    print(f'✗ LearningAgent: {e}')

try:
    from agents.risk_agent import RiskAgent
    print('✓ RiskAgent module')
except Exception as e:
    print(f'✗ RiskAgent: {e}')

print()
print('=== Installation Verification Complete ===')
"

# Check if model was trained
if [ -f "models/regime_detector.pkl" ]; then
    echo -e "${GREEN}✓ Regime detector model trained successfully${NC}"
else
    echo -e "${YELLOW}⚠ Regime detector model not found - may need manual training${NC}"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "To start the trading bot:"
echo "  $COMPOSE_CMD up -d rdt-trading-bot"
echo ""
echo "To view logs:"
echo "  $COMPOSE_CMD logs -f rdt-trading-bot"
echo ""
echo "To run backtest:"
echo "  $COMPOSE_CMD run --rm rdt-trading-bot python main.py backtest --enhanced --watchlist core --days 180"
echo ""
echo "To enter container shell:"
echo "  $COMPOSE_CMD exec rdt-trading-bot bash"
echo ""
echo "To stop:"
echo "  $COMPOSE_CMD down"
echo ""
