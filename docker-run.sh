#!/bin/bash
# RDT Trading System - Docker Runner
# Usage: ./docker-run.sh [scanner|bot|bot-auto|backtest|dashboard|build|stop|logs]

set -e

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="rdt-trading"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════╗"
    echo "║     RDT Trading System - Docker        ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  scanner    - Run signal scanner (semi-automated)"
    echo "  bot        - Run trading bot (manual execution)"
    echo "  bot-auto   - Run trading bot (AUTO EXECUTION - CAUTION!)"
    echo "  backtest   - Run backtesting"
    echo "  dashboard  - Run monitoring dashboard"
    echo "  build      - Build Docker images"
    echo "  stop       - Stop all containers"
    echo "  logs       - View logs"
    echo "  shell      - Open shell in container"
    echo ""
}

check_env() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Warning: .env file not found. Copying from .env.example${NC}"
        if [ -f .env.example ]; then
            cp .env.example .env
            echo -e "${GREEN}Created .env file. Please edit it with your settings.${NC}"
        else
            echo -e "${RED}Error: .env.example not found${NC}"
            exit 1
        fi
    fi
}

build_images() {
    echo -e "${GREEN}Building Docker images...${NC}"
    docker compose -f $COMPOSE_FILE build
    echo -e "${GREEN}Build complete!${NC}"
}

run_scanner() {
    echo -e "${GREEN}Starting Scanner (Semi-Automated Mode)...${NC}"
    docker compose -f $COMPOSE_FILE --profile scanner up -d scanner
    echo -e "${GREEN}Scanner started. View logs with: $0 logs${NC}"
}

run_bot() {
    echo -e "${YELLOW}Starting Trading Bot (Manual Execution Mode)...${NC}"
    docker compose -f $COMPOSE_FILE --profile bot up -d bot
    echo -e "${GREEN}Bot started. View logs with: $0 logs${NC}"
}

run_bot_auto() {
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  WARNING: AUTO-TRADING MODE - REAL MONEY AT RISK!         ║${NC}"
    echo -e "${RED}║  Only proceed if you have:                                 ║${NC}"
    echo -e "${RED}║  - Tested extensively in paper mode                        ║${NC}"
    echo -e "${RED}║  - Set PAPER_TRADING=true in .env for safety              ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        docker compose -f $COMPOSE_FILE --profile bot-auto up -d bot-auto
        echo -e "${GREEN}Auto-trading bot started. View logs with: $0 logs${NC}"
    else
        echo "Cancelled."
    fi
}

run_backtest() {
    echo -e "${GREEN}Running Backtest...${NC}"
    docker compose -f $COMPOSE_FILE --profile backtest run --rm backtest
}

run_dashboard() {
    echo -e "${GREEN}Starting Dashboard...${NC}"
    docker compose -f $COMPOSE_FILE --profile dashboard run --rm dashboard
}

stop_all() {
    echo -e "${YELLOW}Stopping all containers...${NC}"
    docker compose -f $COMPOSE_FILE --profile scanner --profile bot --profile bot-auto --profile dashboard down
    echo -e "${GREEN}All containers stopped.${NC}"
}

view_logs() {
    docker compose -f $COMPOSE_FILE logs -f
}

open_shell() {
    echo -e "${GREEN}Opening shell in container...${NC}"
    docker compose -f $COMPOSE_FILE run --rm --entrypoint /bin/bash scanner
}

# Main
print_banner
check_env

case "${1:-}" in
    scanner)
        run_scanner
        ;;
    bot)
        run_bot
        ;;
    bot-auto)
        run_bot_auto
        ;;
    backtest)
        run_backtest
        ;;
    dashboard)
        run_dashboard
        ;;
    build)
        build_images
        ;;
    stop)
        stop_all
        ;;
    logs)
        view_logs
        ;;
    shell)
        open_shell
        ;;
    *)
        print_usage
        ;;
esac
