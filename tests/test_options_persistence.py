"""
Tests for options position persistence.

Tests the DB models, repository methods, JSON serialization,
and PaperOptionsExecutor integration with the database.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from options.models import (
    OptionAction, OptionContract, OptionGreeks, OptionLeg,
    OptionRight, OptionsPositionSizeResult, OptionsStrategy,
    StrategyDirection,
)
from options.paper_executor import (
    PaperOptionsExecutor, _legs_to_json, _legs_from_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_contract(symbol="AAPL", strike=150.0, right=OptionRight.CALL, days=30):
    expiry = (datetime.now() + timedelta(days=days)).strftime("%Y%m%d")
    return OptionContract(symbol=symbol, expiry=expiry, strike=strike, right=right)


def make_greeks(delta=0.55, bid=3.00, ask=3.20, implied_vol=0.25):
    return OptionGreeks(
        delta=delta, gamma=0.03, theta=-0.05, vega=0.12,
        bid=bid, ask=ask, option_price=(bid + ask) / 2,
        implied_vol=implied_vol, underlying_price=150.0,
        volume=500, open_interest=5000,
    )


def make_long_call_strategy(symbol="AAPL"):
    contract = make_contract(symbol=symbol, strike=150.0)
    greeks = make_greeks()
    leg = OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1, greeks=greeks)
    return OptionsStrategy(
        name="long_call",
        underlying=symbol,
        direction=StrategyDirection.LONG,
        legs=[leg],
        max_loss=310.0,
        max_profit=float('inf'),
        net_premium=-310.0,
    )


def make_bull_call_spread(symbol="AAPL"):
    long_contract = make_contract(symbol=symbol, strike=150.0)
    short_contract = make_contract(symbol=symbol, strike=155.0)
    long_greeks = make_greeks(delta=0.55, bid=3.00, ask=3.20)
    short_greeks = make_greeks(delta=0.35, bid=1.50, ask=1.70)

    long_leg = OptionLeg(contract=long_contract, action=OptionAction.BUY, quantity=1, greeks=long_greeks)
    short_leg = OptionLeg(contract=short_contract, action=OptionAction.SELL, quantity=1, greeks=short_greeks)

    return OptionsStrategy(
        name="bull_call_spread",
        underlying=symbol,
        direction=StrategyDirection.LONG,
        legs=[long_leg, short_leg],
        max_loss=150.0,
        max_profit=350.0,
        net_premium=-150.0,
    )


def make_mock_provider():
    provider = MagicMock()
    provider.get_greeks.return_value = make_greeks()
    return provider


# ---------------------------------------------------------------------------
# JSON Serialization Tests
# ---------------------------------------------------------------------------

class TestLegsJsonSerialization:
    def test_single_leg_round_trip(self):
        strategy = make_long_call_strategy()
        json_str = _legs_to_json(strategy)
        restored = _legs_from_json(json_str, "long_call", "AAPL", "long")

        assert restored is not None
        assert restored.name == "long_call"
        assert restored.underlying == "AAPL"
        assert restored.direction == StrategyDirection.LONG
        assert len(restored.legs) == 1

        leg = restored.legs[0]
        assert leg.contract.symbol == "AAPL"
        assert leg.contract.strike == 150.0
        assert leg.contract.right == OptionRight.CALL
        assert leg.action == OptionAction.BUY
        assert leg.quantity == 1

    def test_multi_leg_round_trip(self):
        strategy = make_bull_call_spread()
        json_str = _legs_to_json(strategy)
        restored = _legs_from_json(json_str, "bull_call_spread", "AAPL", "long")

        assert restored is not None
        assert len(restored.legs) == 2
        assert restored.legs[0].action == OptionAction.BUY
        assert restored.legs[1].action == OptionAction.SELL
        assert restored.legs[0].contract.strike == 150.0
        assert restored.legs[1].contract.strike == 155.0

    def test_greeks_preserved(self):
        strategy = make_long_call_strategy()
        json_str = _legs_to_json(strategy)
        restored = _legs_from_json(json_str, "long_call", "AAPL", "long")

        greeks = restored.legs[0].greeks
        assert greeks is not None
        assert greeks.delta == 0.55
        assert greeks.gamma == 0.03
        assert greeks.theta == -0.05
        assert greeks.vega == 0.12
        assert greeks.implied_vol == 0.25
        assert greeks.underlying_price == 150.0
        assert greeks.bid == 3.00
        assert greeks.ask == 3.20

    def test_no_greeks_leg(self):
        contract = make_contract()
        leg = OptionLeg(contract=contract, action=OptionAction.BUY, quantity=1, greeks=None)
        strategy = OptionsStrategy(
            name="test", underlying="AAPL",
            direction=StrategyDirection.LONG, legs=[leg],
        )
        json_str = _legs_to_json(strategy)
        restored = _legs_from_json(json_str, "test", "AAPL", "long")

        assert restored is not None
        assert restored.legs[0].greeks is None

    def test_invalid_json_returns_none(self):
        result = _legs_from_json("not valid json", "test", "AAPL", "long")
        assert result is None

    def test_none_json_returns_none(self):
        result = _legs_from_json(None, "test", "AAPL", "long")
        assert result is None

    def test_json_is_valid(self):
        strategy = make_long_call_strategy()
        json_str = _legs_to_json(strategy)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["symbol"] == "AAPL"
        assert parsed[0]["strike"] == 150.0
        assert parsed[0]["right"] == "C"
        assert parsed[0]["action"] == "BUY"


# ---------------------------------------------------------------------------
# DB Persistence Tests (using SQLite in-memory)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_repo():
    """Create an in-memory SQLite database with options tables."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as SASession

    from data.database.models import Base
    from data.database.trades_repository import TradesRepository

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    repo = TradesRepository()
    # Monkey-patch the db_manager to use our in-memory engine
    mock_manager = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def mock_session():
        session = SASession(engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    mock_manager.get_session = mock_session
    mock_manager.create_tables = MagicMock()
    repo._db_manager = mock_manager

    return repo


class TestOptionsPositionDB:
    def test_save_and_retrieve(self, db_repo):
        strategy = make_long_call_strategy()
        legs_json = _legs_to_json(strategy)

        result = db_repo.save_options_position({
            'symbol': 'AAPL',
            'strategy_name': 'long_call',
            'direction': 'long',
            'contracts': 2,
            'entry_premium': 3.10,
            'total_premium': -620.0,
            'entry_iv': 0.25,
            'entry_delta': 0.55,
            'order_ids': json.dumps(["abc123"]),
            'legs_json': legs_json,
            'fill_details_json': json.dumps([{"action": "BUY", "fill_price": 3.10}]),
        })
        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert result['strategy_name'] == 'long_call'
        assert result['contracts'] == 2

        # Retrieve
        pos = db_repo.get_options_position_by_symbol('AAPL')
        assert pos is not None
        assert pos['symbol'] == 'AAPL'
        assert pos['strategy_name'] == 'long_call'
        assert pos['contracts'] == 2
        assert pos['entry_premium'] == pytest.approx(3.10, abs=0.01)
        assert pos['total_premium'] == pytest.approx(-620.0, abs=0.01)

    def test_upsert_updates_existing(self, db_repo):
        strategy = make_long_call_strategy()
        legs_json = _legs_to_json(strategy)

        db_repo.save_options_position({
            'symbol': 'AAPL', 'strategy_name': 'long_call', 'direction': 'long',
            'contracts': 1, 'legs_json': legs_json,
        })
        db_repo.save_options_position({
            'symbol': 'AAPL', 'strategy_name': 'long_call', 'direction': 'long',
            'contracts': 3, 'legs_json': legs_json,
        })

        positions = db_repo.get_all_options_positions()
        assert len(positions) == 1
        assert positions[0]['contracts'] == 3

    def test_get_all_positions(self, db_repo):
        for sym in ['AAPL', 'MSFT', 'GOOG']:
            strategy = make_long_call_strategy(sym)
            db_repo.save_options_position({
                'symbol': sym, 'strategy_name': 'long_call', 'direction': 'long',
                'contracts': 1, 'legs_json': _legs_to_json(strategy),
            })

        positions = db_repo.get_all_options_positions()
        assert len(positions) == 3
        symbols = {p['symbol'] for p in positions}
        assert symbols == {'AAPL', 'MSFT', 'GOOG'}

    def test_close_position_deletes(self, db_repo):
        strategy = make_long_call_strategy()
        db_repo.save_options_position({
            'symbol': 'AAPL', 'strategy_name': 'long_call', 'direction': 'long',
            'contracts': 1, 'legs_json': _legs_to_json(strategy),
        })

        assert db_repo.close_options_position('AAPL') is True
        assert db_repo.get_options_position_by_symbol('AAPL') is None

    def test_close_nonexistent_returns_false(self, db_repo):
        assert db_repo.close_options_position('NOPE') is False

    def test_case_insensitive_symbol(self, db_repo):
        strategy = make_long_call_strategy()
        db_repo.save_options_position({
            'symbol': 'aapl', 'strategy_name': 'long_call', 'direction': 'long',
            'contracts': 1, 'legs_json': _legs_to_json(strategy),
        })

        pos = db_repo.get_options_position_by_symbol('AAPL')
        assert pos is not None
        assert pos['symbol'] == 'AAPL'


class TestOptionsTradeDB:
    def test_save_trade(self, db_repo):
        strategy = make_long_call_strategy()
        result = db_repo.save_options_trade({
            'symbol': 'AAPL',
            'strategy_name': 'long_call',
            'direction': 'long',
            'contracts': 2,
            'entry_time': datetime.utcnow(),
            'exit_time': datetime.utcnow(),
            'entry_premium': 3.10,
            'total_premium': -620.0,
            'exit_premium': -500.0,
            'pnl': 120.0,
            'pnl_percent': 19.35,
            'entry_iv': 0.25,
            'exit_iv': 0.22,
            'entry_delta': 0.55,
            'legs_json': _legs_to_json(strategy),
            'exit_reason': 'manual',
            'status': 'closed',
        })
        assert result is not None
        assert result['pnl'] == pytest.approx(120.0, abs=0.01)
        assert result['status'] == 'closed'

    def test_get_trades(self, db_repo):
        strategy = make_long_call_strategy()
        for i in range(3):
            db_repo.save_options_trade({
                'symbol': 'AAPL', 'strategy_name': 'long_call', 'direction': 'long',
                'contracts': 1, 'entry_time': datetime.utcnow(),
                'legs_json': _legs_to_json(strategy), 'status': 'closed',
            })

        trades = db_repo.get_options_trades()
        assert len(trades) == 3

    def test_get_trades_by_symbol(self, db_repo):
        for sym in ['AAPL', 'AAPL', 'MSFT']:
            strategy = make_long_call_strategy(sym)
            db_repo.save_options_trade({
                'symbol': sym, 'strategy_name': 'long_call', 'direction': 'long',
                'contracts': 1, 'entry_time': datetime.utcnow(),
                'legs_json': _legs_to_json(strategy), 'status': 'closed',
            })

        trades = db_repo.get_options_trades(symbol='AAPL')
        assert len(trades) == 2

    def test_get_trades_by_days(self, db_repo):
        strategy = make_long_call_strategy()
        # Recent trade
        db_repo.save_options_trade({
            'symbol': 'AAPL', 'strategy_name': 'long_call', 'direction': 'long',
            'contracts': 1, 'entry_time': datetime.utcnow(),
            'legs_json': _legs_to_json(strategy), 'status': 'closed',
        })
        # Old trade
        db_repo.save_options_trade({
            'symbol': 'MSFT', 'strategy_name': 'long_call', 'direction': 'long',
            'contracts': 1, 'entry_time': datetime.utcnow() - timedelta(days=60),
            'legs_json': _legs_to_json(strategy), 'status': 'closed',
        })

        trades = db_repo.get_options_trades(days=30)
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'AAPL'


# ---------------------------------------------------------------------------
# Executor + Persistence Integration Tests
# ---------------------------------------------------------------------------

class TestExecutorPersistence:
    def _make_executor_with_db(self, db_repo):
        """Create a PaperOptionsExecutor wired to the test DB."""
        provider = make_mock_provider()
        executor = PaperOptionsExecutor.__new__(PaperOptionsExecutor)
        executor._provider = provider
        executor._config = MagicMock()
        executor._r = 0.05
        executor._positions = {}
        executor._get_repository = MagicMock(return_value=db_repo)
        # Don't load on init — we'll do it explicitly
        return executor

    def test_execute_persists_position(self, db_repo):
        executor = self._make_executor_with_db(db_repo)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=2, max_risk=620, premium_cost=620,
        )

        result = executor.execute_strategy(strategy, size)
        assert result is not None
        assert result["status"] == "Filled"

        # Position should be in DB
        pos = db_repo.get_options_position_by_symbol('AAPL')
        assert pos is not None
        assert pos['strategy_name'] == 'long_call'
        assert pos['contracts'] == 2

    def test_close_removes_from_db_and_creates_trade(self, db_repo):
        executor = self._make_executor_with_db(db_repo)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310,
        )

        executor.execute_strategy(strategy, size)
        assert db_repo.get_options_position_by_symbol('AAPL') is not None

        close_result = executor.close_position('AAPL')
        assert close_result is not None

        # Position should be gone from DB
        assert db_repo.get_options_position_by_symbol('AAPL') is None

        # Trade record should exist
        trades = db_repo.get_options_trades(symbol='AAPL')
        assert len(trades) == 1
        assert trades[0]['status'] == 'closed'
        assert trades[0]['exit_reason'] == 'manual'

    def test_load_restores_position(self, db_repo):
        # First executor opens a position
        executor1 = self._make_executor_with_db(db_repo)
        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310,
        )
        executor1.execute_strategy(strategy, size)

        # Second executor loads from DB
        executor2 = self._make_executor_with_db(db_repo)
        executor2._load_positions()

        pos = executor2.get_position('AAPL')
        assert pos is not None
        assert pos['strategy'].name == 'long_call'
        assert pos['contracts'] == 1
        assert len(pos['strategy'].legs) == 1
        assert pos['strategy'].legs[0].contract.strike == 150.0

    def test_roll_creates_trade_and_new_position(self, db_repo):
        executor = self._make_executor_with_db(db_repo)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310,
        )
        executor.execute_strategy(strategy, size)

        new_expiry = (datetime.now() + timedelta(days=60)).strftime("%Y%m%d")
        result = executor.roll_position("AAPL", new_expiry)
        assert result is not None

        # Should have a trade record (from the close)
        trades = db_repo.get_options_trades(symbol='AAPL')
        assert len(trades) == 1

        # Should have a new position in DB
        pos = db_repo.get_options_position_by_symbol('AAPL')
        assert pos is not None

    def test_multiple_positions_persist(self, db_repo):
        executor = self._make_executor_with_db(db_repo)

        for sym in ['AAPL', 'MSFT', 'GOOG']:
            strategy = make_long_call_strategy(sym)
            size = OptionsPositionSizeResult(
                strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310,
            )
            executor.execute_strategy(strategy, size)

        positions = db_repo.get_all_options_positions()
        assert len(positions) == 3

        # Close one
        executor.close_position('MSFT')
        positions = db_repo.get_all_options_positions()
        assert len(positions) == 2
        symbols = {p['symbol'] for p in positions}
        assert 'MSFT' not in symbols

    def test_no_db_still_works(self):
        """Executor should work even if DB is unavailable."""
        provider = make_mock_provider()
        executor = PaperOptionsExecutor.__new__(PaperOptionsExecutor)
        executor._provider = provider
        executor._config = MagicMock()
        executor._r = 0.05
        executor._positions = {}
        executor._get_repository = MagicMock(return_value=None)

        strategy = make_long_call_strategy()
        size = OptionsPositionSizeResult(
            strategy_name="long_call", contracts=1, max_risk=310, premium_cost=310,
        )

        result = executor.execute_strategy(strategy, size)
        assert result is not None
        assert result["status"] == "Filled"

        # Position tracked in memory
        pos = executor.get_position('AAPL')
        assert pos is not None

        # Close still works
        close_result = executor.close_position('AAPL')
        assert close_result is not None
        assert executor.get_position('AAPL') is None
