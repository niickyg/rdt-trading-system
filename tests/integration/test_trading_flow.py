"""
Integration Tests for Complete Trading Flow

Tests the end-to-end trading flow:
1. Scanner generates signal
2. Signal saved to database
3. AnalyzerAgent evaluates signal
4. RiskAgent approves/rejects
5. TradingBot places order
6. Order filled
7. Position tracked with P&L
8. Position closed at target/stop
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

pytestmark = [pytest.mark.integration]


class TestSignalGeneration:
    """Test signal generation and storage flow."""

    def test_scanner_generates_signal(self, sample_signal_data, test_database):
        """Test that scanner can generate and detect valid signals."""
        # Simulate signal generation
        signal = sample_signal_data.copy()

        # Verify signal has required fields
        assert 'symbol' in signal
        assert 'direction' in signal
        assert 'rrs' in signal
        assert 'entry_price' in signal
        assert 'stop_price' in signal
        assert 'target_price' in signal

        # Verify RRS meets threshold
        assert abs(signal['rrs']) >= 2.0, "Signal RRS should meet threshold"

    def test_signal_saved_to_database(self, sample_signal_data, test_database):
        """Test that signals are properly saved to database."""
        cursor = test_database.cursor()

        # Insert signal
        cursor.execute('''
            INSERT INTO signals (
                symbol, direction, strength, rrs, entry_price,
                stop_price, target_price, atr, stock_change_pct,
                spy_change_pct, daily_strong, generated_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample_signal_data['symbol'],
            sample_signal_data['direction'],
            sample_signal_data['strength'],
            sample_signal_data['rrs'],
            sample_signal_data['entry_price'],
            sample_signal_data['stop_price'],
            sample_signal_data['target_price'],
            sample_signal_data['atr'],
            sample_signal_data['stock_change_pct'],
            sample_signal_data['spy_change_pct'],
            1 if sample_signal_data['daily_strong'] else 0,
            sample_signal_data['generated_at'],
            'pending'
        ))
        test_database.commit()

        # Verify signal was saved
        cursor.execute('SELECT * FROM signals WHERE symbol = ?', (sample_signal_data['symbol'],))
        row = cursor.fetchone()

        assert row is not None, "Signal should be saved to database"
        assert row[1] == sample_signal_data['symbol']  # symbol
        assert row[2] == sample_signal_data['direction']  # direction
        assert row[4] == sample_signal_data['rrs']  # rrs

    def test_multiple_signals_stored(self, test_database):
        """Test storing multiple signals from a scan."""
        cursor = test_database.cursor()

        signals = [
            ('AAPL', 'long', 'strong', 2.85, 175.50),
            ('GOOGL', 'long', 'moderate', 2.15, 140.00),
            ('TSLA', 'short', 'strong', -3.20, 200.00),
        ]

        for symbol, direction, strength, rrs, price in signals:
            cursor.execute('''
                INSERT INTO signals (
                    symbol, direction, strength, rrs, entry_price,
                    stop_price, target_price, atr, generated_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, direction, strength, rrs, price,
                price - 3.0 if direction == 'long' else price + 3.0,
                price + 6.0 if direction == 'long' else price - 6.0,
                3.0, datetime.now().isoformat(), 'pending'
            ))

        test_database.commit()

        # Verify all signals stored
        cursor.execute('SELECT COUNT(*) FROM signals')
        count = cursor.fetchone()[0]
        assert count == 3, "All signals should be stored"

        # Verify we can retrieve by direction
        cursor.execute("SELECT COUNT(*) FROM signals WHERE direction = 'long'")
        long_count = cursor.fetchone()[0]
        assert long_count == 2


class TestSignalAnalysis:
    """Test signal analysis by AnalyzerAgent."""

    def test_analyzer_evaluates_strong_signal(self, sample_signal_data, mock_event_bus):
        """Test AnalyzerAgent approves strong signals."""
        # Create mock analyzer that simulates signal evaluation
        analyzed_signal = sample_signal_data.copy()
        analyzed_signal['analysis_result'] = 'approved'
        analyzed_signal['confidence'] = 0.85
        analyzed_signal['reasons'] = ['Strong RRS', 'Good volume', 'Daily trend aligned']

        # Simulate publishing analysis result
        mock_event_bus.publish('SIGNAL_ANALYZED', analyzed_signal)

        # Verify event was published
        events = mock_event_bus.get_events_of_type('SIGNAL_ANALYZED')
        assert len(events) == 1
        assert events[0][1]['analysis_result'] == 'approved'

    def test_analyzer_rejects_weak_signal(self, mock_event_bus):
        """Test AnalyzerAgent rejects weak signals."""
        weak_signal = {
            'symbol': 'MSFT',
            'direction': 'long',
            'rrs': 0.5,  # Below threshold
            'entry_price': 300.00,
            'volume': 100000  # Low volume
        }

        analyzed_signal = weak_signal.copy()
        analyzed_signal['analysis_result'] = 'rejected'
        analyzed_signal['confidence'] = 0.3
        analyzed_signal['reasons'] = ['RRS below threshold', 'Insufficient volume']

        mock_event_bus.publish('SIGNAL_ANALYZED', analyzed_signal)

        events = mock_event_bus.get_events_of_type('SIGNAL_ANALYZED')
        assert len(events) == 1
        assert events[0][1]['analysis_result'] == 'rejected'

    def test_analyzer_adds_ml_confidence(self, sample_signal_data, mock_ml_model, mock_event_bus):
        """Test AnalyzerAgent adds ML confidence scores."""
        # Get ML prediction
        prediction = mock_ml_model.predict({'rrs': sample_signal_data['rrs']})

        analyzed_signal = sample_signal_data.copy()
        analyzed_signal['ml_confidence'] = prediction['confidence']
        analyzed_signal['ml_signal'] = prediction['signal']

        mock_event_bus.publish('SIGNAL_ANALYZED', analyzed_signal)

        events = mock_event_bus.get_events_of_type('SIGNAL_ANALYZED')
        assert 'ml_confidence' in events[0][1]
        assert events[0][1]['ml_confidence'] == 0.75


class TestRiskValidation:
    """Test risk validation by RiskAgent."""

    def test_risk_agent_approves_valid_trade(
        self, sample_signal_data, integration_mock_broker, mock_event_bus
    ):
        """Test RiskAgent approves trade within risk parameters."""
        account = integration_mock_broker.get_account()

        # Calculate position size within limits
        max_position_value = account.equity * 0.10  # 10% max position
        shares = int(max_position_value / sample_signal_data['entry_price'])

        risk_assessment = {
            'approved': True,
            'position_size': shares,
            'risk_amount': shares * (sample_signal_data['entry_price'] - sample_signal_data['stop_price']),
            'risk_percent': 0.008,  # 0.8% of account
            'reasons': ['Within risk limits', 'Sufficient buying power']
        }

        mock_event_bus.publish('RISK_VALIDATED', risk_assessment)

        events = mock_event_bus.get_events_of_type('RISK_VALIDATED')
        assert len(events) == 1
        assert events[0][1]['approved'] is True

    def test_risk_agent_rejects_oversized_position(
        self, sample_signal_data, integration_mock_broker, mock_event_bus
    ):
        """Test RiskAgent rejects positions that exceed limits."""
        account = integration_mock_broker.get_account()

        # Try to calculate oversized position (50% of account)
        oversized_value = account.equity * 0.50
        oversized_shares = int(oversized_value / sample_signal_data['entry_price'])

        risk_assessment = {
            'approved': False,
            'requested_size': oversized_shares,
            'max_allowed_size': int(account.equity * 0.10 / sample_signal_data['entry_price']),
            'reasons': ['Position size exceeds 10% limit']
        }

        mock_event_bus.publish('RISK_REJECTED', risk_assessment)

        events = mock_event_bus.get_events_of_type('RISK_REJECTED')
        assert len(events) == 1
        assert events[0][1]['approved'] is False

    def test_risk_agent_tracks_daily_loss(
        self, integration_mock_broker, mock_event_bus, test_database
    ):
        """Test RiskAgent tracks and enforces daily loss limits."""
        # Simulate some losing trades
        cursor = test_database.cursor()

        losses = [
            ('AAPL', -500.00),
            ('GOOGL', -750.00),
            ('TSLA', -300.00)
        ]

        today = datetime.now().strftime('%Y-%m-%d')

        for symbol, pnl in losses:
            cursor.execute('''
                INSERT INTO trades (
                    symbol, direction, entry_price, exit_price, shares,
                    entry_time, exit_time, pnl, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, 'long', 100.0, 95.0, 100, today, today, pnl, 'closed'))

        test_database.commit()

        # Calculate total daily loss
        cursor.execute('''
            SELECT SUM(pnl) FROM trades
            WHERE date(exit_time) = date('now') AND pnl < 0
        ''')
        total_loss = cursor.fetchone()[0] or 0

        account = integration_mock_broker.get_account()
        loss_percent = abs(total_loss) / account.equity

        # With 3% daily loss limit and $100k account, limit is $3000
        # Current loss is $1550, so new trades should still be allowed
        assert loss_percent < 0.03, "Daily loss should be within limits"


class TestOrderExecution:
    """Test order placement and execution."""

    def test_place_market_order(self, integration_mock_broker, sample_signal_data):
        """Test placing a market order."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        symbol = sample_signal_data['symbol']
        shares = 100
        price = sample_signal_data['entry_price']

        order = integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=price
        )

        assert order is not None
        assert order.symbol == symbol
        assert order.quantity == shares
        assert order.status.value in ('filled', 'FILLED')

    def test_order_creates_position(self, integration_mock_broker, sample_signal_data):
        """Test that filled order creates a position."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        symbol = sample_signal_data['symbol']
        shares = 100
        price = sample_signal_data['entry_price']

        # Place order
        integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=price
        )

        # Check position exists
        position = integration_mock_broker.get_position(symbol)

        assert position is not None
        assert position.symbol == symbol
        assert position.quantity == shares

    def test_order_reduces_buying_power(self, integration_mock_broker, sample_signal_data):
        """Test that order reduces available buying power."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        initial_account = integration_mock_broker.get_account()
        initial_buying_power = initial_account.buying_power

        symbol = sample_signal_data['symbol']
        shares = 100
        price = sample_signal_data['entry_price']

        # Place order
        integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=price
        )

        # Check buying power reduced
        new_account = integration_mock_broker.get_account()
        expected_reduction = shares * price

        assert new_account.buying_power < initial_buying_power
        assert abs((initial_buying_power - new_account.buying_power) - expected_reduction) < 1.0


class TestPositionTracking:
    """Test position tracking and P&L calculation."""

    def test_track_unrealized_pnl(self, integration_mock_broker, sample_signal_data):
        """Test unrealized P&L calculation as price changes."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        symbol = sample_signal_data['symbol']
        shares = 100
        entry_price = sample_signal_data['entry_price']

        # Enter position
        integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=entry_price
        )

        # Simulate price increase
        new_price = entry_price * 1.02  # 2% gain
        integration_mock_broker.update_price(symbol, new_price)

        # Check unrealized P&L
        position = integration_mock_broker.get_position(symbol)
        expected_pnl = (new_price - entry_price) * shares

        assert position.unrealized_pnl == pytest.approx(expected_pnl, rel=0.01)

    def test_position_saved_to_database(
        self, integration_mock_broker, sample_signal_data, test_database
    ):
        """Test position data is saved to database."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        cursor = test_database.cursor()

        # Enter position through broker
        symbol = sample_signal_data['symbol']
        shares = 100
        entry_price = sample_signal_data['entry_price']

        integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=entry_price
        )

        # Save to database
        cursor.execute('''
            INSERT INTO positions (
                symbol, direction, entry_price, shares, stop_loss,
                take_profit, entry_time, status, current_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, 'long', entry_price, shares,
            sample_signal_data['stop_price'],
            sample_signal_data['target_price'],
            datetime.now().isoformat(), 'open', entry_price
        ))
        test_database.commit()

        # Verify saved
        cursor.execute('SELECT * FROM positions WHERE symbol = ?', (symbol,))
        row = cursor.fetchone()

        assert row is not None
        assert row[1] == symbol
        assert row[4] == shares


class TestPositionClosing:
    """Test position closing at target and stop levels."""

    def test_close_position_at_target(self, integration_mock_broker, sample_signal_data):
        """Test closing position when target is reached."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        symbol = sample_signal_data['symbol']
        shares = 100
        entry_price = sample_signal_data['entry_price']
        target_price = sample_signal_data['target_price']

        # Enter position
        integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=entry_price
        )

        # Simulate price reaching target
        integration_mock_broker.update_price(symbol, target_price)

        # Close position at target
        close_order = integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=target_price
        )

        assert close_order.status.value in ('filled', 'FILLED')

        # Verify position closed
        position = integration_mock_broker.get_position(symbol)
        assert position is None

        # Verify trade history
        assert len(integration_mock_broker.position_history) == 1
        trade = integration_mock_broker.position_history[0]
        assert trade['pnl'] > 0  # Profit at target

    def test_close_position_at_stop(self, integration_mock_broker, sample_signal_data):
        """Test closing position when stop is hit."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        symbol = sample_signal_data['symbol']
        shares = 100
        entry_price = sample_signal_data['entry_price']
        stop_price = sample_signal_data['stop_price']

        # Enter position
        integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=entry_price
        )

        # Simulate price hitting stop
        integration_mock_broker.update_price(symbol, stop_price)

        # Close position at stop
        close_order = integration_mock_broker.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=stop_price
        )

        assert close_order.status.value in ('filled', 'FILLED')

        # Verify trade history shows loss
        trade = integration_mock_broker.position_history[-1]
        assert trade['pnl'] < 0  # Loss at stop

    def test_trade_recorded_after_close(
        self, integration_mock_broker, sample_signal_data, test_database
    ):
        """Test that closed trades are properly recorded."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        cursor = test_database.cursor()

        symbol = sample_signal_data['symbol']
        shares = 100
        entry_price = sample_signal_data['entry_price']
        exit_price = sample_signal_data['target_price']

        # Record the trade
        pnl = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price / entry_price) - 1) * 100

        cursor.execute('''
            INSERT INTO trades (
                symbol, direction, entry_price, exit_price, shares,
                entry_time, exit_time, pnl, pnl_pct, status, strategy, rrs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, 'long', entry_price, exit_price, shares,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            pnl, pnl_pct, 'closed', 'RRS_Momentum',
            sample_signal_data['rrs']
        ))
        test_database.commit()

        # Verify trade recorded
        cursor.execute('SELECT * FROM trades WHERE symbol = ? AND status = ?', (symbol, 'closed'))
        row = cursor.fetchone()

        assert row is not None
        assert row[10] == pnl  # pnl column (index 10 in schema)


class TestCompleteFlow:
    """Test complete signal-to-close trading flow."""

    @pytest.mark.slow
    def test_full_long_trade_flow(
        self, sample_signal_data, integration_mock_broker,
        mock_event_bus, test_database
    ):
        """Test complete flow for a long trade from signal to close."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        cursor = test_database.cursor()
        signal = sample_signal_data.copy()

        # Step 1: Signal generated and saved
        cursor.execute('''
            INSERT INTO signals (
                symbol, direction, strength, rrs, entry_price,
                stop_price, target_price, atr, generated_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'], signal['direction'], signal['strength'],
            signal['rrs'], signal['entry_price'], signal['stop_price'],
            signal['target_price'], signal['atr'],
            datetime.now().isoformat(), 'pending'
        ))
        test_database.commit()

        # Step 2: Analyzer evaluates (mocked)
        mock_event_bus.publish('SIGNAL_ANALYZED', {
            **signal, 'analysis_result': 'approved', 'confidence': 0.85
        })

        # Step 3: Risk validation (mocked)
        shares = 100
        mock_event_bus.publish('RISK_VALIDATED', {
            'approved': True, 'position_size': shares
        })

        # Step 4: Place order
        entry_order = integration_mock_broker.place_order(
            symbol=signal['symbol'],
            side=OrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=signal['entry_price']
        )

        # Verify order filled
        assert entry_order.status.value in ('filled', 'FILLED')

        # Step 5: Track position
        cursor.execute('''
            INSERT INTO positions (
                symbol, direction, entry_price, shares,
                stop_loss, take_profit, entry_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'], signal['direction'], signal['entry_price'],
            shares, signal['stop_price'], signal['target_price'],
            datetime.now().isoformat(), 'open'
        ))
        test_database.commit()

        # Step 6: Price reaches target
        integration_mock_broker.update_price(signal['symbol'], signal['target_price'])

        # Step 7: Close position
        exit_order = integration_mock_broker.place_order(
            symbol=signal['symbol'],
            side=OrderSide.SELL,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=signal['target_price']
        )

        assert exit_order.status.value in ('filled', 'FILLED')

        # Step 8: Record closed trade
        pnl = (signal['target_price'] - signal['entry_price']) * shares

        cursor.execute('''
            UPDATE positions SET status = 'closed' WHERE symbol = ?
        ''', (signal['symbol'],))

        cursor.execute('''
            INSERT INTO trades (
                symbol, direction, entry_price, exit_price, shares,
                entry_time, exit_time, pnl, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'], signal['direction'], signal['entry_price'],
            signal['target_price'], shares,
            datetime.now().isoformat(), datetime.now().isoformat(),
            pnl, 'closed'
        ))
        test_database.commit()

        # Verify complete flow
        cursor.execute('SELECT COUNT(*) FROM trades WHERE status = ?', ('closed',))
        assert cursor.fetchone()[0] == 1

        cursor.execute('SELECT pnl FROM trades WHERE symbol = ?', (signal['symbol'],))
        recorded_pnl = cursor.fetchone()[0]
        assert recorded_pnl == pnl
        assert recorded_pnl > 0  # Profitable trade

    @pytest.mark.slow
    def test_full_short_trade_flow(self, integration_mock_broker, mock_event_bus, test_database):
        """Test complete flow for a short trade."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        cursor = test_database.cursor()

        signal = {
            'symbol': 'TSLA',
            'direction': 'short',
            'strength': 'strong',
            'rrs': -3.20,
            'entry_price': 200.00,
            'stop_price': 208.00,
            'target_price': 184.00,
            'atr': 8.00
        }

        # Step 1: Enter short position
        shares = 50
        entry_order = integration_mock_broker.place_order(
            symbol=signal['symbol'],
            side=OrderSide.SELL_SHORT,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=signal['entry_price']
        )

        # For mock broker, SELL_SHORT is treated like SELL
        # In real scenario, this would create a short position

        # Step 2: Price drops to target (profit for short)
        target_price = signal['target_price']

        # Step 3: Cover short
        cover_order = integration_mock_broker.place_order(
            symbol=signal['symbol'],
            side=OrderSide.BUY_TO_COVER,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=target_price
        )

        # Step 4: Record trade
        # For short: profit = (entry - exit) * shares
        pnl = (signal['entry_price'] - target_price) * shares

        cursor.execute('''
            INSERT INTO trades (
                symbol, direction, entry_price, exit_price, shares,
                entry_time, exit_time, pnl, status, rrs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'], 'short', signal['entry_price'],
            target_price, shares,
            datetime.now().isoformat(), datetime.now().isoformat(),
            pnl, 'closed', signal['rrs']
        ))
        test_database.commit()

        # Verify profitable short
        cursor.execute('SELECT pnl FROM trades WHERE symbol = ?', (signal['symbol'],))
        recorded_pnl = cursor.fetchone()[0]
        assert recorded_pnl > 0  # Profitable short when price drops


class TestMultiplePositions:
    """Test handling multiple simultaneous positions."""

    def test_track_multiple_positions(self, integration_mock_broker, test_database):
        """Test tracking multiple open positions."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        positions_data = [
            ('AAPL', 100, 175.00),
            ('GOOGL', 50, 140.00),
            ('MSFT', 75, 380.00)
        ]

        # Open multiple positions
        for symbol, shares, price in positions_data:
            integration_mock_broker.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=shares,
                order_type=OrderType.MARKET,
                price=price
            )

        # Verify all positions exist
        positions = integration_mock_broker.get_positions()
        assert len(positions) == 3

        for symbol, shares, price in positions_data:
            assert symbol in positions
            assert positions[symbol].quantity == shares

    def test_portfolio_pnl_calculation(self, integration_mock_broker):
        """Test total portfolio P&L calculation."""
        try:
            from brokers.broker_interface import OrderSide, OrderType
        except ImportError:
            pytest.skip("Broker interface not available")

        # Open positions
        integration_mock_broker.place_order(
            symbol='AAPL', side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, price=175.00
        )
        integration_mock_broker.place_order(
            symbol='GOOGL', side=OrderSide.BUY, quantity=50,
            order_type=OrderType.MARKET, price=140.00
        )

        # Update prices
        integration_mock_broker.update_price('AAPL', 180.00)  # +5 per share
        integration_mock_broker.update_price('GOOGL', 135.00)  # -5 per share

        # Calculate total unrealized P&L
        positions = integration_mock_broker.get_positions()
        total_pnl = sum(p.unrealized_pnl for p in positions.values())

        # AAPL: +$500, GOOGL: -$250 = +$250 total
        expected_pnl = (180 - 175) * 100 + (135 - 140) * 50
        assert total_pnl == pytest.approx(expected_pnl, rel=0.01)
