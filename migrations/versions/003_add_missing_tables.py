"""Add all missing tables to sync with models.py

Revision ID: 003_add_missing_tables
Revises: 002_add_foreign_keys
Create Date: 2026-02-05

This migration adds all tables defined in models.py that were missing:
- order_executions: Order execution tracking with slippage metrics
- audit_logs: System audit trail
- event_records: Event persistence for recovery
- webhook_events: Stripe webhook deduplication
- market_data_cache: Market data caching
- subscriptions: User subscription management
- payment_history: Payment records
- user_sessions: Session management
- alert_schedules: Alert scheduling preferences
- queued_alerts: Queued alerts for quiet hours

Also updates existing tables with missing columns.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_missing_tables'
down_revision: Union[str, None] = '002_add_foreign_keys'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all missing tables and add missing columns."""

    # Check dialect for database-specific handling
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == 'sqlite'

    # =========================================================================
    # Update existing tables with missing columns
    # =========================================================================

    # Add missing columns to users table
    with op.batch_alter_table('users', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('failed_login_attempts', sa.Integer(), nullable=False, server_default='0'))
        except Exception:
            pass  # Column may already exist
        try:
            batch_op.add_column(sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('stripe_customer_id', sa.String(255), nullable=True, unique=True))
        except Exception:
            pass

    # Add missing columns to api_users table
    with op.batch_alter_table('api_users', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('last_request_at', sa.DateTime(timezone=True), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('request_count', sa.Integer(), nullable=False, server_default='0'))
        except Exception:
            pass

    # Add missing columns to trades table
    with op.batch_alter_table('trades', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('broker_order_id', sa.String(64), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('notes', sa.Text(), nullable=True))
        except Exception:
            pass

    # Add missing columns to positions table
    with op.batch_alter_table('positions', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))
        except Exception:
            pass

    # Add missing columns to signals table
    with op.batch_alter_table('signals', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('volume', sa.BigInteger(), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('market_regime', sa.String(32), nullable=True))
        except Exception:
            pass

    # Add missing columns to daily_stats table
    with op.batch_alter_table('daily_stats', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('pnl_percent', sa.Numeric(8, 4), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('win_rate', sa.Numeric(5, 2), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('avg_win', sa.Numeric(12, 2), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('avg_loss', sa.Numeric(12, 2), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('largest_win', sa.Numeric(12, 2), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('largest_loss', sa.Numeric(12, 2), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('market_regime', sa.String(32), nullable=True))
        except Exception:
            pass

    # Add missing columns to watchlist table
    with op.batch_alter_table('watchlist', schema=None) as batch_op:
        try:
            batch_op.add_column(sa.Column('sector', sa.String(64), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('industry', sa.String(128), nullable=True))
        except Exception:
            pass
        try:
            batch_op.add_column(sa.Column('market_cap', sa.Numeric(16, 2), nullable=True))
        except Exception:
            pass

    # =========================================================================
    # Create new tables
    # =========================================================================

    # Create order_executions table
    op.create_table(
        'order_executions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('order_id', sa.String(64), nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('side', sa.String(20), nullable=False),
        sa.Column('expected_price', sa.Numeric(12, 4), nullable=False),
        sa.Column('fill_price', sa.Numeric(12, 4), nullable=False),
        sa.Column('slippage', sa.Numeric(12, 4), nullable=False, server_default='0'),
        sa.Column('slippage_pct', sa.Numeric(8, 4), nullable=False, server_default='0'),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('filled_quantity', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('order_submitted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('fill_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('time_to_fill_seconds', sa.Numeric(10, 3), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='filled'),
        sa.Column('broker_order_id', sa.String(64), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('quantity > 0', name='ck_order_executions_quantity_positive'),
        sa.CheckConstraint('filled_quantity >= 0', name='ck_order_executions_filled_quantity_non_negative'),
    )
    op.create_index('ix_order_executions_order_id', 'order_executions', ['order_id'], unique=False)
    op.create_index('ix_order_executions_symbol', 'order_executions', ['symbol'], unique=False)
    op.create_index('ix_order_executions_fill_time', 'order_executions', ['fill_time'], unique=False)
    op.create_index('ix_order_executions_status', 'order_executions', ['status'], unique=False)
    op.create_index('ix_order_executions_symbol_fill_time', 'order_executions', ['symbol', 'fill_time'], unique=False)

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('event_type', sa.String(64), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('entity_type', sa.String(64), nullable=True),
        sa.Column('entity_id', sa.String(64), nullable=True),
        sa.Column('action', sa.String(32), nullable=False),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(256), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL') if not is_sqlite else None,
    )
    op.create_index('ix_audit_logs_timestamp', 'audit_logs', ['timestamp'], unique=False)
    op.create_index('ix_audit_logs_event_type', 'audit_logs', ['event_type'], unique=False)
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'], unique=False)
    op.create_index('ix_audit_logs_entity', 'audit_logs', ['entity_type', 'entity_id'], unique=False)

    # Create event_records table
    op.create_table(
        'event_records',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('event_id', sa.String(64), nullable=False, unique=True),
        sa.Column('event_type', sa.String(64), nullable=False),
        sa.Column('source', sa.String(64), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('data', sa.Text(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_event_records_status', 'event_records', ['status'], unique=False)
    op.create_index('ix_event_records_event_type', 'event_records', ['event_type'], unique=False)
    op.create_index('ix_event_records_created_at', 'event_records', ['created_at'], unique=False)
    op.create_index('ix_event_records_status_created', 'event_records', ['status', 'created_at'], unique=False)

    # Create webhook_events table
    op.create_table(
        'webhook_events',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('event_id', sa.String(64), nullable=False, unique=True),
        sa.Column('event_type', sa.String(64), nullable=False),
        sa.Column('source', sa.String(32), nullable=False, server_default='stripe'),
        sa.Column('payload', sa.Text(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('result', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_webhook_events_event_id', 'webhook_events', ['event_id'], unique=False)
    op.create_index('ix_webhook_events_event_type', 'webhook_events', ['event_type'], unique=False)
    op.create_index('ix_webhook_events_status', 'webhook_events', ['status'], unique=False)
    op.create_index('ix_webhook_events_created_at', 'webhook_events', ['created_at'], unique=False)

    # Create market_data_cache table
    op.create_table(
        'market_data_cache',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('data_type', sa.String(32), nullable=False),
        sa.Column('timeframe', sa.String(16), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('data', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'data_type', 'timeframe', 'timestamp', name='uq_market_data_cache'),
    )
    op.create_index('ix_market_data_cache_symbol', 'market_data_cache', ['symbol'], unique=False)
    op.create_index('ix_market_data_cache_expires', 'market_data_cache', ['expires_at'], unique=False)
    op.create_index('ix_market_data_cache_lookup', 'market_data_cache', ['symbol', 'data_type', 'timeframe'], unique=False)

    # Create subscriptions table
    op.create_table(
        'subscriptions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('stripe_subscription_id', sa.String(255), nullable=False, unique=True),
        sa.Column('stripe_customer_id', sa.String(255), nullable=False),
        sa.Column('plan_id', sa.String(50), nullable=False),
        sa.Column('status', sa.String(30), nullable=False, server_default='incomplete'),
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('trial_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('trial_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancel_at_period_end', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('canceled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE') if not is_sqlite else None,
    )
    op.create_index('ix_subscriptions_user_id', 'subscriptions', ['user_id'], unique=False)
    op.create_index('ix_subscriptions_stripe_subscription_id', 'subscriptions', ['stripe_subscription_id'], unique=False)
    op.create_index('ix_subscriptions_stripe_customer_id', 'subscriptions', ['stripe_customer_id'], unique=False)
    op.create_index('ix_subscriptions_status', 'subscriptions', ['status'], unique=False)
    op.create_index('ix_subscriptions_plan_id', 'subscriptions', ['plan_id'], unique=False)
    op.create_index('ix_subscriptions_user_status', 'subscriptions', ['user_id', 'status'], unique=False)

    # Create payment_history table
    op.create_table(
        'payment_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('subscription_id', sa.Integer(), nullable=True),
        sa.Column('stripe_invoice_id', sa.String(255), nullable=True),
        sa.Column('stripe_payment_intent_id', sa.String(255), nullable=True),
        sa.Column('amount', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False, server_default='usd'),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('description', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('paid_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE') if not is_sqlite else None,
        sa.ForeignKeyConstraint(['subscription_id'], ['subscriptions.id'], ondelete='SET NULL') if not is_sqlite else None,
    )
    op.create_index('ix_payment_history_user_id', 'payment_history', ['user_id'], unique=False)
    op.create_index('ix_payment_history_subscription_id', 'payment_history', ['subscription_id'], unique=False)
    op.create_index('ix_payment_history_stripe_invoice_id', 'payment_history', ['stripe_invoice_id'], unique=False)
    op.create_index('ix_payment_history_status', 'payment_history', ['status'], unique=False)
    op.create_index('ix_payment_history_created_at', 'payment_history', ['created_at'], unique=False)

    # Create user_sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('session_token', sa.String(128), nullable=False, unique=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('device_info', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_activity', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE') if not is_sqlite else None,
    )
    op.create_index('ix_user_sessions_user_id', 'user_sessions', ['user_id'], unique=False)
    op.create_index('ix_user_sessions_session_token', 'user_sessions', ['session_token'], unique=False)
    op.create_index('ix_user_sessions_is_active', 'user_sessions', ['is_active'], unique=False)
    op.create_index('ix_user_sessions_user_active', 'user_sessions', ['user_id', 'is_active'], unique=False)
    op.create_index('ix_user_sessions_last_activity', 'user_sessions', ['last_activity'], unique=False)

    # Create alert_schedules table
    op.create_table(
        'alert_schedules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False, server_default='Default'),
        sa.Column('start_time', sa.String(5), nullable=False),
        sa.Column('end_time', sa.String(5), nullable=False),
        sa.Column('timezone', sa.String(50), nullable=False, server_default='America/New_York'),
        sa.Column('days_of_week', sa.String(100), nullable=True),
        sa.Column('channels', sa.String(200), nullable=True),
        sa.Column('alert_types', sa.String(200), nullable=True),
        sa.Column('override_critical', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('priority_threshold', sa.String(20), nullable=False, server_default='critical'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE') if not is_sqlite else None,
    )
    op.create_index('ix_alert_schedules_user_id', 'alert_schedules', ['user_id'], unique=False)
    op.create_index('ix_alert_schedules_is_active', 'alert_schedules', ['is_active'], unique=False)
    op.create_index('ix_alert_schedules_user_active', 'alert_schedules', ['user_id', 'is_active'], unique=False)

    # Create queued_alerts table
    op.create_table(
        'queued_alerts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('alert_id', sa.String(64), nullable=False, unique=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('priority', sa.String(20), nullable=False, server_default='normal'),
        sa.Column('alert_type', sa.String(50), nullable=False, server_default='general'),
        sa.Column('channels', sa.String(200), nullable=False),
        sa.Column('extra_data', sa.Text(), nullable=True),
        sa.Column('queued_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('scheduled_for', sa.DateTime(timezone=True), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_attempt_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL') if not is_sqlite else None,
    )
    op.create_index('ix_queued_alerts_alert_id', 'queued_alerts', ['alert_id'], unique=False)
    op.create_index('ix_queued_alerts_user_id', 'queued_alerts', ['user_id'], unique=False)
    op.create_index('ix_queued_alerts_status', 'queued_alerts', ['status'], unique=False)
    op.create_index('ix_queued_alerts_scheduled_for', 'queued_alerts', ['scheduled_for'], unique=False)
    op.create_index('ix_queued_alerts_user_status', 'queued_alerts', ['user_id', 'status'], unique=False)


def downgrade() -> None:
    """Remove all created tables and added columns."""

    # Drop new tables in reverse order (respecting foreign key dependencies)
    op.drop_index('ix_queued_alerts_user_status', table_name='queued_alerts')
    op.drop_index('ix_queued_alerts_scheduled_for', table_name='queued_alerts')
    op.drop_index('ix_queued_alerts_status', table_name='queued_alerts')
    op.drop_index('ix_queued_alerts_user_id', table_name='queued_alerts')
    op.drop_index('ix_queued_alerts_alert_id', table_name='queued_alerts')
    op.drop_table('queued_alerts')

    op.drop_index('ix_alert_schedules_user_active', table_name='alert_schedules')
    op.drop_index('ix_alert_schedules_is_active', table_name='alert_schedules')
    op.drop_index('ix_alert_schedules_user_id', table_name='alert_schedules')
    op.drop_table('alert_schedules')

    op.drop_index('ix_user_sessions_last_activity', table_name='user_sessions')
    op.drop_index('ix_user_sessions_user_active', table_name='user_sessions')
    op.drop_index('ix_user_sessions_is_active', table_name='user_sessions')
    op.drop_index('ix_user_sessions_session_token', table_name='user_sessions')
    op.drop_index('ix_user_sessions_user_id', table_name='user_sessions')
    op.drop_table('user_sessions')

    op.drop_index('ix_payment_history_created_at', table_name='payment_history')
    op.drop_index('ix_payment_history_status', table_name='payment_history')
    op.drop_index('ix_payment_history_stripe_invoice_id', table_name='payment_history')
    op.drop_index('ix_payment_history_subscription_id', table_name='payment_history')
    op.drop_index('ix_payment_history_user_id', table_name='payment_history')
    op.drop_table('payment_history')

    op.drop_index('ix_subscriptions_user_status', table_name='subscriptions')
    op.drop_index('ix_subscriptions_plan_id', table_name='subscriptions')
    op.drop_index('ix_subscriptions_status', table_name='subscriptions')
    op.drop_index('ix_subscriptions_stripe_customer_id', table_name='subscriptions')
    op.drop_index('ix_subscriptions_stripe_subscription_id', table_name='subscriptions')
    op.drop_index('ix_subscriptions_user_id', table_name='subscriptions')
    op.drop_table('subscriptions')

    op.drop_index('ix_market_data_cache_lookup', table_name='market_data_cache')
    op.drop_index('ix_market_data_cache_expires', table_name='market_data_cache')
    op.drop_index('ix_market_data_cache_symbol', table_name='market_data_cache')
    op.drop_table('market_data_cache')

    op.drop_index('ix_webhook_events_created_at', table_name='webhook_events')
    op.drop_index('ix_webhook_events_status', table_name='webhook_events')
    op.drop_index('ix_webhook_events_event_type', table_name='webhook_events')
    op.drop_index('ix_webhook_events_event_id', table_name='webhook_events')
    op.drop_table('webhook_events')

    op.drop_index('ix_event_records_status_created', table_name='event_records')
    op.drop_index('ix_event_records_created_at', table_name='event_records')
    op.drop_index('ix_event_records_event_type', table_name='event_records')
    op.drop_index('ix_event_records_status', table_name='event_records')
    op.drop_table('event_records')

    op.drop_index('ix_audit_logs_entity', table_name='audit_logs')
    op.drop_index('ix_audit_logs_user_id', table_name='audit_logs')
    op.drop_index('ix_audit_logs_event_type', table_name='audit_logs')
    op.drop_index('ix_audit_logs_timestamp', table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index('ix_order_executions_symbol_fill_time', table_name='order_executions')
    op.drop_index('ix_order_executions_status', table_name='order_executions')
    op.drop_index('ix_order_executions_fill_time', table_name='order_executions')
    op.drop_index('ix_order_executions_symbol', table_name='order_executions')
    op.drop_index('ix_order_executions_order_id', table_name='order_executions')
    op.drop_table('order_executions')

    # Drop added columns from existing tables
    with op.batch_alter_table('watchlist', schema=None) as batch_op:
        try:
            batch_op.drop_column('market_cap')
            batch_op.drop_column('industry')
            batch_op.drop_column('sector')
        except Exception:
            pass

    with op.batch_alter_table('daily_stats', schema=None) as batch_op:
        try:
            batch_op.drop_column('market_regime')
            batch_op.drop_column('largest_loss')
            batch_op.drop_column('largest_win')
            batch_op.drop_column('avg_loss')
            batch_op.drop_column('avg_win')
            batch_op.drop_column('win_rate')
            batch_op.drop_column('pnl_percent')
        except Exception:
            pass

    with op.batch_alter_table('signals', schema=None) as batch_op:
        try:
            batch_op.drop_column('market_regime')
            batch_op.drop_column('volume')
        except Exception:
            pass

    with op.batch_alter_table('positions', schema=None) as batch_op:
        try:
            batch_op.drop_column('updated_at')
        except Exception:
            pass

    with op.batch_alter_table('trades', schema=None) as batch_op:
        try:
            batch_op.drop_column('notes')
            batch_op.drop_column('broker_order_id')
        except Exception:
            pass

    with op.batch_alter_table('api_users', schema=None) as batch_op:
        try:
            batch_op.drop_column('request_count')
            batch_op.drop_column('last_request_at')
        except Exception:
            pass

    with op.batch_alter_table('users', schema=None) as batch_op:
        try:
            batch_op.drop_column('stripe_customer_id')
            batch_op.drop_column('locked_until')
            batch_op.drop_column('failed_login_attempts')
        except Exception:
            pass
