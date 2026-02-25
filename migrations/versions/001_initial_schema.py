"""Initial schema - create all tables

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-01-30

This migration creates all initial tables for the RDT Trading System:
- users: Dashboard user authentication
- api_users: API user authentication and access control
- trades: Trade history and tracking
- positions: Current open positions
- signals: Trading signals from the scanner
- daily_stats: Daily performance statistics
- watchlist: Symbols being monitored
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.String(length=80), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('password_hash', sa.String(length=256), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_admin', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # Create api_users table
    op.create_table(
        'api_users',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('api_key', sa.String(length=64), nullable=False),
        sa.Column('api_secret_hash', sa.String(length=128), nullable=True),
        sa.Column('tier', sa.String(length=20), nullable=False, default='free'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('rate_limit', sa.Integer(), nullable=False, default=100),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('api_key'),
    )
    op.create_index('ix_api_users_email', 'api_users', ['email'], unique=True)
    op.create_index('ix_api_users_api_key', 'api_users', ['api_key'], unique=True)

    # Create trades table
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('exit_price', sa.Float(), nullable=True),
        sa.Column('shares', sa.Integer(), nullable=False),
        sa.Column('entry_time', sa.DateTime(), nullable=False),
        sa.Column('exit_time', sa.DateTime(), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('pnl_percent', sa.Float(), nullable=True),
        sa.Column('rrs_at_entry', sa.Float(), nullable=True),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True, default='open'),
        sa.Column('exit_reason', sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('shares > 0', name='ck_trades_shares_positive'),
    )
    op.create_index('ix_trades_symbol', 'trades', ['symbol'], unique=False)
    op.create_index('ix_trades_symbol_entry_time', 'trades', ['symbol', 'entry_time'], unique=False)

    # Create positions table
    op.create_table(
        'positions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('shares', sa.Integer(), nullable=False),
        sa.Column('entry_time', sa.DateTime(), nullable=False),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('unrealized_pnl', sa.Float(), nullable=True),
        sa.Column('rrs_at_entry', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol'),
    )
    op.create_index('ix_positions_symbol', 'positions', ['symbol'], unique=True)

    # Create signals table
    op.create_table(
        'signals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('rrs', sa.Float(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True, default='pending'),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('atr', sa.Float(), nullable=True),
        sa.Column('daily_strong', sa.Boolean(), nullable=True),
        sa.Column('daily_weak', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_signals_symbol', 'signals', ['symbol'], unique=False)
    op.create_index('ix_signals_timestamp', 'signals', ['timestamp'], unique=False)

    # Create daily_stats table
    op.create_table(
        'daily_stats',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('starting_balance', sa.Float(), nullable=False),
        sa.Column('ending_balance', sa.Float(), nullable=False),
        sa.Column('pnl', sa.Float(), nullable=False, default=0.0),
        sa.Column('num_trades', sa.Integer(), nullable=False, default=0),
        sa.Column('winners', sa.Integer(), nullable=False, default=0),
        sa.Column('losers', sa.Integer(), nullable=False, default=0),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date'),
    )
    op.create_index('ix_daily_stats_date', 'daily_stats', ['date'], unique=True)

    # Create watchlist table
    op.create_table(
        'watchlist',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('added_date', sa.Date(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, default=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol'),
    )
    op.create_index('ix_watchlist_symbol', 'watchlist', ['symbol'], unique=True)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('ix_watchlist_symbol', table_name='watchlist')
    op.drop_table('watchlist')

    op.drop_index('ix_daily_stats_date', table_name='daily_stats')
    op.drop_table('daily_stats')

    op.drop_index('ix_signals_timestamp', table_name='signals')
    op.drop_index('ix_signals_symbol', table_name='signals')
    op.drop_table('signals')

    op.drop_index('ix_positions_symbol', table_name='positions')
    op.drop_table('positions')

    op.drop_index('ix_trades_symbol_entry_time', table_name='trades')
    op.drop_index('ix_trades_symbol', table_name='trades')
    op.drop_table('trades')

    op.drop_index('ix_api_users_api_key', table_name='api_users')
    op.drop_index('ix_api_users_email', table_name='api_users')
    op.drop_table('api_users')

    op.drop_index('ix_users_email', table_name='users')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_table('users')
