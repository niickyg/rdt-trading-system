"""Add daily_bars table for IBKR historical data cache

Revision ID: 008_add_daily_bars_table
Revises: 007_add_strategy_name_columns
Create Date: 2026-03-05

Stores daily OHLCV bars fetched from IBKR to avoid rate limit issues.
Used by HistoricalBarCache as the primary historical data source.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '008_add_daily_bars_table'
down_revision: Union[str, None] = '007_add_strategy_name_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create daily_bars table."""
    op.create_table(
        'daily_bars',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('bar_date', sa.Date(), nullable=False),
        sa.Column('open', sa.Numeric(12, 4), nullable=False),
        sa.Column('high', sa.Numeric(12, 4), nullable=False),
        sa.Column('low', sa.Numeric(12, 4), nullable=False),
        sa.Column('close', sa.Numeric(12, 4), nullable=False),
        sa.Column('volume', sa.BigInteger(), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'bar_date', name='uq_daily_bars_symbol_date'),
    )
    op.create_index('ix_daily_bars_symbol_date', 'daily_bars', ['symbol', 'bar_date'])


def downgrade() -> None:
    """Drop daily_bars table."""
    op.drop_index('ix_daily_bars_symbol_date', table_name='daily_bars')
    op.drop_table('daily_bars')
