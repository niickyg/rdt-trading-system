"""Add trade monitoring and learning tables

Revision ID: 004_add_monitoring_tables
Revises: 003_add_missing_tables
Create Date: 2026-02-21

Adds tables for:
- rejected_signals: Track rejected signals with full context for outcome analysis
- equity_snapshots: Track equity curve and drawdowns over time
- parameter_changes: Historical record of adaptive learner parameter adjustments
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '004_add_monitoring_tables'
down_revision: Union[str, None] = '003_add_missing_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create monitoring and learning tables."""

    # =========================================================================
    # rejected_signals table
    # =========================================================================
    op.create_table(
        'rejected_signals',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('direction', sa.String(10), nullable=False),
        sa.Column('rrs', sa.Numeric(8, 4), nullable=False),
        sa.Column('price', sa.Numeric(12, 4), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('rejection_reasons', sa.Text(), nullable=False),
        sa.Column('market_regime', sa.String(32), nullable=True),
        sa.Column('daily_strong', sa.Boolean(), nullable=True),
        sa.Column('daily_weak', sa.Boolean(), nullable=True),
        sa.Column('atr', sa.Numeric(12, 4), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('ml_probability', sa.Numeric(8, 4), nullable=True),
        sa.Column('ml_confidence', sa.Numeric(8, 4), nullable=True),
        # Outcome tracking columns (filled later by OutcomeTracker)
        sa.Column('price_after_1h', sa.Numeric(12, 4), nullable=True),
        sa.Column('price_after_4h', sa.Numeric(12, 4), nullable=True),
        sa.Column('price_after_1d', sa.Numeric(12, 4), nullable=True),
        sa.Column('would_have_pnl_1h', sa.Numeric(12, 4), nullable=True),
        sa.Column('would_have_pnl_4h', sa.Numeric(12, 4), nullable=True),
        sa.Column('would_have_pnl_1d', sa.Numeric(12, 4), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_rejected_signals_symbol', 'rejected_signals', ['symbol'])
    op.create_index('ix_rejected_signals_timestamp', 'rejected_signals', ['timestamp'])
    op.create_index('ix_rejected_signals_symbol_timestamp', 'rejected_signals',
                     ['symbol', 'timestamp'])

    # =========================================================================
    # equity_snapshots table
    # =========================================================================
    op.create_table(
        'equity_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('equity_value', sa.Numeric(14, 2), nullable=False),
        sa.Column('cash', sa.Numeric(14, 2), nullable=True),
        sa.Column('positions_value', sa.Numeric(14, 2), nullable=True),
        sa.Column('open_positions_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('drawdown_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('high_water_mark', sa.Numeric(14, 2), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_equity_snapshots_timestamp', 'equity_snapshots', ['timestamp'])

    # =========================================================================
    # parameter_changes table
    # =========================================================================
    op.create_table(
        'parameter_changes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('parameter_name', sa.String(64), nullable=False),
        sa.Column('old_value', sa.Numeric(10, 4), nullable=False),
        sa.Column('new_value', sa.Numeric(10, 4), nullable=False),
        sa.Column('reason', sa.String(255), nullable=False),
        sa.Column('trade_count_basis', sa.Integer(), nullable=True),
        sa.Column('win_rate_at_change', sa.Numeric(5, 4), nullable=True),
        sa.Column('regime', sa.String(32), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_parameter_changes_timestamp', 'parameter_changes', ['timestamp'])
    op.create_index('ix_parameter_changes_parameter_name', 'parameter_changes',
                     ['parameter_name'])


def downgrade() -> None:
    """Drop monitoring and learning tables."""

    op.drop_index('ix_parameter_changes_parameter_name', table_name='parameter_changes')
    op.drop_index('ix_parameter_changes_timestamp', table_name='parameter_changes')
    op.drop_table('parameter_changes')

    op.drop_index('ix_equity_snapshots_timestamp', table_name='equity_snapshots')
    op.drop_table('equity_snapshots')

    op.drop_index('ix_rejected_signals_symbol_timestamp', table_name='rejected_signals')
    op.drop_index('ix_rejected_signals_timestamp', table_name='rejected_signals')
    op.drop_index('ix_rejected_signals_symbol', table_name='rejected_signals')
    op.drop_table('rejected_signals')
