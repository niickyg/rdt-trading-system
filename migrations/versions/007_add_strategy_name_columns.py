"""Add strategy_name columns to core trading tables

Revision ID: 007_add_strategy_name_columns
Revises: 006_add_ml_training_tables
Create Date: 2026-03-04

Adds strategy_name column to signals, rejected_signals, trades, and positions
tables to support multi-strategy architecture. Defaults to 'rrs_momentum' for
existing data.

Uses batch_alter_table (required for SQLite ALTER TABLE support).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '007_add_strategy_name_columns'
down_revision: Union[str, None] = '006_add_ml_training_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add strategy_name column to core trading tables."""
    # Signals table
    with op.batch_alter_table('signals', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('strategy_name', sa.String(50), nullable=True, server_default='rrs_momentum')
        )
        batch_op.create_index('ix_signals_strategy_name', ['strategy_name'])

    # Rejected signals table
    with op.batch_alter_table('rejected_signals', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('strategy_name', sa.String(50), nullable=True, server_default='rrs_momentum')
        )
        batch_op.create_index('ix_rejected_signals_strategy_name', ['strategy_name'])

    # Trades table
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('strategy_name', sa.String(50), nullable=True, server_default='rrs_momentum')
        )
        batch_op.create_index('ix_trades_strategy_name', ['strategy_name'])

    # Positions table
    with op.batch_alter_table('positions', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('strategy_name', sa.String(50), nullable=True, server_default='rrs_momentum')
        )
        batch_op.create_index('ix_positions_strategy_name', ['strategy_name'])


def downgrade() -> None:
    """Remove strategy_name columns."""
    with op.batch_alter_table('positions', schema=None) as batch_op:
        batch_op.drop_index('ix_positions_strategy_name')
        batch_op.drop_column('strategy_name')

    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.drop_index('ix_trades_strategy_name')
        batch_op.drop_column('strategy_name')

    with op.batch_alter_table('rejected_signals', schema=None) as batch_op:
        batch_op.drop_index('ix_rejected_signals_strategy_name')
        batch_op.drop_column('strategy_name')

    with op.batch_alter_table('signals', schema=None) as batch_op:
        batch_op.drop_index('ix_signals_strategy_name')
        batch_op.drop_column('strategy_name')
