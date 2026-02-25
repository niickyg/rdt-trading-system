"""Add trade metadata columns for filter evaluation

Revision ID: 005_add_trade_metadata_columns
Revises: 004_add_monitoring_tables
Create Date: 2026-02-24

Adds the 16 metadata columns defined in Trade model (models.py) that are
missing from the actual SQLite table, causing every save_trade() call to
fail with "no column named vix_regime".

Uses batch_alter_table (required for SQLite ALTER TABLE support).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005_add_trade_metadata_columns'
down_revision: Union[str, None] = '004_add_monitoring_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add missing trade metadata columns."""
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.add_column(sa.Column('vix_regime', sa.String(20), nullable=True))
        batch_op.add_column(sa.Column('vix_value', sa.Numeric(8, 2), nullable=True))
        batch_op.add_column(sa.Column('market_regime', sa.String(30), nullable=True))
        batch_op.add_column(sa.Column('sector_name', sa.String(30), nullable=True))
        batch_op.add_column(sa.Column('sector_rs', sa.Numeric(8, 4), nullable=True))
        batch_op.add_column(sa.Column('spy_trend', sa.String(20), nullable=True))
        batch_op.add_column(sa.Column('ml_confidence', sa.Numeric(8, 4), nullable=True))
        batch_op.add_column(sa.Column('signal_strategy', sa.String(30), nullable=True))
        batch_op.add_column(sa.Column('news_sentiment', sa.Numeric(6, 3), nullable=True))
        batch_op.add_column(sa.Column('news_warning', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('regime_rrs_threshold', sa.Numeric(8, 4), nullable=True))
        batch_op.add_column(sa.Column('regime_stop_multiplier', sa.Numeric(6, 3), nullable=True))
        batch_op.add_column(sa.Column('regime_target_multiplier', sa.Numeric(6, 3), nullable=True))
        batch_op.add_column(sa.Column('vix_position_size_mult', sa.Numeric(6, 3), nullable=True))
        batch_op.add_column(sa.Column('sector_boost', sa.Numeric(6, 3), nullable=True))
        batch_op.add_column(sa.Column('first_hour_filtered', sa.Boolean(), nullable=True, default=False))
        batch_op.create_index('ix_trades_market_regime', ['market_regime'])
        batch_op.create_index('ix_trades_vix_regime', ['vix_regime'])


def downgrade() -> None:
    """Remove trade metadata columns."""
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.drop_index('ix_trades_vix_regime')
        batch_op.drop_index('ix_trades_market_regime')
        batch_op.drop_column('first_hour_filtered')
        batch_op.drop_column('sector_boost')
        batch_op.drop_column('vix_position_size_mult')
        batch_op.drop_column('regime_target_multiplier')
        batch_op.drop_column('regime_stop_multiplier')
        batch_op.drop_column('regime_rrs_threshold')
        batch_op.drop_column('news_warning')
        batch_op.drop_column('news_sentiment')
        batch_op.drop_column('signal_strategy')
        batch_op.drop_column('ml_confidence')
        batch_op.drop_column('spy_trend')
        batch_op.drop_column('sector_rs')
        batch_op.drop_column('sector_name')
        batch_op.drop_column('market_regime')
        batch_op.drop_column('vix_value')
        batch_op.drop_column('vix_regime')
