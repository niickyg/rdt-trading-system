"""Add ML training data tables and MFE/MAE columns

Revision ID: 006_add_ml_training_tables
Revises: 005_add_trade_metadata_columns
Create Date: 2026-02-25

Adds 7 new tables for ML model training data collection:
- intraday_bars: 5-minute OHLCV bars
- technical_indicators: Daily technical indicators
- trade_snapshots: Point-in-time position snapshots for MFE/MAE
- market_regime_daily: Daily market context
- sector_data: Daily sector relative strength
- options_greeks_history: Greeks snapshots during options positions
- earnings_calendar: Earnings dates and surprises

Also adds 8 MFE/MAE columns to the trades table.

Uses batch_alter_table (required for SQLite ALTER TABLE support).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '006_add_ml_training_tables'
down_revision: Union[str, None] = '005_add_trade_metadata_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create ML training tables and add MFE/MAE columns to trades."""

    # 1. intraday_bars
    op.create_table(
        'intraday_bars',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open', sa.Numeric(12, 4), nullable=False),
        sa.Column('high', sa.Numeric(12, 4), nullable=False),
        sa.Column('low', sa.Numeric(12, 4), nullable=False),
        sa.Column('close', sa.Numeric(12, 4), nullable=False),
        sa.Column('volume', sa.BigInteger(), nullable=False),
        sa.Column('vwap', sa.Numeric(12, 4), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timestamp', name='uq_intraday_bars_symbol_timestamp'),
    )
    op.create_index('ix_intraday_bars_symbol_timestamp', 'intraday_bars', ['symbol', 'timestamp'])

    # 2. technical_indicators
    op.create_table(
        'technical_indicators',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('rsi_14', sa.Numeric(8, 4), nullable=True),
        sa.Column('macd_line', sa.Numeric(12, 4), nullable=True),
        sa.Column('macd_signal', sa.Numeric(12, 4), nullable=True),
        sa.Column('macd_histogram', sa.Numeric(12, 4), nullable=True),
        sa.Column('bb_upper', sa.Numeric(12, 4), nullable=True),
        sa.Column('bb_middle', sa.Numeric(12, 4), nullable=True),
        sa.Column('bb_lower', sa.Numeric(12, 4), nullable=True),
        sa.Column('bb_width', sa.Numeric(8, 4), nullable=True),
        sa.Column('ema_9', sa.Numeric(12, 4), nullable=True),
        sa.Column('ema_21', sa.Numeric(12, 4), nullable=True),
        sa.Column('ema_50', sa.Numeric(12, 4), nullable=True),
        sa.Column('ema_200', sa.Numeric(12, 4), nullable=True),
        sa.Column('adx', sa.Numeric(8, 4), nullable=True),
        sa.Column('obv', sa.BigInteger(), nullable=True),
        sa.Column('atr_14', sa.Numeric(12, 4), nullable=True),
        sa.Column('close_price', sa.Numeric(12, 4), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'date', name='uq_technical_indicators_symbol_date'),
    )
    op.create_index('ix_technical_indicators_symbol_date', 'technical_indicators', ['symbol', 'date'])

    # 3. trade_snapshots
    op.create_table(
        'trade_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('trade_id', sa.Integer(), sa.ForeignKey('trades.id', ondelete='CASCADE'), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_price', sa.Numeric(12, 4), nullable=False),
        sa.Column('unrealized_pnl', sa.Numeric(12, 2), nullable=True),
        sa.Column('unrealized_pnl_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('unrealized_r', sa.Numeric(8, 4), nullable=True),
        sa.Column('mfe', sa.Numeric(12, 2), nullable=True),
        sa.Column('mae', sa.Numeric(12, 2), nullable=True),
        sa.Column('mfe_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('mae_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('bars_held', sa.Integer(), nullable=True),
        sa.Column('rsi_at_snapshot', sa.Numeric(8, 4), nullable=True),
        sa.Column('distance_to_stop_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('distance_to_target_pct', sa.Numeric(8, 4), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_trade_snapshots_trade_id_timestamp', 'trade_snapshots', ['trade_id', 'timestamp'])

    # 4. market_regime_daily
    op.create_table(
        'market_regime_daily',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False, unique=True),
        sa.Column('vix_close', sa.Numeric(8, 2), nullable=True),
        sa.Column('vix_regime', sa.String(20), nullable=True),
        sa.Column('spy_close', sa.Numeric(12, 4), nullable=True),
        sa.Column('spy_trend', sa.String(20), nullable=True),
        sa.Column('spy_above_200ema', sa.Boolean(), nullable=True),
        sa.Column('spy_above_50ema', sa.Boolean(), nullable=True),
        sa.Column('advance_decline_ratio', sa.Numeric(8, 4), nullable=True),
        sa.Column('new_highs', sa.Integer(), nullable=True),
        sa.Column('new_lows', sa.Integer(), nullable=True),
        sa.Column('breadth_thrust', sa.Numeric(8, 4), nullable=True),
        sa.Column('put_call_ratio', sa.Numeric(8, 4), nullable=True),
        sa.Column('regime_label', sa.String(30), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_market_regime_daily_date', 'market_regime_daily', ['date'])

    # 5. sector_data
    op.create_table(
        'sector_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('sector', sa.String(30), nullable=False),
        sa.Column('etf_symbol', sa.String(10), nullable=False),
        sa.Column('close_price', sa.Numeric(12, 4), nullable=True),
        sa.Column('daily_return_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('relative_strength_5d', sa.Numeric(8, 4), nullable=True),
        sa.Column('relative_strength_20d', sa.Numeric(8, 4), nullable=True),
        sa.Column('relative_strength_60d', sa.Numeric(8, 4), nullable=True),
        sa.Column('sector_rank', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date', 'sector', name='uq_sector_data_date_sector'),
    )
    op.create_index('ix_sector_data_date_sector', 'sector_data', ['date', 'sector'])

    # 6. options_greeks_history
    op.create_table(
        'options_greeks_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('options_trade_id', sa.Integer(),
                  sa.ForeignKey('options_trades.id', ondelete='SET NULL'), nullable=True),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('underlying_price', sa.Numeric(12, 4), nullable=True),
        sa.Column('delta', sa.Numeric(8, 4), nullable=True),
        sa.Column('gamma', sa.Numeric(8, 6), nullable=True),
        sa.Column('theta', sa.Numeric(8, 4), nullable=True),
        sa.Column('vega', sa.Numeric(8, 4), nullable=True),
        sa.Column('iv', sa.Numeric(8, 4), nullable=True),
        sa.Column('premium', sa.Numeric(12, 4), nullable=True),
        sa.Column('dte', sa.Integer(), nullable=True),
        sa.Column('moneyness', sa.Numeric(8, 4), nullable=True),
        sa.Column('intrinsic_value', sa.Numeric(12, 4), nullable=True),
        sa.Column('extrinsic_value', sa.Numeric(12, 4), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_options_greeks_history_symbol_timestamp', 'options_greeks_history',
                    ['symbol', 'timestamp'])

    # 7. earnings_calendar
    op.create_table(
        'earnings_calendar',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('earnings_date', sa.Date(), nullable=False),
        sa.Column('timing', sa.String(10), nullable=True),
        sa.Column('eps_estimate', sa.Numeric(10, 4), nullable=True),
        sa.Column('eps_actual', sa.Numeric(10, 4), nullable=True),
        sa.Column('eps_surprise_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('revenue_estimate', sa.Numeric(16, 2), nullable=True),
        sa.Column('revenue_actual', sa.Numeric(16, 2), nullable=True),
        sa.Column('revenue_surprise_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('price_change_1d_pct', sa.Numeric(8, 4), nullable=True),
        sa.Column('iv_rank_before', sa.Numeric(8, 4), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'earnings_date', name='uq_earnings_calendar_symbol_date'),
    )
    op.create_index('ix_earnings_calendar_symbol_date', 'earnings_calendar',
                    ['symbol', 'earnings_date'])

    # 8. Add MFE/MAE columns to trades table
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.add_column(sa.Column('peak_mfe', sa.Numeric(12, 2), nullable=True))
        batch_op.add_column(sa.Column('peak_mae', sa.Numeric(12, 2), nullable=True))
        batch_op.add_column(sa.Column('peak_mfe_pct', sa.Numeric(8, 4), nullable=True))
        batch_op.add_column(sa.Column('peak_mae_pct', sa.Numeric(8, 4), nullable=True))
        batch_op.add_column(sa.Column('peak_mfe_r', sa.Numeric(8, 4), nullable=True))
        batch_op.add_column(sa.Column('peak_mae_r', sa.Numeric(8, 4), nullable=True))
        batch_op.add_column(sa.Column('bars_to_mfe', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('bars_held', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Drop ML training tables and remove MFE/MAE columns from trades."""

    # Remove MFE/MAE columns from trades
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.drop_column('bars_held')
        batch_op.drop_column('bars_to_mfe')
        batch_op.drop_column('peak_mae_r')
        batch_op.drop_column('peak_mfe_r')
        batch_op.drop_column('peak_mae_pct')
        batch_op.drop_column('peak_mfe_pct')
        batch_op.drop_column('peak_mae')
        batch_op.drop_column('peak_mfe')

    # Drop tables in reverse order
    op.drop_table('earnings_calendar')
    op.drop_table('options_greeks_history')
    op.drop_table('sector_data')
    op.drop_table('market_regime_daily')
    op.drop_table('trade_snapshots')
    op.drop_table('technical_indicators')
    op.drop_table('intraday_bars')
