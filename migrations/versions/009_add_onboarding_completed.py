"""Add onboarding_completed column to users table

Revision ID: 009_add_onboarding_completed
Revises: 008_add_daily_bars_table
Create Date: 2026-03-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '009_add_onboarding_completed'
down_revision: Union[str, None] = '008_add_daily_bars_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add onboarding_completed boolean to users."""
    op.add_column('users', sa.Column('onboarding_completed', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    """Remove onboarding_completed from users."""
    op.drop_column('users', 'onboarding_completed')
