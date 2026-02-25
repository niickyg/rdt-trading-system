"""Add foreign key constraints and indexes

Revision ID: 002_add_foreign_keys
Revises: 001_initial_schema
Create Date: 2026-02-02

This migration adds foreign key constraints to all tables that reference users:
- subscriptions.user_id -> users.id (CASCADE)
- payment_history.user_id -> users.id (CASCADE)
- payment_history.subscription_id -> subscriptions.id (SET NULL)
- user_sessions.user_id -> users.id (CASCADE)
- alert_schedules.user_id -> users.id (CASCADE)
- audit_logs.user_id -> users.id (SET NULL)
- queued_alerts.user_id -> users.id (SET NULL)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_add_foreign_keys'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add foreign key constraints and ensure indexes exist."""

    # Note: SQLite has limited ALTER TABLE support for foreign keys
    # These operations work with PostgreSQL and MySQL
    # For SQLite, foreign keys are enforced at table creation time

    # Check if we're using SQLite (limited FK support via ALTER TABLE)
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == 'sqlite'

    if is_sqlite:
        # SQLite: Foreign keys must be defined at table creation
        # For existing SQLite databases, we just add indexes
        _create_indexes_only()
    else:
        # PostgreSQL/MySQL: Full FK support via ALTER TABLE
        _create_foreign_keys()
        _create_indexes_only()


def _create_foreign_keys():
    """Create foreign key constraints (PostgreSQL/MySQL only)."""

    # subscriptions.user_id -> users.id
    with op.batch_alter_table('subscriptions', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_subscriptions_user_id',
            'users',
            ['user_id'],
            ['id'],
            ondelete='CASCADE'
        )

    # payment_history.user_id -> users.id
    with op.batch_alter_table('payment_history', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_payment_history_user_id',
            'users',
            ['user_id'],
            ['id'],
            ondelete='CASCADE'
        )
        batch_op.create_foreign_key(
            'fk_payment_history_subscription_id',
            'subscriptions',
            ['subscription_id'],
            ['id'],
            ondelete='SET NULL'
        )

    # user_sessions.user_id -> users.id
    with op.batch_alter_table('user_sessions', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_user_sessions_user_id',
            'users',
            ['user_id'],
            ['id'],
            ondelete='CASCADE'
        )

    # alert_schedules.user_id -> users.id
    with op.batch_alter_table('alert_schedules', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_alert_schedules_user_id',
            'users',
            ['user_id'],
            ['id'],
            ondelete='CASCADE'
        )

    # audit_logs.user_id -> users.id (SET NULL)
    with op.batch_alter_table('audit_logs', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_audit_logs_user_id',
            'users',
            ['user_id'],
            ['id'],
            ondelete='SET NULL'
        )

    # queued_alerts.user_id -> users.id (SET NULL)
    with op.batch_alter_table('queued_alerts', schema=None) as batch_op:
        batch_op.create_foreign_key(
            'fk_queued_alerts_user_id',
            'users',
            ['user_id'],
            ['id'],
            ondelete='SET NULL'
        )


def _create_indexes_only():
    """Create indexes on foreign key columns (all databases)."""

    # Add indexes if they don't exist
    # Using execute() to handle 'IF NOT EXISTS' which isn't standard in all DBs

    indexes = [
        ('ix_subscriptions_user_id_fk', 'subscriptions', ['user_id']),
        ('ix_payment_history_user_id_fk', 'payment_history', ['user_id']),
        ('ix_payment_history_subscription_id_fk', 'payment_history', ['subscription_id']),
        ('ix_user_sessions_user_id_fk', 'user_sessions', ['user_id']),
        ('ix_alert_schedules_user_id_fk', 'alert_schedules', ['user_id']),
        ('ix_audit_logs_user_id_fk', 'audit_logs', ['user_id']),
        ('ix_queued_alerts_user_id_fk', 'queued_alerts', ['user_id']),
    ]

    for index_name, table_name, columns in indexes:
        try:
            op.create_index(index_name, table_name, columns, unique=False)
        except Exception:
            # Index may already exist
            pass


def downgrade() -> None:
    """Remove foreign key constraints and indexes."""

    bind = op.get_bind()
    is_sqlite = bind.dialect.name == 'sqlite'

    # Drop indexes
    indexes = [
        ('ix_subscriptions_user_id_fk', 'subscriptions'),
        ('ix_payment_history_user_id_fk', 'payment_history'),
        ('ix_payment_history_subscription_id_fk', 'payment_history'),
        ('ix_user_sessions_user_id_fk', 'user_sessions'),
        ('ix_alert_schedules_user_id_fk', 'alert_schedules'),
        ('ix_audit_logs_user_id_fk', 'audit_logs'),
        ('ix_queued_alerts_user_id_fk', 'queued_alerts'),
    ]

    for index_name, table_name in indexes:
        try:
            op.drop_index(index_name, table_name=table_name)
        except Exception:
            pass

    if not is_sqlite:
        # Drop foreign keys (PostgreSQL/MySQL only)
        with op.batch_alter_table('subscriptions', schema=None) as batch_op:
            batch_op.drop_constraint('fk_subscriptions_user_id', type_='foreignkey')

        with op.batch_alter_table('payment_history', schema=None) as batch_op:
            batch_op.drop_constraint('fk_payment_history_user_id', type_='foreignkey')
            batch_op.drop_constraint('fk_payment_history_subscription_id', type_='foreignkey')

        with op.batch_alter_table('user_sessions', schema=None) as batch_op:
            batch_op.drop_constraint('fk_user_sessions_user_id', type_='foreignkey')

        with op.batch_alter_table('alert_schedules', schema=None) as batch_op:
            batch_op.drop_constraint('fk_alert_schedules_user_id', type_='foreignkey')

        with op.batch_alter_table('audit_logs', schema=None) as batch_op:
            batch_op.drop_constraint('fk_audit_logs_user_id', type_='foreignkey')

        with op.batch_alter_table('queued_alerts', schema=None) as batch_op:
            batch_op.drop_constraint('fk_queued_alerts_user_id', type_='foreignkey')
