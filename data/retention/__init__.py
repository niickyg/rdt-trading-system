"""
Data Retention Module for RDT Trading System.

Provides data retention policy management, archiving, and cleanup functionality.
"""

from .config import RetentionConfig, get_retention_config
from .policies import (
    RetentionPolicy,
    DataType,
    RetentionAction,
    get_default_policies,
    get_policy_for_data_type,
)
from .archiver import DataArchiver, ArchiveFormat, ArchiveManifest
from .cleaner import DataCleaner, CleanupResult, CleanupProgress

__all__ = [
    # Config
    'RetentionConfig',
    'get_retention_config',
    # Policies
    'RetentionPolicy',
    'DataType',
    'RetentionAction',
    'get_default_policies',
    'get_policy_for_data_type',
    # Archiver
    'DataArchiver',
    'ArchiveFormat',
    'ArchiveManifest',
    # Cleaner
    'DataCleaner',
    'CleanupResult',
    'CleanupProgress',
]
