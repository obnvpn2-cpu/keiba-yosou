"""
Database utilities for keiba-yosou.

Provides:
- schema_migration: Schema migrations and safety checks
- upsert: Idempotent UPSERT functions for all tables
"""

from .schema_migration import (
    run_migrations,
    get_schema_status,
    print_schema_status,
    table_exists,
    index_exists,
    clean_duplicates,
)

from .upsert import (
    UpsertHelper,
    upsert_dataframe,
    upsert_feature_table_v2,
    upsert_feature_table_v3,
)

__all__ = [
    # schema_migration
    "run_migrations",
    "get_schema_status",
    "print_schema_status",
    "table_exists",
    "index_exists",
    "clean_duplicates",
    # upsert
    "UpsertHelper",
    "upsert_dataframe",
    "upsert_feature_table_v2",
    "upsert_feature_table_v3",
]
