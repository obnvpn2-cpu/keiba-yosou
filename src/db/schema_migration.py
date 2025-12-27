#!/usr/bin/env python3
"""
schema_migration.py - Database Schema Migration & Safety

Provides idempotent schema migration for keiba-yosou database.
Ensures all tables have proper PRIMARY KEY / UNIQUE constraints.

Key Features:
- Add UNIQUE INDEX on feature_table_v2 and feature_table_v3
- Clean up duplicates before adding UNIQUE constraint
- Idempotent (safe to run multiple times)
- No sqlite3 CLI required (pure Python)

Usage:
    from src.db.schema_migration import run_migrations
    run_migrations(conn)

    # Or CLI
    python -m src.db.schema_migration --db netkeiba.db
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# Migration Definitions
# ============================================================

MIGRATIONS = [
    {
        "id": "001_unique_index_feature_table_v2",
        "description": "Add UNIQUE INDEX on feature_table_v2 (race_id, horse_id)",
        "up": """
            -- First, clean up duplicates keeping the latest row
            DELETE FROM feature_table_v2
            WHERE rowid NOT IN (
                SELECT MAX(rowid) FROM feature_table_v2
                GROUP BY race_id, horse_id
            );

            -- Then create UNIQUE INDEX
            CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_table_v2_pk
            ON feature_table_v2 (race_id, horse_id);
        """,
        "check": """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='index'
            AND name='idx_feature_table_v2_pk';
        """,
    },
    {
        "id": "002_unique_index_feature_table_v3",
        "description": "Add UNIQUE INDEX on feature_table_v3 (race_id, horse_id)",
        "up": """
            -- First, clean up duplicates keeping the latest row
            DELETE FROM feature_table_v3
            WHERE rowid NOT IN (
                SELECT MAX(rowid) FROM feature_table_v3
                GROUP BY race_id, horse_id
            );

            -- Then create UNIQUE INDEX
            CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_table_v3_pk
            ON feature_table_v3 (race_id, horse_id);
        """,
        "check": """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='index'
            AND name='idx_feature_table_v3_pk';
        """,
    },
    {
        "id": "003_indexes_horse_results",
        "description": "Add indexes on horse_results for performance",
        "up": """
            CREATE INDEX IF NOT EXISTS idx_horse_results_race_id ON horse_results(race_id);
            CREATE INDEX IF NOT EXISTS idx_horse_results_race_date ON horse_results(race_date);
        """,
        "check": """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='index'
            AND name IN ('idx_horse_results_race_id', 'idx_horse_results_race_date');
        """,
    },
    {
        "id": "004_migration_tracking_table",
        "description": "Create migrations tracking table",
        "up": """
            CREATE TABLE IF NOT EXISTS _migrations (
                id TEXT PRIMARY KEY,
                description TEXT,
                applied_at TEXT DEFAULT (datetime('now', 'localtime'))
            );
        """,
        "check": """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='table'
            AND name='_migrations';
        """,
    },
    {
        "id": "005_odds_snapshots_table",
        "description": "Create odds_snapshots table for pre-day cutoff evaluation",
        "up": """
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                race_id TEXT NOT NULL,
                horse_no INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                win_odds REAL,
                popularity INTEGER,
                source TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (race_id, horse_no, observed_at)
            );

            -- Index for efficient lookups by race_id and observed_at
            CREATE INDEX IF NOT EXISTS idx_odds_snapshots_race_observed
            ON odds_snapshots (race_id, observed_at);
        """,
        "check": """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='table'
            AND name='odds_snapshots';
        """,
    },
]


# ============================================================
# Migration Functions
# ============================================================


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone()[0] > 0


def index_exists(conn: sqlite3.Connection, index_name: str) -> bool:
    """Check if an index exists in the database."""
    cursor = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,)
    )
    return cursor.fetchone()[0] > 0


def get_applied_migrations(conn: sqlite3.Connection) -> set:
    """Get set of already applied migration IDs."""
    if not table_exists(conn, "_migrations"):
        return set()

    cursor = conn.execute("SELECT id FROM _migrations")
    return {row[0] for row in cursor.fetchall()}


def record_migration(conn: sqlite3.Connection, migration_id: str, description: str) -> None:
    """Record that a migration has been applied."""
    conn.execute(
        "INSERT OR REPLACE INTO _migrations (id, description, applied_at) VALUES (?, ?, ?)",
        (migration_id, description, datetime.now().isoformat())
    )


def count_duplicates(conn: sqlite3.Connection, table_name: str, key_cols: List[str]) -> int:
    """Count duplicate rows based on key columns."""
    if not table_exists(conn, table_name):
        return 0

    key_col_str = ", ".join(key_cols)
    sql = f"""
        SELECT SUM(cnt - 1) FROM (
            SELECT COUNT(*) as cnt
            FROM {table_name}
            GROUP BY {key_col_str}
            HAVING COUNT(*) > 1
        )
    """
    cursor = conn.execute(sql)
    result = cursor.fetchone()[0]
    return result if result else 0


def clean_duplicates(
    conn: sqlite3.Connection,
    table_name: str,
    key_cols: List[str],
    keep: str = "last"  # "first" or "last"
) -> int:
    """
    Remove duplicates from a table, keeping first or last row.

    Args:
        conn: SQLite connection
        table_name: Table to clean
        key_cols: Columns that define uniqueness
        keep: "first" keeps lowest rowid, "last" keeps highest rowid

    Returns:
        Number of rows deleted
    """
    if not table_exists(conn, table_name):
        logger.warning(f"Table {table_name} does not exist, skipping duplicate cleanup")
        return 0

    # Count before
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    count_before = cursor.fetchone()[0]

    # Count duplicates
    dup_count = count_duplicates(conn, table_name, key_cols)
    if dup_count == 0:
        logger.info(f"No duplicates found in {table_name}")
        return 0

    logger.info(f"Found {dup_count:,} duplicate rows in {table_name}")

    key_col_str = ", ".join(key_cols)
    agg_func = "MAX" if keep == "last" else "MIN"

    sql = f"""
        DELETE FROM {table_name}
        WHERE rowid NOT IN (
            SELECT {agg_func}(rowid) FROM {table_name}
            GROUP BY {key_col_str}
        )
    """
    conn.execute(sql)
    conn.commit()

    # Count after
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    count_after = cursor.fetchone()[0]
    deleted = count_before - count_after

    logger.info(f"Deleted {deleted:,} duplicate rows from {table_name}")
    return deleted


def create_unique_index_safe(
    conn: sqlite3.Connection,
    table_name: str,
    index_name: str,
    key_cols: List[str],
    clean_duplicates_first: bool = True
) -> bool:
    """
    Create a UNIQUE INDEX safely by cleaning duplicates first.

    Args:
        conn: SQLite connection
        table_name: Table name
        index_name: Index name to create
        key_cols: Columns for the unique constraint
        clean_duplicates_first: If True, clean duplicates before creating index

    Returns:
        True if index was created, False if already existed or table doesn't exist
    """
    if not table_exists(conn, table_name):
        logger.warning(f"Table {table_name} does not exist, skipping index creation")
        return False

    if index_exists(conn, index_name):
        logger.info(f"Index {index_name} already exists")
        return False

    if clean_duplicates_first:
        clean_duplicates(conn, table_name, key_cols, keep="last")

    key_col_str = ", ".join(key_cols)
    sql = f"CREATE UNIQUE INDEX {index_name} ON {table_name} ({key_col_str})"

    try:
        conn.execute(sql)
        conn.commit()
        logger.info(f"Created UNIQUE INDEX {index_name} on {table_name}({key_col_str})")
        return True
    except sqlite3.IntegrityError as e:
        logger.error(f"Failed to create UNIQUE INDEX {index_name}: {e}")
        logger.error("There may still be duplicates. Run clean_duplicates() first.")
        raise


def run_migration(conn: sqlite3.Connection, migration: Dict[str, Any]) -> bool:
    """
    Run a single migration if not already applied.

    Returns:
        True if migration was applied, False if already applied or skipped
    """
    migration_id = migration["id"]
    description = migration["description"]

    applied = get_applied_migrations(conn)
    if migration_id in applied:
        logger.debug(f"Migration {migration_id} already applied, skipping")
        return False

    logger.info(f"Running migration: {migration_id} - {description}")

    # Handle special migrations
    if migration_id == "001_unique_index_feature_table_v2":
        if not table_exists(conn, "feature_table_v2"):
            logger.info("feature_table_v2 does not exist, skipping migration")
            return False
        create_unique_index_safe(conn, "feature_table_v2", "idx_feature_table_v2_pk", ["race_id", "horse_id"])

    elif migration_id == "002_unique_index_feature_table_v3":
        if not table_exists(conn, "feature_table_v3"):
            logger.info("feature_table_v3 does not exist, skipping migration")
            return False
        create_unique_index_safe(conn, "feature_table_v3", "idx_feature_table_v3_pk", ["race_id", "horse_id"])

    elif migration_id == "004_migration_tracking_table":
        # This one creates the tracking table itself
        conn.executescript(migration["up"])
        conn.commit()

    else:
        # Generic migration - run SQL directly
        try:
            conn.executescript(migration["up"])
            conn.commit()
        except Exception as e:
            logger.error(f"Migration {migration_id} failed: {e}")
            conn.rollback()
            raise

    # Record migration (need tracking table to exist first)
    if table_exists(conn, "_migrations"):
        record_migration(conn, migration_id, description)

    logger.info(f"Migration {migration_id} completed")
    return True


def run_migrations(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """
    Run all pending migrations.

    Args:
        conn: SQLite connection
        dry_run: If True, only show what would be done

    Returns:
        Number of migrations applied
    """
    logger.info("=" * 60)
    logger.info("Running database migrations...")
    logger.info("=" * 60)

    # First, ensure tracking table exists
    if not table_exists(conn, "_migrations"):
        if not dry_run:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    applied_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.commit()

    applied_count = 0
    applied_set = get_applied_migrations(conn)

    for migration in MIGRATIONS:
        migration_id = migration["id"]
        description = migration["description"]

        if migration_id in applied_set:
            logger.debug(f"[SKIP] {migration_id} - already applied")
            continue

        if dry_run:
            logger.info(f"[DRY-RUN] Would apply: {migration_id} - {description}")
            applied_count += 1
        else:
            if run_migration(conn, migration):
                applied_count += 1

    logger.info("=" * 60)
    logger.info(f"Migrations complete. Applied: {applied_count}")
    logger.info("=" * 60)

    return applied_count


def get_schema_status(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get current schema status for diagnostics.

    Returns:
        Dict with table counts, index info, etc.
    """
    status = {
        "tables": {},
        "indexes": {},
        "migrations_applied": [],
        "feature_table_v2_has_unique": False,
        "feature_table_v3_has_unique": False,
    }

    # Get table row counts
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    for row in cursor.fetchall():
        table_name = row[0]
        if table_name.startswith("sqlite_"):
            continue
        try:
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            status["tables"][table_name] = count_cursor.fetchone()[0]
        except Exception as e:
            status["tables"][table_name] = f"ERROR: {e}"

    # Get indexes
    cursor = conn.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' ORDER BY name"
    )
    for row in cursor.fetchall():
        if row[0]:  # Some internal indexes have NULL name
            status["indexes"][row[0]] = {
                "table": row[1],
                "sql": row[2]
            }

    # Check feature table unique indexes
    status["feature_table_v2_has_unique"] = index_exists(conn, "idx_feature_table_v2_pk")
    status["feature_table_v3_has_unique"] = index_exists(conn, "idx_feature_table_v3_pk")

    # Get applied migrations
    if table_exists(conn, "_migrations"):
        cursor = conn.execute("SELECT id, applied_at FROM _migrations ORDER BY applied_at")
        status["migrations_applied"] = [(row[0], row[1]) for row in cursor.fetchall()]

    return status


def print_schema_status(conn: sqlite3.Connection) -> None:
    """Print schema status in a readable format."""
    status = get_schema_status(conn)

    print("\n" + "=" * 60)
    print("DATABASE SCHEMA STATUS")
    print("=" * 60)

    print("\n📊 Tables:")
    for table, count in sorted(status["tables"].items()):
        print(f"  {table}: {count:,} rows" if isinstance(count, int) else f"  {table}: {count}")

    print("\n🔑 Key Indexes:")
    print(f"  feature_table_v2 UNIQUE (race_id, horse_id): {'✅' if status['feature_table_v2_has_unique'] else '❌'}")
    print(f"  feature_table_v3 UNIQUE (race_id, horse_id): {'✅' if status['feature_table_v3_has_unique'] else '❌'}")

    print("\n📋 All Indexes:")
    for name, info in sorted(status["indexes"].items()):
        print(f"  {name} on {info['table']}")

    print("\n🔄 Applied Migrations:")
    if status["migrations_applied"]:
        for mid, applied_at in status["migrations_applied"]:
            print(f"  [{applied_at}] {mid}")
    else:
        print("  (none)")

    print("=" * 60 + "\n")


# ============================================================
# CLI
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run database schema migrations for keiba-yosou"
    )
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current schema status and exit"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        if args.status:
            print_schema_status(conn)
        else:
            run_migrations(conn, dry_run=args.dry_run)
            print_schema_status(conn)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
