#!/usr/bin/env python3
"""
run_migrations.py - Run database schema migrations

This script ensures:
1. UNIQUE indexes exist on feature_table_v2 and feature_table_v3
2. Duplicates are cleaned up before creating unique constraints
3. Migrations are tracked for idempotency

Usage:
    # Show current status
    python scripts/run_migrations.py --db netkeiba.db --status

    # Dry run (show what would be done)
    python scripts/run_migrations.py --db netkeiba.db --dry-run

    # Apply migrations
    python scripts/run_migrations.py --db netkeiba.db

    # With verbose logging
    python scripts/run_migrations.py --db netkeiba.db -v
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.schema_migration import (
    run_migrations,
    print_schema_status,
    count_duplicates,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run database schema migrations for keiba-yosou"
    )
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)"
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
        "--check-duplicates",
        action="store_true",
        help="Check for duplicates in feature tables"
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
    logger = logging.getLogger(__name__)

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        if args.check_duplicates:
            print("\n" + "=" * 60)
            print("DUPLICATE CHECK")
            print("=" * 60)

            tables_to_check = [
                ("feature_table", ["race_id", "horse_id"]),
                ("feature_table_v2", ["race_id", "horse_id"]),
                ("feature_table_v3", ["race_id", "horse_id"]),
                ("race_results", ["race_id", "horse_id"]),
                ("horse_results", ["horse_id", "race_id"]),
            ]

            for table, keys in tables_to_check:
                try:
                    dup_count = count_duplicates(conn, table, keys)
                    status = "❌" if dup_count > 0 else "✅"
                    print(f"  {status} {table}: {dup_count:,} duplicates")
                except Exception as e:
                    print(f"  ⚠️  {table}: {e}")

            print("=" * 60 + "\n")

        if args.status:
            print_schema_status(conn)
        else:
            count = run_migrations(conn, dry_run=args.dry_run)

            if args.dry_run:
                print(f"\n[DRY-RUN] Would apply {count} migrations")
            else:
                print(f"\nApplied {count} migrations")

            # Show status after migrations
            print_schema_status(conn)

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
