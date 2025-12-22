#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_masters.py - Master Data Fetcher for Road 2

Fetches and stores master data for horses, jockeys, and trainers
from netkeiba.com with resume capability.

Features:
- Extract entity IDs from race_results (2021-2024)
- Resume capability via fetch_status table
- Rate limiting and retry logic
- Progress reporting
- Coverage statistics

Usage:
    # Dry run - show what would be fetched
    python scripts/fetch_masters.py --db netkeiba.db --dry-run

    # Fetch horses only (limit 100)
    python scripts/fetch_masters.py --db netkeiba.db --entity horse --limit 100

    # Fetch all entities
    python scripts/fetch_masters.py --db netkeiba.db --entity all

    # Show coverage report only
    python scripts/fetch_masters.py --db netkeiba.db --report

    # Resume failed fetches
    python scripts/fetch_masters.py --db netkeiba.db --entity horse --retry-failed

    # Verbose logging
    python scripts/fetch_masters.py --db netkeiba.db --entity horse -v
"""

import argparse
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from bs4 import BeautifulSoup

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.masters_migration import (
    run_road2_migrations,
    FetchStatusManager,
)
from src.ingestion.scraper import NetkeibaClient
from src.ingestion.parser_horse_extended import HorseExtendedParser, HorseExtendedRecord
from src.ingestion.parser_jockey import JockeyParser, JockeyRecord
from src.ingestion.parser_trainer import TrainerParser, TrainerRecord

logger = logging.getLogger(__name__)


# ============================================================
# Database Operations
# ============================================================


def extract_entity_ids(
    conn: sqlite3.Connection,
    entity_type: str,
    start_year: int = 2021,
    end_year: int = 2024,
) -> List[str]:
    """
    Extract unique entity IDs from race_results table.

    Args:
        conn: SQLite connection
        entity_type: 'horse', 'jockey', or 'trainer'
        start_year: Start year for filtering
        end_year: End year for filtering

    Returns:
        List of unique entity IDs
    """
    column_map = {
        "horse": "horse_id",
        "jockey": "jockey_id",
        "trainer": "trainer_id",
    }

    column = column_map.get(entity_type)
    if not column:
        raise ValueError(f"Unknown entity type: {entity_type}")

    # Check if column exists
    cursor = conn.execute("PRAGMA table_info(race_results)")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        logger.warning(f"Column {column} not found in race_results")
        return []

    # Extract unique IDs with date filter
    query = f"""
        SELECT DISTINCT {column}
        FROM race_results
        WHERE {column} IS NOT NULL
        AND {column} != ''
        AND race_id LIKE ? || '%'
    """

    all_ids = set()
    for year in range(start_year, end_year + 1):
        cursor = conn.execute(query, (str(year),))
        ids = [row[0] for row in cursor.fetchall()]
        all_ids.update(ids)
        logger.debug(f"Year {year}: found {len(ids)} {entity_type} IDs")

    result = sorted(all_ids)
    logger.info(f"Total unique {entity_type} IDs: {len(result)}")
    return result


def upsert_horse(conn: sqlite3.Connection, record: HorseExtendedRecord) -> None:
    """UPSERT a horse record."""
    conn.execute("""
        INSERT INTO horses (
            horse_id, horse_name, sex, birth_date, coat_color,
            breeder, breeder_region, owner, owner_id,
            sire_id, sire_name, dam_id, dam_name,
            broodmare_sire_id, broodmare_sire_name,
            sire_sire_name, sire_dam_name, dam_dam_name,
            total_prize, total_starts, total_wins,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ON CONFLICT(horse_id) DO UPDATE SET
            horse_name = excluded.horse_name,
            sex = excluded.sex,
            birth_date = excluded.birth_date,
            coat_color = excluded.coat_color,
            breeder = excluded.breeder,
            breeder_region = excluded.breeder_region,
            owner = excluded.owner,
            owner_id = excluded.owner_id,
            sire_id = excluded.sire_id,
            sire_name = excluded.sire_name,
            dam_id = excluded.dam_id,
            dam_name = excluded.dam_name,
            broodmare_sire_id = excluded.broodmare_sire_id,
            broodmare_sire_name = excluded.broodmare_sire_name,
            sire_sire_name = excluded.sire_sire_name,
            sire_dam_name = excluded.sire_dam_name,
            dam_dam_name = excluded.dam_dam_name,
            total_prize = excluded.total_prize,
            total_starts = excluded.total_starts,
            total_wins = excluded.total_wins,
            updated_at = datetime('now', 'localtime')
    """, (
        record.horse_id, record.horse_name, record.sex, record.birth_date,
        record.coat_color, record.breeder, record.breeder_region,
        record.owner, record.owner_id,
        record.sire_id, record.sire_name, record.dam_id, record.dam_name,
        record.broodmare_sire_id, record.broodmare_sire_name,
        record.sire_sire_name, record.sire_dam_name, record.dam_dam_name,
        record.total_prize, record.total_starts, record.total_wins,
    ))
    conn.commit()


def upsert_jockey(conn: sqlite3.Connection, record: JockeyRecord) -> None:
    """UPSERT a jockey record."""
    conn.execute("""
        INSERT INTO jockeys (
            jockey_id, jockey_name, name_kana, birth_date,
            affiliation, debut_year,
            career_wins, career_in3, career_starts,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ON CONFLICT(jockey_id) DO UPDATE SET
            jockey_name = excluded.jockey_name,
            name_kana = excluded.name_kana,
            birth_date = excluded.birth_date,
            affiliation = excluded.affiliation,
            debut_year = excluded.debut_year,
            career_wins = excluded.career_wins,
            career_in3 = excluded.career_in3,
            career_starts = excluded.career_starts,
            updated_at = datetime('now', 'localtime')
    """, (
        record.jockey_id, record.jockey_name, record.name_kana,
        record.birth_date, record.affiliation, record.debut_year,
        record.career_wins, record.career_in3, record.career_starts,
    ))
    conn.commit()


def upsert_trainer(conn: sqlite3.Connection, record: TrainerRecord) -> None:
    """UPSERT a trainer record."""
    conn.execute("""
        INSERT INTO trainers (
            trainer_id, trainer_name, name_kana, birth_date,
            affiliation,
            career_wins, career_in3, career_starts,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ON CONFLICT(trainer_id) DO UPDATE SET
            trainer_name = excluded.trainer_name,
            name_kana = excluded.name_kana,
            birth_date = excluded.birth_date,
            affiliation = excluded.affiliation,
            career_wins = excluded.career_wins,
            career_in3 = excluded.career_in3,
            career_starts = excluded.career_starts,
            updated_at = datetime('now', 'localtime')
    """, (
        record.trainer_id, record.trainer_name, record.name_kana,
        record.birth_date, record.affiliation,
        record.career_wins, record.career_in3, record.career_starts,
    ))
    conn.commit()


# ============================================================
# Fetcher Class
# ============================================================


class MasterFetcher:
    """
    Fetcher for master data with resume capability.

    Features:
    - Tracks fetch status in fetch_status table
    - Automatic retry for failed fetches
    - Rate limiting
    - Progress reporting
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        client: Optional[NetkeibaClient] = None,
        dry_run: bool = False,
    ):
        self.conn = conn
        self.client = client or NetkeibaClient()
        self.dry_run = dry_run
        self.status_manager = FetchStatusManager(conn)
        self.status_manager.ensure_table()

        # Parsers
        self.horse_parser = HorseExtendedParser()
        self.jockey_parser = JockeyParser()
        self.trainer_parser = TrainerParser()

        # Stats
        self.stats = {
            "fetched": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }

    def init_pending(
        self,
        entity_type: str,
        entity_ids: List[str],
    ) -> int:
        """
        Initialize pending entities in fetch_status table.

        Args:
            entity_type: 'horse', 'jockey', or 'trainer'
            entity_ids: List of entity IDs to add

        Returns:
            Number of new entities added
        """
        return self.status_manager.add_pending(entity_type, entity_ids)

    def fetch_entities(
        self,
        entity_type: str,
        limit: int = 100,
        max_retries: int = 3,
    ) -> Dict[str, int]:
        """
        Fetch pending entities.

        Args:
            entity_type: 'horse', 'jockey', or 'trainer'
            limit: Maximum number of entities to fetch
            max_retries: Skip entities with more retries

        Returns:
            Stats dict
        """
        pending = self.status_manager.get_pending(entity_type, limit, max_retries)

        if not pending:
            logger.info(f"No pending {entity_type} entities to fetch")
            return self.stats

        logger.info(f"Fetching {len(pending)} {entity_type} entities")

        for i, entity_id in enumerate(pending, 1):
            logger.info(f"[{i}/{len(pending)}] Fetching {entity_type} {entity_id}")

            if self.dry_run:
                logger.info(f"  [DRY-RUN] Would fetch {entity_type}/{entity_id}")
                self.stats["skipped"] += 1
                continue

            try:
                self._fetch_one(entity_type, entity_id)
                self.stats["success"] += 1
            except Exception as e:
                logger.error(f"  Failed: {e}")
                self.status_manager.mark_failed(entity_type, entity_id, str(e))
                self.stats["failed"] += 1

            self.stats["fetched"] += 1

            # Progress every 10 items
            if i % 10 == 0:
                progress = self.status_manager.get_progress(entity_type)
                logger.info(
                    f"  Progress: {progress['success']}/{progress['total']} "
                    f"({progress['progress_pct']:.1f}%)"
                )

        return self.stats

    def _fetch_one(self, entity_type: str, entity_id: str) -> None:
        """Fetch and parse one entity."""
        url_map = {
            "horse": f"/horse/{entity_id}/",
            "jockey": f"/jockey/{entity_id}/",
            "trainer": f"/trainer/{entity_id}/",
        }

        url = url_map.get(entity_type)
        if not url:
            raise ValueError(f"Unknown entity type: {entity_type}")

        # Fetch HTML
        response = self.client.get(url)

        if response.status_code == 404:
            logger.warning(f"  Not found: {entity_type}/{entity_id}")
            self.status_manager.mark_skipped(entity_type, entity_id, "404 Not Found")
            return

        response.raise_for_status()

        # Decode HTML
        html = self.client._decode_response(response)
        soup = BeautifulSoup(html, "html.parser")

        # Parse and store
        if entity_type == "horse":
            record = self.horse_parser.parse(entity_id, soup)
            upsert_horse(self.conn, record)
        elif entity_type == "jockey":
            record = self.jockey_parser.parse(entity_id, soup)
            upsert_jockey(self.conn, record)
        elif entity_type == "trainer":
            record = self.trainer_parser.parse(entity_id, soup)
            upsert_trainer(self.conn, record)

        self.status_manager.mark_success(entity_type, entity_id)
        logger.debug(f"  Stored {entity_type}/{entity_id}")


# ============================================================
# Coverage Report
# ============================================================


def print_coverage_report(conn: sqlite3.Connection) -> None:
    """Print coverage statistics."""
    print("\n" + "=" * 60)
    print("MASTER DATA COVERAGE REPORT")
    print("=" * 60)

    manager = FetchStatusManager(conn)

    for entity_type in ["horse", "jockey", "trainer"]:
        progress = manager.get_progress(entity_type)
        stats = manager.get_stats(entity_type)

        print(f"\n{entity_type.upper()}")
        print("-" * 40)
        print(f"  Total tracked:  {progress['total']:,}")
        print(f"  Success:        {progress['success']:,} ({progress['progress_pct']:.1f}%)")
        print(f"  Failed:         {progress['failed']:,}")
        print(f"  Pending:        {progress['pending']:,}")
        print(f"  Skipped:        {progress['skipped']:,}")

        # Count actual records in master tables
        table_map = {
            "horse": "horses",
            "jockey": "jockeys",
            "trainer": "trainers",
        }
        table = table_map.get(entity_type)
        if table:
            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  In {table} table: {count:,}")
            except sqlite3.OperationalError:
                print(f"  Table {table} not found")

    print("\n" + "=" * 60)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fetch master data (horses, jockeys, trainers) from netkeiba.com"
    )
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)"
    )
    parser.add_argument(
        "--entity",
        choices=["horse", "jockey", "trainer", "all"],
        default="all",
        help="Entity type to fetch (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum entities to fetch per run (default: 100)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year for entity extraction (default: 2021)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year for entity extraction (default: 2024)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without actually fetching"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show coverage report only"
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize pending entities without fetching"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed fetches"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    # Check database
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        # Run migrations
        logger.info("Running Road 2 migrations...")
        count = run_road2_migrations(conn)
        if count > 0:
            logger.info(f"Applied {count} migrations")

        # Report only mode
        if args.report:
            print_coverage_report(conn)
            return 0

        # Determine entity types
        if args.entity == "all":
            entity_types = ["horse", "jockey", "trainer"]
        else:
            entity_types = [args.entity]

        # Initialize fetcher
        client = None if args.dry_run else NetkeibaClient()
        fetcher = MasterFetcher(conn, client, dry_run=args.dry_run)

        try:
            for entity_type in entity_types:
                print(f"\n{'=' * 60}")
                print(f"Processing {entity_type.upper()}")
                print("=" * 60)

                # Extract entity IDs from race_results
                logger.info(f"Extracting {entity_type} IDs from race_results...")
                entity_ids = extract_entity_ids(
                    conn, entity_type,
                    args.start_year, args.end_year
                )

                if not entity_ids:
                    logger.warning(f"No {entity_type} IDs found")
                    continue

                # Initialize pending
                added = fetcher.init_pending(entity_type, entity_ids)
                logger.info(f"Added {added} new {entity_type} entities to fetch queue")

                if args.init_only:
                    continue

                # Fetch
                stats = fetcher.fetch_entities(entity_type, args.limit)
                logger.info(
                    f"Completed: success={stats['success']}, "
                    f"failed={stats['failed']}, skipped={stats['skipped']}"
                )

        finally:
            if client:
                client.close()

        # Print final report
        print_coverage_report(conn)

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
