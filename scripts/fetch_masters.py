#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_masters.py - Master Data Fetcher for Road 2 + PR2.5

Fetches and stores master data for horses, jockeys, trainers, and 5-gen pedigrees
from netkeiba.com with resume capability.

Features:
- Extract entity IDs from race_results (2021-2024)
- Resume capability via fetch_status table
- User-Agent rotation (12 UAs, random per request)
- Rate limiting (configurable sleep between requests)
- Exponential backoff retry for errors
- Safe stop after consecutive failures
- Graceful shutdown (SIGINT/Ctrl+C)
- Progress snapshots to artifacts/
- ETA estimation
- Failure classification

Usage:
    # Dry run - show what would be fetched
    python scripts/fetch_masters.py --db netkeiba.db --dry-run

    # Fetch horses only (limit 100)
    python scripts/fetch_masters.py --db netkeiba.db --entity horse --limit 100

    # Fetch 5-generation pedigrees only
    python scripts/fetch_masters.py --db netkeiba.db --entity horse_pedigree --limit 100

    # Fetch all entities (including pedigrees)
    python scripts/fetch_masters.py --db netkeiba.db --entity all

    # Run until all pending are fetched (no limit)
    python scripts/fetch_masters.py --db netkeiba.db --entity horse --run-until-empty

    # Custom sleep interval (slower for safety)
    python scripts/fetch_masters.py --db netkeiba.db --entity horse --sleep-min 3.0 --sleep-max 5.0

    # Show coverage report only
    python scripts/fetch_masters.py --db netkeiba.db --report

    # Resume failed fetches
    python scripts/fetch_masters.py --db netkeiba.db --entity horse --retry-failed

    # Active only (2021-2024 race participants)
    python scripts/fetch_masters.py --db netkeiba.db --entity jockey --active-only

    # Priority ordering
    python scripts/fetch_masters.py --db netkeiba.db --entity horse --priority frequency

    # Verbose logging
    python scripts/fetch_masters.py --db netkeiba.db --entity horse -v

Safe Stop Behavior:
    The fetcher will automatically stop after MAX_CONSECUTIVE_FAILURES (10) consecutive
    failures. This prevents runaway failure loops from hammering the server.

    When safe stop triggers:
    1. All successfully fetched data is already committed to DB
    2. Failed entities are marked in fetch_status with error messages
    3. Resume by running the same command again

    To diagnose failures:
    - Check the error log messages
    - Use --report to see failure counts
    - Use --retry-failed to retry only failed entities

Graceful Shutdown:
    Press Ctrl+C (SIGINT) to stop gracefully:
    - Current request completes or is aborted
    - All data committed to DB remains intact
    - fetch_status is consistent (no partial states)
    - Progress snapshot is saved to artifacts/
    - Resume by running the same command again
"""

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
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
from src.ingestion.parser_pedigree_5gen import Pedigree5GenParser, PedigreeAncestor

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Safe stop after this many consecutive failures
MAX_CONSECUTIVE_FAILURES = 10

# Default batch size for run-until-empty
DEFAULT_BATCH_SIZE = 100

# Snapshot interval (save progress every N fetches)
SNAPSHOT_INTERVAL = 50

# Failure categories for retry classification
FAILURE_CATEGORIES = {
    "timeout": {
        "patterns": ["timeout", "timed out", "read timed out"],
        "retry_immediately": False,
        "description": "Network timeout - server may be slow or overloaded",
    },
    "http_4xx": {
        "patterns": ["400", "401", "403", "429"],
        "retry_immediately": False,
        "description": "Client error - may need auth refresh or rate limit backoff",
    },
    "http_5xx": {
        "patterns": ["500", "502", "503", "504"],
        "retry_immediately": True,
        "description": "Server error - usually temporary, retry recommended",
    },
    "parse": {
        "patterns": ["parse", "attribute", "keyerror", "indexerror"],
        "retry_immediately": False,
        "description": "Parse error - page structure may have changed",
    },
    "network": {
        "patterns": ["connection", "refused", "reset", "broken pipe"],
        "retry_immediately": False,
        "description": "Network error - check connectivity",
    },
}


# ============================================================
# Graceful Shutdown Handler
# ============================================================

class GracefulShutdown:
    """
    Handle SIGINT (Ctrl+C) for graceful shutdown.

    Usage:
        shutdown = GracefulShutdown()
        while not shutdown.should_stop:
            do_work()
    """

    def __init__(self):
        self.should_stop = False
        self._original_handler = None

    def install(self):
        """Install signal handler."""
        self._original_handler = signal.signal(signal.SIGINT, self._handler)
        logger.debug("Graceful shutdown handler installed (Ctrl+C to stop)")

    def uninstall(self):
        """Restore original signal handler."""
        if self._original_handler:
            signal.signal(signal.SIGINT, self._original_handler)

    def _handler(self, signum, frame):
        """Handle SIGINT."""
        if self.should_stop:
            # Second Ctrl+C - force exit
            logger.warning("Force exit requested")
            sys.exit(1)

        logger.warning("\n[SIGINT] Graceful shutdown requested. Finishing current operation...")
        logger.warning("         (Press Ctrl+C again to force exit)")
        self.should_stop = True


# ============================================================
# Progress Snapshot
# ============================================================

@dataclass
class ProgressSnapshot:
    """Snapshot of fetch progress for resumability."""
    timestamp: str
    entity_type: str
    total: int
    success: int
    failed: int
    pending: int
    skipped: int
    rate_per_minute: float
    eta_minutes: Optional[float]
    consecutive_failures: int
    last_entity_id: Optional[str]
    failure_summary: Dict[str, int]


def save_progress_snapshot(
    snapshot: ProgressSnapshot,
    output_dir: str = "artifacts",
) -> str:
    """
    Save progress snapshot to JSON file.

    Args:
        snapshot: ProgressSnapshot to save
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fetch_progress_{snapshot.entity_type}_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(snapshot), f, ensure_ascii=False, indent=2)

    logger.info(f"Progress snapshot saved: {filepath}")
    return str(filepath)


# ============================================================
# ETA Estimator
# ============================================================

class ETAEstimator:
    """Estimate time remaining based on processing rate."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.timestamps: List[float] = []
        self.start_time: Optional[float] = None

    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self.timestamps = []

    def record_completion(self):
        """Record a successful completion."""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

    def get_rate_per_minute(self) -> float:
        """Get current processing rate (items per minute)."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed * 60

    def get_eta_minutes(self, remaining: int) -> Optional[float]:
        """
        Get estimated time to completion in minutes.

        Args:
            remaining: Number of items remaining

        Returns:
            Estimated minutes, or None if not enough data
        """
        rate = self.get_rate_per_minute()
        if rate <= 0:
            return None
        return remaining / rate

    def format_eta(self, remaining: int) -> str:
        """Format ETA as human-readable string."""
        eta = self.get_eta_minutes(remaining)
        if eta is None:
            return "calculating..."

        if eta < 60:
            return f"{eta:.0f} min"
        elif eta < 1440:  # 24 hours
            hours = int(eta / 60)
            mins = int(eta % 60)
            return f"{hours}h {mins}m"
        else:
            days = int(eta / 1440)
            hours = int((eta % 1440) / 60)
            return f"{days}d {hours}h"


# ============================================================
# Failure Classifier
# ============================================================

def classify_failure(error_message: str) -> str:
    """
    Classify failure by error message.

    Args:
        error_message: Error message string

    Returns:
        Failure category name
    """
    error_lower = error_message.lower()

    for category, info in FAILURE_CATEGORIES.items():
        for pattern in info["patterns"]:
            if pattern.lower() in error_lower:
                return category

    return "unknown"


def get_failure_summary(conn: sqlite3.Connection, entity_type: str) -> Dict[str, int]:
    """
    Get summary of failures by category.

    Args:
        conn: SQLite connection
        entity_type: Entity type

    Returns:
        Dict of category -> count
    """
    cursor = conn.execute("""
        SELECT error_message
        FROM fetch_status
        WHERE entity_type = ? AND status = 'failed'
    """, (entity_type,))

    summary: Dict[str, int] = {}
    for (error_message,) in cursor.fetchall():
        if error_message:
            category = classify_failure(error_message)
            summary[category] = summary.get(category, 0) + 1

    return summary


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
        entity_type: 'horse', 'jockey', 'trainer', or 'horse_pedigree'
        start_year: Start year for filtering
        end_year: End year for filtering

    Returns:
        List of unique entity IDs
    """
    # horse_pedigree uses same IDs as horse
    if entity_type == "horse_pedigree":
        entity_type = "horse"

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


def get_active_entity_ids(
    conn: sqlite3.Connection,
    entity_type: str,
    start_year: int = 2021,
    end_year: int = 2024,
) -> List[str]:
    """
    Get entity IDs that have race activity in the specified period.

    For jockey/trainer, returns only those who have ridden/trained
    in races during 2021-2024.

    Args:
        conn: SQLite connection
        entity_type: 'jockey' or 'trainer'
        start_year: Start year
        end_year: End year

    Returns:
        List of active entity IDs
    """
    if entity_type not in ("jockey", "trainer"):
        return extract_entity_ids(conn, entity_type, start_year, end_year)

    column = f"{entity_type}_id"

    query = f"""
        SELECT DISTINCT {column}
        FROM race_results
        WHERE {column} IS NOT NULL
        AND {column} != ''
        AND race_id >= ? || '0000000000'
        AND race_id < ? || '0000000000'
    """

    cursor = conn.execute(query, (str(start_year), str(end_year + 1)))
    ids = [row[0] for row in cursor.fetchall()]
    logger.info(f"Found {len(ids)} active {entity_type} IDs ({start_year}-{end_year})")
    return ids


def get_entity_frequency(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_ids: List[str],
) -> Dict[str, int]:
    """
    Get frequency (race appearance count) for each entity.

    Args:
        conn: SQLite connection
        entity_type: 'horse', 'jockey', or 'trainer'
        entity_ids: List of entity IDs

    Returns:
        Dict of entity_id -> frequency
    """
    if entity_type == "horse_pedigree":
        entity_type = "horse"

    column = f"{entity_type}_id"

    # Build query with placeholders
    placeholders = ",".join(["?"] * len(entity_ids))
    query = f"""
        SELECT {column}, COUNT(*) as cnt
        FROM race_results
        WHERE {column} IN ({placeholders})
        GROUP BY {column}
    """

    cursor = conn.execute(query, entity_ids)
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_entity_last_race(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_ids: List[str],
) -> Dict[str, str]:
    """
    Get last race date for each entity.

    Args:
        conn: SQLite connection
        entity_type: 'horse', 'jockey', or 'trainer'
        entity_ids: List of entity IDs

    Returns:
        Dict of entity_id -> last_race_id
    """
    if entity_type == "horse_pedigree":
        entity_type = "horse"

    column = f"{entity_type}_id"

    placeholders = ",".join(["?"] * len(entity_ids))
    query = f"""
        SELECT {column}, MAX(race_id) as last_race
        FROM race_results
        WHERE {column} IN ({placeholders})
        GROUP BY {column}
    """

    cursor = conn.execute(query, entity_ids)
    return {row[0]: row[1] for row in cursor.fetchall()}


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


def upsert_pedigree_ancestors(
    conn: sqlite3.Connection,
    ancestors: List[PedigreeAncestor],
) -> int:
    """
    UPSERT pedigree ancestors to horse_pedigree table.

    Args:
        conn: SQLite connection
        ancestors: List of PedigreeAncestor records

    Returns:
        Number of records upserted
    """
    if not ancestors:
        return 0

    conn.executemany("""
        INSERT INTO horse_pedigree (
            horse_id, generation, position, ancestor_id, ancestor_name,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ON CONFLICT(horse_id, generation, position) DO UPDATE SET
            ancestor_id = excluded.ancestor_id,
            ancestor_name = excluded.ancestor_name,
            updated_at = datetime('now', 'localtime')
    """, [
        (a.horse_id, a.generation, a.position, a.ancestor_id, a.ancestor_name)
        for a in ancestors
    ])
    conn.commit()
    return len(ancestors)


# ============================================================
# Fetcher Class
# ============================================================


class MasterFetcher:
    """
    Fetcher for master data with resume capability.

    Features:
    - Tracks fetch status in fetch_status table
    - Automatic retry for failed fetches
    - Rate limiting (configurable)
    - Safe stop after consecutive failures
    - Graceful shutdown (SIGINT)
    - Progress snapshots
    - ETA estimation
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        client: Optional[NetkeibaClient] = None,
        dry_run: bool = False,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
        shutdown_handler: Optional[GracefulShutdown] = None,
    ):
        self.conn = conn
        self.client = client
        self.dry_run = dry_run
        self.max_consecutive_failures = max_consecutive_failures
        self.shutdown = shutdown_handler
        self.status_manager = FetchStatusManager(conn)
        self.status_manager.ensure_table()

        # Parsers
        self.horse_parser = HorseExtendedParser()
        self.jockey_parser = JockeyParser()
        self.trainer_parser = TrainerParser()
        self.pedigree_parser = Pedigree5GenParser()

        # Stats
        self.stats = {
            "fetched": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }

        # Safe stop tracking
        self.consecutive_failures = 0
        self.last_entity_id: Optional[str] = None

        # ETA estimation
        self.eta = ETAEstimator()

    def init_pending(
        self,
        entity_type: str,
        entity_ids: List[str],
    ) -> int:
        """
        Initialize pending entities in fetch_status table.

        Args:
            entity_type: 'horse', 'jockey', 'trainer', or 'horse_pedigree'
            entity_ids: List of entity IDs to add

        Returns:
            Number of new entities added
        """
        return self.status_manager.add_pending(entity_type, entity_ids)

    def reorder_pending_by_priority(
        self,
        entity_type: str,
        priority_mode: str,
    ) -> None:
        """
        Reorder pending entities by priority.

        Note: This updates the priority column in fetch_status.
        The get_pending method will order by priority DESC.

        Args:
            entity_type: Entity type
            priority_mode: 'frequency' or 'recent'
        """
        # Get pending IDs
        cursor = self.conn.execute("""
            SELECT entity_id FROM fetch_status
            WHERE entity_type = ? AND status = 'pending'
        """, (entity_type,))
        pending_ids = [row[0] for row in cursor.fetchall()]

        if not pending_ids:
            return

        logger.info(f"Reordering {len(pending_ids)} pending by {priority_mode}...")

        if priority_mode == "frequency":
            # Higher frequency = higher priority
            freq_map = get_entity_frequency(self.conn, entity_type, pending_ids)
            for entity_id, freq in freq_map.items():
                self.conn.execute("""
                    UPDATE fetch_status
                    SET priority = ?
                    WHERE entity_type = ? AND entity_id = ?
                """, (freq, entity_type, entity_id))

        elif priority_mode == "recent":
            # More recent = higher priority (use race_id as proxy)
            last_race_map = get_entity_last_race(self.conn, entity_type, pending_ids)
            for entity_id, last_race in last_race_map.items():
                # Convert race_id to priority (higher race_id = more recent)
                priority = int(last_race[:8]) if last_race else 0
                self.conn.execute("""
                    UPDATE fetch_status
                    SET priority = ?
                    WHERE entity_type = ? AND entity_id = ?
                """, (priority, entity_type, entity_id))

        self.conn.commit()
        logger.info(f"Reordered by {priority_mode}")

    def fetch_entities(
        self,
        entity_type: str,
        limit: Optional[int] = None,
        max_retries: int = 3,
    ) -> Dict[str, int]:
        """
        Fetch pending entities.

        Args:
            entity_type: 'horse', 'jockey', 'trainer', or 'horse_pedigree'
            limit: Maximum entities to fetch (None = unlimited)
            max_retries: Skip entities with more retries

        Returns:
            Stats dict
        """
        batch_size = limit if limit else DEFAULT_BATCH_SIZE
        total_fetched = 0
        snapshot_counter = 0

        # Start ETA tracking
        self.eta.start()

        while True:
            # Check for graceful shutdown
            if self.shutdown and self.shutdown.should_stop:
                logger.info("Graceful shutdown - stopping fetch loop")
                break

            # Get next batch (ordered by priority DESC)
            pending = self.status_manager.get_pending(
                entity_type, batch_size, max_retries
            )

            if not pending:
                if total_fetched == 0:
                    logger.info(f"No pending {entity_type} entities to fetch")
                break

            # Get progress for ETA display
            progress = self.status_manager.get_progress(entity_type)
            remaining = progress["pending"]

            logger.info(f"Fetching batch of {len(pending)} {entity_type} entities")
            logger.info(f"  Remaining: {remaining:,} | ETA: {self.eta.format_eta(remaining)}")

            for i, entity_id in enumerate(pending, 1):
                # Check for graceful shutdown
                if self.shutdown and self.shutdown.should_stop:
                    logger.info("Graceful shutdown - stopping mid-batch")
                    break

                # Check for safe stop
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.error(
                        f"\n{'='*60}\n"
                        f"SAFE STOP: {self.consecutive_failures} consecutive failures\n"
                        f"{'='*60}\n"
                        f"This prevents runaway failure loops.\n"
                        f"\n"
                        f"To diagnose:\n"
                        f"  1. Check error messages above\n"
                        f"  2. Run with --report to see failure counts\n"
                        f"  3. Check network/auth status\n"
                        f"\n"
                        f"To resume:\n"
                        f"  Run the same command again\n"
                        f"{'='*60}"
                    )
                    self._save_snapshot(entity_type, progress)
                    return self.stats

                self.last_entity_id = entity_id

                logger.info(
                    f"[{total_fetched + i}/{total_fetched + len(pending)}] "
                    f"Fetching {entity_type} {entity_id}"
                )

                if self.dry_run:
                    logger.info(f"  [DRY-RUN] Would fetch {entity_type}/{entity_id}")
                    self.stats["skipped"] += 1
                    continue

                try:
                    self._fetch_one(entity_type, entity_id)
                    self.stats["success"] += 1
                    self.consecutive_failures = 0  # Reset on success
                    self.eta.record_completion()
                except Exception as e:
                    error_msg = str(e)
                    category = classify_failure(error_msg)
                    logger.error(f"  Failed ({category}): {error_msg}")
                    self.status_manager.mark_failed(entity_type, entity_id, error_msg)
                    self.stats["failed"] += 1
                    self.consecutive_failures += 1

                self.stats["fetched"] += 1
                snapshot_counter += 1

                # Progress every 10 items
                if (total_fetched + i) % 10 == 0:
                    progress = self.status_manager.get_progress(entity_type)
                    remaining = progress["pending"]
                    rate = self.eta.get_rate_per_minute()
                    logger.info(
                        f"  Progress: {progress['success']:,}/{progress['total']:,} "
                        f"({progress['progress_pct']:.1f}%) | "
                        f"Rate: {rate:.1f}/min | ETA: {self.eta.format_eta(remaining)}"
                    )

                # Save snapshot periodically
                if snapshot_counter >= SNAPSHOT_INTERVAL:
                    progress = self.status_manager.get_progress(entity_type)
                    self._save_snapshot(entity_type, progress)
                    snapshot_counter = 0

            total_fetched += len(pending)

            # Stop if limit was specified and reached
            if limit and total_fetched >= limit:
                break

        # Final snapshot
        progress = self.status_manager.get_progress(entity_type)
        self._save_snapshot(entity_type, progress)

        return self.stats

    def _save_snapshot(
        self,
        entity_type: str,
        progress: Dict[str, Any],
    ) -> None:
        """Save progress snapshot to artifacts/."""
        failure_summary = get_failure_summary(self.conn, entity_type)
        remaining = progress["pending"]

        snapshot = ProgressSnapshot(
            timestamp=datetime.now().isoformat(),
            entity_type=entity_type,
            total=progress["total"],
            success=progress["success"],
            failed=progress["failed"],
            pending=progress["pending"],
            skipped=progress["skipped"],
            rate_per_minute=self.eta.get_rate_per_minute(),
            eta_minutes=self.eta.get_eta_minutes(remaining),
            consecutive_failures=self.consecutive_failures,
            last_entity_id=self.last_entity_id,
            failure_summary=failure_summary,
        )

        try:
            save_progress_snapshot(snapshot)
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")

    def _fetch_one(self, entity_type: str, entity_id: str) -> None:
        """Fetch and parse one entity."""
        url_map = {
            "horse": f"/horse/{entity_id}/",
            "jockey": f"/jockey/{entity_id}/",
            "trainer": f"/trainer/{entity_id}/",
            "horse_pedigree": f"/horse/ped/{entity_id}/",
        }

        url = url_map.get(entity_type)
        if not url:
            raise ValueError(f"Unknown entity type: {entity_type}")

        if not self.client:
            raise RuntimeError("Client not initialized (dry-run mode?)")

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
        elif entity_type == "horse_pedigree":
            ancestors = self.pedigree_parser.parse(entity_id, soup)
            count = upsert_pedigree_ancestors(self.conn, ancestors)
            logger.debug(f"  Stored {count} ancestors for {entity_id}")

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

    entity_types = ["horse", "jockey", "trainer", "horse_pedigree"]
    table_map = {
        "horse": "horses",
        "jockey": "jockeys",
        "trainer": "trainers",
        "horse_pedigree": "horse_pedigree",
    }

    for entity_type in entity_types:
        progress = manager.get_progress(entity_type)

        # Skip if no data tracked for this entity
        if progress['total'] == 0:
            continue

        print(f"\n{entity_type.upper()}")
        print("-" * 40)
        print(f"  Total tracked:  {progress['total']:,}")
        print(f"  Success:        {progress['success']:,} ({progress['progress_pct']:.1f}%)")
        print(f"  Failed:         {progress['failed']:,}")
        print(f"  Pending:        {progress['pending']:,}")
        print(f"  Skipped:        {progress['skipped']:,}")

        # Failure breakdown
        if progress['failed'] > 0:
            failure_summary = get_failure_summary(conn, entity_type)
            if failure_summary:
                print(f"\n  Failure Breakdown:")
                for category, count in sorted(failure_summary.items(), key=lambda x: -x[1]):
                    info = FAILURE_CATEGORIES.get(category, {"description": "Unknown error"})
                    print(f"    {category}: {count:,} - {info['description']}")

        # Count actual records in master tables
        table = table_map.get(entity_type)
        if table:
            try:
                if entity_type == "horse_pedigree":
                    # Count unique horses in pedigree table
                    cursor = conn.execute(
                        "SELECT COUNT(DISTINCT horse_id) FROM horse_pedigree"
                    )
                else:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"\n  In {table} table: {count:,}")
            except sqlite3.OperationalError:
                print(f"  Table {table} not found")

    print("\n" + "=" * 60)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fetch master data (horses, jockeys, trainers, pedigrees) from netkeiba.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 100 horses
  %(prog)s --db netkeiba.db --entity horse --limit 100

  # Fetch all pending horses (no limit)
  %(prog)s --db netkeiba.db --entity horse --run-until-empty

  # Fetch 5-gen pedigrees with slower rate
  %(prog)s --db netkeiba.db --entity horse_pedigree --sleep-min 3.0 --sleep-max 5.0

  # Show progress only
  %(prog)s --db netkeiba.db --report

  # Fetch only active jockeys (2021-2024 participants)
  %(prog)s --db netkeiba.db --entity jockey --active-only

  # Prioritize by frequency (most frequent first)
  %(prog)s --db netkeiba.db --entity horse --priority frequency

  # Prioritize by recency (most recent first)
  %(prog)s --db netkeiba.db --entity horse --priority recent

Rate Limiting:
  - User-Agent is rotated randomly per request (12 different UAs)
  - Sleep between requests: --sleep-min to --sleep-max seconds
  - Exponential backoff on 400/429 errors
  - Safe stop after 10 consecutive failures (resume with same command)

Safe Operation with --run-until-empty:
  - Recommended sleep: --sleep-min 3.0 --sleep-max 5.0 (for long runs)
  - Monitor with: tail -f logs or check artifacts/ for snapshots
  - Graceful stop: Ctrl+C saves state and exits cleanly
  - Resume: Run same command again
"""
    )
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)"
    )
    parser.add_argument(
        "--entity",
        choices=["horse", "jockey", "trainer", "horse_pedigree", "all"],
        default="all",
        help="Entity type to fetch (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum entities to fetch per run (default: 100, or unlimited with --run-until-empty)"
    )
    parser.add_argument(
        "--run-until-empty",
        action="store_true",
        help="Fetch until all pending entities are processed (no limit)"
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
        "--sleep-min",
        type=float,
        default=2.0,
        help="Minimum sleep between requests in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=3.5,
        help="Maximum sleep between requests in seconds (default: 3.5)"
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="For jockey/trainer: only fetch those active in 2021-2024 races"
    )
    parser.add_argument(
        "--priority",
        choices=["frequency", "recent"],
        default=None,
        help="Priority ordering: 'frequency' (most common first) or 'recent' (newest first)"
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

    # Handle limit logic
    if args.run_until_empty:
        limit = None  # Unlimited
        # Recommend slower rate for long runs
        if args.sleep_min < 3.0:
            logger.warning(
                "TIP: For --run-until-empty, consider using --sleep-min 3.0 --sleep-max 5.0 "
                "to reduce server load"
            )
    elif args.limit is not None:
        limit = args.limit
    else:
        limit = 100  # Default

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Check database
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)

    # Install graceful shutdown handler
    shutdown = GracefulShutdown()
    shutdown.install()

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
            entity_types = ["horse", "jockey", "trainer", "horse_pedigree"]
        else:
            entity_types = [args.entity]

        # Initialize client with custom sleep settings
        client = None
        if not args.dry_run:
            logger.info(
                f"Rate limiting: sleep {args.sleep_min:.1f}-{args.sleep_max:.1f}s between requests"
            )
            client = NetkeibaClient(
                min_sleep=args.sleep_min,
                max_sleep=args.sleep_max,
            )

        fetcher = MasterFetcher(
            conn, client,
            dry_run=args.dry_run,
            shutdown_handler=shutdown,
        )

        try:
            for entity_type in entity_types:
                # Check for graceful shutdown
                if shutdown.should_stop:
                    logger.info("Graceful shutdown - skipping remaining entity types")
                    break

                print(f"\n{'=' * 60}")
                print(f"Processing {entity_type.upper()}")
                print("=" * 60)

                # Extract entity IDs from race_results
                logger.info(f"Extracting {entity_type} IDs from race_results...")

                if args.active_only and entity_type in ("jockey", "trainer"):
                    entity_ids = get_active_entity_ids(
                        conn, entity_type,
                        args.start_year, args.end_year
                    )
                else:
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

                # Apply priority ordering if specified
                if args.priority:
                    fetcher.reorder_pending_by_priority(entity_type, args.priority)

                if args.init_only:
                    continue

                # Show ETA before starting
                progress = fetcher.status_manager.get_progress(entity_type)
                if progress["pending"] > 0:
                    avg_sleep = (args.sleep_min + args.sleep_max) / 2
                    est_minutes = progress["pending"] * avg_sleep / 60
                    logger.info(
                        f"Estimated time for {progress['pending']:,} pending: "
                        f"~{est_minutes:.0f} min at current rate"
                    )

                # Fetch
                stats = fetcher.fetch_entities(entity_type, limit)
                logger.info(
                    f"Completed: success={stats['success']}, "
                    f"failed={stats['failed']}, skipped={stats['skipped']}"
                )

                # Check for safe stop
                if fetcher.consecutive_failures >= fetcher.max_consecutive_failures:
                    logger.warning("Stopping due to consecutive failures")
                    break

        finally:
            if client:
                client.close()

        # Print final report
        print_coverage_report(conn)

    finally:
        shutdown.uninstall()
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
