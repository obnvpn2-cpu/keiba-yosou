#!/usr/bin/env python3
"""
masters_migration.py - Schema Migration for Master Tables (Road 2 + PR2.5)

Adds tables for:
- horses: Horse master data with basic pedigree
- jockeys: Jockey master data
- trainers: Trainer master data
- fetch_status: Track fetch progress for resume capability
- horse_pedigree: 5-generation pedigree (normalized, PR2.5)

These migrations integrate with src/db/schema_migration.py

【重要: 未来リーク防止ルール】
Master tables contain mostly "static" attributes that don't change over time:
- horse: name, sex, birth_date, coat_color, breeder, pedigree
- jockey/trainer: name, affiliation, birth_date

The following fields are TIME-SENSITIVE and may cause future leakage:
- total_prize, total_starts, total_wins, career_wins, career_in3, career_starts
- These values grow over time as the entity participates in more races

*** DO NOT USE CAREER STATS FOR ML FEATURES ***

If you need career stats, compute them as-of the race date from race_results.
The scraped_at / updated_at fields indicate when the data was fetched.
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================
# Table DDL
# ============================================================

CREATE_HORSES_TABLE = """
CREATE TABLE IF NOT EXISTS horses (
    horse_id TEXT PRIMARY KEY,
    horse_name TEXT,
    sex TEXT,                    -- 牡/牝/セン
    birth_date TEXT,             -- YYYY-MM-DD or YYYY
    coat_color TEXT,             -- 毛色
    breeder TEXT,
    breeder_region TEXT,         -- 生産地域
    owner TEXT,
    owner_id TEXT,

    -- Pedigree (血統) - Basic 3-gen from horse profile
    sire_id TEXT,                -- 父 horse_id
    sire_name TEXT,              -- 父 名前
    dam_id TEXT,                 -- 母 horse_id
    dam_name TEXT,               -- 母 名前
    broodmare_sire_id TEXT,      -- 母父 horse_id
    broodmare_sire_name TEXT,    -- 母父 名前

    -- Additional pedigree (取れれば)
    sire_sire_name TEXT,         -- 父父
    sire_dam_name TEXT,          -- 父母
    dam_dam_name TEXT,           -- 母母

    -- Stats from profile page
    -- *** WARNING: TIME-SENSITIVE - DO NOT USE FOR ML FEATURES ***
    -- These values grow over time. Use as-of computation from race_results instead.
    total_prize INTEGER,         -- 総獲得賞金 (scraped_at時点)
    total_starts INTEGER,        -- 出走数 (scraped_at時点)
    total_wins INTEGER,          -- 勝利数 (scraped_at時点)

    -- Metadata
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
);
"""

# PR2.5: 5-generation pedigree table (normalized)
CREATE_HORSE_PEDIGREE_TABLE = """
CREATE TABLE IF NOT EXISTS horse_pedigree (
    horse_id TEXT NOT NULL,
    generation INTEGER NOT NULL,      -- 1..5 (1=父母, 2=祖父母, ...)
    position TEXT NOT NULL,           -- "s", "d", "ss", "sd", ... (sire/dam path)
    ancestor_id TEXT,                 -- 祖先の horse_id (リンクがあれば)
    ancestor_name TEXT NOT NULL,      -- 祖先の名前

    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),

    PRIMARY KEY (horse_id, generation, position)
);
"""

CREATE_JOCKEYS_TABLE = """
CREATE TABLE IF NOT EXISTS jockeys (
    jockey_id TEXT PRIMARY KEY,
    jockey_name TEXT,
    name_kana TEXT,              -- ふりがな
    birth_date TEXT,
    affiliation TEXT,            -- 所属 (美浦/栗東/地方)
    debut_year INTEGER,

    -- Career stats (from profile, as-of fetch time)
    -- *** WARNING: TIME-SENSITIVE - DO NOT USE FOR ML FEATURES ***
    -- These values grow over time. Use as-of computation from race_results instead.
    career_wins INTEGER,         -- 勝利数 (scraped_at時点)
    career_in3 INTEGER,          -- 3着内数 (scraped_at時点)
    career_starts INTEGER,       -- 出走数 (scraped_at時点)

    -- Metadata
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
);
"""

CREATE_TRAINERS_TABLE = """
CREATE TABLE IF NOT EXISTS trainers (
    trainer_id TEXT PRIMARY KEY,
    trainer_name TEXT,
    name_kana TEXT,
    birth_date TEXT,
    affiliation TEXT,            -- 所属 (美浦/栗東)

    -- Career stats (from profile, as-of fetch time)
    -- *** WARNING: TIME-SENSITIVE - DO NOT USE FOR ML FEATURES ***
    -- These values grow over time. Use as-of computation from race_results instead.
    career_wins INTEGER,         -- 勝利数 (scraped_at時点)
    career_in3 INTEGER,          -- 3着内数 (scraped_at時点)
    career_starts INTEGER,       -- 出走数 (scraped_at時点)

    -- Metadata
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
);
"""

CREATE_FETCH_STATUS_TABLE = """
CREATE TABLE IF NOT EXISTS fetch_status (
    entity_type TEXT NOT NULL,   -- 'horse', 'jockey', 'trainer'
    entity_id TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- 'pending', 'success', 'failed', 'skipped'
    fetched_at TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    last_attempt_at TEXT,

    PRIMARY KEY (entity_type, entity_id)
);
"""

CREATE_INDEXES = """
-- Horses indexes
CREATE INDEX IF NOT EXISTS idx_horses_sire_id ON horses(sire_id);
CREATE INDEX IF NOT EXISTS idx_horses_dam_id ON horses(dam_id);
CREATE INDEX IF NOT EXISTS idx_horses_broodmare_sire_id ON horses(broodmare_sire_id);

-- Jockeys indexes
CREATE INDEX IF NOT EXISTS idx_jockeys_affiliation ON jockeys(affiliation);

-- Trainers indexes
CREATE INDEX IF NOT EXISTS idx_trainers_affiliation ON trainers(affiliation);

-- Fetch status indexes
CREATE INDEX IF NOT EXISTS idx_fetch_status_status ON fetch_status(status);
CREATE INDEX IF NOT EXISTS idx_fetch_status_type_status ON fetch_status(entity_type, status);
"""

# PR2.5: Indexes for horse_pedigree
CREATE_PEDIGREE_INDEXES = """
-- horse_pedigree indexes for ancestry queries
CREATE INDEX IF NOT EXISTS idx_horse_pedigree_horse_id ON horse_pedigree(horse_id);
CREATE INDEX IF NOT EXISTS idx_horse_pedigree_ancestor_id ON horse_pedigree(ancestor_id);
CREATE INDEX IF NOT EXISTS idx_horse_pedigree_gen ON horse_pedigree(generation);
"""


# ============================================================
# Migration Definitions (for schema_migration.py)
# ============================================================

ROAD2_MIGRATIONS = [
    {
        "id": "010_create_horses_table",
        "description": "Create horses master table with pedigree",
        "up": CREATE_HORSES_TABLE,
    },
    {
        "id": "011_create_jockeys_table",
        "description": "Create jockeys master table",
        "up": CREATE_JOCKEYS_TABLE,
    },
    {
        "id": "012_create_trainers_table",
        "description": "Create trainers master table",
        "up": CREATE_TRAINERS_TABLE,
    },
    {
        "id": "013_create_fetch_status_table",
        "description": "Create fetch_status table for resume capability",
        "up": CREATE_FETCH_STATUS_TABLE,
    },
    {
        "id": "014_create_masters_indexes",
        "description": "Create indexes for master tables",
        "up": CREATE_INDEXES,
    },
    # PR2.5: 5-generation pedigree
    {
        "id": "015_create_horse_pedigree_table",
        "description": "Create horse_pedigree table for 5-gen normalized pedigree",
        "up": CREATE_HORSE_PEDIGREE_TABLE,
    },
    {
        "id": "016_create_pedigree_indexes",
        "description": "Create indexes for horse_pedigree table",
        "up": CREATE_PEDIGREE_INDEXES,
    },
]


# ============================================================
# Migration Functions
# ============================================================


def run_road2_migrations(conn: sqlite3.Connection) -> int:
    """
    Run Road 2 migrations for master tables.

    Args:
        conn: SQLite connection

    Returns:
        Number of migrations applied
    """
    from src.db.schema_migration import (
        table_exists,
        get_applied_migrations,
        record_migration,
    )

    applied = get_applied_migrations(conn)
    count = 0

    for migration in ROAD2_MIGRATIONS:
        migration_id = migration["id"]
        description = migration["description"]

        if migration_id in applied:
            logger.debug(f"[SKIP] {migration_id} - already applied")
            continue

        logger.info(f"Applying: {migration_id} - {description}")

        try:
            conn.executescript(migration["up"])
            conn.commit()
            record_migration(conn, migration_id, description)
            count += 1
            logger.info(f"Applied: {migration_id}")
        except Exception as e:
            logger.error(f"Failed to apply {migration_id}: {e}")
            conn.rollback()
            raise

    return count


def create_master_tables(conn: sqlite3.Connection) -> None:
    """
    Create all master tables (idempotent).

    This is a simpler alternative to run_road2_migrations()
    for when you just want to ensure tables exist.
    """
    conn.executescript(CREATE_HORSES_TABLE)
    conn.executescript(CREATE_JOCKEYS_TABLE)
    conn.executescript(CREATE_TRAINERS_TABLE)
    conn.executescript(CREATE_FETCH_STATUS_TABLE)
    conn.executescript(CREATE_HORSE_PEDIGREE_TABLE)  # PR2.5
    conn.executescript(CREATE_INDEXES)
    conn.executescript(CREATE_PEDIGREE_INDEXES)  # PR2.5
    conn.commit()
    logger.info("Master tables created/verified")


# ============================================================
# Fetch Status Management
# ============================================================


class FetchStatusManager:
    """
    Manage fetch status for resume capability.

    Example:
        manager = FetchStatusManager(conn)

        # Get pending items
        pending = manager.get_pending('horse', limit=100)

        # Mark as success
        manager.mark_success('horse', '2020104385')

        # Mark as failed with error
        manager.mark_failed('horse', '2020104386', 'HTTP 404')
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def ensure_table(self) -> None:
        """Ensure fetch_status table exists."""
        self.conn.executescript(CREATE_FETCH_STATUS_TABLE)
        self.conn.commit()

    def add_pending(
        self,
        entity_type: str,
        entity_ids: List[str],
        skip_existing: bool = True,
    ) -> int:
        """
        Add entities as pending if not already tracked.

        Args:
            entity_type: 'horse', 'jockey', 'trainer'
            entity_ids: List of entity IDs
            skip_existing: If True, skip already tracked entities

        Returns:
            Number of entities added
        """
        if not entity_ids:
            return 0

        if skip_existing:
            # Get existing IDs
            placeholders = ",".join(["?"] * len(entity_ids))
            cursor = self.conn.execute(
                f"""SELECT entity_id FROM fetch_status
                   WHERE entity_type = ? AND entity_id IN ({placeholders})""",
                [entity_type] + list(entity_ids)
            )
            existing = {row[0] for row in cursor.fetchall()}
            entity_ids = [eid for eid in entity_ids if eid not in existing]

        if not entity_ids:
            return 0

        self.conn.executemany(
            """INSERT OR IGNORE INTO fetch_status (entity_type, entity_id, status)
               VALUES (?, ?, 'pending')""",
            [(entity_type, eid) for eid in entity_ids]
        )
        self.conn.commit()
        return len(entity_ids)

    def get_pending(
        self,
        entity_type: str,
        limit: int = 100,
        max_retries: int = 3,
    ) -> List[str]:
        """
        Get pending entity IDs to fetch.

        Args:
            entity_type: 'horse', 'jockey', 'trainer'
            limit: Maximum number of IDs to return
            max_retries: Skip items with retry_count >= max_retries

        Returns:
            List of entity IDs
        """
        cursor = self.conn.execute(
            """SELECT entity_id FROM fetch_status
               WHERE entity_type = ?
               AND (status = 'pending' OR (status = 'failed' AND retry_count < ?))
               ORDER BY retry_count ASC, entity_id ASC
               LIMIT ?""",
            (entity_type, max_retries, limit)
        )
        return [row[0] for row in cursor.fetchall()]

    def mark_success(self, entity_type: str, entity_id: str) -> None:
        """Mark an entity as successfully fetched."""
        self.conn.execute(
            """UPDATE fetch_status
               SET status = 'success',
                   fetched_at = datetime('now', 'localtime'),
                   error_message = NULL
               WHERE entity_type = ? AND entity_id = ?""",
            (entity_type, entity_id)
        )
        self.conn.commit()

    def mark_failed(
        self,
        entity_type: str,
        entity_id: str,
        error_message: str,
    ) -> None:
        """Mark an entity as failed with error message."""
        self.conn.execute(
            """UPDATE fetch_status
               SET status = 'failed',
                   error_message = ?,
                   retry_count = retry_count + 1,
                   last_attempt_at = datetime('now', 'localtime')
               WHERE entity_type = ? AND entity_id = ?""",
            (error_message, entity_type, entity_id)
        )
        self.conn.commit()

    def mark_skipped(self, entity_type: str, entity_id: str, reason: str = None) -> None:
        """Mark an entity as skipped."""
        self.conn.execute(
            """UPDATE fetch_status
               SET status = 'skipped',
                   error_message = ?
               WHERE entity_type = ? AND entity_id = ?""",
            (reason, entity_type, entity_id)
        )
        self.conn.commit()

    def get_stats(self, entity_type: str = None) -> Dict[str, Any]:
        """
        Get fetch status statistics.

        Args:
            entity_type: Filter by type, or None for all

        Returns:
            Dict with counts by status
        """
        if entity_type:
            cursor = self.conn.execute(
                """SELECT status, COUNT(*) FROM fetch_status
                   WHERE entity_type = ?
                   GROUP BY status""",
                (entity_type,)
            )
        else:
            cursor = self.conn.execute(
                """SELECT status, COUNT(*) FROM fetch_status
                   GROUP BY status"""
            )

        stats = dict(cursor.fetchall())
        stats["total"] = sum(stats.values())
        return stats

    def get_progress(self, entity_type: str) -> Dict[str, Any]:
        """Get detailed progress for an entity type."""
        cursor = self.conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                   SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                   SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) as skipped
               FROM fetch_status
               WHERE entity_type = ?""",
            (entity_type,)
        )
        row = cursor.fetchone()
        return {
            "total": row[0] or 0,
            "success": row[1] or 0,
            "failed": row[2] or 0,
            "pending": row[3] or 0,
            "skipped": row[4] or 0,
            "progress_pct": 100 * (row[1] or 0) / row[0] if row[0] else 0,
        }


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="Run Road 2 migrations")
    parser.add_argument("--db", default="netkeiba.db", help="Database path")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    if not Path(args.db).exists():
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    try:
        count = run_road2_migrations(conn)
        logger.info(f"Applied {count} migrations")
    finally:
        conn.close()
