#!/usr/bin/env python3
"""
test_db_idempotency.py - Tests for database idempotency

Tests that:
1. UPSERT operations don't create duplicates
2. UNIQUE indexes exist on feature tables
3. Migrations are idempotent (can run multiple times)
4. Schema status is correctly reported

Usage:
    python -m pytest tests/test_db_idempotency.py -v
    python tests/test_db_idempotency.py  # Direct run
"""

import sqlite3
import tempfile
import os
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db.schema_migration import (
    run_migrations,
    get_schema_status,
    table_exists,
    index_exists,
    clean_duplicates,
    count_duplicates,
)
from db.upsert import (
    UpsertHelper,
    upsert_dataframe,
    build_upsert_sql,
    create_table_from_df,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)

    # Create test tables
    conn.executescript("""
        -- Simple test table with PK
        CREATE TABLE test_races (
            race_id TEXT PRIMARY KEY,
            name TEXT,
            date TEXT,
            updated_at TEXT
        );

        -- Test table with composite PK
        CREATE TABLE test_results (
            race_id TEXT NOT NULL,
            horse_id TEXT NOT NULL,
            finish_order INTEGER,
            PRIMARY KEY (race_id, horse_id)
        );

        -- Simulated feature_table_v2 (no unique constraint like pandas to_sql)
        CREATE TABLE feature_table_v2 (
            race_id TEXT,
            horse_id TEXT,
            target_win INTEGER,
            created_at TEXT,
            updated_at TEXT
        );

        -- Simulated feature_table_v3
        CREATE TABLE feature_table_v3 (
            race_id TEXT,
            horse_id TEXT,
            target_win INTEGER,
            target_in3 INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
    """)
    conn.commit()

    yield conn, path

    conn.close()
    os.unlink(path)


@pytest.fixture
def real_db():
    """Connect to actual netkeiba.db if exists."""
    db_path = Path("netkeiba.db")
    if not db_path.exists():
        pytest.skip("netkeiba.db not found")
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


# ============================================================
# Schema Migration Tests
# ============================================================


class TestSchemaMigration:
    """Tests for schema migration functionality."""

    def test_table_exists(self, temp_db):
        """Test table_exists function."""
        conn, _ = temp_db
        assert table_exists(conn, "test_races") is True
        assert table_exists(conn, "nonexistent_table") is False

    def test_index_exists(self, temp_db):
        """Test index_exists function."""
        conn, _ = temp_db
        # Create an index
        conn.execute("CREATE INDEX idx_test ON test_races(name)")
        conn.commit()

        assert index_exists(conn, "idx_test") is True
        assert index_exists(conn, "nonexistent_index") is False

    def test_count_duplicates_no_dups(self, temp_db):
        """Test counting duplicates when none exist."""
        conn, _ = temp_db
        conn.executemany(
            "INSERT INTO feature_table_v2 (race_id, horse_id, target_win) VALUES (?, ?, ?)",
            [("R1", "H1", 1), ("R1", "H2", 0), ("R2", "H1", 1)]
        )
        conn.commit()

        count = count_duplicates(conn, "feature_table_v2", ["race_id", "horse_id"])
        assert count == 0

    def test_count_duplicates_with_dups(self, temp_db):
        """Test counting duplicates when they exist."""
        conn, _ = temp_db
        conn.executemany(
            "INSERT INTO feature_table_v2 (race_id, horse_id, target_win) VALUES (?, ?, ?)",
            [
                ("R1", "H1", 1),
                ("R1", "H1", 0),  # Duplicate
                ("R1", "H2", 0),
                ("R2", "H1", 1),
                ("R2", "H1", 0),  # Duplicate
                ("R2", "H1", 1),  # Triple
            ]
        )
        conn.commit()

        count = count_duplicates(conn, "feature_table_v2", ["race_id", "horse_id"])
        assert count == 3  # 1 extra for R1H1, 2 extra for R2H1

    def test_clean_duplicates(self, temp_db):
        """Test cleaning duplicates keeps last row."""
        conn, _ = temp_db
        conn.executemany(
            "INSERT INTO feature_table_v2 (race_id, horse_id, target_win) VALUES (?, ?, ?)",
            [
                ("R1", "H1", 1),  # First
                ("R1", "H1", 2),  # Second (keep this one)
            ]
        )
        conn.commit()

        deleted = clean_duplicates(conn, "feature_table_v2", ["race_id", "horse_id"], keep="last")
        assert deleted == 1

        cursor = conn.execute(
            "SELECT target_win FROM feature_table_v2 WHERE race_id='R1' AND horse_id='H1'"
        )
        result = cursor.fetchone()
        assert result[0] == 2  # Should keep the last inserted value

    def test_migrations_idempotent(self, temp_db):
        """Test that migrations can run multiple times safely."""
        conn, _ = temp_db

        # Run migrations first time
        count1 = run_migrations(conn)

        # Run migrations second time
        count2 = run_migrations(conn)

        # Should not apply any migrations the second time
        assert count2 == 0

        # Check that unique indexes exist
        assert index_exists(conn, "idx_feature_table_v2_pk")
        assert index_exists(conn, "idx_feature_table_v3_pk")

    def test_get_schema_status(self, temp_db):
        """Test getting schema status."""
        conn, _ = temp_db
        run_migrations(conn)

        status = get_schema_status(conn)

        assert "tables" in status
        assert "feature_table_v2" in status["tables"]
        assert status["feature_table_v2_has_unique"] is True
        assert status["feature_table_v3_has_unique"] is True


# ============================================================
# UPSERT Tests
# ============================================================


class TestUpsert:
    """Tests for UPSERT functionality."""

    def test_build_upsert_sql(self):
        """Test SQL generation."""
        sql = build_upsert_sql(
            "test_table",
            ["id", "name", "value"],
            ["id"],
            ["name", "value"],
        )
        assert "INSERT INTO test_table" in sql
        assert "ON CONFLICT(id)" in sql
        assert "DO UPDATE SET" in sql
        assert "name = excluded.name" in sql

    def test_build_upsert_sql_composite_key(self):
        """Test SQL generation with composite key."""
        sql = build_upsert_sql(
            "test_table",
            ["race_id", "horse_id", "value"],
            ["race_id", "horse_id"],
            ["value"],
        )
        assert "ON CONFLICT(race_id, horse_id)" in sql

    def test_upsert_helper_insert(self, temp_db):
        """Test UpsertHelper inserts new rows."""
        conn, _ = temp_db
        helper = UpsertHelper(conn)

        rows = [
            {"race_id": "R1", "horse_id": "H1", "finish_order": 1},
            {"race_id": "R1", "horse_id": "H2", "finish_order": 2},
        ]
        count = helper.upsert_rows("test_results", rows)

        assert count == 2

        cursor = conn.execute("SELECT COUNT(*) FROM test_results")
        assert cursor.fetchone()[0] == 2

    def test_upsert_helper_update(self, temp_db):
        """Test UpsertHelper updates existing rows."""
        conn, _ = temp_db
        helper = UpsertHelper(conn)

        # Insert initial data
        rows1 = [
            {"race_id": "R1", "horse_id": "H1", "finish_order": 1},
        ]
        helper.upsert_rows("test_results", rows1)

        # Update with new value
        rows2 = [
            {"race_id": "R1", "horse_id": "H1", "finish_order": 3},
        ]
        helper.upsert_rows("test_results", rows2)

        # Should still have only 1 row
        cursor = conn.execute("SELECT COUNT(*) FROM test_results")
        assert cursor.fetchone()[0] == 1

        # Value should be updated
        cursor = conn.execute("SELECT finish_order FROM test_results WHERE race_id='R1'")
        assert cursor.fetchone()[0] == 3

    def test_upsert_idempotent(self, temp_db):
        """Test that upserting same data twice doesn't create duplicates."""
        conn, _ = temp_db

        # First, run migrations to add UNIQUE index
        run_migrations(conn)

        # Create sample data
        df = pd.DataFrame({
            "race_id": ["R1", "R1", "R2"],
            "horse_id": ["H1", "H2", "H1"],
            "target_win": [1, 0, 1],
        })

        # UPSERT first time
        count1 = upsert_dataframe(conn, df, "feature_table_v2", ["race_id", "horse_id"])
        assert count1 == 3

        # UPSERT second time with same data
        count2 = upsert_dataframe(conn, df, "feature_table_v2", ["race_id", "horse_id"])
        assert count2 == 3

        # Should still have only 3 rows (not 6)
        cursor = conn.execute("SELECT COUNT(*) FROM feature_table_v2")
        assert cursor.fetchone()[0] == 3

    def test_upsert_updates_existing(self, temp_db):
        """Test that UPSERT updates existing values."""
        conn, _ = temp_db
        run_migrations(conn)

        # Insert initial data
        df1 = pd.DataFrame({
            "race_id": ["R1"],
            "horse_id": ["H1"],
            "target_win": [0],
        })
        upsert_dataframe(conn, df1, "feature_table_v2", ["race_id", "horse_id"])

        # Update with new value
        df2 = pd.DataFrame({
            "race_id": ["R1"],
            "horse_id": ["H1"],
            "target_win": [1],
        })
        upsert_dataframe(conn, df2, "feature_table_v2", ["race_id", "horse_id"])

        # Check value was updated
        cursor = conn.execute(
            "SELECT target_win FROM feature_table_v2 WHERE race_id='R1' AND horse_id='H1'"
        )
        assert cursor.fetchone()[0] == 1

    def test_create_table_from_df(self, temp_db):
        """Test creating table with UNIQUE index from DataFrame."""
        conn, _ = temp_db

        # Drop the pre-created table
        conn.execute("DROP TABLE IF EXISTS feature_table_v2")
        conn.commit()

        df = pd.DataFrame({
            "race_id": ["R1", "R2"],
            "horse_id": ["H1", "H1"],
            "target_win": [1, 0],
            "some_feature": [0.5, 0.3],
        })

        create_table_from_df(conn, df, "feature_table_v2", ["race_id", "horse_id"])

        # Check table was created with data
        cursor = conn.execute("SELECT COUNT(*) FROM feature_table_v2")
        assert cursor.fetchone()[0] == 2

        # Check index was created
        assert index_exists(conn, "idx_feature_table_v2_pk")

        # Test that duplicate insert fails properly with UPSERT
        df_dup = pd.DataFrame({
            "race_id": ["R1"],
            "horse_id": ["H1"],
            "target_win": [0],
            "some_feature": [0.9],
        })
        upsert_dataframe(conn, df_dup, "feature_table_v2", ["race_id", "horse_id"])

        # Should still have 2 rows
        cursor = conn.execute("SELECT COUNT(*) FROM feature_table_v2")
        assert cursor.fetchone()[0] == 2


# ============================================================
# Integration Tests (require netkeiba.db)
# ============================================================


class TestRealDatabase:
    """Tests against real netkeiba.db (skipped if not available)."""

    def test_schema_status(self, real_db):
        """Test getting schema status from real DB."""
        status = get_schema_status(real_db)
        assert "tables" in status
        assert "races" in status["tables"]
        assert "race_results" in status["tables"]

    def test_migrations_on_real_db(self, real_db):
        """Test that migrations work on real DB."""
        # This is non-destructive - migrations are idempotent
        # and only add UNIQUE indexes (won't break existing data)
        count = run_migrations(real_db, dry_run=True)
        # Just check it doesn't crash
        assert count >= 0


# ============================================================
# CLI Runner
# ============================================================


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"],
        cwd=Path(__file__).parent.parent
    )
    sys.exit(result.returncode)
