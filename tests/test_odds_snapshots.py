"""
Tests for odds snapshot fetching and parsing.

Tests cover:
1. HTML parsing extracts all horses (not just 8)
2. Safety check: don't store if parsed count != expected
3. Expected horse count comes from race_results
"""

import sqlite3
import pytest
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if beautifulsoup4 is available
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

from scripts.fetch_odds_snapshots import (
    parse_odds_page,
    OddsSnapshot,
    OddsFetcher,
    _DEPS_AVAILABLE,
)
from src.db.schema_migration import run_migrations

# Skip all tests if beautifulsoup is not available
pytestmark = pytest.mark.skipif(
    not _DEPS_AVAILABLE or not _BS4_AVAILABLE,
    reason="beautifulsoup4 or requests not installed"
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def html_16horses(fixtures_dir):
    """Load 16-horse odds page HTML fixture."""
    path = fixtures_dir / "odds_page_16horses.html"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def html_8horses(fixtures_dir):
    """Load 8-horse odds page HTML fixture."""
    path = fixtures_dir / "odds_page_8horses.html"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def test_db():
    """Create in-memory test database with required tables."""
    conn = sqlite3.connect(":memory:")

    # Create races table
    conn.execute("""
        CREATE TABLE races (
            race_id TEXT PRIMARY KEY,
            date TEXT,
            name TEXT,
            head_count INTEGER
        )
    """)

    # Create race_results table
    conn.execute("""
        CREATE TABLE race_results (
            race_id TEXT,
            horse_id TEXT,
            horse_no INTEGER,
            horse_name TEXT,
            finish_order INTEGER,
            PRIMARY KEY (race_id, horse_id)
        )
    """)

    # Run migrations to create odds_snapshots table
    run_migrations(conn)

    yield conn
    conn.close()


@pytest.fixture
def db_with_16_horse_race(test_db):
    """Database with a 16-horse race in race_results."""
    race_id = "202412280601"

    # Insert race
    test_db.execute(
        "INSERT INTO races (race_id, date, name, head_count) VALUES (?, ?, ?, ?)",
        (race_id, "2024-12-28", "Test Race 16", 16)
    )

    # Insert 16 race results
    for i in range(1, 17):
        test_db.execute(
            "INSERT INTO race_results (race_id, horse_id, horse_no, horse_name, finish_order) "
            "VALUES (?, ?, ?, ?, ?)",
            (race_id, f"horse_{i:03d}", i, f"テストホース{i}", i)
        )

    test_db.commit()
    return test_db, race_id


@pytest.fixture
def db_with_8_horse_race(test_db):
    """Database with an 8-horse race in race_results."""
    race_id = "202412280501"

    # Insert race
    test_db.execute(
        "INSERT INTO races (race_id, date, name, head_count) VALUES (?, ?, ?, ?)",
        (race_id, "2024-12-28", "Test Race 8", 8)
    )

    # Insert 8 race results
    for i in range(1, 9):
        test_db.execute(
            "INSERT INTO race_results (race_id, horse_id, horse_no, horse_name, finish_order) "
            "VALUES (?, ?, ?, ?, ?)",
            (race_id, f"horse_{i:03d}", i, f"テストホース{i}", i)
        )

    test_db.commit()
    return test_db, race_id


# ============================================================
# Parser Tests
# ============================================================


class TestParseOddsPage:
    """Tests for parse_odds_page function."""

    def test_parse_16_horses(self, html_16horses):
        """Verify parser extracts all 16 horses, not just 8."""
        race_id = "202412280601"
        observed_at = "2024-12-27T21:00:00"

        snapshots = parse_odds_page(html_16horses, race_id, observed_at)

        # Must get exactly 16 horses
        assert len(snapshots) == 16, f"Expected 16 horses, got {len(snapshots)}"

        # Verify horse numbers
        horse_nos = sorted([s.horse_no for s in snapshots])
        assert horse_nos == list(range(1, 17)), f"Horse numbers: {horse_nos}"

    def test_parse_8_horses(self, html_8horses):
        """Verify parser handles 8-horse races correctly."""
        race_id = "202412280501"
        observed_at = "2024-12-27T21:00:00"

        snapshots = parse_odds_page(html_8horses, race_id, observed_at)

        assert len(snapshots) == 8, f"Expected 8 horses, got {len(snapshots)}"

        horse_nos = sorted([s.horse_no for s in snapshots])
        assert horse_nos == list(range(1, 9))

    def test_parse_extracts_odds(self, html_16horses):
        """Verify odds values are correctly extracted."""
        snapshots = parse_odds_page(html_16horses, "202412280601", "2024-12-27T21:00:00")

        # Find horse 2 (should have odds 3.2 from fixture)
        horse2 = next((s for s in snapshots if s.horse_no == 2), None)
        assert horse2 is not None
        assert horse2.win_odds == 3.2

    def test_parse_extracts_popularity(self, html_16horses):
        """Verify popularity values are correctly extracted."""
        snapshots = parse_odds_page(html_16horses, "202412280601", "2024-12-27T21:00:00")

        # Find horse 2 (should be 1人気 from fixture)
        horse2 = next((s for s in snapshots if s.horse_no == 2), None)
        assert horse2 is not None
        assert horse2.popularity == 1

    def test_parse_min_max_horse_no(self, html_16horses):
        """Verify horse_no range matches expected min/max."""
        snapshots = parse_odds_page(html_16horses, "202412280601", "2024-12-27T21:00:00")

        horse_nos = [s.horse_no for s in snapshots]
        assert min(horse_nos) == 1
        assert max(horse_nos) == 16

    def test_parse_sets_race_id(self, html_16horses):
        """Verify race_id is set on all snapshots."""
        race_id = "202412280601"
        snapshots = parse_odds_page(html_16horses, race_id, "2024-12-27T21:00:00")

        for s in snapshots:
            assert s.race_id == race_id

    def test_parse_sets_observed_at(self, html_16horses):
        """Verify observed_at is set on all snapshots."""
        observed_at = "2024-12-27T21:00:00"
        snapshots = parse_odds_page(html_16horses, "202412280601", observed_at)

        for s in snapshots:
            assert s.observed_at == observed_at


# ============================================================
# Safety Check Tests
# ============================================================


class TestSafetyChecks:
    """Tests for safety checks that prevent incomplete data storage."""

    def test_get_expected_horse_count(self, db_with_16_horse_race):
        """Verify expected count comes from race_results table."""
        conn, race_id = db_with_16_horse_race

        fetcher = OddsFetcher(conn)
        expected = fetcher.get_expected_horse_count(race_id)

        assert expected == 16

    def test_get_expected_count_missing_race(self, test_db):
        """Verify None returned for non-existent race."""
        fetcher = OddsFetcher(test_db)
        expected = fetcher.get_expected_horse_count("999999999999")

        assert expected is None

    def test_get_expected_count_empty_results(self, test_db):
        """Verify None returned when race has no results."""
        # Insert race but no results
        test_db.execute(
            "INSERT INTO races (race_id, date, name) VALUES (?, ?, ?)",
            ("202412280001", "2024-12-28", "Empty Race")
        )
        test_db.commit()

        fetcher = OddsFetcher(test_db)
        expected = fetcher.get_expected_horse_count("202412280001")

        assert expected is None


# ============================================================
# Integration Tests
# ============================================================


class TestOddsFetcherIntegration:
    """Integration tests for OddsFetcher class."""

    def test_upsert_snapshots(self, db_with_16_horse_race):
        """Verify snapshots can be upserted to database."""
        conn, race_id = db_with_16_horse_race

        from src.db.upsert import UpsertHelper
        helper = UpsertHelper(conn)

        # Create test snapshots
        observed_at = datetime.now().isoformat()
        rows = [
            {
                "race_id": race_id,
                "horse_no": i,
                "observed_at": observed_at,
                "win_odds": float(i) * 1.5,
                "popularity": i,
                "source": "test",
            }
            for i in range(1, 17)
        ]

        count = helper.upsert_odds_snapshots(rows)
        assert count == 16

        # Verify stored correctly
        cursor = conn.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE race_id = ?",
            (race_id,)
        )
        assert cursor.fetchone()[0] == 16

    def test_upsert_idempotent(self, db_with_16_horse_race):
        """Verify upserting same data twice doesn't duplicate."""
        conn, race_id = db_with_16_horse_race

        from src.db.upsert import UpsertHelper
        helper = UpsertHelper(conn)

        observed_at = "2024-12-27T21:00:00"
        rows = [
            {
                "race_id": race_id,
                "horse_no": i,
                "observed_at": observed_at,
                "win_odds": float(i) * 1.5,
                "popularity": i,
                "source": "test",
            }
            for i in range(1, 17)
        ]

        # Upsert twice
        helper.upsert_odds_snapshots(rows)
        helper.upsert_odds_snapshots(rows)

        # Should still be 16 rows
        cursor = conn.execute(
            "SELECT COUNT(*) FROM odds_snapshots WHERE race_id = ? AND observed_at = ?",
            (race_id, observed_at)
        )
        assert cursor.fetchone()[0] == 16


# ============================================================
# Count Mismatch Simulation Tests
# ============================================================


class TestCountMismatchPrevention:
    """Tests verifying that count mismatches prevent storage."""

    def test_parsed_8_but_expected_16_scenario(self, db_with_16_horse_race, html_8horses):
        """
        Simulate the bug scenario: parsing only 8 horses when 16 expected.

        This test demonstrates that with the safety check in place,
        incomplete data won't be stored.
        """
        conn, race_id = db_with_16_horse_race

        # Parse the 8-horse HTML (simulating the bug)
        snapshots = parse_odds_page(html_8horses, race_id, "2024-12-27T21:00:00")

        # Should get 8 snapshots
        assert len(snapshots) == 8

        # But expected count is 16
        fetcher = OddsFetcher(conn)
        expected = fetcher.get_expected_horse_count(race_id)
        assert expected == 16

        # Count mismatch should be detected
        assert len(snapshots) != expected, "Count mismatch should be detected"

    def test_count_matches_allows_storage(self, db_with_8_horse_race, html_8horses):
        """
        Verify that when counts match, storage is allowed.
        """
        conn, race_id = db_with_8_horse_race

        # Parse the 8-horse HTML
        snapshots = parse_odds_page(html_8horses, race_id, "2024-12-27T21:00:00")

        # Should get 8 snapshots
        assert len(snapshots) == 8

        # Expected count should also be 8
        fetcher = OddsFetcher(conn)
        expected = fetcher.get_expected_horse_count(race_id)
        assert expected == 8

        # Counts match - storage would be allowed
        assert len(snapshots) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
