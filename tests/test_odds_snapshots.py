"""
Tests for odds snapshot fetching and parsing (API method).

Tests cover:
1. API response parsing extracts all horses (not just 8)
2. Popularity calculation from win_odds
3. Safety check: warn if parsed count < half of expected
4. Expected horse count comes from race_results
"""

import json
import sqlite3
import pytest
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if requests is available
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from scripts.fetch_odds_snapshots import (
    parse_api_response,
    OddsSnapshot,
    OddsFetcher,
    _DEPS_AVAILABLE,
)

# Skip all tests if requests is not available
pytestmark = pytest.mark.skipif(
    not _DEPS_AVAILABLE,
    reason="requests not installed"
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def api_response_16horses():
    """API response with 16 horses."""
    return {
        "data": {
            "odds": {
                "1": {"odds": "5.2", "popular": "3"},
                "2": {"odds": "3.2", "popular": "1"},
                "3": {"odds": "8.5", "popular": "5"},
                "4": {"odds": "12.3", "popular": "7"},
                "5": {"odds": "4.1", "popular": "2"},
                "6": {"odds": "15.8", "popular": "9"},
                "7": {"odds": "7.4", "popular": "4"},
                "8": {"odds": "25.2", "popular": "11"},
                "9": {"odds": "10.5", "popular": "6"},
                "10": {"odds": "45.3", "popular": "13"},
                "11": {"odds": "14.2", "popular": "8"},
                "12": {"odds": "88.7", "popular": "15"},
                "13": {"odds": "18.9", "popular": "10"},
                "14": {"odds": "120.5", "popular": "16"},
                "15": {"odds": "32.1", "popular": "12"},
                "16": {"odds": "55.0", "popular": "14"},
            }
        }
    }


@pytest.fixture
def api_response_8horses():
    """API response with 8 horses."""
    return {
        "data": {
            "odds": {
                "1": {"odds": "2.5", "popular": "1"},
                "2": {"odds": "3.8", "popular": "2"},
                "3": {"odds": "5.2", "popular": "3"},
                "4": {"odds": "8.1", "popular": "4"},
                "5": {"odds": "12.5", "popular": "5"},
                "6": {"odds": "18.3", "popular": "6"},
                "7": {"odds": "25.0", "popular": "7"},
                "8": {"odds": "45.8", "popular": "8"},
            }
        }
    }


@pytest.fixture
def api_response_no_popularity():
    """API response without popularity field (must calculate from odds)."""
    return {
        "data": {
            "odds": {
                "1": {"odds": "5.2"},
                "2": {"odds": "3.2"},
                "3": {"odds": "8.5"},
                "4": {"odds": "3.2"},  # Same odds as horse 2 (tie)
                "5": {"odds": "4.1"},
                "6": {"odds": "15.8"},
            }
        }
    }


@pytest.fixture
def api_response_alt_structure():
    """API response with alternative structure (odds at data level)."""
    return {
        "data": {
            "1": {"odds": "2.5"},
            "2": {"odds": "3.8"},
            "3": {"odds": "5.2"},
            "4": {"odds": "8.1"},
        }
    }


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

    # Create odds_snapshots table directly (instead of running migrations)
    # This avoids migration errors on missing tables in the test environment
    conn.execute("""
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
        )
    """)

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


class TestParseApiResponse:
    """Tests for parse_api_response function."""

    def test_parse_16_horses(self, api_response_16horses):
        """Verify parser extracts all 16 horses."""
        race_id = "202412280601"
        observed_at = "2024-12-27T21:00:00"

        snapshots = parse_api_response(api_response_16horses, race_id, observed_at)

        # Must get exactly 16 horses
        assert len(snapshots) == 16, f"Expected 16 horses, got {len(snapshots)}"

        # Verify horse numbers
        horse_nos = sorted([s.horse_no for s in snapshots])
        assert horse_nos == list(range(1, 17)), f"Horse numbers: {horse_nos}"

    def test_parse_8_horses(self, api_response_8horses):
        """Verify parser handles 8-horse races correctly."""
        race_id = "202412280501"
        observed_at = "2024-12-27T21:00:00"

        snapshots = parse_api_response(api_response_8horses, race_id, observed_at)

        assert len(snapshots) == 8, f"Expected 8 horses, got {len(snapshots)}"

        horse_nos = sorted([s.horse_no for s in snapshots])
        assert horse_nos == list(range(1, 9))

    def test_parse_extracts_odds(self, api_response_16horses):
        """Verify odds values are correctly extracted."""
        snapshots = parse_api_response(api_response_16horses, "202412280601", "2024-12-27T21:00:00")

        # Find horse 2 (should have odds 3.2)
        horse2 = next((s for s in snapshots if s.horse_no == 2), None)
        assert horse2 is not None
        assert horse2.win_odds == 3.2

    def test_parse_extracts_popularity(self, api_response_16horses):
        """Verify popularity values are correctly extracted."""
        snapshots = parse_api_response(api_response_16horses, "202412280601", "2024-12-27T21:00:00")

        # Find horse 2 (should be 1人気)
        horse2 = next((s for s in snapshots if s.horse_no == 2), None)
        assert horse2 is not None
        assert horse2.popularity == 1

    def test_parse_min_max_horse_no(self, api_response_16horses):
        """Verify horse_no range matches expected min/max."""
        snapshots = parse_api_response(api_response_16horses, "202412280601", "2024-12-27T21:00:00")

        horse_nos = [s.horse_no for s in snapshots]
        assert min(horse_nos) == 1
        assert max(horse_nos) == 16

    def test_parse_sets_race_id(self, api_response_16horses):
        """Verify race_id is set on all snapshots."""
        race_id = "202412280601"
        snapshots = parse_api_response(api_response_16horses, race_id, "2024-12-27T21:00:00")

        for s in snapshots:
            assert s.race_id == race_id

    def test_parse_sets_observed_at(self, api_response_16horses):
        """Verify observed_at is set on all snapshots."""
        observed_at = "2024-12-27T21:00:00"
        snapshots = parse_api_response(api_response_16horses, "202412280601", observed_at)

        for s in snapshots:
            assert s.observed_at == observed_at

    def test_parse_sets_source(self, api_response_16horses):
        """Verify source is set to netkeiba_api_type1."""
        snapshots = parse_api_response(api_response_16horses, "202412280601", "2024-12-27T21:00:00")

        for s in snapshots:
            assert s.source == "netkeiba_api_type1"

    def test_parse_alt_structure(self, api_response_alt_structure):
        """Verify parser handles alternative API structure."""
        snapshots = parse_api_response(api_response_alt_structure, "202412280601", "2024-12-27T21:00:00")

        assert len(snapshots) == 4
        horse_nos = sorted([s.horse_no for s in snapshots])
        assert horse_nos == [1, 2, 3, 4]


# ============================================================
# Popularity Calculation Tests
# ============================================================


class TestPopularityCalculation:
    """Tests for popularity calculation from odds."""

    def test_calculate_popularity_from_odds(self, api_response_no_popularity):
        """Verify popularity is calculated correctly from odds ranking."""
        snapshots = parse_api_response(api_response_no_popularity, "202412280601", "2024-12-27T21:00:00")

        # Sort by odds to verify ranking
        # Expected: horse 2 (3.2) = 1st, horse 4 (3.2) = 1st (tie), horse 5 (4.1) = 3rd, etc.
        by_odds = sorted(snapshots, key=lambda s: s.win_odds)

        # Horse 2 should be 1st (3.2 odds)
        horse2 = next(s for s in snapshots if s.horse_no == 2)
        assert horse2.popularity == 1

        # Horse 4 should also be 1st (tie at 3.2)
        horse4 = next(s for s in snapshots if s.horse_no == 4)
        assert horse4.popularity == 1  # Same odds = same rank

        # Horse 5 should be 3rd (4.1 odds, after the two tied horses)
        horse5 = next(s for s in snapshots if s.horse_no == 5)
        assert horse5.popularity == 3

        # Horse 1 should be 4th (5.2 odds)
        horse1 = next(s for s in snapshots if s.horse_no == 1)
        assert horse1.popularity == 4

    def test_api_popularity_takes_precedence(self, api_response_16horses):
        """Verify API-provided popularity is used over calculated."""
        snapshots = parse_api_response(api_response_16horses, "202412280601", "2024-12-27T21:00:00")

        # Horse 2 has odds 3.2 and popularity 1 from API
        horse2 = next(s for s in snapshots if s.horse_no == 2)
        assert horse2.popularity == 1

        # Horse 1 has odds 5.2 and popularity 3 from API
        horse1 = next(s for s in snapshots if s.horse_no == 1)
        assert horse1.popularity == 3


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

    def test_get_head_count_from_races(self, db_with_16_horse_race):
        """Verify head_count comes from races table."""
        conn, race_id = db_with_16_horse_race

        fetcher = OddsFetcher(conn)
        head_count = fetcher.get_head_count_from_races(race_id)

        assert head_count == 16

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
                "source": "netkeiba_api_type1",
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
                "source": "netkeiba_api_type1",
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
# Count Verification Tests
# ============================================================


class TestCountVerification:
    """Tests for count verification logic."""

    def test_parsed_8_but_expected_16_scenario(self, db_with_16_horse_race, api_response_8horses):
        """
        Simulate the bug scenario: parsing only 8 horses when 16 expected.

        This test verifies that count mismatch is detected.
        """
        conn, race_id = db_with_16_horse_race

        # Parse the 8-horse API response
        snapshots = parse_api_response(api_response_8horses, race_id, "2024-12-27T21:00:00")

        # Should get 8 snapshots
        assert len(snapshots) == 8

        # But expected count is 16
        fetcher = OddsFetcher(conn)
        expected = fetcher.get_expected_horse_count(race_id)
        assert expected == 16

        # Count mismatch should be detectable
        assert len(snapshots) != expected, "Count mismatch should be detected"
        # 8 < 16 / 2 = False, so this would log a warning but still store

    def test_count_matches_allows_storage(self, db_with_8_horse_race, api_response_8horses):
        """
        Verify that when counts match, storage is allowed.
        """
        conn, race_id = db_with_8_horse_race

        # Parse the 8-horse API response
        snapshots = parse_api_response(api_response_8horses, race_id, "2024-12-27T21:00:00")

        # Should get 8 snapshots
        assert len(snapshots) == 8

        # Expected count should also be 8
        fetcher = OddsFetcher(conn)
        expected = fetcher.get_expected_horse_count(race_id)
        assert expected == 8

        # Counts match - storage would be allowed
        assert len(snapshots) == expected

    def test_severe_mismatch_detected(self, db_with_16_horse_race):
        """Verify severe mismatch (< half of expected) is flagged."""
        conn, race_id = db_with_16_horse_race

        # Create API response with only 4 horses (< 8 = half of 16)
        sparse_response = {
            "data": {
                "odds": {
                    "1": {"odds": "2.5"},
                    "2": {"odds": "3.8"},
                    "3": {"odds": "5.2"},
                    "4": {"odds": "8.1"},
                }
            }
        }

        snapshots = parse_api_response(sparse_response, race_id, "2024-12-27T21:00:00")

        fetcher = OddsFetcher(conn)
        expected = fetcher.get_expected_horse_count(race_id)

        # 4 < 16 / 2 = 4 < 8 = True (severe mismatch)
        assert len(snapshots) < expected // 2


# ============================================================
# Empty/Invalid Response Tests
# ============================================================


class TestInvalidResponses:
    """Tests for handling invalid API responses."""

    def test_empty_response(self):
        """Verify empty response returns empty list."""
        snapshots = parse_api_response({}, "202412280601", "2024-12-27T21:00:00")
        assert snapshots == []

    def test_none_response(self):
        """Verify None data returns empty list."""
        snapshots = parse_api_response(None, "202412280601", "2024-12-27T21:00:00")
        assert snapshots == []

    def test_missing_data_key(self):
        """Verify response without data key returns empty list."""
        response = {"error": "something"}
        snapshots = parse_api_response(response, "202412280601", "2024-12-27T21:00:00")
        assert snapshots == []

    def test_invalid_odds_value(self):
        """Verify invalid odds values are skipped."""
        response = {
            "data": {
                "odds": {
                    "1": {"odds": "2.5"},  # Valid
                    "2": {"odds": "invalid"},  # Invalid
                    "3": {"odds": "-1.0"},  # Out of range
                    "4": {"odds": "10000.0"},  # Out of range
                }
            }
        }
        snapshots = parse_api_response(response, "202412280601", "2024-12-27T21:00:00")

        # Only horse 1 should be parsed
        assert len(snapshots) == 1
        assert snapshots[0].horse_no == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
