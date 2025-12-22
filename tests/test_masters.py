#!/usr/bin/env python3
"""
test_masters.py - Tests for master data functionality (Road 2 + PR2.5)

Tests for:
1. Master table migrations (including horse_pedigree)
2. Horse/Jockey/Trainer parsers
3. 5-generation pedigree parser (PR2.5)
4. FetchStatusManager
5. UPSERT operations
"""

import sqlite3
import tempfile
import os
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.masters_migration import (
    run_road2_migrations,
    create_master_tables,
    FetchStatusManager,
    ROAD2_MIGRATIONS,
)
from src.ingestion.parser_horse_extended import HorseExtendedParser, HorseExtendedRecord
from src.ingestion.parser_jockey import JockeyParser, JockeyRecord
from src.ingestion.parser_trainer import TrainerParser, TrainerRecord
from src.ingestion.parser_pedigree_5gen import Pedigree5GenParser, PedigreeAncestor


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)

    # Create minimal schema for testing
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id TEXT PRIMARY KEY,
            description TEXT,
            applied_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
    """)
    conn.commit()

    yield conn, path

    conn.close()
    os.unlink(path)


@pytest.fixture
def db_with_masters(temp_db):
    """Database with master tables created."""
    conn, path = temp_db
    create_master_tables(conn)
    yield conn, path


@pytest.fixture
def horse_parser():
    """HorseExtendedParser instance."""
    return HorseExtendedParser()


@pytest.fixture
def jockey_parser():
    """JockeyParser instance."""
    return JockeyParser()


@pytest.fixture
def trainer_parser():
    """TrainerParser instance."""
    return TrainerParser()


@pytest.fixture
def pedigree_parser():
    """Pedigree5GenParser instance."""
    return Pedigree5GenParser()


# ============================================================
# Migration Tests
# ============================================================


class TestMasterMigrations:
    """Tests for master table migrations."""

    def test_run_road2_migrations(self, temp_db):
        """Test that Road 2 migrations run correctly."""
        conn, _ = temp_db

        # Run migrations
        count = run_road2_migrations(conn)

        # Should apply all migrations
        assert count == len(ROAD2_MIGRATIONS)

        # Check tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('horses', 'jockeys', 'trainers', 'fetch_status')
        """)
        tables = {row[0] for row in cursor.fetchall()}
        assert "horses" in tables
        assert "jockeys" in tables
        assert "trainers" in tables
        assert "fetch_status" in tables

    def test_migrations_idempotent(self, temp_db):
        """Test that migrations can run multiple times."""
        conn, _ = temp_db

        # Run first time
        count1 = run_road2_migrations(conn)
        assert count1 > 0

        # Run second time
        count2 = run_road2_migrations(conn)
        assert count2 == 0  # No new migrations

    def test_create_master_tables_idempotent(self, temp_db):
        """Test that create_master_tables is idempotent."""
        conn, _ = temp_db

        # Run multiple times
        create_master_tables(conn)
        create_master_tables(conn)
        create_master_tables(conn)

        # Should not raise any errors
        cursor = conn.execute("SELECT COUNT(*) FROM horses")
        assert cursor.fetchone()[0] == 0


# ============================================================
# FetchStatusManager Tests
# ============================================================


class TestFetchStatusManager:
    """Tests for FetchStatusManager."""

    def test_add_pending(self, db_with_masters):
        """Test adding pending entities."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        # Add some pending
        count = manager.add_pending("horse", ["H1", "H2", "H3"])
        assert count == 3

        # Check they're in the table
        cursor = conn.execute("SELECT COUNT(*) FROM fetch_status WHERE entity_type='horse'")
        assert cursor.fetchone()[0] == 3

    def test_add_pending_skip_existing(self, db_with_masters):
        """Test that existing entities are skipped."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        # Add initial
        count1 = manager.add_pending("horse", ["H1", "H2"])
        assert count1 == 2

        # Add overlapping
        count2 = manager.add_pending("horse", ["H2", "H3"])
        assert count2 == 1  # Only H3 is new

    def test_get_pending(self, db_with_masters):
        """Test getting pending entities."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        manager.add_pending("horse", ["H1", "H2", "H3"])

        pending = manager.get_pending("horse", limit=2)
        assert len(pending) == 2
        assert pending[0] == "H1"

    def test_mark_success(self, db_with_masters):
        """Test marking entity as success."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        manager.add_pending("horse", ["H1"])
        manager.mark_success("horse", "H1")

        # Should not appear in pending
        pending = manager.get_pending("horse")
        assert "H1" not in pending

        # Check status
        cursor = conn.execute("""
            SELECT status FROM fetch_status
            WHERE entity_type='horse' AND entity_id='H1'
        """)
        assert cursor.fetchone()[0] == "success"

    def test_mark_failed(self, db_with_masters):
        """Test marking entity as failed."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        manager.add_pending("horse", ["H1"])
        manager.mark_failed("horse", "H1", "HTTP 500")

        # Check status and retry count
        cursor = conn.execute("""
            SELECT status, retry_count, error_message FROM fetch_status
            WHERE entity_type='horse' AND entity_id='H1'
        """)
        row = cursor.fetchone()
        assert row[0] == "failed"
        assert row[1] == 1
        assert row[2] == "HTTP 500"

    def test_get_progress(self, db_with_masters):
        """Test getting progress statistics."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        manager.add_pending("horse", ["H1", "H2", "H3", "H4"])
        manager.mark_success("horse", "H1")
        manager.mark_success("horse", "H2")
        manager.mark_failed("horse", "H3", "error")

        progress = manager.get_progress("horse")
        assert progress["total"] == 4
        assert progress["success"] == 2
        assert progress["failed"] == 1
        assert progress["pending"] == 1
        assert progress["progress_pct"] == 50.0

    def test_retry_failed_entities(self, db_with_masters):
        """Test that failed entities with low retry count are retried."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        manager.add_pending("horse", ["H1"])
        manager.mark_failed("horse", "H1", "error")

        # Should appear in pending (retry_count < max_retries)
        pending = manager.get_pending("horse", max_retries=3)
        assert "H1" in pending

    def test_skip_max_retries(self, db_with_masters):
        """Test that entities exceeding max_retries are skipped."""
        conn, _ = db_with_masters
        manager = FetchStatusManager(conn)

        manager.add_pending("horse", ["H1"])

        # Fail 3 times
        for _ in range(3):
            manager.mark_failed("horse", "H1", "error")

        # Should not appear in pending (retry_count >= 3)
        pending = manager.get_pending("horse", max_retries=3)
        assert "H1" not in pending


# ============================================================
# Parser Tests
# ============================================================


class TestHorseExtendedParser:
    """Tests for HorseExtendedParser."""

    def test_parse_basic_info(self, horse_parser):
        """Test parsing basic horse info."""
        html = """
        <html>
        <body>
            <div class="db_head">
                <h1>ドウデュース</h1>
            </div>
            <table>
                <tr><th>性齢</th><td>牡4</td></tr>
                <tr><th>生年月日</th><td>2019年5月7日</td></tr>
                <tr><th>毛色</th><td>鹿毛</td></tr>
                <tr><th>生産者</th><td><a href="/breeder/12345/">ノーザンファーム</a></td></tr>
            </table>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        record = horse_parser.parse("2019104385", soup)

        assert record.horse_id == "2019104385"
        assert record.horse_name == "ドウデュース"
        assert record.sex == "牡"
        assert record.birth_date == "2019-05-07"
        assert record.coat_color == "鹿毛"
        assert record.breeder == "ノーザンファーム"

    def test_parse_sex_variations(self, horse_parser):
        """Test parsing different sex values."""
        for sex_text, expected in [("牡3", "牡"), ("牝5", "牝"), ("セン4", "セン")]:
            html = f"""
            <html>
            <body>
                <h1>テスト馬</h1>
                <table>
                    <tr><th>性齢</th><td>{sex_text}</td></tr>
                </table>
            </body>
            </html>
            """
            soup = BeautifulSoup(html, "html.parser")
            record = horse_parser.parse("0000000001", soup)
            assert record.sex == expected

    def test_parse_raises_on_missing_name(self, horse_parser):
        """Test that parser raises on missing horse name."""
        html = """
        <html><body><table></table></body></html>
        """
        soup = BeautifulSoup(html, "html.parser")

        with pytest.raises(ValueError, match="馬名を取得できませんでした"):
            horse_parser.parse("0000000001", soup)


class TestJockeyParser:
    """Tests for JockeyParser."""

    def test_parse_basic_info(self, jockey_parser):
        """Test parsing basic jockey info."""
        html = """
        <html>
        <body>
            <div class="db_head">
                <h1>川田将雅</h1>
            </div>
            <table>
                <tr><th>生年月日</th><td>1985年10月15日</td></tr>
                <tr><th>所属</th><td>栗東</td></tr>
            </table>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        record = jockey_parser.parse("01089", soup)

        assert record.jockey_id == "01089"
        assert record.jockey_name == "川田将雅"
        assert record.birth_date == "1985-10-15"
        assert record.affiliation == "栗東"

    def test_parse_affiliation_normalization(self, jockey_parser):
        """Test that affiliation is normalized."""
        for affil_text, expected in [
            ("栗東トレーニングセンター", "栗東"),
            ("美浦TC", "美浦"),
            ("地方競馬", "地方"),
        ]:
            html = f"""
            <html>
            <body>
                <h1>テスト騎手</h1>
                <table>
                    <tr><th>所属</th><td>{affil_text}</td></tr>
                </table>
            </body>
            </html>
            """
            soup = BeautifulSoup(html, "html.parser")
            record = jockey_parser.parse("00001", soup)
            assert record.affiliation == expected


class TestTrainerParser:
    """Tests for TrainerParser."""

    def test_parse_basic_info(self, trainer_parser):
        """Test parsing basic trainer info."""
        html = """
        <html>
        <body>
            <div class="db_head">
                <h1>国枝栄</h1>
            </div>
            <table>
                <tr><th>生年月日</th><td>1955年4月3日</td></tr>
                <tr><th>所属</th><td>美浦</td></tr>
            </table>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        record = trainer_parser.parse("01012", soup)

        assert record.trainer_id == "01012"
        assert record.trainer_name == "国枝栄"
        assert record.birth_date == "1955-04-03"
        assert record.affiliation == "美浦"


class TestPedigree5GenParser:
    """Tests for Pedigree5GenParser (PR2.5)."""

    def test_parse_simple_pedigree(self, pedigree_parser):
        """Test parsing a simple pedigree table."""
        html = """
        <html>
        <body>
            <table class="blood_table">
                <tr>
                    <td rowspan="16"><a href="/horse/2010101234/">ディープインパクト</a></td>
                    <td rowspan="8"><a href="/horse/2000101111/">サンデーサイレンス</a></td>
                </tr>
                <tr>
                    <td rowspan="8"><a href="/horse/2000102222/">ウインドインハーヘア</a></td>
                </tr>
                <tr>
                    <td rowspan="16"><a href="/horse/2012102345/">ダストアンドダイヤモンズ</a></td>
                    <td rowspan="8"><a href="/horse/2001103333/">Vindication</a></td>
                </tr>
                <tr>
                    <td rowspan="8"><a href="/horse/2001104444/">Majesty's Crown</a></td>
                </tr>
            </table>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        ancestors = pedigree_parser.parse("2019104385", soup)

        assert len(ancestors) >= 2  # At least sire and dam

        # Check that all have required fields
        for a in ancestors:
            assert a.horse_id == "2019104385"
            assert a.generation in range(1, 6)
            assert a.position  # Not empty
            assert a.ancestor_name  # Not empty

    def test_position_calculation(self, pedigree_parser):
        """Test that position paths are correctly calculated."""
        # Gen 1
        assert pedigree_parser._calc_position(1, 0) == "s"
        assert pedigree_parser._calc_position(1, 1) == "d"

        # Gen 2
        assert pedigree_parser._calc_position(2, 0) == "ss"
        assert pedigree_parser._calc_position(2, 1) == "sd"
        assert pedigree_parser._calc_position(2, 2) == "ds"
        assert pedigree_parser._calc_position(2, 3) == "dd"

        # Gen 3
        assert pedigree_parser._calc_position(3, 0) == "sss"
        assert pedigree_parser._calc_position(3, 4) == "dss"
        assert pedigree_parser._calc_position(3, 7) == "ddd"

    def test_parse_empty_table(self, pedigree_parser):
        """Test parsing an empty/missing pedigree table."""
        html = """
        <html>
        <body>
            <table><tr><td>No pedigree data</td></tr></table>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        ancestors = pedigree_parser.parse("0000000001", soup)

        # Should return empty list, not raise
        assert ancestors == []

    def test_parse_partial_pedigree(self, pedigree_parser):
        """Test parsing a pedigree with missing links."""
        html = """
        <html>
        <body>
            <table class="blood_table">
                <tr>
                    <td rowspan="16"><a href="/horse/2010101234/">ディープインパクト</a></td>
                </tr>
                <tr>
                    <td rowspan="16">Unknown Mare (no link)</td>
                </tr>
            </table>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        ancestors = pedigree_parser.parse("2019104385", soup)

        # Should still parse what's available
        # The exact count depends on implementation, but should not crash
        assert isinstance(ancestors, list)


# ============================================================
# UPSERT Tests
# ============================================================


class TestMasterUpsert:
    """Tests for master data UPSERT operations."""

    def test_horse_upsert(self, db_with_masters):
        """Test UPSERT for horse records."""
        conn, _ = db_with_masters

        # Import upsert function
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from scripts.fetch_masters import upsert_horse

        # Create test record
        record = HorseExtendedRecord(
            horse_id="2019104385",
            horse_name="ドウデュース",
            sex="牡",
            birth_date="2019-05-07",
            coat_color="鹿毛",
            breeder="ノーザンファーム",
            breeder_region=None,
            owner="キーファーズ",
            owner_id="abc123",
            sire_id="2010101234",
            sire_name="ハーツクライ",
            dam_id="2012102345",
            dam_name="ダストアンドダイヤモンズ",
            broodmare_sire_id=None,
            broodmare_sire_name=None,
            sire_sire_name=None,
            sire_dam_name=None,
            dam_dam_name=None,
            total_prize=None,
            total_starts=None,
            total_wins=None,
        )

        # Insert
        upsert_horse(conn, record)

        # Verify
        cursor = conn.execute("SELECT horse_name, sire_name FROM horses WHERE horse_id=?", (record.horse_id,))
        row = cursor.fetchone()
        assert row[0] == "ドウデュース"
        assert row[1] == "ハーツクライ"

        # Update
        record.sire_name = "Updated Sire"
        upsert_horse(conn, record)

        cursor = conn.execute("SELECT sire_name FROM horses WHERE horse_id=?", (record.horse_id,))
        row = cursor.fetchone()
        assert row[0] == "Updated Sire"

        # Still only one row
        cursor = conn.execute("SELECT COUNT(*) FROM horses")
        assert cursor.fetchone()[0] == 1

    def test_pedigree_upsert(self, db_with_masters):
        """Test UPSERT for pedigree records."""
        conn, _ = db_with_masters

        # Import upsert function
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from scripts.fetch_masters import upsert_pedigree_ancestors

        # Create test records
        ancestors = [
            PedigreeAncestor(
                horse_id="2019104385",
                generation=1,
                position="s",
                ancestor_id="2010101234",
                ancestor_name="ディープインパクト",
            ),
            PedigreeAncestor(
                horse_id="2019104385",
                generation=1,
                position="d",
                ancestor_id="2012102345",
                ancestor_name="ダストアンドダイヤモンズ",
            ),
            PedigreeAncestor(
                horse_id="2019104385",
                generation=2,
                position="ss",
                ancestor_id="2000101111",
                ancestor_name="サンデーサイレンス",
            ),
        ]

        # Insert
        count = upsert_pedigree_ancestors(conn, ancestors)
        assert count == 3

        # Verify
        cursor = conn.execute(
            "SELECT ancestor_name FROM horse_pedigree WHERE horse_id=? AND position=?",
            ("2019104385", "s")
        )
        row = cursor.fetchone()
        assert row[0] == "ディープインパクト"

        # Update
        ancestors[0] = PedigreeAncestor(
            horse_id="2019104385",
            generation=1,
            position="s",
            ancestor_id="2010101234",
            ancestor_name="Updated Sire Name",
        )
        upsert_pedigree_ancestors(conn, ancestors[:1])

        cursor = conn.execute(
            "SELECT ancestor_name FROM horse_pedigree WHERE horse_id=? AND position=?",
            ("2019104385", "s")
        )
        row = cursor.fetchone()
        assert row[0] == "Updated Sire Name"

        # Still only 3 rows (UPSERT, not INSERT)
        cursor = conn.execute("SELECT COUNT(*) FROM horse_pedigree")
        assert cursor.fetchone()[0] == 3

    def test_pedigree_table_exists_after_migration(self, db_with_masters):
        """Test that horse_pedigree table is created by migrations."""
        conn, _ = db_with_masters

        # Check table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='horse_pedigree'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        assert "horse_pedigree" in tables

        # Check PRIMARY KEY constraint
        cursor = conn.execute("PRAGMA table_info(horse_pedigree)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert "horse_id" in columns
        assert "generation" in columns
        assert "position" in columns
        assert "ancestor_name" in columns


# ============================================================
# CLI Runner
# ============================================================


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"],
        cwd=Path(__file__).parent.parent
    )
    sys.exit(result.returncode)
