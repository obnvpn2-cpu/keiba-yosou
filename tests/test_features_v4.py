# -*- coding: utf-8 -*-
"""
test_features_v4.py - Tests for FeaturePack v1 module

Tests for:
1. DDL creation and column counts
2. As-of aggregation (leak prevention)
3. Encoding mappings
4. Quality report generation
"""

import sqlite3
import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_v4.feature_table_v4 import (
    CREATE_FEATURE_TABLE_V4,
    create_feature_table_v4,
    get_feature_v4_columns,
    get_feature_groups,
    get_pedigree_hash_column_names,
)

from src.features_v4.asof_aggregator import (
    AsOfAggregator,
    map_distance_to_cat,
    map_class_to_id,
    PLACE_MAP,
    SURFACE_MAP,
    TRACK_CONDITION_MAP,
)

from src.features_v4.quality_report import (
    MasterQualityReporter,
    generate_quality_report,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def in_memory_db():
    """In-memory SQLite database for testing"""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def populated_db(in_memory_db):
    """Database with test data"""
    conn = in_memory_db

    # Create races table
    conn.execute("""
        CREATE TABLE races (
            race_id TEXT PRIMARY KEY,
            date TEXT,
            place TEXT,
            course_type TEXT,
            distance INTEGER,
            track_condition TEXT,
            race_class TEXT,
            grade TEXT,
            race_no INTEGER,
            course_turn TEXT,
            course_inout TEXT,
            head_count INTEGER
        )
    """)

    # Create race_results table
    conn.execute("""
        CREATE TABLE race_results (
            race_id TEXT NOT NULL,
            horse_id TEXT NOT NULL,
            finish_order INTEGER,
            last_3f REAL,
            body_weight INTEGER,
            body_weight_diff INTEGER,
            passing_order TEXT,
            win_odds REAL,
            popularity INTEGER,
            prize_money REAL,
            jockey_id TEXT,
            trainer_id TEXT,
            sex TEXT,
            age INTEGER,
            weight REAL,
            frame_no INTEGER,
            horse_no INTEGER,
            PRIMARY KEY (race_id, horse_id)
        )
    """)

    # Create horses table
    conn.execute("""
        CREATE TABLE horses (
            horse_id TEXT PRIMARY KEY,
            horse_name TEXT,
            sex TEXT,
            birth_date TEXT,
            sire_id TEXT,
            dam_id TEXT
        )
    """)

    # Create jockeys table
    conn.execute("""
        CREATE TABLE jockeys (
            jockey_id TEXT PRIMARY KEY,
            jockey_name TEXT,
            affiliation TEXT
        )
    """)

    # Create trainers table
    conn.execute("""
        CREATE TABLE trainers (
            trainer_id TEXT PRIMARY KEY,
            trainer_name TEXT,
            affiliation TEXT
        )
    """)

    # Create horse_pedigree table
    conn.execute("""
        CREATE TABLE horse_pedigree (
            horse_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            position TEXT NOT NULL,
            ancestor_id TEXT,
            ancestor_name TEXT,
            PRIMARY KEY (horse_id, generation, position)
        )
    """)

    # Insert test data
    # Race 1: 2024-01-01
    conn.execute("""
        INSERT INTO races VALUES
        ('202401010101', '2024-01-01', '中山', '芝', 2000, '良', '3勝', NULL, 11, '右', '内', 16)
    """)

    # Race 2: 2024-01-08
    conn.execute("""
        INSERT INTO races VALUES
        ('202401010801', '2024-01-08', '中山', '芝', 2000, '良', '3勝', NULL, 11, '右', '内', 14)
    """)

    # Race 3: 2024-01-15 (target race for as-of testing)
    conn.execute("""
        INSERT INTO races VALUES
        ('202401011501', '2024-01-15', '中山', '芝', 2000, '良', '3勝', NULL, 11, '右', '内', 12)
    """)

    # Horse A participated in Race 1 and Race 2 (before Race 3)
    conn.execute("""
        INSERT INTO race_results VALUES
        ('202401010101', 'HORSE_A', 1, 34.5, 500, 2, '1-1-1-1', 2.5, 1, 5000, 'JOCKEY_X', 'TRAINER_Y', '牡', 4, 57.0, 1, 1),
        ('202401010101', 'HORSE_B', 3, 35.0, 480, -2, '5-4-3-2', 8.0, 3, 2000, 'JOCKEY_Y', 'TRAINER_Y', '牝', 4, 55.0, 2, 3),
        ('202401010801', 'HORSE_A', 2, 34.2, 502, 2, '2-2-2-1', 3.0, 2, 3000, 'JOCKEY_X', 'TRAINER_Y', '牡', 4, 57.0, 3, 5),
        ('202401011501', 'HORSE_A', NULL, NULL, 504, 2, NULL, 4.0, 2, NULL, 'JOCKEY_X', 'TRAINER_Y', '牡', 4, 57.0, 2, 4)
    """)

    # Insert horse master
    conn.execute("""
        INSERT INTO horses VALUES
        ('HORSE_A', 'テストホースA', '牡', '2020-04-01', 'SIRE_1', 'DAM_1'),
        ('HORSE_B', 'テストホースB', '牝', '2020-05-15', 'SIRE_2', 'DAM_2')
    """)

    # Insert jockey master
    conn.execute("""
        INSERT INTO jockeys VALUES
        ('JOCKEY_X', 'テスト騎手X', '美浦'),
        ('JOCKEY_Y', 'テスト騎手Y', '栗東')
    """)

    # Insert trainer master
    conn.execute("""
        INSERT INTO trainers VALUES
        ('TRAINER_Y', 'テスト調教師Y', '栗東')
    """)

    # Insert pedigree
    conn.execute("""
        INSERT INTO horse_pedigree VALUES
        ('HORSE_A', 1, 's', 'SIRE_1', 'テスト父'),
        ('HORSE_A', 1, 'd', 'DAM_1', 'テスト母'),
        ('HORSE_A', 2, 'ss', NULL, 'テスト父父'),
        ('HORSE_A', 2, 'sd', NULL, 'テスト父母')
    """)

    conn.commit()
    return conn


# =============================================================================
# DDL Tests
# =============================================================================

class TestFeatureTableV4DDL:
    """feature_table_v4 DDL のテスト"""

    def test_ddl_contains_required_sections(self):
        """DDL に必要なセクションが含まれているか"""
        ddl = CREATE_FEATURE_TABLE_V4

        assert "race_id TEXT NOT NULL" in ddl
        assert "horse_id TEXT NOT NULL" in ddl
        assert "target_win INTEGER" in ddl
        assert "target_in3 INTEGER" in ddl
        assert "h_n_starts INTEGER" in ddl
        assert "j_n_starts INTEGER" in ddl
        assert "t_n_starts INTEGER" in ddl
        assert "market_win_odds REAL" in ddl

    def test_create_table_v4(self, in_memory_db):
        """テーブル作成が成功するか"""
        create_feature_table_v4(in_memory_db, include_pedigree_hash=False)

        # テーブルが存在することを確認
        cursor = in_memory_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_table_v4'"
        )
        assert cursor.fetchone() is not None

    def test_feature_column_count(self):
        """特徴量カラム数が 200+ あるか"""
        columns = get_feature_v4_columns(include_pedigree=True, include_market=True)
        assert len(columns) >= 200, f"Expected 200+ columns, got {len(columns)}"

    def test_feature_column_count_without_pedigree(self):
        """血統ハッシュなしでも基本特徴量が揃っているか"""
        columns = get_feature_v4_columns(include_pedigree=False, include_market=False)
        assert len(columns) >= 100, f"Expected 100+ columns, got {len(columns)}"

    def test_pedigree_hash_column_count(self):
        """血統ハッシュカラム数が 640 (512+128) あるか"""
        columns = get_pedigree_hash_column_names()
        assert len(columns) == 640, f"Expected 640 columns, got {len(columns)}"

        # ped_hash が 512
        ped_hash_cols = [c for c in columns if c.startswith("ped_hash_")]
        assert len(ped_hash_cols) == 512

        # anc_hash が 128
        anc_hash_cols = [c for c in columns if c.startswith("anc_hash_")]
        assert len(anc_hash_cols) == 128

    def test_feature_groups(self):
        """特徴量グループが正しく定義されているか"""
        groups = get_feature_groups()

        assert "base_race" in groups
        assert "horse_form" in groups
        assert "pace_position" in groups
        assert "class_prize" in groups
        assert "jockey_trainer" in groups
        assert "pedigree" in groups
        assert "market" in groups

        # 各グループのカラム数をチェック
        assert len(groups["base_race"]) >= 15
        assert len(groups["horse_form"]) >= 30
        assert len(groups["jockey_trainer"]) >= 30


# =============================================================================
# Encoding Tests
# =============================================================================

class TestEncodings:
    """エンコーディングマップのテスト"""

    def test_distance_to_cat(self):
        """距離カテゴリマッピング"""
        assert map_distance_to_cat(1000) == 1000
        assert map_distance_to_cat(1200) == 1200
        assert map_distance_to_cat(1600) == 1600
        assert map_distance_to_cat(2000) == 2000
        assert map_distance_to_cat(2400) == 2400
        assert map_distance_to_cat(3200) == 3000
        assert map_distance_to_cat(None) is None

    def test_class_to_id(self):
        """クラスIDマッピング"""
        assert map_class_to_id("G1") == 0
        assert map_class_to_id("G2") == 1
        assert map_class_to_id("G3") == 2
        assert map_class_to_id("オープン") == 3
        assert map_class_to_id("3勝クラス") == 4
        assert map_class_to_id("未勝利") == 7
        assert map_class_to_id("新馬") == 8

    def test_place_map(self):
        """開催場マッピング"""
        assert PLACE_MAP["東京"] == 4
        assert PLACE_MAP["中山"] == 5
        assert PLACE_MAP["阪神"] == 8

    def test_surface_map(self):
        """馬場マッピング"""
        assert SURFACE_MAP["芝"] == 0
        assert SURFACE_MAP["ダ"] == 1
        assert SURFACE_MAP["ダート"] == 1
        assert SURFACE_MAP["障"] == 2

    def test_track_condition_map(self):
        """馬場状態マッピング"""
        assert TRACK_CONDITION_MAP["良"] == 0
        assert TRACK_CONDITION_MAP["稍"] == 1
        assert TRACK_CONDITION_MAP["稍重"] == 1
        assert TRACK_CONDITION_MAP["重"] == 2
        assert TRACK_CONDITION_MAP["不良"] == 3


# =============================================================================
# As-Of Aggregation Tests (Leak Prevention)
# =============================================================================

class TestAsOfAggregation:
    """As-Of 統計計算のテスト (リーク防止)"""

    def test_horse_asof_stats_excludes_future(self, populated_db):
        """馬の統計が未来データを含まないことを確認"""
        agg = AsOfAggregator(populated_db)

        # Race 3 (2024-01-15) の時点での HORSE_A の統計
        # Race 1 (2024-01-01) と Race 2 (2024-01-08) のみ含まれるべき
        stats = agg.compute_horse_asof_stats(
            horse_id="HORSE_A",
            race_date="2024-01-15",
            distance_cat=2000,
            surface="芝",
        )

        assert stats["h_n_starts"] == 2  # Race 1 と Race 2 のみ
        assert stats["h_n_wins"] == 1    # Race 1 で1着
        assert stats["h_win_rate"] == 0.5
        assert stats["h_is_first_run"] == 0

    def test_horse_asof_stats_first_run(self, populated_db):
        """新馬フラグが正しく設定されるか"""
        agg = AsOfAggregator(populated_db)

        # HORSE_B は Race 1 にしか出走していない
        # Race 1 の日付より前の統計を取ると新馬扱い
        stats = agg.compute_horse_asof_stats(
            horse_id="HORSE_B",
            race_date="2024-01-01",
        )

        assert stats["h_n_starts"] == 0
        assert stats["h_is_first_run"] == 1

    def test_jockey_asof_stats(self, populated_db):
        """騎手の as-of 統計"""
        agg = AsOfAggregator(populated_db)

        # JOCKEY_X は Race 1 で1着、Race 2 で2着
        stats = agg.compute_jockey_asof_stats(
            jockey_id="JOCKEY_X",
            race_date="2024-01-15",
            horse_id="HORSE_A",
        )

        assert stats["j_n_starts"] == 2
        assert stats["j_n_wins"] == 1
        assert stats["j_win_rate"] == 0.5
        assert stats["jh_n_combos"] == 2  # JOCKEY_X × HORSE_A

    def test_trainer_asof_stats(self, populated_db):
        """調教師の as-of 統計"""
        agg = AsOfAggregator(populated_db)

        # TRAINER_Y は Race 1 で2頭出走
        stats = agg.compute_trainer_asof_stats(
            trainer_id="TRAINER_Y",
            race_date="2024-01-15",
        )

        assert stats["t_n_starts"] >= 2
        assert stats["t_n_wins"] >= 1

    def test_no_same_day_data(self, populated_db):
        """同日データが含まれないことを確認"""
        agg = AsOfAggregator(populated_db)

        # Race 1 (2024-01-01) 当日の統計
        # HORSE_A はこの日が初出走なので n_starts = 0
        stats = agg.compute_horse_asof_stats(
            horse_id="HORSE_A",
            race_date="2024-01-01",
        )

        assert stats["h_n_starts"] == 0, "Same-day data should not be included"


# =============================================================================
# Quality Report Tests
# =============================================================================

class TestQualityReport:
    """品質レポートのテスト"""

    def test_report_generation(self, populated_db):
        """レポートが正常に生成されるか"""
        reporter = MasterQualityReporter(populated_db)
        report = reporter.generate_report()

        assert report is not None
        assert report.generated_at is not None
        assert "races" in report.table_stats
        assert "race_results" in report.table_stats

    def test_table_stats(self, populated_db):
        """テーブル統計が正しいか"""
        reporter = MasterQualityReporter(populated_db)
        report = reporter.generate_report()

        # races テーブル: 3 レース
        assert report.table_stats["races"].row_count == 3

        # race_results テーブル: 4 エントリ
        assert report.table_stats["race_results"].row_count == 4

    def test_reference_integrity(self, populated_db):
        """参照整合性チェック"""
        reporter = MasterQualityReporter(populated_db)
        report = reporter.generate_report()

        # horse_id の参照整合性
        horse_ref = None
        for ref in report.reference_integrity:
            if ref.source_column == "horse_id" and ref.target_table == "horses":
                horse_ref = ref
                break

        assert horse_ref is not None
        assert horse_ref.match_rate > 0

    def test_pedigree_coverage(self, populated_db):
        """血統カバレッジチェック"""
        reporter = MasterQualityReporter(populated_db)
        report = reporter.generate_report()

        pc = report.pedigree_coverage
        assert pc["horses_with_pedigree"] == 1  # HORSE_A のみ
        assert pc["ancestor_count"] == 4  # 4 祖先

    def test_print_report(self, populated_db):
        """レポート出力が成功するか"""
        reporter = MasterQualityReporter(populated_db)
        output = reporter.print_report()

        assert "Master Data Quality Report" in output
        assert "Table Coverage" in output
        assert "Reference Integrity" in output


# =============================================================================
# Leak Detection Tests
# =============================================================================

class TestLeakDetection:
    """リーク検出テスト"""

    def test_no_future_data_in_stats(self, populated_db):
        """
        統計に未来データが含まれていないことを確認

        これは最も重要なテスト: as-of 統計が正しく計算されているか
        """
        agg = AsOfAggregator(populated_db)

        # Race 2 (2024-01-08) 時点の統計
        stats_at_race2 = agg.compute_horse_asof_stats(
            horse_id="HORSE_A",
            race_date="2024-01-08",
        )

        # Race 3 (2024-01-15) 時点の統計
        stats_at_race3 = agg.compute_horse_asof_stats(
            horse_id="HORSE_A",
            race_date="2024-01-15",
        )

        # Race 2 時点では Race 1 のみ含まれる
        assert stats_at_race2["h_n_starts"] == 1

        # Race 3 時点では Race 1 と Race 2 が含まれる
        assert stats_at_race3["h_n_starts"] == 2

        # 統計が正しく増加している
        assert stats_at_race3["h_n_starts"] > stats_at_race2["h_n_starts"]

    def test_market_columns_are_isolated(self):
        """market_* カラムが分離されているか"""
        columns_no_market = get_feature_v4_columns(include_market=False)
        columns_with_market = get_feature_v4_columns(include_market=True)

        market_cols = set(columns_with_market) - set(columns_no_market)

        # market_ プレフィックスのカラムのみが追加される
        for col in market_cols:
            assert col.startswith("market_"), f"Non-market column in difference: {col}"

        assert "market_win_odds" in market_cols
        assert "market_popularity" in market_cols


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
