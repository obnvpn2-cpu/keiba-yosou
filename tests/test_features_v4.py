# -*- coding: utf-8 -*-
"""
test_features_v4.py - Tests for FeaturePack v1 module

Tests for:
1. DDL creation and column counts
2. As-of aggregation (leak prevention)
3. Encoding mappings
4. Quality report generation
"""

import json
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

from src.features_v4.feature_builder_v4 import (
    FeatureBuilderV4,
    is_invalid_finish_status,
    INVALID_FINISH_STATUS_EXACT,
    INVALID_FINISH_STATUS_PATTERNS,
)

from src.features_v4.train_eval_v4 import (
    evaluate_ranking,
    RankingResult,
    evaluate_roi_strategy,
    ROIStrategyResult,
    ROIEvalResult,
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


@pytest.fixture
def db_with_invalid_records(in_memory_db):
    """Database with invalid records (中/除/取/降格) for testing exclusion"""
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

    # Create race_results table with finish_status
    conn.execute("""
        CREATE TABLE race_results (
            race_id TEXT NOT NULL,
            horse_id TEXT NOT NULL,
            finish_order INTEGER,
            finish_status TEXT,
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

    # Insert race
    conn.execute("""
        INSERT INTO races VALUES
        ('202401010101', '2024-01-01', '中山', '芝', 2000, '良', '3勝', NULL, 11, '右', '内', 10)
    """)

    # Insert mix of valid and invalid race results
    conn.executemany("""
        INSERT INTO race_results
        (race_id, horse_id, finish_order, finish_status, last_3f, body_weight,
         body_weight_diff, passing_order, win_odds, popularity, prize_money,
         jockey_id, trainer_id, sex, age, weight, frame_no, horse_no)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        # Valid: normal finish
        ('202401010101', 'HORSE_1', 1, '正常', 34.5, 500, 2, '1-1-1-1', 2.5, 1, 5000, 'J1', 'T1', '牡', 4, 57.0, 1, 1),
        ('202401010101', 'HORSE_2', 2, '正常', 34.8, 480, 0, '2-2-2-2', 5.0, 2, 2000, 'J2', 'T1', '牝', 4, 55.0, 2, 2),
        ('202401010101', 'HORSE_3', 3, '正常', 35.0, 490, -2, '3-3-3-3', 8.0, 3, 1000, 'J3', 'T2', '牡', 5, 57.0, 3, 3),

        # Invalid: 中 (競走中止)
        ('202401010101', 'HORSE_4', None, '中', None, 485, 0, None, 10.0, 4, None, 'J4', 'T2', '牡', 4, 57.0, 4, 4),

        # Invalid: 除 (除外)
        ('202401010101', 'HORSE_5', None, '除', None, 500, 5, None, 15.0, 5, None, 'J5', 'T3', '牝', 3, 54.0, 5, 5),

        # Invalid: 取 (取消)
        ('202401010101', 'HORSE_6', None, '取', None, 495, -3, None, 20.0, 6, None, 'J1', 'T3', '牡', 5, 57.0, 6, 6),

        # Invalid: 降格 (3(降))
        ('202401010101', 'HORSE_7', 4, '3(降)', 35.5, 510, 2, '4-4-4-4', 12.0, 7, 500, 'J2', 'T1', '牡', 4, 57.0, 7, 7),

        # Valid: finish_order is set, normal status
        ('202401010101', 'HORSE_8', 4, '正常', 35.2, 505, 1, '5-5-5-5', 25.0, 8, 300, 'J3', 'T2', '牝', 4, 55.0, 8, 8),
        ('202401010101', 'HORSE_9', 5, '正常', 35.8, 488, -1, '6-6-6-6', 30.0, 9, 200, 'J4', 'T3', '牡', 6, 58.0, 9, 9),
        ('202401010101', 'HORSE_10', 6, '正常', 36.0, 492, 3, '7-7-7-7', 50.0, 10, 100, 'J5', 'T1', '牝', 3, 54.0, 10, 10),
    ])

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
# Invalid Record Exclusion Tests (A-3)
# =============================================================================

class TestInvalidRecordExclusion:
    """無効レコード除外のテスト (中/除/取/降格)"""

    def test_is_invalid_finish_status_exact_match(self):
        """完全一致の無効ステータス判定"""
        # 無効なステータス
        assert is_invalid_finish_status("中") is True
        assert is_invalid_finish_status("除") is True
        assert is_invalid_finish_status("取") is True

        # 正常なステータス
        assert is_invalid_finish_status("正常") is False
        assert is_invalid_finish_status(None) is False
        assert is_invalid_finish_status("") is False

    def test_is_invalid_finish_status_pattern_match(self):
        """パターン一致の無効ステータス判定 (降格)"""
        assert is_invalid_finish_status("3(降)") is True
        assert is_invalid_finish_status("1(降)") is True
        assert is_invalid_finish_status("降格") is True

        # 類似だが無効でないもの
        assert is_invalid_finish_status("降雨") is True  # 降を含むので True

    def test_invalid_constants(self):
        """無効ステータス定数の確認"""
        assert "中" in INVALID_FINISH_STATUS_EXACT
        assert "除" in INVALID_FINISH_STATUS_EXACT
        assert "取" in INVALID_FINISH_STATUS_EXACT
        assert "降" in INVALID_FINISH_STATUS_PATTERNS

    def test_feature_builder_excludes_invalid_records(self, db_with_invalid_records):
        """FeatureBuilderV4 が無効レコードを除外することを確認"""
        builder = FeatureBuilderV4(db_with_invalid_records)

        # レース 202401010101 の特徴量を構築
        df = builder.build_features_for_race(
            race_id="202401010101",
            include_pedigree=False,
        )

        # 10頭中、有効なのは 6頭 (HORSE_1, 2, 3, 8, 9, 10)
        # 無効: HORSE_4 (中), HORSE_5 (除), HORSE_6 (取), HORSE_7 (降格)
        assert len(df) == 6, f"Expected 6 valid entries, got {len(df)}"

        # 無効なレコードが含まれていないことを確認
        horse_ids = set(df["horse_id"].tolist())
        assert "HORSE_4" not in horse_ids, "中 record should be excluded"
        assert "HORSE_5" not in horse_ids, "除 record should be excluded"
        assert "HORSE_6" not in horse_ids, "取 record should be excluded"
        assert "HORSE_7" not in horse_ids, "降格 record should be excluded"

        # 有効なレコードが含まれていることを確認
        assert "HORSE_1" in horse_ids
        assert "HORSE_2" in horse_ids
        assert "HORSE_3" in horse_ids
        assert "HORSE_8" in horse_ids
        assert "HORSE_9" in horse_ids
        assert "HORSE_10" in horse_ids

    def test_feature_builder_targets_are_valid(self, db_with_invalid_records):
        """生成された特徴量の target が 0/1 のみであることを確認"""
        builder = FeatureBuilderV4(db_with_invalid_records)

        df = builder.build_features_for_race(
            race_id="202401010101",
            include_pedigree=False,
        )

        # target_win は 0 または 1 のみ
        assert df["target_win"].isin([0, 1]).all(), "target_win should be 0 or 1"

        # target_in3 は 0 または 1 のみ
        assert df["target_in3"].isin([0, 1]).all(), "target_in3 should be 0 or 1"

        # target_quinella は 0 または 1 のみ
        assert df["target_quinella"].isin([0, 1]).all(), "target_quinella should be 0 or 1"

        # NaN がないことを確認
        assert df["target_win"].isna().sum() == 0, "target_win should have no NaN"
        assert df["target_in3"].isna().sum() == 0, "target_in3 should have no NaN"
        assert df["target_quinella"].isna().sum() == 0, "target_quinella should have no NaN"

    def test_quality_report_invalid_results(self, db_with_invalid_records):
        """品質レポートが無効レコード統計を含むことを確認"""
        reporter = MasterQualityReporter(db_with_invalid_records)
        report = reporter.generate_report()

        # 無効レコード統計が存在する
        assert report.invalid_results is not None

        # 総件数 10, 無効件数 4 (中/除/取/降格)
        assert report.invalid_results.total_entries == 10
        assert report.invalid_results.invalid_count == 4
        assert report.invalid_results.invalid_rate == pytest.approx(0.4)

        # breakdown が正しい
        breakdown = report.invalid_results.breakdown
        assert "中" in breakdown or "(NULL finish_order)" in breakdown


# =============================================================================
# Ranking Evaluation Tests
# =============================================================================

class TestRankingEvaluation:
    """ランキング評価 (per-race metrics) のテスト"""

    @pytest.fixture
    def mock_model(self):
        """予測確率を返すモックモデル"""
        import numpy as np

        class MockModel:
            def __init__(self, predictions):
                self.predictions = predictions
                self.best_iteration = 1

            def predict(self, X, num_iteration=None):
                return np.array(self.predictions[:len(X)])

        return MockModel

    @pytest.fixture
    def ranking_test_df(self):
        """ランキング評価用のテストデータ"""
        import pandas as pd

        # 2レース分のデータ
        # Race 1: 4頭 (勝ち馬は HORSE_A)
        # Race 2: 3頭 (勝ち馬は HORSE_E)
        return pd.DataFrame({
            "race_id": ["R1", "R1", "R1", "R1", "R2", "R2", "R2"],
            "horse_id": ["A", "B", "C", "D", "E", "F", "G"],
            "target_win": [1, 0, 0, 0, 1, 0, 0],
            # 特徴量（ダミー）
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "feat2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        })

    def test_evaluate_ranking_top1_hit(self, mock_model, ranking_test_df):
        """Top1 Hit Rate のテスト - 勝ち馬が予測1位の場合"""
        import numpy as np

        # 勝ち馬が両レースで予測1位になるよう設定
        # Race 1: HORSE_A (勝ち) が最高確率
        # Race 2: HORSE_E (勝ち) が最高確率
        predictions = [0.9, 0.2, 0.1, 0.05, 0.8, 0.3, 0.1]
        model = mock_model(predictions)

        result = evaluate_ranking(
            model=model,
            df=ranking_test_df,
            feature_cols=["feat1", "feat2"],
            target_col="target_win",
            dataset_name="test",
        )

        assert isinstance(result, RankingResult)
        assert result.n_races == 2
        assert result.n_entries == 7
        assert result.top1_hit_rate == 1.0  # 2/2 = 100%
        assert result.top3_hit_rate == 1.0  # 両方 Top3 以内
        assert result.mrr == 1.0  # 両方1位なので 1/1 の平均 = 1.0

    def test_evaluate_ranking_top3_hit(self, mock_model, ranking_test_df):
        """Top3 Hit Rate のテスト - 勝ち馬が予測2位,3位の場合"""
        import numpy as np

        # Race 1: HORSE_A (勝ち) が2位
        # Race 2: HORSE_E (勝ち) が3位
        predictions = [0.4, 0.5, 0.1, 0.05, 0.2, 0.8, 0.3]
        model = mock_model(predictions)

        result = evaluate_ranking(
            model=model,
            df=ranking_test_df,
            feature_cols=["feat1", "feat2"],
            target_col="target_win",
            dataset_name="test",
        )

        assert result.top1_hit_rate == 0.0  # 0/2
        assert result.top3_hit_rate == 1.0  # 2/2 = 100%
        # MRR: (1/2 + 1/3) / 2 ≈ 0.4167
        assert 0.40 < result.mrr < 0.45

    def test_evaluate_ranking_histogram(self, mock_model, ranking_test_df):
        """勝ち馬順位ヒストグラムのテスト"""
        import numpy as np

        # Race 1: HORSE_A (勝ち) が1位
        # Race 2: HORSE_E (勝ち) が2位
        predictions = [0.9, 0.2, 0.1, 0.05, 0.4, 0.8, 0.1]
        model = mock_model(predictions)

        result = evaluate_ranking(
            model=model,
            df=ranking_test_df,
            feature_cols=["feat1", "feat2"],
            target_col="target_win",
            dataset_name="test",
        )

        # ヒストグラム確認
        assert result.winner_rank_histogram.get("1") == 1
        assert result.winner_rank_histogram.get("2") == 1

    def test_evaluate_ranking_empty_df(self, mock_model):
        """空のDataFrameの場合"""
        import pandas as pd

        empty_df = pd.DataFrame({
            "race_id": [],
            "horse_id": [],
            "target_win": [],
            "feat1": [],
            "feat2": [],
        })

        model = mock_model([])

        result = evaluate_ranking(
            model=model,
            df=empty_df,
            feature_cols=["feat1", "feat2"],
            target_col="target_win",
            dataset_name="test",
        )

        assert result.n_races == 0
        assert result.n_entries == 0
        assert result.top1_hit_rate == 0.0
        assert result.mrr == 0.0

    def test_evaluate_ranking_no_winners(self, mock_model):
        """勝ち馬がいないレースの場合"""
        import pandas as pd

        no_winner_df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1"],
            "horse_id": ["A", "B", "C"],
            "target_win": [0, 0, 0],  # 勝ち馬なし
            "feat1": [1.0, 2.0, 3.0],
            "feat2": [0.1, 0.2, 0.3],
        })

        model = mock_model([0.5, 0.3, 0.2])

        result = evaluate_ranking(
            model=model,
            df=no_winner_df,
            feature_cols=["feat1", "feat2"],
            target_col="target_win",
            dataset_name="test",
        )

        assert result.n_races == 0  # 勝ち馬がいたレースは0
        assert result.n_entries == 3

    def test_evaluate_ranking_mrr_calculation(self, mock_model, ranking_test_df):
        """MRR (Mean Reciprocal Rank) の計算テスト"""
        import numpy as np

        # Race 1: HORSE_A (勝ち) が4位（最下位）
        # Race 2: HORSE_E (勝ち) が1位
        predictions = [0.05, 0.3, 0.5, 0.9, 0.9, 0.2, 0.1]
        model = mock_model(predictions)

        result = evaluate_ranking(
            model=model,
            df=ranking_test_df,
            feature_cols=["feat1", "feat2"],
            target_col="target_win",
            dataset_name="test",
        )

        # MRR: (1/4 + 1/1) / 2 = 0.625
        expected_mrr = (1/4 + 1/1) / 2
        assert abs(result.mrr - expected_mrr) < 0.001

    def test_evaluate_ranking_avg_field_size(self, mock_model, ranking_test_df):
        """平均フィールドサイズのテスト"""
        predictions = [0.9, 0.2, 0.1, 0.05, 0.8, 0.3, 0.1]
        model = mock_model(predictions)

        result = evaluate_ranking(
            model=model,
            df=ranking_test_df,
            feature_cols=["feat1", "feat2"],
            target_col="target_win",
            dataset_name="test",
        )

        # Race 1: 4頭, Race 2: 3頭 → 平均 3.5
        assert result.avg_field_size == 3.5


# =============================================================================
# ROI Evaluation Tests
# =============================================================================

class TestROIEvaluation:
    """ROI評価 (単勝・複勝) のテスト"""

    @pytest.fixture
    def roi_payouts_df(self):
        """払戻データ (payouts テーブル相当)"""
        import pandas as pd

        # 2レース分の払戻データ
        # Race R1: 馬番1が1着 (単勝500円, 複勝120円), 馬番2が2着 (複勝150円), 馬番3が3着 (複勝180円)
        # Race R2: 馬番5が1着 (単勝800円, 複勝200円), 馬番6が2着 (複勝130円), 馬番7が3着 (複勝110円)
        return pd.DataFrame({
            "race_id": ["R1", "R1", "R1", "R1", "R2", "R2", "R2", "R2"],
            "bet_type": ["単勝", "複勝", "複勝", "複勝", "単勝", "複勝", "複勝", "複勝"],
            "combination": ["1", "1", "2", "3", "5", "5", "6", "7"],
            "payout": [500, 120, 150, 180, 800, 200, 130, 110],
        })

    @pytest.fixture
    def roi_race_results_df(self):
        """レース結果データ (race_results テーブル相当)"""
        import pandas as pd

        # 2レース分のデータ
        # Race R1: 4頭 (A:1着, B:2着, C:3着, D:4着)
        # Race R2: 4頭 (E:1着, F:2着, G:3着, H:4着)
        return pd.DataFrame({
            "race_id": ["R1", "R1", "R1", "R1", "R2", "R2", "R2", "R2"],
            "horse_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "horse_no": [1, 2, 3, 4, 5, 6, 7, 8],
            "horse_no_str": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "finish_order": [1, 2, 3, 4, 1, 2, 3, 4],
            "popularity": [2, 1, 3, 4, 1, 3, 2, 4],
        })

    def test_evaluate_roi_strategy_tansho_win(self, roi_payouts_df, roi_race_results_df):
        """単勝: 勝ち馬に賭けた場合のROI計算"""
        import pandas as pd

        # 1着馬に賭けるケース (Race R1: A, Race R2: E)
        bets_df = pd.DataFrame({
            "race_id": ["R1", "R2"],
            "horse_id": ["A", "E"],
        })

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="単勝",
            strategy_name="TestWinner",
            bet_amount=100.0,
        )

        assert isinstance(result, ROIStrategyResult)
        assert result.strategy == "TestWinner"
        assert result.bet_type == "単勝"
        assert result.n_races == 2
        assert result.n_bets == 2
        assert result.stake_total_yen == 200.0  # 100 * 2
        # 払戻: R1=500円, R2=800円 → 1300円
        assert result.return_total_yen == 1300.0
        assert result.roi == pytest.approx(6.5)  # 1300 / 200 = 6.5
        assert result.hit_rate == 1.0  # 2/2 = 100%
        assert result.n_hits == 2
        assert result.avg_payout == pytest.approx(650.0)  # (500+800)/2

    def test_evaluate_roi_strategy_tansho_lose(self, roi_payouts_df, roi_race_results_df):
        """単勝: 負け馬に賭けた場合のROI計算"""
        import pandas as pd

        # 2着馬に賭けるケース (Race R1: B, Race R2: F)
        bets_df = pd.DataFrame({
            "race_id": ["R1", "R2"],
            "horse_id": ["B", "F"],
        })

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="単勝",
            strategy_name="TestLoser",
            bet_amount=100.0,
        )

        assert result.n_bets == 2
        assert result.stake_total_yen == 200.0
        assert result.return_total_yen == 0.0  # 2着なので単勝は外れ
        assert result.roi == 0.0
        assert result.hit_rate == 0.0
        assert result.n_hits == 0

    def test_evaluate_roi_strategy_fukusho_in3(self, roi_payouts_df, roi_race_results_df):
        """複勝: 3着内馬に賭けた場合のROI計算"""
        import pandas as pd

        # 2着馬に賭けるケース (Race R1: B, Race R2: F)
        bets_df = pd.DataFrame({
            "race_id": ["R1", "R2"],
            "horse_id": ["B", "F"],
        })

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="複勝",
            strategy_name="TestPlace",
            bet_amount=100.0,
        )

        assert result.bet_type == "複勝"
        assert result.n_bets == 2
        assert result.stake_total_yen == 200.0
        # 払戻: R1馬番2=150円, R2馬番6=130円 → 280円
        assert result.return_total_yen == pytest.approx(280.0)
        assert result.roi == pytest.approx(1.4)  # 280 / 200 = 1.4
        assert result.hit_rate == 1.0  # 2/2 = 100%
        assert result.n_hits == 2
        assert result.avg_payout == pytest.approx(140.0)  # (150+130)/2

    def test_evaluate_roi_strategy_fukusho_out(self, roi_payouts_df, roi_race_results_df):
        """複勝: 4着馬に賭けた場合 (外れ)"""
        import pandas as pd

        # 4着馬に賭けるケース (Race R1: D, Race R2: H)
        bets_df = pd.DataFrame({
            "race_id": ["R1", "R2"],
            "horse_id": ["D", "H"],
        })

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="複勝",
            strategy_name="TestOut",
            bet_amount=100.0,
        )

        assert result.n_bets == 2
        assert result.return_total_yen == 0.0  # 4着なので複勝も外れ
        assert result.roi == 0.0
        assert result.hit_rate == 0.0
        assert result.n_hits == 0

    def test_evaluate_roi_strategy_mixed(self, roi_payouts_df, roi_race_results_df):
        """単勝: 勝ち1回、負け1回の場合"""
        import pandas as pd

        # R1: 勝ち馬A、R2: 2着馬F
        bets_df = pd.DataFrame({
            "race_id": ["R1", "R2"],
            "horse_id": ["A", "F"],
        })

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="単勝",
            strategy_name="TestMixed",
            bet_amount=100.0,
        )

        assert result.n_bets == 2
        assert result.stake_total_yen == 200.0
        # 払戻: R1=500円 (勝ち), R2=0円 (2着なので外れ)
        assert result.return_total_yen == 500.0
        assert result.roi == pytest.approx(2.5)  # 500 / 200 = 2.5
        assert result.hit_rate == pytest.approx(0.5)  # 1/2 = 50%
        assert result.n_hits == 1
        assert result.avg_payout == 500.0  # 1回だけ当たり

    def test_evaluate_roi_strategy_empty_bets(self, roi_payouts_df, roi_race_results_df):
        """空の賭けデータの場合"""
        import pandas as pd

        bets_df = pd.DataFrame(columns=["race_id", "horse_id"])

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="単勝",
            strategy_name="TestEmpty",
            bet_amount=100.0,
        )

        assert result.n_races == 0
        assert result.n_bets == 0
        assert result.stake_total_yen == 0.0
        assert result.return_total_yen == 0.0
        assert result.roi == 0.0

    def test_evaluate_roi_strategy_top3_bets(self, roi_payouts_df, roi_race_results_df):
        """Top3戦略: 1レースに3頭賭けた場合"""
        import pandas as pd

        # R1のみ、上位3頭 (A, B, C) に賭ける
        bets_df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1"],
            "horse_id": ["A", "B", "C"],
        })

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="単勝",
            strategy_name="TestTop3",
            bet_amount=100.0,
        )

        assert result.n_races == 1
        assert result.n_bets == 3  # 3頭に賭けた
        assert result.stake_total_yen == 300.0  # 100 * 3
        # A(1着)のみ当たり: 500円
        assert result.return_total_yen == 500.0
        assert result.roi == pytest.approx(500.0 / 300.0)
        assert result.hit_rate == pytest.approx(1.0 / 3.0)  # 1/3
        assert result.n_hits == 1

    def test_evaluate_roi_strategy_fukusho_top3(self, roi_payouts_df, roi_race_results_df):
        """複勝Top3戦略: 1レースに3頭賭けた場合"""
        import pandas as pd

        # R1のみ、上位3頭 (A, B, C) に賭ける
        bets_df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1"],
            "horse_id": ["A", "B", "C"],
        })

        result = evaluate_roi_strategy(
            bets_df=bets_df,
            payouts_df=roi_payouts_df,
            race_results_df=roi_race_results_df,
            bet_type="複勝",
            strategy_name="TestTop3Fukusho",
            bet_amount=100.0,
        )

        assert result.n_bets == 3
        assert result.stake_total_yen == 300.0
        # A(1着)=120円, B(2着)=150円, C(3着)=180円 → 450円
        assert result.return_total_yen == pytest.approx(450.0)
        assert result.roi == pytest.approx(1.5)  # 450 / 300 = 1.5
        assert result.hit_rate == 1.0  # 3/3 = 100%
        assert result.n_hits == 3
        assert result.avg_payout == pytest.approx(150.0)  # (120+150+180)/3


# =============================================================================
# Selective Betting Tests
# =============================================================================

class TestSelectiveBetting:
    """選択的ベッティング戦略のテスト (Prob/Gap Threshold)"""

    @pytest.fixture
    def selective_test_df(self):
        """選択的ベッティング用テストデータ"""
        import pandas as pd

        # 4レース分のデータ
        # Race R1: 4頭 (A, B, C, D) - Top1確率: 0.25, gap: 0.10
        # Race R2: 4頭 (E, F, G, H) - Top1確率: 0.40, gap: 0.25
        # Race R3: 4頭 (I, J, K, L) - Top1確率: 0.15, gap: 0.02
        # Race R4: 4頭 (M, N, O, P) - Top1確率: 0.30, gap: 0.05
        return pd.DataFrame({
            "race_id": ["R1"] * 4 + ["R2"] * 4 + ["R3"] * 4 + ["R4"] * 4,
            "horse_id": list("ABCDEFGHIJKLMNOP"),
            "feat1": [1, 2, 3, 4] * 4,  # dummy features
            "feat2": [0.5, 0.3, 0.2, 0.1] * 4,
        })

    @pytest.fixture
    def selective_mock_model(self):
        """確率が閾値テスト用になるモックモデル"""
        class MockModel:
            def __init__(self, predictions):
                self.predictions = predictions
                self.best_iteration = 100

            def predict(self, X, num_iteration=None):
                return self.predictions[:len(X)]

        # R1: Top1=0.25, Top2=0.15, Gap=0.10
        # R2: Top1=0.40, Top2=0.15, Gap=0.25
        # R3: Top1=0.15, Top2=0.13, Gap=0.02
        # R4: Top1=0.30, Top2=0.25, Gap=0.05
        predictions = [
            0.25, 0.15, 0.05, 0.02,  # R1
            0.40, 0.15, 0.10, 0.05,  # R2
            0.15, 0.13, 0.08, 0.04,  # R3
            0.30, 0.25, 0.12, 0.08,  # R4
        ]
        return MockModel(predictions)

    @pytest.fixture
    def selective_payouts_df(self):
        """選択的ベッティング用払戻データ"""
        import pandas as pd

        return pd.DataFrame({
            "race_id": ["R1", "R1", "R2", "R2", "R3", "R3", "R4", "R4"],
            "bet_type": ["単勝", "複勝", "単勝", "複勝", "単勝", "複勝", "単勝", "複勝"],
            "combination": ["1", "1", "5", "5", "9", "9", "13", "13"],
            "payout": [500, 120, 300, 110, 800, 200, 400, 130],
        })

    @pytest.fixture
    def selective_race_results_df(self):
        """選択的ベッティング用レース結果データ"""
        import pandas as pd

        # R2: E wins (finish_order=1), but F is popularity=1
        # This tests the case where model wins but pop loses
        return pd.DataFrame({
            "race_id": ["R1"] * 4 + ["R2"] * 4 + ["R3"] * 4 + ["R4"] * 4,
            "horse_id": list("ABCDEFGHIJKLMNOP"),
            "horse_no": list(range(1, 17)),
            "horse_no_str": [str(i) for i in range(1, 17)],
            "finish_order": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            # R1: pop=1 is B (2nd place)
            # R2: pop=1 is F (2nd place) - Model predicts E (1st), Pop picks F (2nd)
            # R3: pop=1 is J (2nd place)
            # R4: pop=1 is N (2nd place)
            "popularity": [2, 1, 3, 4, 2, 1, 3, 4, 3, 1, 2, 4, 2, 1, 3, 4],
        })

    def test_prob_threshold_bets_filters_correctly(self, selective_test_df, selective_mock_model):
        """確率閾値でレースがフィルタされることをテスト"""
        from src.features_v4.train_eval_v4 import generate_model_top1_prob_threshold_bets

        # 閾値 0.20: R1(0.25), R2(0.40), R4(0.30) が対象 (R3は0.15で除外)
        bets_df, pred_df = generate_model_top1_prob_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            prob_threshold=0.20,
        )

        assert len(bets_df) == 3  # R1, R2, R4
        assert set(bets_df["race_id"]) == {"R1", "R2", "R4"}
        assert "R3" not in set(bets_df["race_id"])

    def test_prob_threshold_high_value_filters_more(self, selective_test_df, selective_mock_model):
        """高い確率閾値でより多くのレースがフィルタされる"""
        from src.features_v4.train_eval_v4 import generate_model_top1_prob_threshold_bets

        # 閾値 0.35: R2(0.40) のみ対象
        bets_df, _ = generate_model_top1_prob_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            prob_threshold=0.35,
        )

        assert len(bets_df) == 1
        assert set(bets_df["race_id"]) == {"R2"}

    def test_gap_threshold_bets_filters_correctly(self, selective_test_df, selective_mock_model):
        """ギャップ閾値でレースがフィルタされることをテスト"""
        from src.features_v4.train_eval_v4 import generate_model_top1_gap_threshold_bets

        # 閾値 0.04: R1(0.10), R2(0.25), R4(0.05) が対象 (R3は0.02で除外)
        # Note: R4 gap = 0.05, which is >= 0.04
        bets_df, pred_df = generate_model_top1_gap_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            gap_threshold=0.04,
        )

        assert len(bets_df) == 3  # R1, R2, R4
        assert set(bets_df["race_id"]) == {"R1", "R2", "R4"}
        assert "R3" not in set(bets_df["race_id"])

    def test_gap_threshold_high_value_filters_more(self, selective_test_df, selective_mock_model):
        """高いギャップ閾値でより多くのレースがフィルタされる"""
        from src.features_v4.train_eval_v4 import generate_model_top1_gap_threshold_bets

        # 閾値 0.15: R2(0.25) のみ対象
        bets_df, _ = generate_model_top1_gap_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            gap_threshold=0.15,
        )

        assert len(bets_df) == 1
        assert set(bets_df["race_id"]) == {"R2"}

    def test_selective_bet_result_coverage(
        self,
        selective_test_df,
        selective_mock_model,
        selective_payouts_df,
        selective_race_results_df,
    ):
        """SelectiveBetResult の coverage 計算が正しいことをテスト"""
        from src.features_v4.train_eval_v4 import (
            generate_model_top1_prob_threshold_bets,
            evaluate_selective_bet,
        )

        # 閾値 0.20: 3レース/4レース = 75%
        bets_df, _ = generate_model_top1_prob_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            prob_threshold=0.20,
        )
        race_ids_subset = bets_df["race_id"].unique().tolist()

        result = evaluate_selective_bet(
            bets_df,
            race_ids_subset,
            selective_payouts_df,
            selective_race_results_df,
            bet_type="単勝",
            threshold_type="prob",
            threshold_value=0.20,
            n_total_races=4,
            bet_amount=100.0,
        )

        assert result.n_total_races == 4
        assert result.n_races_bet == 3
        assert result.coverage == pytest.approx(0.75)

    def test_selective_bet_result_subset_comparison(
        self,
        selective_test_df,
        selective_mock_model,
        selective_payouts_df,
        selective_race_results_df,
    ):
        """subset comparison で同じレース集合を使っていることをテスト"""
        from src.features_v4.train_eval_v4 import (
            generate_model_top1_prob_threshold_bets,
            evaluate_selective_bet,
        )

        # 閾値 0.35: R2 のみ
        bets_df, _ = generate_model_top1_prob_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            prob_threshold=0.35,
        )
        race_ids_subset = bets_df["race_id"].unique().tolist()

        result = evaluate_selective_bet(
            bets_df,
            race_ids_subset,
            selective_payouts_df,
            selective_race_results_df,
            bet_type="単勝",
            threshold_type="prob",
            threshold_value=0.35,
            n_total_races=4,
            bet_amount=100.0,
        )

        # R2 のみ賭ける
        assert result.n_races_bet == 1
        assert result.coverage == pytest.approx(0.25)

        # Model: R2 で A (horse_id E) に賭ける (finish_order=1なので当たり、払戻300円)
        # Model の予測1位 = horse E (pred_proba=0.40)
        # finish_order=1 なので当たり
        assert result.model_stake_total == 100.0
        assert result.model_return_total == 300.0  # 300 / 100 * 100 = 300
        assert result.model_roi == pytest.approx(3.0)  # 300 / 100 = 3.0
        assert result.model_hit_rate == 1.0
        assert result.model_n_hits == 1

        # Pop: R2 の popularity=1 は F (horse_id F) だが、finish_order=2 なので外れ
        # (race_results_df で R2 の popularity=1 は horse_id F = finish_order=2)
        assert result.pop_stake_total == 100.0
        assert result.pop_return_total == 0.0
        assert result.pop_roi == 0.0
        assert result.pop_hit_rate == 0.0
        assert result.pop_n_hits == 0

        # 差分
        assert result.roi_diff == pytest.approx(3.0)  # 3.0 - 0.0

    def test_selective_bet_empty_result(
        self,
        selective_test_df,
        selective_mock_model,
        selective_payouts_df,
        selective_race_results_df,
    ):
        """閾値が高すぎて0レースになった場合"""
        from src.features_v4.train_eval_v4 import (
            generate_model_top1_prob_threshold_bets,
            evaluate_selective_bet,
        )

        # 閾値 0.90: 該当レースなし
        bets_df, _ = generate_model_top1_prob_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            prob_threshold=0.90,
        )
        race_ids_subset = bets_df["race_id"].unique().tolist()

        result = evaluate_selective_bet(
            bets_df,
            race_ids_subset,
            selective_payouts_df,
            selective_race_results_df,
            bet_type="単勝",
            threshold_type="prob",
            threshold_value=0.90,
            n_total_races=4,
            bet_amount=100.0,
        )

        assert result.n_races_bet == 0
        assert result.coverage == 0.0
        assert result.model_roi == 0.0
        assert result.pop_roi == 0.0

    def test_pred_df_contains_gap_info(self, selective_test_df, selective_mock_model):
        """pred_df にギャップ情報が含まれていることをテスト"""
        from src.features_v4.train_eval_v4 import generate_model_top1_prob_threshold_bets

        _, pred_df = generate_model_top1_prob_threshold_bets(
            selective_test_df,
            selective_mock_model,
            ["feat1", "feat2"],
            prob_threshold=0.0,  # すべてのレース
        )

        # 必要なカラムが含まれている
        assert "pred_proba" in pred_df.columns
        assert "pred_rank" in pred_df.columns
        assert "top1_proba" in pred_df.columns
        assert "top2_proba" in pred_df.columns
        assert "gap" in pred_df.columns

        # R1 のギャップを確認 (0.25 - 0.15 = 0.10)
        r1_top1 = pred_df[(pred_df["race_id"] == "R1") & (pred_df["pred_rank"] == 1)].iloc[0]
        assert r1_top1["top1_proba"] == pytest.approx(0.25)
        assert r1_top1["top2_proba"] == pytest.approx(0.15)
        assert r1_top1["gap"] == pytest.approx(0.10)


class TestROISweepResult:
    """ROISweepResult のテスト"""

    def test_roi_sweep_result_structure(self):
        """ROISweepResult の構造テスト"""
        from src.features_v4.train_eval_v4 import ROISweepResult, SelectiveBetResult

        # SelectiveBetResult を作成
        sbr = SelectiveBetResult(
            threshold_type="prob",
            threshold_value=0.10,
            bet_type="単勝",
            n_total_races=100,
            n_races_bet=50,
            coverage=0.5,
            model_stake_total=5000.0,
            model_return_total=6000.0,
            model_roi=1.2,
            model_hit_rate=0.3,
            model_n_hits=15,
            model_avg_payout=400.0,
            pop_stake_total=5000.0,
            pop_return_total=4000.0,
            pop_roi=0.8,
            pop_hit_rate=0.25,
            pop_n_hits=12,
            pop_avg_payout=333.0,
            roi_diff=0.4,
            hit_diff=0.05,
        )

        # ROISweepResult を作成
        rsr = ROISweepResult(
            dataset="test",
            sweep_type="prob",
            bet_type="単勝",
            results=[sbr],
        )

        assert rsr.dataset == "test"
        assert rsr.sweep_type == "prob"
        assert rsr.bet_type == "単勝"
        assert len(rsr.results) == 1
        assert rsr.results[0].coverage == 0.5
        assert rsr.results[0].roi_diff == 0.4


# =============================================================================
# Odds Snapshots Tests
# =============================================================================

class TestOddsSnapshots:
    """odds_snapshots テーブルとスナップショットベース評価のテスト"""

    @pytest.fixture
    def db_with_snapshots(self):
        """odds_snapshots テーブルを含むデータベース"""
        import pandas as pd

        conn = sqlite3.connect(":memory:")

        # odds_snapshots テーブルを作成
        conn.execute("""
            CREATE TABLE odds_snapshots (
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

        # race_results テーブルを作成
        conn.execute("""
            CREATE TABLE race_results (
                race_id TEXT NOT NULL,
                horse_id TEXT NOT NULL,
                horse_no INTEGER,
                finish_order INTEGER,
                popularity INTEGER,
                PRIMARY KEY (race_id, horse_id)
            )
        """)

        # payouts テーブルを作成
        conn.execute("""
            CREATE TABLE payouts (
                race_id TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                combination TEXT NOT NULL,
                payout INTEGER,
                PRIMARY KEY (race_id, bet_type, combination)
            )
        """)

        # テストデータを挿入
        # Race R1: スナップショットあり
        # Race R2: スナップショットあり (異なる時刻で複数)
        conn.executemany(
            "INSERT INTO odds_snapshots (race_id, horse_no, observed_at, win_odds, popularity) VALUES (?, ?, ?, ?, ?)",
            [
                ("R1", 1, "2024-12-27T20:00:00", 5.0, 2),
                ("R1", 2, "2024-12-27T20:00:00", 3.0, 1),
                ("R1", 3, "2024-12-27T20:00:00", 10.0, 3),
                # R2: 複数スナップショット (20:00 と 21:00)
                ("R2", 1, "2024-12-27T20:00:00", 4.0, 1),  # 20:00 時点
                ("R2", 2, "2024-12-27T20:00:00", 6.0, 2),
                ("R2", 1, "2024-12-27T21:00:00", 3.5, 1),  # 21:00 時点 (最新)
                ("R2", 2, "2024-12-27T21:00:00", 7.0, 2),
            ]
        )

        # race_results (最終結果の popularity は異なる場合がある)
        conn.executemany(
            "INSERT INTO race_results (race_id, horse_id, horse_no, finish_order, popularity) VALUES (?, ?, ?, ?, ?)",
            [
                ("R1", "A", 1, 1, 3),  # 最終人気=3 (スナップショット=2 と異なる)
                ("R1", "B", 2, 2, 1),
                ("R1", "C", 3, 3, 2),
                ("R2", "D", 1, 1, 2),  # 最終人気=2 (スナップショット=1 と異なる)
                ("R2", "E", 2, 2, 1),
            ]
        )

        # payouts
        conn.executemany(
            "INSERT INTO payouts (race_id, bet_type, combination, payout) VALUES (?, ?, ?, ?)",
            [
                ("R1", "単勝", "1", 500),
                ("R2", "単勝", "1", 350),
            ]
        )

        conn.commit()
        yield conn
        conn.close()

    def test_load_odds_snapshots_basic(self, db_with_snapshots):
        """基本的なスナップショット読み込み"""
        from src.features_v4.train_eval_v4 import load_odds_snapshots_for_races

        snapshots = load_odds_snapshots_for_races(db_with_snapshots, ["R1"])

        assert snapshots is not None
        assert len(snapshots) == 3  # R1 の 3 馬
        assert set(snapshots["horse_no"]) == {1, 2, 3}

    def test_load_odds_snapshots_latest(self, db_with_snapshots):
        """最新スナップショットの取得"""
        from src.features_v4.train_eval_v4 import load_odds_snapshots_for_races

        # decision_cutoff なし = 最新
        snapshots = load_odds_snapshots_for_races(db_with_snapshots, ["R2"])

        assert snapshots is not None
        r2_horse1 = snapshots[snapshots["horse_no"] == 1].iloc[0]
        # 最新 (21:00) のオッズ
        assert r2_horse1["win_odds"] == 3.5
        assert r2_horse1["observed_at"] == "2024-12-27T21:00:00"

    def test_load_odds_snapshots_with_cutoff(self, db_with_snapshots):
        """decision_cutoff を指定した場合"""
        from src.features_v4.train_eval_v4 import load_odds_snapshots_for_races

        # 20:30 以前 = 20:00 のスナップショットのみ
        snapshots = load_odds_snapshots_for_races(
            db_with_snapshots,
            ["R2"],
            decision_cutoff="2024-12-27T20:30:00",
        )

        assert snapshots is not None
        r2_horse1 = snapshots[snapshots["horse_no"] == 1].iloc[0]
        # 20:00 のオッズ (20:30 以前で最新)
        assert r2_horse1["win_odds"] == 4.0
        assert r2_horse1["observed_at"] == "2024-12-27T20:00:00"

    def test_load_odds_snapshots_no_table(self, in_memory_db):
        """odds_snapshots テーブルがない場合"""
        from src.features_v4.train_eval_v4 import load_odds_snapshots_for_races

        # テーブルなし
        result = load_odds_snapshots_for_races(in_memory_db, ["R1", "R2"])
        assert result is None

    def test_load_race_results_with_snapshots(self, db_with_snapshots):
        """race_results にスナップショット popularity をマージ"""
        from src.features_v4.train_eval_v4 import load_race_results_for_roi

        # スナップショット使用
        results = load_race_results_for_roi(
            db_with_snapshots,
            ["R1"],
            decision_cutoff=None,
            use_snapshots=True,
        )

        # R1 の horse_no=1 は最終 popularity=3、スナップショット popularity=2
        r1_horse1 = results[(results["race_id"] == "R1") & (results["horse_no"] == 1)].iloc[0]
        assert r1_horse1["popularity"] == 2  # スナップショットの値
        assert r1_horse1["popularity_source"] == "odds_snapshot"

    def test_load_race_results_without_snapshots(self, db_with_snapshots):
        """スナップショット使用しない場合"""
        from src.features_v4.train_eval_v4 import load_race_results_for_roi

        # スナップショット不使用
        results = load_race_results_for_roi(
            db_with_snapshots,
            ["R1"],
            decision_cutoff=None,
            use_snapshots=False,
        )

        # race_results の値がそのまま
        r1_horse1 = results[(results["race_id"] == "R1") & (results["horse_no"] == 1)].iloc[0]
        assert r1_horse1["popularity"] == 3  # race_results の値
        assert r1_horse1["popularity_source"] == "race_results"

    def test_load_race_results_partial_snapshots(self, db_with_snapshots):
        """一部のレースのみスナップショットがある場合"""
        from src.features_v4.train_eval_v4 import load_race_results_for_roi

        # R3 を追加 (スナップショットなし)
        db_with_snapshots.execute(
            "INSERT INTO race_results (race_id, horse_id, horse_no, finish_order, popularity) VALUES (?, ?, ?, ?, ?)",
            ("R3", "F", 1, 1, 5),
        )
        db_with_snapshots.commit()

        results = load_race_results_for_roi(
            db_with_snapshots,
            ["R1", "R3"],
            use_snapshots=True,
        )

        # R1 はスナップショットあり
        r1 = results[results["race_id"] == "R1"]
        assert all(r1["popularity_source"] == "odds_snapshot")

        # R3 はスナップショットなし (race_results のまま)
        r3 = results[results["race_id"] == "R3"]
        assert all(r3["popularity_source"] == "race_results")

    def test_coverage_warning_when_missing_races(self, db_with_snapshots, caplog):
        """一部のレースにスナップショットがない場合に警告ログ"""
        import logging
        from src.features_v4.train_eval_v4 import load_odds_snapshots_for_races

        # R3 を追加してもスナップショットがないのでカバレッジ警告が出るはず
        db_with_snapshots.execute(
            "INSERT INTO race_results (race_id, horse_id, horse_no, finish_order, popularity) VALUES (?, ?, ?, ?, ?)",
            ("R3", "G", 1, 1, 5),
        )
        db_with_snapshots.commit()

        with caplog.at_level(logging.WARNING):
            result = load_odds_snapshots_for_races(
                db_with_snapshots,
                ["R1", "R2", "R3"],  # R3 にはスナップショットなし
            )

        # 結果は R1, R2 のスナップショットのみ
        assert result is not None
        assert set(result["race_id"].unique()) == {"R1", "R2"}

        # 警告ログが出力されること
        assert "Snapshot coverage" in caplog.text
        assert "missing: 1 races" in caplog.text or "Missing race_ids" in caplog.text


# =============================================================================
# ROI Sweep Flat Artifacts Tests
# =============================================================================


class TestRoiSweepFlatArtifacts:
    """roi_sweep_flat JSON 出力のテスト"""

    def test_save_roi_sweep_flat_structure(self, tmp_path):
        """フラットなJSON構造が正しく生成されること"""
        from src.features_v4.train_eval_v4 import (
            save_roi_sweep_flat_artifacts,
            SelectiveBetResult,
            ROISweepResult,
        )

        # テストデータ作成
        result1 = SelectiveBetResult(
            threshold_type="prob",
            threshold_value=0.10,
            bet_type="単勝",
            n_total_races=100,
            n_races_bet=30,
            coverage=0.30,
            model_stake_total=3000.0,
            model_return_total=3600.0,
            model_roi=1.20,
            model_hit_rate=0.25,
            model_n_hits=8,
            model_avg_payout=450.0,
            pop_stake_total=3000.0,
            pop_return_total=2700.0,
            pop_roi=0.90,
            pop_hit_rate=0.22,
            pop_n_hits=7,
            pop_avg_payout=386.0,
            roi_diff=0.30,
            hit_diff=0.03,
        )

        sweep_result = ROISweepResult(
            dataset="test",
            sweep_type="prob",
            bet_type="単勝",
            results=[result1],
        )

        all_results = {
            "test": {
                "prob": [sweep_result],
                "gap": [],
            }
        }

        # 保存
        filepath = save_roi_sweep_flat_artifacts(
            all_results, "target_win", "20241227_120000", str(tmp_path)
        )

        # ファイル存在確認
        assert Path(filepath).exists()

        # JSON読み込み
        with open(filepath, "r", encoding="utf-8") as f:
            flat_data = json.load(f)

        # 構造確認
        assert isinstance(flat_data, list)
        assert len(flat_data) == 1

        row = flat_data[0]
        assert row["dataset"] == "test"
        assert row["kind"] == "prob"
        assert row["bet_type"] == "単勝"
        assert row["threshold_value"] == 0.10
        assert row["coverage"] == 0.30
        assert row["model_roi"] == 1.20
        assert row["pop_roi"] == 0.90
        assert row["roi_diff"] == 0.30

    def test_save_roi_sweep_flat_multiple_entries(self, tmp_path):
        """複数のエントリが正しく展開されること"""
        from src.features_v4.train_eval_v4 import (
            save_roi_sweep_flat_artifacts,
            SelectiveBetResult,
            ROISweepResult,
        )

        # 複数の閾値結果
        results = [
            SelectiveBetResult(
                threshold_type="gap",
                threshold_value=t,
                bet_type="複勝",
                n_total_races=100,
                n_races_bet=int(100 - t * 1000),
                coverage=(100 - t * 1000) / 100,
                model_stake_total=1000.0,
                model_return_total=1100.0,
                model_roi=1.10,
                model_hit_rate=0.50,
                model_n_hits=50,
                model_avg_payout=220.0,
                pop_stake_total=1000.0,
                pop_return_total=900.0,
                pop_roi=0.90,
                pop_hit_rate=0.45,
                pop_n_hits=45,
                pop_avg_payout=200.0,
                roi_diff=0.20,
                hit_diff=0.05,
            )
            for t in [0.01, 0.02, 0.03]
        ]

        sweep_result = ROISweepResult(
            dataset="val",
            sweep_type="gap",
            bet_type="複勝",
            results=results,
        )

        all_results = {
            "val": {
                "prob": [],
                "gap": [sweep_result],
            }
        }

        filepath = save_roi_sweep_flat_artifacts(
            all_results, "target_in3", "20241227_130000", str(tmp_path)
        )

        with open(filepath, "r", encoding="utf-8") as f:
            flat_data = json.load(f)

        # 3行あること
        assert len(flat_data) == 3

        # 各行のthreshold_valueが正しいこと
        thresholds = [row["threshold_value"] for row in flat_data]
        assert 0.01 in thresholds
        assert 0.02 in thresholds
        assert 0.03 in thresholds


# =============================================================================
# Snapshot-based Feature Building Tests
# =============================================================================


class TestSnapshotFeatureBuilding:
    """FeatureBuilderV4 のスナップショットベース市場特徴量テスト"""

    @pytest.fixture
    def db_for_feature_builder(self):
        """特徴量ビルダー用のテストデータベース"""
        conn = sqlite3.connect(":memory:")

        # races テーブル
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

        # race_results テーブル
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
                prize_money INTEGER,
                jockey_id TEXT,
                trainer_id TEXT,
                sex TEXT,
                age INTEGER,
                weight REAL,
                frame_no INTEGER,
                horse_no INTEGER,
                finish_status TEXT,
                PRIMARY KEY (race_id, horse_id)
            )
        """)

        # odds_snapshots テーブル
        conn.execute("""
            CREATE TABLE odds_snapshots (
                race_id TEXT NOT NULL,
                horse_no INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                win_odds REAL,
                popularity INTEGER,
                source TEXT,
                PRIMARY KEY (race_id, horse_no, observed_at)
            )
        """)

        # テストデータ: レース
        conn.execute("""
            INSERT INTO races
            (race_id, date, place, course_type, distance, track_condition,
             race_class, grade, race_no, course_turn, course_inout, head_count)
            VALUES
            ('202412280101', '2024-12-28', '東京', 'turf', 1600, '良',
             '3勝クラス', NULL, 1, 'left', 'inner', 3)
        """)

        # テストデータ: 出走馬 (race_resultsの win_odds/popularity は最終値)
        conn.executemany("""
            INSERT INTO race_results
            (race_id, horse_id, finish_order, win_odds, popularity, horse_no, frame_no,
             sex, age, weight, finish_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            ("202412280101", "H1", 1, 2.5, 1, 1, 1, "牡", 4, 55.0, None),   # 最終オッズ 2.5
            ("202412280101", "H2", 2, 5.0, 2, 2, 1, "牝", 3, 54.0, None),   # 最終オッズ 5.0
            ("202412280101", "H3", 3, 8.0, 3, 3, 2, "牡", 5, 56.0, None),   # 最終オッズ 8.0
        ])

        # テストデータ: スナップショット (前日21時点 - 最終値とは異なる)
        conn.executemany("""
            INSERT INTO odds_snapshots (race_id, horse_no, observed_at, win_odds, popularity)
            VALUES (?, ?, ?, ?, ?)
        """, [
            ("202412280101", 1, "2024-12-27T21:00:00", 3.0, 2),   # スナップショットは 3.0, 2番人気
            ("202412280101", 2, "2024-12-27T21:00:00", 4.5, 1),   # スナップショットは 4.5, 1番人気
            ("202412280101", 3, "2024-12-27T21:00:00", 7.5, 3),   # スナップショットは 7.5, 3番人気
        ])

        conn.commit()
        yield conn
        conn.close()

    def test_build_features_without_snapshots(self, db_for_feature_builder):
        """スナップショット無しの場合は race_results の値を使用"""
        builder = FeatureBuilderV4(db_for_feature_builder)
        df = builder.build_features_for_race(
            "202412280101",
            include_pedigree=False,
            include_market=True,
            include_snapshots=False,
        )

        assert len(df) == 3

        # race_results の値 (最終オッズ) が使用される
        h1 = df[df["horse_id"] == "H1"].iloc[0]
        assert h1["market_win_odds"] == 2.5  # 最終オッズ
        assert h1["market_popularity"] == 1  # 最終人気

        h2 = df[df["horse_id"] == "H2"].iloc[0]
        assert h2["market_win_odds"] == 5.0
        assert h2["market_popularity"] == 2

    def test_build_features_with_snapshots(self, db_for_feature_builder):
        """スナップショット有りの場合は odds_snapshots の値を使用"""
        builder = FeatureBuilderV4(db_for_feature_builder)
        df = builder.build_features_for_race(
            "202412280101",
            include_pedigree=False,
            include_market=True,
            include_snapshots=True,
        )

        assert len(df) == 3

        # スナップショットの値 (前日21時点) が使用される
        h1 = df[df["horse_id"] == "H1"].iloc[0]
        assert h1["market_win_odds"] == 3.0  # スナップショット
        assert h1["market_popularity"] == 2  # スナップショット (race_results とは異なる!)

        h2 = df[df["horse_id"] == "H2"].iloc[0]
        assert h2["market_win_odds"] == 4.5
        assert h2["market_popularity"] == 1  # スナップショットでは1番人気

    def test_build_features_with_cutoff(self, db_for_feature_builder):
        """decision_cutoff でフィルタリングされること"""
        conn = db_for_feature_builder

        # 追加のスナップショット (20:00 時点)
        conn.executemany("""
            INSERT INTO odds_snapshots (race_id, horse_no, observed_at, win_odds, popularity)
            VALUES (?, ?, ?, ?, ?)
        """, [
            ("202412280101", 1, "2024-12-27T20:00:00", 3.5, 3),  # 20:00 時点
            ("202412280101", 2, "2024-12-27T20:00:00", 4.0, 2),
            ("202412280101", 3, "2024-12-27T20:00:00", 8.0, 1),
        ])
        conn.commit()

        builder = FeatureBuilderV4(conn)

        # cutoff=20:30 の場合、20:00 のスナップショットが使用される
        df = builder.build_features_for_race(
            "202412280101",
            include_pedigree=False,
            include_market=True,
            include_snapshots=True,
            decision_cutoff="2024-12-27T20:30:00",
        )

        h1 = df[df["horse_id"] == "H1"].iloc[0]
        assert h1["market_win_odds"] == 3.5  # 20:00 のスナップショット
        assert h1["market_popularity"] == 3

        # cutoff=21:30 の場合、21:00 のスナップショットが使用される (最新)
        df2 = builder.build_features_for_race(
            "202412280101",
            include_pedigree=False,
            include_market=True,
            include_snapshots=True,
            decision_cutoff="2024-12-27T21:30:00",
        )

        h1_2 = df2[df2["horse_id"] == "H1"].iloc[0]
        assert h1_2["market_win_odds"] == 3.0  # 21:00 のスナップショット
        assert h1_2["market_popularity"] == 2

    def test_build_features_snapshot_missing_horse(self, db_for_feature_builder):
        """スナップショットにない馬は NULL になること"""
        import pandas as pd

        conn = db_for_feature_builder

        # horse_no=3 のスナップショットを削除
        conn.execute("""
            DELETE FROM odds_snapshots WHERE horse_no = 3
        """)
        conn.commit()

        builder = FeatureBuilderV4(conn)
        df = builder.build_features_for_race(
            "202412280101",
            include_pedigree=False,
            include_market=True,
            include_snapshots=True,
        )

        # H3 の市場情報は NULL/NaN になる
        h3 = df[df["horse_id"] == "H3"].iloc[0]
        assert pd.isna(h3["market_win_odds"])
        assert pd.isna(h3["market_popularity"])

        # H1, H2 は正常
        h1 = df[df["horse_id"] == "H1"].iloc[0]
        assert h1["market_win_odds"] == 3.0

    def test_build_features_no_snapshot_table(self, in_memory_db):
        """odds_snapshots テーブルがない場合も動作すること"""
        conn = in_memory_db

        # 最小限のテーブル
        conn.execute("""
            CREATE TABLE races (
                race_id TEXT PRIMARY KEY, date TEXT, place TEXT, course_type TEXT,
                distance INTEGER, track_condition TEXT, race_class TEXT, grade TEXT,
                race_no INTEGER, course_turn TEXT, course_inout TEXT, head_count INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE race_results (
                race_id TEXT, horse_id TEXT, finish_order INTEGER, last_3f REAL,
                body_weight INTEGER, body_weight_diff INTEGER, passing_order TEXT,
                win_odds REAL, popularity INTEGER, prize_money INTEGER, jockey_id TEXT,
                trainer_id TEXT, sex TEXT, age INTEGER, weight REAL, frame_no INTEGER,
                horse_no INTEGER, finish_status TEXT, PRIMARY KEY (race_id, horse_id)
            )
        """)
        conn.execute("""
            INSERT INTO races VALUES
            ('R1', '2024-12-28', '東京', 'turf', 1600, '良', NULL, NULL, 1, NULL, NULL, 1)
        """)
        conn.execute("""
            INSERT INTO race_results
            (race_id, horse_id, finish_order, win_odds, popularity, horse_no, frame_no,
             sex, age, weight, finish_status)
            VALUES ('R1', 'H1', 1, 2.5, 1, 1, 1, '牡', 4, 55.0, NULL)
        """)
        conn.commit()

        builder = FeatureBuilderV4(conn)
        # スナップショットモードでもテーブルがなければ NULL になる
        df = builder.build_features_for_race(
            "R1",
            include_pedigree=False,
            include_market=True,
            include_snapshots=True,
        )

        import pandas as pd
        h1 = df[df["horse_id"] == "H1"].iloc[0]
        assert pd.isna(h1["market_win_odds"])  # テーブルがないので NULL/NaN


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
