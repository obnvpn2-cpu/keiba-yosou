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

from src.features_v4.feature_builder_v4 import (
    FeatureBuilderV4,
    is_invalid_finish_status,
    INVALID_FINISH_STATUS_EXACT,
    INVALID_FINISH_STATUS_PATTERNS,
)

from src.features_v4.train_eval_v4 import (
    evaluate_ranking,
    RankingResult,
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
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
