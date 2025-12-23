# -*- coding: utf-8 -*-
"""
feature_table_v4.py - FeaturePack v1 DDL (200+ features)

【重要: 未来リーク防止ルール】

1. 当該レースの win_odds / popularity は使用禁止
   - 同日オッズには未来情報が含まれる可能性がある
   - 使用する場合は market_* プレフィックスで明示し、
     学習時に除外可能にする

2. masters テーブルの career_* / total_* フィールドは使用禁止
   - これらは scraped_at 時点の累積値であり、未来情報
   - 代わりに race_results から as-of で再計算する

3. 全ての統計は race_date より前のデータのみで計算
   - race_date 当日は含めない (< race_date, not <=)

4. 血統情報 (horse_pedigree) は静的なので使用可
"""

import logging
import sqlite3
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Table v4 DDL
# =============================================================================

CREATE_FEATURE_TABLE_V4 = """
-- =============================================================================
-- feature_table_v4: FeaturePack v1 (200+ features)
-- =============================================================================
--
-- 【未来リーク防止】
-- - 当該レースの win_odds/popularity は market_* プレフィックスで隔離
-- - 全ての _asof サフィックスは race_date より前のデータで計算
-- - masters の career_*/total_* は使用しない
--
-- 【カラム命名規則】
-- - 馬の過去成績: h_* (horse)
-- - 騎手の成績: j_* (jockey)
-- - 調教師の成績: t_* (trainer)
-- - ペース関連: pace_*
-- - 血統ハッシュ: ped_hash_* (512次元)
-- - 祖先頻度ハッシュ: anc_hash_* (128次元)
-- - 当日市場情報: market_* (学習時除外可能)
-- =============================================================================

CREATE TABLE IF NOT EXISTS feature_table_v4 (
    -- =========================================================================
    -- 1. Identity & Metadata (NOT features)
    -- =========================================================================
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    race_date TEXT NOT NULL,          -- YYYY-MM-DD format

    -- =========================================================================
    -- 2. Target Variables (NOT features)
    -- =========================================================================
    target_win INTEGER,               -- 1着なら1
    target_in3 INTEGER,               -- 3着以内なら1
    target_quinella INTEGER,          -- 2着以内なら1 (馬連用)

    -- =========================================================================
    -- 3. base_race: 基本レース情報 (~25 columns)
    -- =========================================================================
    -- 3.1 Race attributes
    place TEXT,                       -- 開催場 (東京, 中山, etc.)
    place_id INTEGER,                 -- 開催場ID (0-9)
    surface TEXT,                     -- 芝/ダ/障
    surface_id INTEGER,               -- 0:芝, 1:ダ, 2:障
    distance INTEGER,                 -- 距離 (m)
    distance_cat INTEGER,             -- 距離カテゴリ (1000, 1200, ..., 3000)
    track_condition TEXT,             -- 良/稍/重/不
    track_condition_id INTEGER,       -- 0:良, 1:稍, 2:重, 3:不
    course_turn TEXT,                 -- 右/左/直
    course_turn_id INTEGER,           -- 0:右, 1:左, 2:直
    course_inout TEXT,                -- 内/外/直
    course_inout_id INTEGER,          -- 0:内, 1:外, 2:直

    -- 3.2 Race context
    race_year INTEGER,
    race_month INTEGER,
    race_day_of_week INTEGER,         -- 0:Mon, 6:Sun
    race_no INTEGER,                  -- レース番号 (1-12)
    grade TEXT,                       -- G1, G2, G3, OP, Listed, etc.
    grade_id INTEGER,                 -- 0:G1, 1:G2, 2:G3, 3:OP, 4:Listed, 5:その他
    race_class TEXT,                  -- クラス (新馬, 未勝利, 1勝, etc.)
    race_class_id INTEGER,
    field_size INTEGER,               -- 出走頭数

    -- 3.3 Horse position in race
    waku INTEGER,                     -- 枠番 (1-8)
    umaban INTEGER,                   -- 馬番 (1-18)
    umaban_norm REAL,                 -- 馬番 / field_size (正規化)
    sex TEXT,                         -- 牡/牝/セ
    sex_id INTEGER,                   -- 0:牡, 1:牝, 2:セ
    age INTEGER,                      -- 馬齢
    weight_carried REAL,              -- 斤量 (kg)

    -- =========================================================================
    -- 4. horse_form: 馬の過去成績・フォーム (~40 columns)
    -- =========================================================================
    -- 4.1 Global as-of stats (race_date より前の全レース)
    h_n_starts INTEGER,               -- 出走数
    h_n_wins INTEGER,                 -- 勝利数
    h_n_in3 INTEGER,                  -- 3着以内数
    h_win_rate REAL,                  -- 勝率
    h_in3_rate REAL,                  -- 複勝率
    h_avg_finish REAL,                -- 平均着順
    h_std_finish REAL,                -- 着順標準偏差
    h_best_finish INTEGER,            -- 最高着順
    h_worst_finish INTEGER,           -- 最低着順

    -- 4.2 Distance category stats
    h_n_starts_dist INTEGER,          -- 同距離カテゴリ出走数
    h_win_rate_dist REAL,             -- 同距離カテゴリ勝率
    h_in3_rate_dist REAL,             -- 同距離カテゴリ複勝率
    h_avg_finish_dist REAL,           -- 同距離カテゴリ平均着順
    h_avg_last3f_dist REAL,           -- 同距離カテゴリ平均上がり3F

    -- 4.3 Surface stats
    h_n_starts_surface INTEGER,       -- 同馬場出走数
    h_win_rate_surface REAL,          -- 同馬場勝率
    h_in3_rate_surface REAL,          -- 同馬場複勝率

    -- 4.4 Track condition stats
    h_n_starts_track INTEGER,         -- 同馬場状態出走数
    h_win_rate_track REAL,            -- 同馬場状態勝率
    h_in3_rate_track REAL,            -- 同馬場状態複勝率

    -- 4.5 Course (venue) stats
    h_n_starts_course INTEGER,        -- 同コース出走数
    h_win_rate_course REAL,           -- 同コース勝率
    h_in3_rate_course REAL,           -- 同コース複勝率

    -- 4.6 Recent form (last N races)
    h_days_since_last INTEGER,        -- 前走からの日数
    h_recent3_avg_finish REAL,        -- 直近3走平均着順
    h_recent3_best_finish INTEGER,    -- 直近3走最高着順
    h_recent3_avg_last3f REAL,        -- 直近3走平均上がり3F
    h_recent5_avg_finish REAL,        -- 直近5走平均着順
    h_recent5_win_rate REAL,          -- 直近5走勝率
    h_recent5_in3_rate REAL,          -- 直近5走複勝率

    -- 4.7 Weight features
    h_body_weight INTEGER,            -- 馬体重
    h_body_weight_diff INTEGER,       -- 馬体重増減
    h_avg_body_weight REAL,           -- 過去平均馬体重
    h_body_weight_dev REAL,           -- 馬体重偏差 (current - avg)

    -- 4.8 First run flag
    h_is_first_run INTEGER,           -- 新馬/初出走フラグ

    -- =========================================================================
    -- 5. pace_position: ペース・位置取り (~20 columns)
    -- =========================================================================
    -- 5.1 Historical passing position stats
    h_avg_corner1_pos REAL,           -- 過去平均1角位置
    h_avg_corner2_pos REAL,           -- 過去平均2角位置
    h_avg_corner3_pos REAL,           -- 過去平均3角位置
    h_avg_corner4_pos REAL,           -- 過去平均4角位置
    h_avg_early_pos REAL,             -- 過去平均序盤位置 (1-2角平均)
    h_avg_late_pos REAL,              -- 過去平均終盤位置 (3-4角平均)
    h_pos_change_tendency REAL,       -- 位置変化傾向 (late - early, 負=追込)

    -- 5.2 Last 3F (上がり) stats
    h_avg_last3f REAL,                -- 過去平均上がり3F
    h_best_last3f REAL,               -- 過去最速上がり3F
    h_recent3_last3f_rank REAL,       -- 直近3走の上がり順位平均

    -- 5.3 Pace profile from horse_laps (if available)
    h_hlap_early_ratio REAL,          -- 序盤ラップ比率 (vs レース平均)
    h_hlap_mid_ratio REAL,            -- 中盤ラップ比率
    h_hlap_late_ratio REAL,           -- 終盤ラップ比率
    h_hlap_finish_ratio REAL,         -- 上がりラップ比率

    -- 5.4 Running style classification
    h_running_style_id INTEGER,       -- 脚質ID (0:逃, 1:先, 2:差, 3:追)
    h_running_style_entropy REAL,     -- 脚質エントロピー (多様性)

    -- =========================================================================
    -- 6. class_prize: クラス・賞金 (~15 columns)
    -- =========================================================================
    -- 6.1 Prize money aggregations
    h_total_prize REAL,               -- 累計賞金 (as-of)
    h_avg_prize REAL,                 -- 平均賞金
    h_max_prize REAL,                 -- 最高賞金
    h_recent3_total_prize REAL,       -- 直近3走合計賞金
    h_prize_momentum REAL,            -- 賞金モメンタム (recent3 / avg)

    -- 6.2 Class progression
    h_highest_class_id INTEGER,       -- 過去最高クラスID
    h_class_progression REAL,         -- クラス上昇度 (current - avg)
    h_n_class_wins INTEGER,           -- 現クラスでの勝利数
    h_n_class_in3 INTEGER,            -- 現クラスでの複勝数
    h_class_win_rate REAL,            -- 現クラス勝率

    -- 6.3 Grade race experience
    h_n_g1_runs INTEGER,              -- G1出走数
    h_n_g2_runs INTEGER,              -- G2出走数
    h_n_g3_runs INTEGER,              -- G3出走数
    h_n_op_runs INTEGER,              -- OP/重賞出走数
    h_graded_in3_rate REAL,           -- 重賞複勝率

    -- =========================================================================
    -- 7. jockey_trainer: 騎手・調教師の as-of 成績 (~40 columns)
    -- =========================================================================
    -- 7.1 Jockey global stats
    j_n_starts INTEGER,               -- 騎手出走数 (as-of)
    j_n_wins INTEGER,                 -- 騎手勝利数
    j_win_rate REAL,                  -- 騎手勝率
    j_in3_rate REAL,                  -- 騎手複勝率

    -- 7.2 Jockey distance category stats
    j_n_starts_dist INTEGER,          -- 騎手同距離カテゴリ出走数
    j_win_rate_dist REAL,             -- 騎手同距離カテゴリ勝率
    j_in3_rate_dist REAL,             -- 騎手同距離カテゴリ複勝率

    -- 7.3 Jockey surface stats
    j_win_rate_surface REAL,          -- 騎手同馬場勝率
    j_in3_rate_surface REAL,          -- 騎手同馬場複勝率

    -- 7.4 Jockey course stats
    j_win_rate_course REAL,           -- 騎手同コース勝率
    j_in3_rate_course REAL,           -- 騎手同コース複勝率

    -- 7.5 Jockey recent form
    j_recent30d_win_rate REAL,        -- 騎手直近30日勝率
    j_recent30d_in3_rate REAL,        -- 騎手直近30日複勝率
    j_recent30d_n_rides INTEGER,      -- 騎手直近30日騎乗数

    -- 7.6 Jockey-horse combo
    jh_n_combos INTEGER,              -- 騎手×馬コンビ回数
    jh_combo_win_rate REAL,           -- 騎手×馬コンビ勝率
    jh_combo_in3_rate REAL,           -- 騎手×馬コンビ複勝率

    -- 7.7 Trainer global stats
    t_n_starts INTEGER,               -- 調教師出走数 (as-of)
    t_n_wins INTEGER,                 -- 調教師勝利数
    t_win_rate REAL,                  -- 調教師勝率
    t_in3_rate REAL,                  -- 調教師複勝率

    -- 7.8 Trainer distance category stats
    t_n_starts_dist INTEGER,          -- 調教師同距離カテゴリ出走数
    t_win_rate_dist REAL,             -- 調教師同距離カテゴリ勝率
    t_in3_rate_dist REAL,             -- 調教師同距離カテゴリ複勝率

    -- 7.9 Trainer surface stats
    t_win_rate_surface REAL,          -- 調教師同馬場勝率
    t_in3_rate_surface REAL,          -- 調教師同馬場複勝率

    -- 7.10 Trainer course stats
    t_win_rate_course REAL,           -- 調教師同コース勝率
    t_in3_rate_course REAL,           -- 調教師同コース複勝率

    -- 7.11 Trainer recent form
    t_recent30d_win_rate REAL,        -- 調教師直近30日勝率
    t_recent30d_in3_rate REAL,        -- 調教師直近30日複勝率
    t_recent30d_n_entries INTEGER,    -- 調教師直近30日出走数

    -- 7.12 Trainer-jockey combo
    tj_n_combos INTEGER,              -- 調教師×騎手コンビ回数
    tj_combo_win_rate REAL,           -- 調教師×騎手コンビ勝率
    tj_combo_in3_rate REAL,           -- 調教師×騎手コンビ複勝率

    -- =========================================================================
    -- 8. pedigree: 血統ハッシュ (640 columns total)
    -- =========================================================================
    -- 8.1 Sire-Dam Hash (512 dimensions)
    -- 父×母×母父 の組み合わせを Feature Hashing で 512 次元に圧縮
    -- 列名: ped_hash_000 ~ ped_hash_511
    -- (実際のカラムは動的に生成)

    -- 8.2 Ancestor Frequency Hash (128 dimensions)
    -- 5代血統表の全祖先の出現頻度を Feature Hashing で 128 次元に圧縮
    -- 列名: anc_hash_000 ~ anc_hash_127
    -- (実際のカラムは動的に生成)

    -- =========================================================================
    -- 9. market_features: 当日市場情報 (Optional, 学習時除外可能)
    -- =========================================================================
    -- 【警告】これらは当日情報のため、学習時は除外推奨
    -- 推論時のみ使用する場合は別途検討が必要
    market_win_odds REAL,             -- 単勝オッズ (当日)
    market_popularity INTEGER,        -- 人気順 (当日)
    market_odds_rank INTEGER,         -- オッズ順位 (field内)

    -- =========================================================================
    -- Metadata
    -- =========================================================================
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),

    PRIMARY KEY (race_id, horse_id)
);

-- =============================================================================
-- Indexes
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_ftv4_race_id ON feature_table_v4(race_id);
CREATE INDEX IF NOT EXISTS idx_ftv4_horse_id ON feature_table_v4(horse_id);
CREATE INDEX IF NOT EXISTS idx_ftv4_race_date ON feature_table_v4(race_date);
CREATE INDEX IF NOT EXISTS idx_ftv4_race_year ON feature_table_v4(race_year);
CREATE INDEX IF NOT EXISTS idx_ftv4_target_win ON feature_table_v4(target_win);
"""

# =============================================================================
# Pedigree Hash Column Generation
# =============================================================================

def generate_pedigree_hash_columns() -> str:
    """
    血統ハッシュ用のカラム定義を生成

    - ped_hash_000 ~ ped_hash_511 (512 次元)
    - anc_hash_000 ~ anc_hash_127 (128 次元)

    Returns:
        カラム定義の SQL 文字列
    """
    columns = []

    # Sire-Dam Hash (512 dimensions)
    for i in range(512):
        columns.append(f"    ped_hash_{i:03d} REAL DEFAULT 0.0")

    # Ancestor Frequency Hash (128 dimensions)
    for i in range(128):
        columns.append(f"    anc_hash_{i:03d} REAL DEFAULT 0.0")

    return ",\n".join(columns)


def get_pedigree_hash_column_names() -> List[str]:
    """
    血統ハッシュのカラム名リストを取得

    Returns:
        ["ped_hash_000", ..., "ped_hash_511", "anc_hash_000", ..., "anc_hash_127"]
    """
    columns = []
    for i in range(512):
        columns.append(f"ped_hash_{i:03d}")
    for i in range(128):
        columns.append(f"anc_hash_{i:03d}")
    return columns


# =============================================================================
# Full DDL with Pedigree Hash Columns
# =============================================================================

def get_full_feature_table_v4_ddl() -> str:
    """
    血統ハッシュカラムを含む完全な DDL を生成

    Returns:
        CREATE TABLE SQL
    """
    # ベース DDL を解析して血統ハッシュカラムを挿入
    base_ddl = CREATE_FEATURE_TABLE_V4

    # "market_win_odds" の直前に血統ハッシュカラムを挿入
    pedigree_columns = generate_pedigree_hash_columns()

    # 挿入位置を見つける
    insert_marker = "    -- =========================================================================\n    -- 9. market_features"

    pedigree_section = f"""    -- =========================================================================
    -- 8. pedigree: 血統ハッシュ (640 columns)
    -- =========================================================================
    -- Sire-Dam Hash (512 dimensions)
{pedigree_columns},

"""

    full_ddl = base_ddl.replace(insert_marker, pedigree_section + insert_marker)

    return full_ddl


# =============================================================================
# Table Creation Functions
# =============================================================================

def create_feature_table_v4(conn: sqlite3.Connection, include_pedigree_hash: bool = True) -> None:
    """
    feature_table_v4 を作成

    Args:
        conn: SQLite 接続
        include_pedigree_hash: 血統ハッシュカラムを含めるかどうか
                               (640カラムを追加するため、開発時は False 推奨)
    """
    if include_pedigree_hash:
        ddl = get_full_feature_table_v4_ddl()
    else:
        ddl = CREATE_FEATURE_TABLE_V4

    logger.info("Creating feature_table_v4...")
    conn.executescript(ddl)
    conn.commit()
    logger.info("feature_table_v4 created successfully")


def get_feature_v4_columns(
    include_pedigree: bool = True,
    include_market: bool = False,
) -> List[str]:
    """
    feature_table_v4 の特徴量カラム名リストを取得

    Args:
        include_pedigree: 血統ハッシュカラムを含めるか
        include_market: 市場情報カラムを含めるか (学習時は通常 False)

    Returns:
        特徴量カラム名のリスト

    Note:
        以下は除外される:
        - race_id, horse_id, race_date (identity)
        - target_* (targets)
        - created_at, updated_at (metadata)
    """
    # 基本特徴量カラム (手動定義)
    base_columns = [
        # base_race
        "place_id", "surface_id", "distance", "distance_cat",
        "track_condition_id", "course_turn_id", "course_inout_id",
        "race_year", "race_month", "race_day_of_week", "race_no",
        "grade_id", "race_class_id", "field_size",
        "waku", "umaban", "umaban_norm", "sex_id", "age", "weight_carried",

        # horse_form (global)
        "h_n_starts", "h_n_wins", "h_n_in3",
        "h_win_rate", "h_in3_rate", "h_avg_finish", "h_std_finish",
        "h_best_finish", "h_worst_finish",

        # horse_form (distance)
        "h_n_starts_dist", "h_win_rate_dist", "h_in3_rate_dist",
        "h_avg_finish_dist", "h_avg_last3f_dist",

        # horse_form (surface)
        "h_n_starts_surface", "h_win_rate_surface", "h_in3_rate_surface",

        # horse_form (track condition)
        "h_n_starts_track", "h_win_rate_track", "h_in3_rate_track",

        # horse_form (course)
        "h_n_starts_course", "h_win_rate_course", "h_in3_rate_course",

        # horse_form (recent)
        "h_days_since_last", "h_recent3_avg_finish", "h_recent3_best_finish",
        "h_recent3_avg_last3f", "h_recent5_avg_finish",
        "h_recent5_win_rate", "h_recent5_in3_rate",

        # horse_form (weight)
        "h_body_weight", "h_body_weight_diff", "h_avg_body_weight", "h_body_weight_dev",

        # horse_form (first run)
        "h_is_first_run",

        # pace_position
        "h_avg_corner1_pos", "h_avg_corner2_pos",
        "h_avg_corner3_pos", "h_avg_corner4_pos",
        "h_avg_early_pos", "h_avg_late_pos", "h_pos_change_tendency",
        "h_avg_last3f", "h_best_last3f", "h_recent3_last3f_rank",
        "h_hlap_early_ratio", "h_hlap_mid_ratio",
        "h_hlap_late_ratio", "h_hlap_finish_ratio",
        "h_running_style_id", "h_running_style_entropy",

        # class_prize
        "h_total_prize", "h_avg_prize", "h_max_prize",
        "h_recent3_total_prize", "h_prize_momentum",
        "h_highest_class_id", "h_class_progression",
        "h_n_class_wins", "h_n_class_in3", "h_class_win_rate",
        "h_n_g1_runs", "h_n_g2_runs", "h_n_g3_runs",
        "h_n_op_runs", "h_graded_in3_rate",

        # jockey
        "j_n_starts", "j_n_wins", "j_win_rate", "j_in3_rate",
        "j_n_starts_dist", "j_win_rate_dist", "j_in3_rate_dist",
        "j_win_rate_surface", "j_in3_rate_surface",
        "j_win_rate_course", "j_in3_rate_course",
        "j_recent30d_win_rate", "j_recent30d_in3_rate", "j_recent30d_n_rides",
        "jh_n_combos", "jh_combo_win_rate", "jh_combo_in3_rate",

        # trainer
        "t_n_starts", "t_n_wins", "t_win_rate", "t_in3_rate",
        "t_n_starts_dist", "t_win_rate_dist", "t_in3_rate_dist",
        "t_win_rate_surface", "t_in3_rate_surface",
        "t_win_rate_course", "t_in3_rate_course",
        "t_recent30d_win_rate", "t_recent30d_in3_rate", "t_recent30d_n_entries",
        "tj_n_combos", "tj_combo_win_rate", "tj_combo_in3_rate",
    ]

    columns = list(base_columns)

    # 血統ハッシュカラム
    if include_pedigree:
        columns.extend(get_pedigree_hash_column_names())

    # 市場情報カラム
    if include_market:
        columns.extend([
            "market_win_odds",
            "market_popularity",
            "market_odds_rank",
        ])

    return columns


# =============================================================================
# Utility Functions
# =============================================================================

def get_feature_groups() -> dict:
    """
    特徴量グループ別のカラム名辞書を取得

    Returns:
        {"group_name": ["col1", "col2", ...], ...}
    """
    return {
        "base_race": [
            "place_id", "surface_id", "distance", "distance_cat",
            "track_condition_id", "course_turn_id", "course_inout_id",
            "race_year", "race_month", "race_day_of_week", "race_no",
            "grade_id", "race_class_id", "field_size",
            "waku", "umaban", "umaban_norm", "sex_id", "age", "weight_carried",
        ],
        "horse_form": [
            "h_n_starts", "h_n_wins", "h_n_in3",
            "h_win_rate", "h_in3_rate", "h_avg_finish", "h_std_finish",
            "h_best_finish", "h_worst_finish",
            "h_n_starts_dist", "h_win_rate_dist", "h_in3_rate_dist",
            "h_avg_finish_dist", "h_avg_last3f_dist",
            "h_n_starts_surface", "h_win_rate_surface", "h_in3_rate_surface",
            "h_n_starts_track", "h_win_rate_track", "h_in3_rate_track",
            "h_n_starts_course", "h_win_rate_course", "h_in3_rate_course",
            "h_days_since_last", "h_recent3_avg_finish", "h_recent3_best_finish",
            "h_recent3_avg_last3f", "h_recent5_avg_finish",
            "h_recent5_win_rate", "h_recent5_in3_rate",
            "h_body_weight", "h_body_weight_diff", "h_avg_body_weight", "h_body_weight_dev",
            "h_is_first_run",
        ],
        "pace_position": [
            "h_avg_corner1_pos", "h_avg_corner2_pos",
            "h_avg_corner3_pos", "h_avg_corner4_pos",
            "h_avg_early_pos", "h_avg_late_pos", "h_pos_change_tendency",
            "h_avg_last3f", "h_best_last3f", "h_recent3_last3f_rank",
            "h_hlap_early_ratio", "h_hlap_mid_ratio",
            "h_hlap_late_ratio", "h_hlap_finish_ratio",
            "h_running_style_id", "h_running_style_entropy",
        ],
        "class_prize": [
            "h_total_prize", "h_avg_prize", "h_max_prize",
            "h_recent3_total_prize", "h_prize_momentum",
            "h_highest_class_id", "h_class_progression",
            "h_n_class_wins", "h_n_class_in3", "h_class_win_rate",
            "h_n_g1_runs", "h_n_g2_runs", "h_n_g3_runs",
            "h_n_op_runs", "h_graded_in3_rate",
        ],
        "jockey_trainer": [
            "j_n_starts", "j_n_wins", "j_win_rate", "j_in3_rate",
            "j_n_starts_dist", "j_win_rate_dist", "j_in3_rate_dist",
            "j_win_rate_surface", "j_in3_rate_surface",
            "j_win_rate_course", "j_in3_rate_course",
            "j_recent30d_win_rate", "j_recent30d_in3_rate", "j_recent30d_n_rides",
            "jh_n_combos", "jh_combo_win_rate", "jh_combo_in3_rate",
            "t_n_starts", "t_n_wins", "t_win_rate", "t_in3_rate",
            "t_n_starts_dist", "t_win_rate_dist", "t_in3_rate_dist",
            "t_win_rate_surface", "t_in3_rate_surface",
            "t_win_rate_course", "t_in3_rate_course",
            "t_recent30d_win_rate", "t_recent30d_in3_rate", "t_recent30d_n_entries",
            "tj_n_combos", "tj_combo_win_rate", "tj_combo_in3_rate",
        ],
        "pedigree": get_pedigree_hash_column_names(),
        "market": [
            "market_win_odds",
            "market_popularity",
            "market_odds_rank",
        ],
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    # カラム数を表示
    groups = get_feature_groups()
    total = 0
    print("=" * 60)
    print("feature_table_v4 Column Summary")
    print("=" * 60)
    for group, cols in groups.items():
        print(f"  {group}: {len(cols)} columns")
        total += len(cols)
    print("-" * 60)
    print(f"  TOTAL: {total} columns")
    print("=" * 60)

    # 全カラム名を表示 (pedigree 除く)
    print("\nFeature columns (excluding pedigree):")
    for col in get_feature_v4_columns(include_pedigree=False, include_market=True):
        print(f"  - {col}")
