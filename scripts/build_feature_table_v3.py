#!/usr/bin/env python3
"""
build_feature_table_v3.py

9軸特徴量を feature_table_v2 に追加して feature_table_v3 を構築する。

軸①: 近況・成長（直近の着順・上がり3Fトレンド）
軸②: 距離・コース条件適性
軸③: 地力・レースレベル（クラス・賞金）
軸④: ローテーション・臨戦過程
軸⑤: 枠・ゲート・並び（レース内相対）
軸⑥: 脚質・位置取り × 展開（corners由来）
軸⑦: 安定性・ブレ（メンタル/一貫性の proxy）
軸⑧: 騎手・厩舎（人の力の proxy）
軸⑨: 妙味/人気・オッズ（当日最終は使わず proxy）

重要な原則:
- すべての特徴量は「そのレースより前のレース」のみから計算（shift(1)徹底）
- per-row ループ禁止（groupby + shift/cum系で高速処理）
- 禁止列（target_*, finish_*, payout等）は絶対に入れない
- 当日直前情報（馬体重・最終オッズ）は入れない

Usage:
    python scripts/build_feature_table_v3.py --db netkeiba.db
    python scripts/build_feature_table_v3.py --db netkeiba.db --limit 10000
"""

import argparse
import logging
import sqlite3
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ============================================================
# Data Loading
# ============================================================


def load_feature_table_v2(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load feature_table_v2 as base."""
    logger.info("Loading feature_table_v2...")
    df = pd.read_sql_query("SELECT * FROM feature_table_v2", conn)
    logger.info(f"feature_table_v2 rows: {len(df):,}")
    return df


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get list of column names for a table using PRAGMA."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return columns


def load_horse_results(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load horse_results with dynamic column detection.

    - Uses PRAGMA to check which columns exist in horse_results and race_results
    - JOINs with race_results (uniquified subquery) to get jockey_id/trainer_id if not in horse_results
    - Falls back to NULL columns if data cannot be obtained
    """
    logger.info("Loading horse_results with dynamic column detection...")

    # Get columns from both tables
    hr_cols = set(get_table_columns(conn, "horse_results"))
    rr_cols = set(get_table_columns(conn, "race_results"))

    logger.info(f"  horse_results columns ({len(hr_cols)}): {sorted(hr_cols)}")
    logger.info(f"  race_results columns ({len(rr_cols)}): {sorted(rr_cols)}")

    # Required base columns from horse_results
    hr_base_cols = [
        "horse_id", "race_id", "race_date", "place", "course_type", "distance",
        "track_condition", "field_size", "race_class", "grade", "finish_order",
        "frame_no", "horse_no", "last_3f", "prize_money"
    ]

    # Optional columns that may be in horse_results
    hr_optional_cols = ["win_odds", "popularity"]

    # Build SELECT for horse_results base columns
    hr_select_parts = []
    for col in hr_base_cols:
        if col in hr_cols:
            hr_select_parts.append(f"hr.{col}")
        else:
            logger.warning(f"  Column '{col}' not found in horse_results, using NULL")
            hr_select_parts.append(f"NULL AS {col}")

    for col in hr_optional_cols:
        if col in hr_cols:
            hr_select_parts.append(f"hr.{col}")
        else:
            logger.info(f"  Optional column '{col}' not in horse_results, using NULL")
            hr_select_parts.append(f"NULL AS {col}")

    # Check if jockey_id/trainer_id are in horse_results directly
    jockey_in_hr = "jockey_id" in hr_cols
    trainer_in_hr = "trainer_id" in hr_cols
    jockey_in_rr = "jockey_id" in rr_cols
    trainer_in_rr = "trainer_id" in rr_cols

    logger.info(f"  jockey_id in horse_results: {jockey_in_hr}")
    logger.info(f"  trainer_id in horse_results: {trainer_in_hr}")
    logger.info(f"  jockey_id in race_results: {jockey_in_rr}")
    logger.info(f"  trainer_id in race_results: {trainer_in_rr}")

    if jockey_in_hr:
        hr_select_parts.append("hr.jockey_id")
    if trainer_in_hr:
        hr_select_parts.append("hr.trainer_id")

    # Determine if we need to JOIN with race_results
    need_jockey_from_rr = not jockey_in_hr and jockey_in_rr
    need_trainer_from_rr = not trainer_in_hr and trainer_in_rr
    need_join = need_jockey_from_rr or need_trainer_from_rr

    logger.info(f"  need_join: {need_join} (jockey_from_rr={need_jockey_from_rr}, trainer_from_rr={need_trainer_from_rr})")

    # Determine JOIN key (only safe composite keys)
    join_keys = None
    join_clause = ""
    rr_subquery = ""

    if need_join:
        # Priority: (race_id, horse_id) > (race_id, horse_no)
        # Note: race_id alone is forbidden (many-to-many risk)
        if "race_id" in rr_cols and "horse_id" in rr_cols and "horse_id" in hr_cols:
            join_keys = ["race_id", "horse_id"]
        elif "race_id" in rr_cols and "horse_no" in rr_cols and "horse_no" in hr_cols:
            join_keys = ["race_id", "horse_no"]
        else:
            logger.warning("  FALLBACK: Cannot determine safe JOIN key for race_results")
            logger.warning("    Available in rr: race_id={}, horse_id={}, horse_no={}".format(
                "race_id" in rr_cols, "horse_id" in rr_cols, "horse_no" in rr_cols
            ))
            need_join = False

        if join_keys:
            logger.info(f"  join_keys: {join_keys}")

            # Build uniquifying subquery with GROUP BY
            rr_select_cols = join_keys.copy()
            if need_jockey_from_rr:
                rr_select_cols.append("MAX(jockey_id) AS jockey_id")
            if need_trainer_from_rr:
                rr_select_cols.append("MAX(trainer_id) AS trainer_id")

            rr_select_str = ", ".join(rr_select_cols[:len(join_keys)])  # join keys without MAX
            rr_agg_str = ", ".join(rr_select_cols[len(join_keys):])  # aggregated columns
            if rr_agg_str:
                rr_select_str = rr_select_str + ", " + rr_agg_str

            group_by_str = ", ".join(join_keys)
            rr_subquery = f"(SELECT {rr_select_str} FROM race_results GROUP BY {group_by_str}) rr_uniq"

            join_conditions = " AND ".join([f"hr.{k} = rr_uniq.{k}" for k in join_keys])
            join_clause = f"LEFT JOIN {rr_subquery} ON {join_conditions}"

    # Add jockey_id/trainer_id from race_results subquery or NULL
    if need_jockey_from_rr and need_join:
        hr_select_parts.append("rr_uniq.jockey_id")
        logger.info("  jockey_id will be loaded from race_results via JOIN (uniquified)")
    elif not jockey_in_hr:
        hr_select_parts.append("NULL AS jockey_id")
        logger.warning("  FALLBACK: jockey_id not available anywhere, using NULL")

    if need_trainer_from_rr and need_join:
        hr_select_parts.append("rr_uniq.trainer_id")
        logger.info("  trainer_id will be loaded from race_results via JOIN (uniquified)")
    elif not trainer_in_hr:
        hr_select_parts.append("NULL AS trainer_id")
        logger.warning("  FALLBACK: trainer_id not available anywhere, using NULL")

    # Build final query
    select_clause = ",\n        ".join(hr_select_parts)

    if need_join and join_clause:
        query = f"""
        SELECT
            {select_clause}
        FROM horse_results hr
        {join_clause}
        WHERE hr.race_date IS NOT NULL
        ORDER BY hr.horse_id, hr.race_date, hr.race_id
        """
    else:
        query = f"""
        SELECT
            {select_clause}
        FROM horse_results hr
        WHERE hr.race_date IS NOT NULL
        ORDER BY hr.horse_id, hr.race_date, hr.race_id
        """

    logger.info("  Executing query...")
    df = pd.read_sql_query(query, conn)
    logger.info(f"  horse_results rows: {len(df):,}")

    # Report NULL rates for key columns
    for col in ["jockey_id", "trainer_id", "win_odds", "popularity"]:
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = 100 * null_count / len(df) if len(df) > 0 else 0
            logger.info(f"  {col} null_rate: {null_count:,}/{len(df):,} ({null_pct:.1f}%)")

    return df


def load_corners(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load corners table."""
    logger.info("Loading corners...")
    df = pd.read_sql_query("SELECT * FROM corners", conn)
    logger.info(f"corners rows: {len(df):,}")
    return df


def load_races_for_corners(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load races table for field_size (head_count)."""
    logger.info("Loading races for field_size...")
    df = pd.read_sql_query(
        "SELECT race_id, head_count as field_size FROM races",
        conn
    )
    logger.info(f"races rows: {len(df):,}")
    return df


def load_race_results_for_corners(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load race_results for horse_no mapping."""
    logger.info("Loading race_results for horse_no mapping...")
    df = pd.read_sql_query(
        "SELECT race_id, horse_id, horse_no FROM race_results",
        conn
    )
    logger.info(f"race_results rows: {len(df):,}")
    return df


# ============================================================
# Distance Category Helper
# ============================================================


def get_distance_cat(distance: int) -> str:
    """Classify distance into category."""
    if pd.isna(distance):
        return "unknown"
    if distance < 1400:
        return "short"
    elif distance < 1800:
        return "mile"
    elif distance < 2200:
        return "middle"
    else:
        return "long"


def get_surface_id(course_type: str) -> int:
    """Convert course_type to surface_id."""
    if pd.isna(course_type):
        return -1
    ct = str(course_type).lower()
    if "芝" in ct or "turf" in ct:
        return 0
    elif "ダ" in ct or "dirt" in ct:
        return 1
    elif "障" in ct:
        return 2
    return -1


# ============================================================
# Axis 2: Condition Suitability Features (ax2_)
# ============================================================


def compute_ax2_features(hr: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """
    Compute axis 2 features: condition suitability.

    All features use shift(1) to exclude current race.
    """
    logger.info("Computing ax2_ features (condition suitability)...")

    df = hr.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    # Add derived columns
    df["surface_id"] = df["course_type"].apply(get_surface_id)
    df["distance_cat"] = df["distance"].apply(get_distance_cat)
    df["is_in3"] = df["finish_order"].between(1, 3).astype(int)

    g = df.groupby("horse_id", group_keys=False)

    # ========================================
    # 1. Same Surface Stats (過去のサーフェス別成績)
    # ========================================

    def cum_stats_by_condition(group: pd.DataFrame, cond_col: str, prefix: str) -> pd.DataFrame:
        """Compute cumulative stats for a specific condition."""
        result = pd.DataFrame(index=group.index)

        # Get unique conditions in this group
        for cond_val in group[cond_col].dropna().unique():
            mask = group[cond_col] == cond_val
            group_cond = group[mask].copy()

            if len(group_cond) == 0:
                continue

            # Cumulative count (shift(1) - exclude current)
            starts = group_cond.groupby("horse_id").cumcount()
            wins_cum = group_cond["is_in3"].shift(1).fillna(0).cumsum()
            finish_cum = group_cond["finish_order"].shift(1).fillna(0).cumsum()
            last3f_cum = group_cond["last_3f"].shift(1).fillna(0).cumsum()
            last3f_count = group_cond["last_3f"].shift(1).notna().cumsum()

            # Store in result at original indices
            result.loc[group_cond.index, f"{prefix}_starts_temp"] = starts
            result.loc[group_cond.index, f"{prefix}_in3_cum_temp"] = wins_cum
            result.loc[group_cond.index, f"{prefix}_finish_cum_temp"] = finish_cum
            result.loc[group_cond.index, f"{prefix}_last3f_cum_temp"] = last3f_cum
            result.loc[group_cond.index, f"{prefix}_last3f_count_temp"] = last3f_count

        return result

    # Same surface
    logger.info("  - Computing same surface stats...")
    df["_same_surface_starts"] = 0
    df["_same_surface_in3_cum"] = 0.0
    df["_same_surface_finish_cum"] = 0.0
    df["_same_surface_last3f_cum"] = 0.0
    df["_same_surface_last3f_count"] = 0

    for horse_id, group in df.groupby("horse_id"):
        for surface_id in group["surface_id"].dropna().unique():
            mask = (df["horse_id"] == horse_id) & (df["surface_id"] == surface_id)
            sub = df.loc[mask].copy()
            if len(sub) == 0:
                continue

            starts = sub.groupby("horse_id").cumcount()
            in3_cum = sub["is_in3"].shift(1).fillna(0).cumsum()
            finish_cum = sub["finish_order"].shift(1).fillna(0).cumsum()
            last3f_cum = sub["last_3f"].shift(1).fillna(0).cumsum()
            last3f_count = sub["last_3f"].shift(1).notna().cumsum()

            df.loc[mask, "_same_surface_starts"] = starts.values
            df.loc[mask, "_same_surface_in3_cum"] = in3_cum.values
            df.loc[mask, "_same_surface_finish_cum"] = finish_cum.values
            df.loc[mask, "_same_surface_last3f_cum"] = last3f_cum.values
            df.loc[mask, "_same_surface_last3f_count"] = last3f_count.values

    # Compute rates
    df["ax2_same_surface_starts"] = df["_same_surface_starts"]
    df["ax2_same_surface_in3_rate"] = np.where(
        df["_same_surface_starts"] > 0,
        df["_same_surface_in3_cum"] / df["_same_surface_starts"],
        np.nan
    )
    df["ax2_same_surface_avg_finish"] = np.where(
        df["_same_surface_starts"] > 0,
        df["_same_surface_finish_cum"] / df["_same_surface_starts"],
        np.nan
    )
    df["ax2_same_surface_avg_last3f"] = np.where(
        df["_same_surface_last3f_count"] > 0,
        df["_same_surface_last3f_cum"] / df["_same_surface_last3f_count"],
        np.nan
    )

    # Same course (place)
    logger.info("  - Computing same course stats...")
    df["_same_course_starts"] = 0
    df["_same_course_in3_cum"] = 0.0
    df["_same_course_finish_cum"] = 0.0

    for horse_id, group in df.groupby("horse_id"):
        for place in group["place"].dropna().unique():
            mask = (df["horse_id"] == horse_id) & (df["place"] == place)
            sub = df.loc[mask].copy()
            if len(sub) == 0:
                continue

            starts = sub.groupby("horse_id").cumcount()
            in3_cum = sub["is_in3"].shift(1).fillna(0).cumsum()
            finish_cum = sub["finish_order"].shift(1).fillna(0).cumsum()

            df.loc[mask, "_same_course_starts"] = starts.values
            df.loc[mask, "_same_course_in3_cum"] = in3_cum.values
            df.loc[mask, "_same_course_finish_cum"] = finish_cum.values

    df["ax2_same_course_starts"] = df["_same_course_starts"]
    df["ax2_same_course_in3_rate"] = np.where(
        df["_same_course_starts"] > 0,
        df["_same_course_in3_cum"] / df["_same_course_starts"],
        np.nan
    )
    df["ax2_same_course_avg_finish"] = np.where(
        df["_same_course_starts"] > 0,
        df["_same_course_finish_cum"] / df["_same_course_starts"],
        np.nan
    )

    # Same distance category
    logger.info("  - Computing same distance category stats...")
    df["_same_distcat_starts"] = 0
    df["_same_distcat_in3_cum"] = 0.0
    df["_same_distcat_finish_cum"] = 0.0

    for horse_id, group in df.groupby("horse_id"):
        for distcat in group["distance_cat"].dropna().unique():
            mask = (df["horse_id"] == horse_id) & (df["distance_cat"] == distcat)
            sub = df.loc[mask].copy()
            if len(sub) == 0:
                continue

            starts = sub.groupby("horse_id").cumcount()
            in3_cum = sub["is_in3"].shift(1).fillna(0).cumsum()
            finish_cum = sub["finish_order"].shift(1).fillna(0).cumsum()

            df.loc[mask, "_same_distcat_starts"] = starts.values
            df.loc[mask, "_same_distcat_in3_cum"] = in3_cum.values
            df.loc[mask, "_same_distcat_finish_cum"] = finish_cum.values

    df["ax2_same_distcat_starts"] = df["_same_distcat_starts"]
    df["ax2_same_distcat_in3_rate"] = np.where(
        df["_same_distcat_starts"] > 0,
        df["_same_distcat_in3_cum"] / df["_same_distcat_starts"],
        np.nan
    )
    df["ax2_same_distcat_avg_finish"] = np.where(
        df["_same_distcat_starts"] > 0,
        df["_same_distcat_finish_cum"] / df["_same_distcat_starts"],
        np.nan
    )

    # ========================================
    # 2. Distance/Surface/Course Change from Previous Race
    # ========================================

    logger.info("  - Computing condition changes from previous race...")

    df["_prev_distance"] = g["distance"].shift(1)
    df["_prev_surface_id"] = g["surface_id"].shift(1)
    df["_prev_place"] = g["place"].shift(1)

    df["ax2_dist_change_prev"] = df["distance"] - df["_prev_distance"]
    df["ax2_is_dist_up"] = (df["ax2_dist_change_prev"] > 0).astype(int)
    df["ax2_is_dist_down"] = (df["ax2_dist_change_prev"] < 0).astype(int)
    df["ax2_is_surface_change"] = (df["surface_id"] != df["_prev_surface_id"]).astype(int)
    df["ax2_is_course_change"] = (df["place"] != df["_prev_place"]).astype(int)

    # Handle first race (no previous)
    first_race_mask = df["_prev_distance"].isna()
    df.loc[first_race_mask, "ax2_dist_change_prev"] = 0
    df.loc[first_race_mask, "ax2_is_dist_up"] = 0
    df.loc[first_race_mask, "ax2_is_dist_down"] = 0
    df.loc[first_race_mask, "ax2_is_surface_change"] = 0
    df.loc[first_race_mask, "ax2_is_course_change"] = 0

    # ========================================
    # Output columns
    # ========================================

    ax2_cols = [
        "race_id", "horse_id",
        "ax2_same_surface_starts", "ax2_same_surface_in3_rate",
        "ax2_same_surface_avg_finish", "ax2_same_surface_avg_last3f",
        "ax2_same_course_starts", "ax2_same_course_in3_rate",
        "ax2_same_course_avg_finish",
        "ax2_same_distcat_starts", "ax2_same_distcat_in3_rate",
        "ax2_same_distcat_avg_finish",
        "ax2_dist_change_prev", "ax2_is_dist_up", "ax2_is_dist_down",
        "ax2_is_surface_change", "ax2_is_course_change",
    ]

    result = df[ax2_cols].copy()
    logger.info(f"  - ax2_ features computed: {len([c for c in ax2_cols if c.startswith('ax2_')])} columns")

    return result


# ============================================================
# Axis 1: Recent Form / Growth Features (ax1_)
# ============================================================


def compute_ax1_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute axis 1 features: recent form and growth.

    All features use shift(1) to exclude current race.
    """
    logger.info("Computing ax1_ features (recent form/growth)...")

    df = hr.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    g = df.groupby("horse_id", group_keys=False)

    # ========================================
    # 1. Last race stats (前走成績)
    # ========================================

    df["ax1_last_finish"] = g["finish_order"].shift(1)
    df["ax1_last_last3f"] = g["last_3f"].shift(1)

    # ========================================
    # 2. Recent form trends (直近のトレンド)
    # ========================================

    def rolling_mean_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling mean of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=1).mean()

    def rolling_min_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling min of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=1).min()

    def compute_slope(s: pd.Series, window: int) -> pd.Series:
        """
        Compute slope (trend) over recent N races.
        Negative slope = improving (finish order decreasing)
        Positive slope = declining
        """
        shifted = s.shift(1)

        def get_slope(x):
            if len(x.dropna()) < 2:
                return 0.0
            y = x.dropna().values
            n = len(y)
            if n < 2:
                return 0.0
            # Linear regression slope: (y_last - y_first) / (n - 1)
            return (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0

        return shifted.rolling(window=window, min_periods=2).apply(get_slope, raw=False)

    # Recent 3 stats
    df["ax1_recent3_avg_finish"] = g["finish_order"].apply(lambda x: rolling_mean_prev(x, 3))
    df["ax1_recent3_avg_last3f"] = g["last_3f"].apply(lambda x: rolling_mean_prev(x, 3))

    # Slope/trend features
    df["ax1_recent3_finish_slope"] = g["finish_order"].apply(lambda x: compute_slope(x, 3))
    df["ax1_recent3_last3f_slope"] = g["last_3f"].apply(lambda x: compute_slope(x, 3))

    # Best finish in recent 5
    df["ax1_best_finish_recent5"] = g["finish_order"].apply(lambda x: rolling_min_prev(x, 5))

    # ========================================
    # 3. Improvement indicators
    # ========================================

    # Compare last race to recent average
    df["ax1_last_vs_avg3_finish"] = df["ax1_last_finish"] - df["ax1_recent3_avg_finish"]
    df["ax1_last_vs_avg3_last3f"] = df["ax1_last_last3f"] - df["ax1_recent3_avg_last3f"]

    # ========================================
    # Handle first race (no history)
    # ========================================

    first_mask = df["ax1_last_finish"].isna()
    df.loc[first_mask, "ax1_last_finish"] = 0
    df.loc[first_mask, "ax1_last_last3f"] = 0
    df.loc[first_mask, "ax1_recent3_finish_slope"] = 0
    df.loc[first_mask, "ax1_recent3_last3f_slope"] = 0
    df.loc[first_mask, "ax1_last_vs_avg3_finish"] = 0
    df.loc[first_mask, "ax1_last_vs_avg3_last3f"] = 0

    # ========================================
    # Output columns
    # ========================================

    ax1_cols = [
        "race_id", "horse_id",
        "ax1_last_finish", "ax1_last_last3f",
        "ax1_recent3_avg_finish", "ax1_recent3_avg_last3f",
        "ax1_recent3_finish_slope", "ax1_recent3_last3f_slope",
        "ax1_best_finish_recent5",
        "ax1_last_vs_avg3_finish", "ax1_last_vs_avg3_last3f",
    ]

    result = df[ax1_cols].copy()
    logger.info(f"  - ax1_ features computed: {len([c for c in ax1_cols if c.startswith('ax1_')])} columns")

    return result


# ============================================================
# Axis 3: Class Level / Prize Features (ax3_)
# ============================================================


def compute_ax3_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute axis 3 features: class level and prize money.

    All features use shift(1) to exclude current race.
    """
    logger.info("Computing ax3_ features (class level/prize)...")

    df = hr.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    # Compute class rank for each race
    df["_class_rank"] = df.apply(
        lambda row: get_class_rank(row.get("race_class"), row.get("grade")),
        axis=1
    )

    g = df.groupby("horse_id", group_keys=False)

    # ========================================
    # 1. Class level stats (直近のクラスレベル)
    # ========================================

    def rolling_max_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling max of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=1).max()

    def rolling_mean_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling mean of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=1).mean()

    def rolling_sum_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling sum of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=1).sum()

    # Best class in recent 5 races
    df["ax3_best_class_recent5"] = g["_class_rank"].apply(lambda x: rolling_max_prev(x, 5))

    # Average class in recent 5 races
    df["ax3_avg_class_recent5"] = g["_class_rank"].apply(lambda x: rolling_mean_prev(x, 5))

    # ========================================
    # 2. Prize money stats (賞金関連)
    # ========================================

    # Fill NaN prize_money with 0
    df["prize_money"] = df["prize_money"].fillna(0)

    # Total prize in recent 5 races
    df["ax3_total_prize_recent5"] = g["prize_money"].apply(lambda x: rolling_sum_prev(x, 5))

    # Average prize in recent 3 races
    df["ax3_avg_prize_recent3"] = g["prize_money"].apply(lambda x: rolling_mean_prev(x, 3))

    # Prize trend (recent vs older)
    def compute_prize_trend(s: pd.Series) -> pd.Series:
        """
        Compute prize trend: avg of last 2 races - avg of races 3-5.
        Positive = improving prize earnings
        """
        shifted = s.shift(1)

        def get_trend(x):
            vals = x.dropna().values
            if len(vals) < 3:
                return 0.0
            recent = vals[-2:].mean() if len(vals) >= 2 else vals[-1]
            older = vals[:-2].mean() if len(vals) > 2 else vals[0]
            return recent - older

        return shifted.rolling(window=5, min_periods=3).apply(get_trend, raw=False)

    df["ax3_prize_trend_recent5"] = g["prize_money"].apply(compute_prize_trend)

    # ========================================
    # 3. Class-based performance
    # ========================================

    # Current class vs best class achieved
    df["ax3_curr_vs_best_class"] = df["_class_rank"] - df["ax3_best_class_recent5"].fillna(0)

    # ========================================
    # Handle first race (no history)
    # ========================================

    first_mask = g["race_id"].cumcount() == 0
    df.loc[first_mask, "ax3_best_class_recent5"] = 0
    df.loc[first_mask, "ax3_avg_class_recent5"] = 0
    df.loc[first_mask, "ax3_total_prize_recent5"] = 0
    df.loc[first_mask, "ax3_avg_prize_recent3"] = 0
    df.loc[first_mask, "ax3_prize_trend_recent5"] = 0
    df.loc[first_mask, "ax3_curr_vs_best_class"] = 0

    # ========================================
    # Output columns
    # ========================================

    ax3_cols = [
        "race_id", "horse_id",
        "ax3_best_class_recent5", "ax3_avg_class_recent5",
        "ax3_total_prize_recent5", "ax3_avg_prize_recent3",
        "ax3_prize_trend_recent5", "ax3_curr_vs_best_class",
    ]

    result = df[ax3_cols].copy()
    logger.info(f"  - ax3_ features computed: {len([c for c in ax3_cols if c.startswith('ax3_')])} columns")

    return result


# ============================================================
# Axis 4: Rotation/Schedule Features (ax4_)
# ============================================================


# Class ranking for comparison (higher = stronger class)
CLASS_RANK = {
    "新馬": 1, "未勝利": 2, "1勝": 3, "1勝クラス": 3,
    "2勝": 4, "2勝クラス": 4, "3勝": 5, "3勝クラス": 5,
    "オープン": 6, "OP": 6, "L": 7, "Listed": 7,
    "G3": 8, "G2": 9, "G1": 10,
}


def get_class_rank(race_class: str, grade: str = None) -> int:
    """Get numeric class rank."""
    if pd.isna(race_class):
        return 0

    # Check grade first
    if grade and not pd.isna(grade):
        grade_str = str(grade).upper()
        if "G1" in grade_str:
            return 10
        elif "G2" in grade_str:
            return 9
        elif "G3" in grade_str:
            return 8
        elif grade_str in ["L", "LISTED"]:
            return 7

    # Check race_class
    rc = str(race_class)
    for key, rank in CLASS_RANK.items():
        if key in rc:
            return rank
    return 0


def compute_ax4_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute axis 4 features: rotation and schedule.
    """
    logger.info("Computing ax4_ features (rotation/schedule)...")

    df = hr.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    g = df.groupby("horse_id", group_keys=False)

    # ========================================
    # 1. Rest Days (days since previous race)
    # ========================================

    df["ax4_rest_days"] = g["race_date_dt"].diff().dt.days
    df["ax4_rest_weeks"] = df["ax4_rest_days"] / 7.0
    df["ax4_is_short_turnaround"] = (df["ax4_rest_days"] <= 7).astype(int)
    df["ax4_is_long_break_12w"] = (df["ax4_rest_days"] >= 84).astype(int)

    # Handle first race
    first_mask = df["ax4_rest_days"].isna()
    df.loc[first_mask, "ax4_rest_days"] = 0
    df.loc[first_mask, "ax4_rest_weeks"] = 0
    df.loc[first_mask, "ax4_is_short_turnaround"] = 0
    df.loc[first_mask, "ax4_is_long_break_12w"] = 0

    # ========================================
    # 2. Class Change
    # ========================================

    df["_class_rank"] = df.apply(
        lambda row: get_class_rank(row.get("race_class"), row.get("grade")),
        axis=1
    )
    df["_prev_class_rank"] = g["_class_rank"].shift(1)

    df["ax4_class_diff"] = df["_class_rank"] - df["_prev_class_rank"].fillna(df["_class_rank"])
    df["ax4_is_class_up"] = (df["ax4_class_diff"] > 0).astype(int)
    df["ax4_is_class_down"] = (df["ax4_class_diff"] < 0).astype(int)

    # Handle first race
    df.loc[first_mask, "ax4_class_diff"] = 0
    df.loc[first_mask, "ax4_is_class_up"] = 0
    df.loc[first_mask, "ax4_is_class_down"] = 0

    # ========================================
    # Output columns
    # ========================================

    ax4_cols = [
        "race_id", "horse_id",
        "ax4_rest_days", "ax4_rest_weeks",
        "ax4_is_short_turnaround", "ax4_is_long_break_12w",
        "ax4_class_diff", "ax4_is_class_up", "ax4_is_class_down",
    ]

    result = df[ax4_cols].copy()
    logger.info(f"  - ax4_ features computed: {len([c for c in ax4_cols if c.startswith('ax4_')])} columns")

    return result


# ============================================================
# Axis 6: Running Style Features (ax6_)
# ============================================================


def parse_corner_positions(corners: pd.DataFrame, race_results: pd.DataFrame) -> pd.DataFrame:
    """
    Parse corner passing positions from corners table.

    corners table has: race_id, corner_1, corner_2, corner_3, corner_4
    where corner_X is a comma-separated list of horse_no in passing order.

    Returns DataFrame with: race_id, horse_id, c1_pos, c4_pos
    """
    logger.info("Parsing corner positions...")

    if corners.empty:
        logger.warning("corners table is empty!")
        return pd.DataFrame(columns=["race_id", "horse_id", "c1_pos", "c4_pos"])

    records = []

    for _, row in corners.iterrows():
        race_id = row["race_id"]

        # Get horse_no to horse_id mapping for this race
        race_rr = race_results[race_results["race_id"] == race_id]
        if race_rr.empty:
            continue

        horse_no_to_id = dict(zip(race_rr["horse_no"], race_rr["horse_id"]))

        # Parse corner_1 (first corner)
        c1_positions = {}
        if pd.notna(row.get("corner_1")) and row["corner_1"]:
            try:
                c1_str = str(row["corner_1"]).replace("(", ",").replace(")", ",")
                parts = [p.strip() for p in c1_str.split(",") if p.strip()]
                for pos_idx, part in enumerate(parts, 1):
                    # Handle multiple horses at same position (e.g., "1=2")
                    for horse_no_str in part.split("="):
                        try:
                            horse_no = int(horse_no_str)
                            if horse_no in horse_no_to_id:
                                c1_positions[horse_no_to_id[horse_no]] = pos_idx
                        except ValueError:
                            continue
            except Exception:
                pass

        # Parse corner_4 (last corner)
        c4_positions = {}
        if pd.notna(row.get("corner_4")) and row["corner_4"]:
            try:
                c4_str = str(row["corner_4"]).replace("(", ",").replace(")", ",")
                parts = [p.strip() for p in c4_str.split(",") if p.strip()]
                for pos_idx, part in enumerate(parts, 1):
                    for horse_no_str in part.split("="):
                        try:
                            horse_no = int(horse_no_str)
                            if horse_no in horse_no_to_id:
                                c4_positions[horse_no_to_id[horse_no]] = pos_idx
                        except ValueError:
                            continue
            except Exception:
                pass

        # Create records for all horses in this race
        all_horses = set(c1_positions.keys()) | set(c4_positions.keys())
        for horse_id in all_horses:
            records.append({
                "race_id": race_id,
                "horse_id": horse_id,
                "c1_pos": c1_positions.get(horse_id),
                "c4_pos": c4_positions.get(horse_id),
            })

    result = pd.DataFrame(records)
    logger.info(f"  - Parsed corner positions for {len(result):,} horse-race pairs")

    return result


def compute_ax6_features(
    hr: pd.DataFrame,
    corners: pd.DataFrame,
    race_results: pd.DataFrame,
    races: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute axis 6 features: running style and race pace.

    Returns:
        - Individual horse features (ax6_recent*...)
        - Race-level features (ax6_race_*)
    """
    logger.info("Computing ax6_ features (running style)...")

    # Parse corner positions
    corner_positions = parse_corner_positions(corners, race_results)

    if corner_positions.empty:
        logger.warning("No corner positions found. ax6_ features will be empty.")
        # Return empty DataFrames with correct columns
        horse_df = pd.DataFrame(columns=[
            "race_id", "horse_id",
            "ax6_recent3_c1_pos_pct_mean", "ax6_recent5_c1_pos_pct_mean",
            "ax6_recent3_c4_pos_pct_mean", "ax6_recent5_c4_pos_pct_mean",
            "ax6_recent3_pos_gain_mean", "ax6_recent5_pos_gain_mean",
        ])
        race_df = pd.DataFrame(columns=[
            "race_id",
            "ax6_race_front_runner_count", "ax6_race_front_runner_share",
            "ax6_race_leader_count",
        ])
        return horse_df, race_df

    # Merge with field_size from races
    corner_positions = corner_positions.merge(
        races[["race_id", "field_size"]],
        on="race_id",
        how="left"
    )

    # Merge with race_date from horse_results
    race_dates = hr[["race_id", "race_date"]].drop_duplicates()
    corner_positions = corner_positions.merge(
        race_dates,
        on="race_id",
        how="left"
    )

    # Calculate position percentages
    corner_positions["c1_pos_pct"] = corner_positions["c1_pos"] / corner_positions["field_size"]
    corner_positions["c4_pos_pct"] = corner_positions["c4_pos"] / corner_positions["field_size"]
    corner_positions["pos_gain"] = (
        corner_positions["c1_pos"] - corner_positions["c4_pos"]
    ) / corner_positions["field_size"]

    # Sort by horse and date
    corner_positions["race_date_dt"] = pd.to_datetime(corner_positions["race_date"])
    corner_positions = corner_positions.sort_values(
        ["horse_id", "race_date_dt", "race_id"]
    ).reset_index(drop=True)

    # ========================================
    # Individual horse running style features
    # ========================================

    logger.info("  - Computing individual running style features...")
    g = corner_positions.groupby("horse_id", group_keys=False)

    def rolling_mean_prev(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window=window, min_periods=1).mean()

    corner_positions["ax6_recent3_c1_pos_pct_mean"] = g["c1_pos_pct"].apply(
        lambda x: rolling_mean_prev(x, 3)
    )
    corner_positions["ax6_recent5_c1_pos_pct_mean"] = g["c1_pos_pct"].apply(
        lambda x: rolling_mean_prev(x, 5)
    )
    corner_positions["ax6_recent3_c4_pos_pct_mean"] = g["c4_pos_pct"].apply(
        lambda x: rolling_mean_prev(x, 3)
    )
    corner_positions["ax6_recent5_c4_pos_pct_mean"] = g["c4_pos_pct"].apply(
        lambda x: rolling_mean_prev(x, 5)
    )
    corner_positions["ax6_recent3_pos_gain_mean"] = g["pos_gain"].apply(
        lambda x: rolling_mean_prev(x, 3)
    )
    corner_positions["ax6_recent5_pos_gain_mean"] = g["pos_gain"].apply(
        lambda x: rolling_mean_prev(x, 5)
    )

    horse_cols = [
        "race_id", "horse_id",
        "ax6_recent3_c1_pos_pct_mean", "ax6_recent5_c1_pos_pct_mean",
        "ax6_recent3_c4_pos_pct_mean", "ax6_recent5_c4_pos_pct_mean",
        "ax6_recent3_pos_gain_mean", "ax6_recent5_pos_gain_mean",
    ]
    horse_df = corner_positions[horse_cols].copy()

    # ========================================
    # Race-level pace pressure features
    # ========================================

    logger.info("  - Computing race-level pace pressure features...")

    # For each race, count front runners based on their past running style
    # (using ax6_recent5_c1_pos_pct_mean <= 0.30)
    race_stats = []

    for race_id, race_df in corner_positions.groupby("race_id"):
        # Filter to horses with valid running style data
        valid = race_df[race_df["ax6_recent5_c1_pos_pct_mean"].notna()]
        n_valid = len(valid)

        if n_valid == 0:
            race_stats.append({
                "race_id": race_id,
                "ax6_race_front_runner_count": 0,
                "ax6_race_front_runner_share": 0.0,
                "ax6_race_leader_count": 0,
            })
            continue

        # Front runners: typically in top 30%
        front_runner_count = (valid["ax6_recent5_c1_pos_pct_mean"] <= 0.30).sum()
        front_runner_share = front_runner_count / n_valid

        # Leaders: typically in top 20%
        leader_count = (valid["ax6_recent5_c1_pos_pct_mean"] <= 0.20).sum()

        race_stats.append({
            "race_id": race_id,
            "ax6_race_front_runner_count": int(front_runner_count),
            "ax6_race_front_runner_share": float(front_runner_share),
            "ax6_race_leader_count": int(leader_count),
        })

    race_df = pd.DataFrame(race_stats)

    # Log missing rate
    total_horses = len(corner_positions)
    missing_c1 = corner_positions["c1_pos"].isna().sum()
    missing_c4 = corner_positions["c4_pos"].isna().sum()
    missing_style = corner_positions["ax6_recent5_c1_pos_pct_mean"].isna().sum()

    logger.info(f"  - Corner data coverage:")
    logger.info(f"    - c1_pos missing: {missing_c1:,}/{total_horses:,} ({100*missing_c1/total_horses:.1f}%)")
    logger.info(f"    - c4_pos missing: {missing_c4:,}/{total_horses:,} ({100*missing_c4/total_horses:.1f}%)")
    logger.info(f"    - running style (recent5) missing: {missing_style:,}/{total_horses:,} ({100*missing_style/total_horses:.1f}%)")

    logger.info(f"  - ax6_ horse features computed: {len([c for c in horse_cols if c.startswith('ax6_')])} columns")
    logger.info(f"  - ax6_ race features computed: 3 columns")

    return horse_df, race_df


# ============================================================
# Axis 5: Gate/Position Features (ax5_)
# ============================================================


def compute_ax5_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute axis 5 features: gate and position (race-level relative).

    These are computed from current race data (frame_no, horse_no, field_size)
    and do not require shift(1) since they are fixed at race entry time.
    """
    logger.info("Computing ax5_ features (gate/position)...")

    df = hr[["race_id", "horse_id", "frame_no", "horse_no", "field_size"]].copy()

    # Compute relative positions
    df["ax5_waku_pct"] = df["frame_no"] / df["field_size"].clip(lower=1)
    df["ax5_umaban_pct"] = df["horse_no"] / df["field_size"].clip(lower=1)

    # Inner/outer flags
    df["ax5_is_inner"] = (df["ax5_waku_pct"] <= 0.5).astype(int)
    df["ax5_is_outer"] = (df["ax5_waku_pct"] > 0.5).astype(int)

    # Gate distance to edge (normalized)
    # Uses field_size as approximation for max_waku
    max_waku = df["field_size"].clip(lower=1)
    waku = df["frame_no"].fillna(1)
    dist_to_edge = np.minimum(waku - 1, max_waku - waku)
    df["ax5_gate_distance_to_edge_norm"] = dist_to_edge / (max_waku - 1).clip(lower=1)

    # Handle missing values
    for col in ["ax5_waku_pct", "ax5_umaban_pct", "ax5_gate_distance_to_edge_norm"]:
        df[col] = df[col].fillna(0.5)
    df["ax5_is_inner"] = df["ax5_is_inner"].fillna(0).astype(int)
    df["ax5_is_outer"] = df["ax5_is_outer"].fillna(0).astype(int)

    ax5_cols = [
        "race_id", "horse_id",
        "ax5_waku_pct", "ax5_umaban_pct",
        "ax5_is_inner", "ax5_is_outer",
        "ax5_gate_distance_to_edge_norm",
    ]

    result = df[ax5_cols].copy()
    logger.info(f"  - ax5_ features computed: {len([c for c in ax5_cols if c.startswith('ax5_')])} columns")

    return result


# ============================================================
# Axis 7: Stability/Variance Features (ax7_)
# ============================================================


def compute_ax7_features(hr: pd.DataFrame, corner_positions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute axis 7 features: stability and variance (mental/consistency proxy).

    All features use shift(1) to exclude current race.
    """
    logger.info("Computing ax7_ features (stability/variance)...")

    df = hr.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    g = df.groupby("horse_id", group_keys=False)

    def rolling_std_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling std of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=2).std()

    # Finish order variance
    df["ax7_finish_std_recent5"] = g["finish_order"].apply(lambda x: rolling_std_prev(x, 5))

    # Last 3F variance
    df["ax7_last3f_std_recent5"] = g["last_3f"].apply(lambda x: rolling_std_prev(x, 5))

    # Popularity variance (if available)
    if "popularity" in df.columns:
        df["ax7_popularity_std_recent5"] = g["popularity"].apply(lambda x: rolling_std_prev(x, 5))
    else:
        df["ax7_popularity_std_recent5"] = np.nan

    # Position gain variance (if corner_positions available)
    if corner_positions is not None and not corner_positions.empty:
        # Merge corner positions
        cp = corner_positions[["race_id", "horse_id", "pos_gain"]].copy()
        df = df.merge(cp, on=["race_id", "horse_id"], how="left")
        g = df.groupby("horse_id", group_keys=False)
        df["ax7_pos_gain_std_recent5"] = g["pos_gain"].apply(lambda x: rolling_std_prev(x, 5))
    else:
        df["ax7_pos_gain_std_recent5"] = np.nan

    # Fill NaN with 0 (no variance for first races)
    for col in ["ax7_finish_std_recent5", "ax7_last3f_std_recent5",
                "ax7_popularity_std_recent5", "ax7_pos_gain_std_recent5"]:
        df[col] = df[col].fillna(0)

    ax7_cols = [
        "race_id", "horse_id",
        "ax7_finish_std_recent5", "ax7_last3f_std_recent5",
        "ax7_popularity_std_recent5", "ax7_pos_gain_std_recent5",
    ]

    result = df[ax7_cols].copy()
    logger.info(f"  - ax7_ features computed: {len([c for c in ax7_cols if c.startswith('ax7_')])} columns")

    return result


# ============================================================
# Axis 8: Jockey/Trainer Features (ax8_)
# ============================================================


def compute_ax8_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute axis 8 features: jockey and trainer performance (as-of).

    All features are cumulative stats up to (but not including) current race.
    Handles cases where jockey_id/trainer_id are all NULL gracefully.
    """
    logger.info("Computing ax8_ features (jockey/trainer)...")

    df = hr.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date_dt", "race_id"]).reset_index(drop=True)

    # Add is_in3 target for cumulative calculation
    df["is_in3"] = df["finish_order"].between(1, 3).astype(int)

    # Check data availability
    jockey_available = "jockey_id" in df.columns and df["jockey_id"].notna().any()
    trainer_available = "trainer_id" in df.columns and df["trainer_id"].notna().any()

    if not jockey_available:
        logger.warning("  jockey_id is all NULL or missing - ax8_jockey_* will be 0")
    if not trainer_available:
        logger.warning("  trainer_id is all NULL or missing - ax8_trainer_* will be 0")

    # ========================================
    # Jockey stats (as-of)
    # ========================================

    logger.info("  - Computing jockey stats...")
    jockey_stats = []

    if jockey_available:
        # Sort by date for cumulative calculation
        df_sorted = df.sort_values(["jockey_id", "race_date_dt", "race_id"]).reset_index(drop=True)

        for jockey_id, group in df_sorted.groupby("jockey_id"):
            if pd.isna(jockey_id):
                continue

            # Cumulative count (shift(1) - exclude current)
            starts = group.groupby("jockey_id").cumcount()
            in3_cum = group["is_in3"].shift(1).fillna(0).cumsum()

            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                n_starts = starts.iloc[idx]
                in3_rate = in3_cum.iloc[idx] / n_starts if n_starts > 0 else 0.0
                jockey_stats.append({
                    "race_id": row["race_id"],
                    "horse_id": row["horse_id"],
                    "ax8_jockey_in3_rate_total_asof": in3_rate,
                })

    if jockey_stats:
        jockey_df = pd.DataFrame(jockey_stats)
    else:
        jockey_df = pd.DataFrame(columns=["race_id", "horse_id", "ax8_jockey_in3_rate_total_asof"])

    # ========================================
    # Trainer stats (as-of)
    # ========================================

    logger.info("  - Computing trainer stats...")
    trainer_stats = []

    if trainer_available:
        df_sorted = df.sort_values(["trainer_id", "race_date_dt", "race_id"]).reset_index(drop=True)

        for trainer_id, group in df_sorted.groupby("trainer_id"):
            if pd.isna(trainer_id):
                continue

            starts = group.groupby("trainer_id").cumcount()
            in3_cum = group["is_in3"].shift(1).fillna(0).cumsum()

            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                n_starts = starts.iloc[idx]
                in3_rate = in3_cum.iloc[idx] / n_starts if n_starts > 0 else 0.0
                trainer_stats.append({
                    "race_id": row["race_id"],
                    "horse_id": row["horse_id"],
                    "ax8_trainer_in3_rate_total_asof": in3_rate,
                })

    if trainer_stats:
        trainer_df = pd.DataFrame(trainer_stats)
    else:
        trainer_df = pd.DataFrame(columns=["race_id", "horse_id", "ax8_trainer_in3_rate_total_asof"])

    # Merge jockey and trainer stats
    result = df[["race_id", "horse_id"]].copy()
    result = result.merge(jockey_df, on=["race_id", "horse_id"], how="left")
    result = result.merge(trainer_df, on=["race_id", "horse_id"], how="left")

    # Fill NaN with 0
    result["ax8_jockey_in3_rate_total_asof"] = result["ax8_jockey_in3_rate_total_asof"].fillna(0)
    result["ax8_trainer_in3_rate_total_asof"] = result["ax8_trainer_in3_rate_total_asof"].fillna(0)

    ax8_cols = [
        "race_id", "horse_id",
        "ax8_jockey_in3_rate_total_asof",
        "ax8_trainer_in3_rate_total_asof",
    ]

    result = result[ax8_cols].drop_duplicates(subset=["race_id", "horse_id"])
    logger.info(f"  - ax8_ features computed: {len([c for c in ax8_cols if c.startswith('ax8_')])} columns")

    return result


# ============================================================
# Axis 9: Odds/Popularity Proxy Features (ax9_)
# ============================================================


def compute_ax9_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute axis 9 features: odds and popularity proxy (as-of).

    Uses past odds/popularity data, NOT current race odds.
    All features use shift(1) to exclude current race.
    """
    logger.info("Computing ax9_ features (odds/popularity proxy)...")

    df = hr.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    g = df.groupby("horse_id", group_keys=False)

    def rolling_mean_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling mean of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=1).mean()

    def rolling_std_prev(s: pd.Series, window: int) -> pd.Series:
        """Rolling std of previous N races (excluding current)."""
        return s.shift(1).rolling(window=window, min_periods=2).std()

    def compute_slope(s: pd.Series, window: int) -> pd.Series:
        """Compute slope (trend) over recent N races."""
        shifted = s.shift(1)

        def get_slope(x):
            vals = x.dropna().values
            if len(vals) < 2:
                return 0.0
            return (vals[-1] - vals[0]) / (len(vals) - 1)

        return shifted.rolling(window=window, min_periods=2).apply(get_slope, raw=False)

    # Average win odds in recent 5 (past odds, not current)
    if "win_odds" in df.columns:
        df["ax9_avg_win_odds_recent5"] = g["win_odds"].apply(lambda x: rolling_mean_prev(x, 5))
        df["ax9_std_win_odds_recent5"] = g["win_odds"].apply(lambda x: rolling_std_prev(x, 5))
    else:
        df["ax9_avg_win_odds_recent5"] = np.nan
        df["ax9_std_win_odds_recent5"] = np.nan

    # Finish vs popularity difference (performance vs expectation)
    if "popularity" in df.columns and "finish_order" in df.columns:
        df["_finish_minus_popularity"] = df["finish_order"] - df["popularity"]
        df["ax9_finish_minus_popularity_mean_recent5"] = g["_finish_minus_popularity"].apply(
            lambda x: rolling_mean_prev(x, 5)
        )
        df["ax9_finish_minus_popularity_slope_recent5"] = g["_finish_minus_popularity"].apply(
            lambda x: compute_slope(x, 5)
        )
    else:
        df["ax9_finish_minus_popularity_mean_recent5"] = np.nan
        df["ax9_finish_minus_popularity_slope_recent5"] = np.nan

    # Fill NaN with 0
    for col in ["ax9_avg_win_odds_recent5", "ax9_std_win_odds_recent5",
                "ax9_finish_minus_popularity_mean_recent5", "ax9_finish_minus_popularity_slope_recent5"]:
        df[col] = df[col].fillna(0)

    ax9_cols = [
        "race_id", "horse_id",
        "ax9_avg_win_odds_recent5", "ax9_std_win_odds_recent5",
        "ax9_finish_minus_popularity_mean_recent5", "ax9_finish_minus_popularity_slope_recent5",
    ]

    result = df[ax9_cols].copy()
    logger.info(f"  - ax9_ features computed: {len([c for c in ax9_cols if c.startswith('ax9_')])} columns")

    return result


# ============================================================
# Build Feature Table V3
# ============================================================


def build_feature_table_v3(
    conn: sqlite3.Connection,
    target_table: str = "feature_table_v3",
    limit: Optional[int] = None,
) -> None:
    """Build feature_table_v3 by adding all 9-axis features to v2."""

    # Load base table
    base = load_feature_table_v2(conn)

    if limit:
        logger.info(f"Limiting to first {limit} rows for development")
        base = base.head(limit)

    # Load supporting tables
    hr = load_horse_results(conn)
    corners = load_corners(conn)
    races = load_races_for_corners(conn)
    race_results = load_race_results_for_corners(conn)

    # Compute axis 1 features (recent form/growth)
    ax1 = compute_ax1_features(hr)

    # Compute axis 2 features (condition suitability)
    ax2 = compute_ax2_features(hr, base)

    # Compute axis 3 features (class level/prize)
    ax3 = compute_ax3_features(hr)

    # Compute axis 4 features (rotation/schedule)
    ax4 = compute_ax4_features(hr)

    # Compute axis 5 features (gate/position)
    ax5 = compute_ax5_features(hr)

    # Compute axis 6 features (running style)
    ax6_horse, ax6_race = compute_ax6_features(hr, corners, race_results, races)

    # Prepare corner positions for ax7 (need pos_gain column)
    corner_positions = parse_corner_positions(corners, race_results)
    if not corner_positions.empty:
        corner_positions = corner_positions.merge(
            races[["race_id", "field_size"]], on="race_id", how="left"
        )
        corner_positions["pos_gain"] = (
            corner_positions["c1_pos"] - corner_positions["c4_pos"]
        ) / corner_positions["field_size"].clip(lower=1)

    # Compute axis 7 features (stability/variance)
    ax7 = compute_ax7_features(hr, corner_positions if not corner_positions.empty else None)

    # Compute axis 8 features (jockey/trainer)
    ax8 = compute_ax8_features(hr)

    # Compute axis 9 features (odds/popularity proxy)
    ax9 = compute_ax9_features(hr)

    # Merge all features
    logger.info("Merging all features...")

    # Start with base
    merged = base.copy()

    # Merge ax1
    ax1_cols_to_merge = [c for c in ax1.columns if c.startswith("ax1_")]
    ax1_for_merge = ax1[["race_id", "horse_id"] + ax1_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax1_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax2
    ax2_cols_to_merge = [c for c in ax2.columns if c.startswith("ax2_")]
    ax2_for_merge = ax2[["race_id", "horse_id"] + ax2_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax2_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax3
    ax3_cols_to_merge = [c for c in ax3.columns if c.startswith("ax3_")]
    ax3_for_merge = ax3[["race_id", "horse_id"] + ax3_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax3_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax4
    ax4_cols_to_merge = [c for c in ax4.columns if c.startswith("ax4_")]
    ax4_for_merge = ax4[["race_id", "horse_id"] + ax4_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax4_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax5
    ax5_cols_to_merge = [c for c in ax5.columns if c.startswith("ax5_")]
    ax5_for_merge = ax5[["race_id", "horse_id"] + ax5_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax5_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax6 horse features
    ax6_horse_cols = [c for c in ax6_horse.columns if c.startswith("ax6_")]
    if ax6_horse_cols:
        ax6_horse_for_merge = ax6_horse[["race_id", "horse_id"] + ax6_horse_cols].drop_duplicates(
            subset=["race_id", "horse_id"]
        )
        merged = merged.merge(ax6_horse_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax6 race features
    if not ax6_race.empty:
        merged = merged.merge(ax6_race, on="race_id", how="left")

    # Merge ax7
    ax7_cols_to_merge = [c for c in ax7.columns if c.startswith("ax7_")]
    ax7_for_merge = ax7[["race_id", "horse_id"] + ax7_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax7_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax8
    ax8_cols_to_merge = [c for c in ax8.columns if c.startswith("ax8_")]
    ax8_for_merge = ax8[["race_id", "horse_id"] + ax8_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax8_for_merge, on=["race_id", "horse_id"], how="left")

    # Merge ax9
    ax9_cols_to_merge = [c for c in ax9.columns if c.startswith("ax9_")]
    ax9_for_merge = ax9[["race_id", "horse_id"] + ax9_cols_to_merge].drop_duplicates(
        subset=["race_id", "horse_id"]
    )
    merged = merged.merge(ax9_for_merge, on=["race_id", "horse_id"], how="left")

    logger.info(f"Final merged rows: {len(merged):,}")

    # Log new column counts
    ax1_final = [c for c in merged.columns if c.startswith("ax1_")]
    ax2_final = [c for c in merged.columns if c.startswith("ax2_")]
    ax3_final = [c for c in merged.columns if c.startswith("ax3_")]
    ax4_final = [c for c in merged.columns if c.startswith("ax4_")]
    ax5_final = [c for c in merged.columns if c.startswith("ax5_")]
    ax6_final = [c for c in merged.columns if c.startswith("ax6_")]
    ax7_final = [c for c in merged.columns if c.startswith("ax7_")]
    ax8_final = [c for c in merged.columns if c.startswith("ax8_")]
    ax9_final = [c for c in merged.columns if c.startswith("ax9_")]

    logger.info(f"ax1_ columns: {len(ax1_final)}")
    logger.info(f"ax2_ columns: {len(ax2_final)}")
    logger.info(f"ax3_ columns: {len(ax3_final)}")
    logger.info(f"ax4_ columns: {len(ax4_final)}")
    logger.info(f"ax5_ columns: {len(ax5_final)}")
    logger.info(f"ax6_ columns: {len(ax6_final)}")
    logger.info(f"ax7_ columns: {len(ax7_final)}")
    logger.info(f"ax8_ columns: {len(ax8_final)}")
    logger.info(f"ax9_ columns: {len(ax9_final)}")

    # Report missing rates for new columns
    all_ax_cols = (ax1_final + ax2_final + ax3_final + ax4_final +
                   ax5_final + ax6_final + ax7_final + ax8_final + ax9_final)
    logger.info(f"\nTotal ax*_ columns in feature_table_v3: {len(all_ax_cols)}")
    logger.info("\nMissing rates for ax*_ columns:")
    for col in all_ax_cols:
        missing = merged[col].isna().sum()
        missing_pct = 100 * missing / len(merged)
        if missing_pct > 0:
            logger.info(f"  {col}: {missing:,} ({missing_pct:.1f}%)")

    # Save to SQLite with proper UNIQUE constraint
    logger.info(f"\nSaving {target_table} with UPSERT support...")

    # Import UPSERT utilities
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from src.db.upsert import create_table_from_df
        # Use new idempotent table creation
        create_table_from_df(
            conn, merged, target_table,
            key_columns=["race_id", "horse_id"],
            if_exists="replace"
        )
        logger.info(f"Wrote table {target_table}: {len(merged):,} rows, {len(merged.columns)} columns")
        logger.info(f"Created UNIQUE INDEX on ({target_table}.race_id, {target_table}.horse_id)")
    except ImportError:
        # Fallback to old method if src.db not available
        logger.warning("src.db.upsert not available, using pandas to_sql (no UNIQUE constraint)")
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {target_table}")
        conn.commit()
        merged.to_sql(target_table, conn, index=False)
        logger.info(f"Wrote table {target_table}: {len(merged):,} rows, {len(merged.columns)} columns")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature_table_v3 with all 9-axis features (ax1-9)"
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB (e.g., netkeiba.db)",
    )
    parser.add_argument(
        "--target-table",
        type=str,
        default="feature_table_v3",
        help="Target feature table name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows for development (optional)",
    )
    args = parser.parse_args()

    import os
    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"Database: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        build_feature_table_v3(
            conn,
            target_table=args.target_table,
            limit=args.limit,
        )
        logger.info("Done!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
