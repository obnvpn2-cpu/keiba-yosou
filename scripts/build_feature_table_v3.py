#!/usr/bin/env python3
"""
build_feature_table_v3.py

9軸特徴量のうち5軸（①②③④⑥）を feature_table_v2 に追加して feature_table_v3 を構築する。

軸①: 近況・成長（直近の着順・上がり3Fトレンド）
軸②: 距離・コース条件適性
軸③: 地力・レースレベル（クラス・賞金）
軸④: ローテーション・臨戦過程
軸⑥: 脚質・位置取り × 展開（corners由来）

重要な原則:
- すべての特徴量は「そのレースより前のレース」のみから計算（shift(1)徹底）
- per-row ループ禁止（groupby + shift/cum系で高速処理）
- 禁止列（target_*, finish_*, payout等）は絶対に入れない

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


def load_horse_results(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load horse_results with surface/course info from races table."""
    logger.info("Loading horse_results...")

    query = """
    SELECT
        hr.horse_id,
        hr.race_id,
        hr.race_date,
        hr.place,
        hr.course_type,
        hr.distance,
        hr.track_condition,
        hr.field_size,
        hr.race_class,
        hr.grade,
        hr.finish_order,
        hr.frame_no,
        hr.horse_no,
        hr.last_3f,
        hr.prize_money
    FROM horse_results hr
    WHERE hr.race_date IS NOT NULL
    ORDER BY hr.horse_id, hr.race_date, hr.race_id
    """
    df = pd.read_sql_query(query, conn)
    logger.info(f"horse_results rows: {len(df):,}")
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
# Build Feature Table V3
# ============================================================


def build_feature_table_v3(
    conn: sqlite3.Connection,
    target_table: str = "feature_table_v3",
    limit: Optional[int] = None,
) -> None:
    """Build feature_table_v3 by adding ax1_/ax2_/ax3_/ax4_/ax6_ features to v2."""

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

    # Compute axis 6 features (running style)
    ax6_horse, ax6_race = compute_ax6_features(hr, corners, race_results, races)

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

    logger.info(f"Final merged rows: {len(merged):,}")

    # Log new column counts
    ax1_final = [c for c in merged.columns if c.startswith("ax1_")]
    ax2_final = [c for c in merged.columns if c.startswith("ax2_")]
    ax3_final = [c for c in merged.columns if c.startswith("ax3_")]
    ax4_final = [c for c in merged.columns if c.startswith("ax4_")]
    ax6_final = [c for c in merged.columns if c.startswith("ax6_")]

    logger.info(f"New ax1_ columns: {len(ax1_final)}")
    logger.info(f"New ax2_ columns: {len(ax2_final)}")
    logger.info(f"New ax3_ columns: {len(ax3_final)}")
    logger.info(f"New ax4_ columns: {len(ax4_final)}")
    logger.info(f"New ax6_ columns: {len(ax6_final)}")
    logger.info(f"ax1_ columns: {ax1_final}")
    logger.info(f"ax2_ columns: {ax2_final}")
    logger.info(f"ax3_ columns: {ax3_final}")
    logger.info(f"ax4_ columns: {ax4_final}")
    logger.info(f"ax6_ columns: {ax6_final}")

    # Report missing rates for new columns
    all_ax_cols = ax1_final + ax2_final + ax3_final + ax4_final + ax6_final
    logger.info(f"\nTotal new ax*_ columns: {len(all_ax_cols)}")
    logger.info("\nMissing rates for new columns:")
    for col in all_ax_cols:
        missing = merged[col].isna().sum()
        missing_pct = 100 * missing / len(merged)
        if missing_pct > 0:
            logger.info(f"  {col}: {missing:,} ({missing_pct:.1f}%)")

    # Save to SQLite
    logger.info(f"\nDropping and creating {target_table}...")
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {target_table}")
    conn.commit()

    merged.to_sql(target_table, conn, index=False)
    logger.info(f"Wrote table {target_table}: {len(merged):,} rows, {len(merged.columns)} columns")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature_table_v3 with ax1_/ax2_/ax3_/ax4_/ax6_ features"
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
