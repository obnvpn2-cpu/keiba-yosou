# -*- coding: utf-8 -*-
"""
asof_aggregator.py - As-Of 統計計算モジュール

【設計原則】
1. 全ての統計は race_date より前 (< race_date) のデータのみで計算
2. 当該レースのデータは絶対に含めない
3. SQL ベースの集計で高速化 (Python ループ禁止)
4. キャッシュを活用してパフォーマンス最適化

【使用例】
    agg = AsOfAggregator(conn)

    # 馬の as-of 統計を一括計算
    horse_stats = agg.compute_horse_stats_batch(target_races_df)

    # 騎手の as-of 統計を計算
    jockey_stats = agg.compute_jockey_stats_batch(target_races_df)

    # 調教師の as-of 統計を計算
    trainer_stats = agg.compute_trainer_stats_batch(target_races_df)

【As-Of 安全性レベル】
現在の実装: Level 1 (日付ベース)
    - race_date < current_race_date で統計を計算
    - 同日の他レース結果は含まれない (安全)
    - ただし同日の朝イチ発走結果も除外される (やや保守的)

将来の拡張案: Level 2 (発走時刻ベース)
    - races.start_time を使用して同日でも発走時刻で比較
    - 例: 15:40発走のレースなら、同日14:00発走のレース結果は使用可能
    - 実装時の注意点:
        * start_time のパース処理が必要
        * NULL の場合のフォールバック (日付ベースに戻す)
        * タイムゾーン考慮 (JST前提)

TODO: Level 2 (発走時刻ベース) の実装
    1. races.start_time カラムの存在確認
    2. 時刻パース関数の追加 (_parse_start_time)
    3. compute_*_asof_stats に use_start_time: bool オプション追加
    4. WHERE句を race_date < ? OR (race_date = ? AND start_time < ?) に変更
    5. 単体テスト追加 (同日の前後レースで結果が変わることを確認)

注: 現時点ではJRAのレースは同日に同じ馬が複数回走ることは稀なため、
    Level 1 で十分な安全性が確保されている。
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Encoding Maps
# =============================================================================

PLACE_MAP = {
    "札幌": 0, "函館": 1, "福島": 2, "新潟": 3, "東京": 4,
    "中山": 5, "中京": 6, "京都": 7, "阪神": 8, "小倉": 9,
}

SURFACE_MAP = {
    "芝": 0, "ダ": 1, "ダート": 1, "障": 2, "障害": 2,
}

TRACK_CONDITION_MAP = {
    "良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3,
}

GRADE_MAP = {
    "G1": 0, "G2": 1, "G3": 2, "OP": 3, "オープン": 3,
    "L": 4, "Listed": 4, "リステッド": 4,
}

SEX_MAP = {
    "牡": 0, "牝": 1, "セ": 2, "せん": 2,
}

TURN_MAP = {
    "右": 0, "左": 1, "直": 2,
}

INOUT_MAP = {
    "内": 0, "外": 1, "直": 2,
}


def map_distance_to_cat(distance: Optional[int]) -> Optional[int]:
    """距離を距離カテゴリにマッピング"""
    if distance is None:
        return None
    if distance < 1100:
        return 1000
    elif distance < 1300:
        return 1200
    elif distance < 1500:
        return 1400
    elif distance < 1700:
        return 1600
    elif distance < 1900:
        return 1800
    elif distance < 2100:
        return 2000
    elif distance < 2300:
        return 2200
    elif distance < 2500:
        return 2400
    elif distance < 2700:
        return 2500
    else:
        return 3000


def map_class_to_id(race_class: Optional[str]) -> Optional[int]:
    """レースクラスをIDにマッピング"""
    if race_class is None:
        return None

    class_str = str(race_class).strip()

    # 優先度順にマッチング
    if "G1" in class_str or "GI" in class_str:
        return 0
    elif "G2" in class_str or "GII" in class_str:
        return 1
    elif "G3" in class_str or "GIII" in class_str:
        return 2
    elif "オープン" in class_str or "OP" in class_str or "Open" in class_str:
        return 3
    elif "3勝" in class_str or "1600万" in class_str:
        return 4
    elif "2勝" in class_str or "1000万" in class_str:
        return 5
    elif "1勝" in class_str or "500万" in class_str:
        return 6
    elif "未勝利" in class_str:
        return 7
    elif "新馬" in class_str:
        return 8
    else:
        return 9


# =============================================================================
# As-Of Aggregator Class
# =============================================================================

class AsOfAggregator:
    """
    As-Of 統計計算クラス

    全ての統計は race_date より前のデータのみで計算し、
    未来情報リークを防止する。

    【パフォーマンス】
    - SQL ベースの集計で高速化
    - バッチ処理でクエリ数を最小化
    - キャッシュを活用して重複計算を回避
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Args:
            conn: SQLite 接続
        """
        self.conn = conn
        self._cache: Dict[str, pd.DataFrame] = {}

    # =========================================================================
    # Helper: Date Normalization
    # =========================================================================

    def _normalize_date(self, date_value: Any) -> Optional[str]:
        """日付を YYYY-MM-DD 形式に正規化"""
        if date_value is None or pd.isna(date_value):
            return None

        if isinstance(date_value, datetime):
            return date_value.strftime("%Y-%m-%d")

        if isinstance(date_value, pd.Timestamp):
            return date_value.strftime("%Y-%m-%d")

        date_str = str(date_value).strip()
        if not date_str:
            return None

        # 各種フォーマットを試行
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    # =========================================================================
    # Load Race Results for As-Of Computation
    # =========================================================================

    def load_race_results(self) -> pd.DataFrame:
        """
        race_results テーブルを races と JOIN してロード

        Returns:
            DataFrame with race_date, horse_id, jockey_id, trainer_id,
            finish_order, place, surface, distance, track_condition, etc.
        """
        if "race_results_full" in self._cache:
            return self._cache["race_results_full"]

        logger.info("Loading race_results with race info...")

        sql = """
        SELECT
            rr.race_id,
            rr.horse_id,
            rr.finish_order,
            rr.last_3f,
            rr.body_weight,
            rr.body_weight_diff,
            rr.passing_order,
            rr.win_odds,
            rr.popularity,
            rr.prize_money,
            rr.jockey_id,
            rr.trainer_id,
            rr.sex,
            rr.age,
            rr.weight AS weight_carried,
            rr.frame_no,
            rr.horse_no,
            r.date AS race_date,
            r.place,
            r.course_type AS surface,
            r.distance,
            r.track_condition,
            r.race_class,
            r.grade,
            r.race_no,
            r.course_turn,
            r.course_inout,
            r.head_count AS field_size
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        WHERE rr.finish_order IS NOT NULL
          AND r.date IS NOT NULL
        ORDER BY r.date, rr.race_id
        """

        df = pd.read_sql_query(sql, self.conn)

        # 日付を正規化
        df["race_date"] = df["race_date"].apply(self._normalize_date)
        df = df.dropna(subset=["race_date"])

        # エンコーディング
        df["place_id"] = df["place"].map(PLACE_MAP)
        df["surface_id"] = df["surface"].map(SURFACE_MAP)
        df["track_condition_id"] = df["track_condition"].map(TRACK_CONDITION_MAP)
        df["distance_cat"] = df["distance"].apply(map_distance_to_cat)
        df["race_class_id"] = df["race_class"].apply(map_class_to_id)
        df["grade_id"] = df["grade"].map(GRADE_MAP)
        df["sex_id"] = df["sex"].map(SEX_MAP)
        df["course_turn_id"] = df["course_turn"].map(TURN_MAP)
        df["course_inout_id"] = df["course_inout"].map(INOUT_MAP)

        # フィールドサイズが NULL の場合は race_id ごとにカウント
        if df["field_size"].isna().any():
            field_sizes = df.groupby("race_id").size().reset_index(name="field_size_calc")
            df = df.merge(field_sizes, on="race_id", how="left")
            df["field_size"] = df["field_size"].fillna(df["field_size_calc"])
            df.drop(columns=["field_size_calc"], inplace=True)

        logger.info("Loaded %d race_results rows", len(df))
        self._cache["race_results_full"] = df

        return df

    # =========================================================================
    # Horse As-Of Stats
    # =========================================================================

    def compute_horse_asof_stats(
        self,
        horse_id: str,
        race_date: str,
        distance_cat: Optional[int] = None,
        surface: Optional[str] = None,
        track_condition: Optional[str] = None,
        place: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        指定した馬の as-of 統計を計算

        Args:
            horse_id: 馬ID
            race_date: 基準日 (この日より前のデータのみ使用)
            distance_cat: 距離カテゴリ (条件別統計用)
            surface: 馬場 (条件別統計用)
            track_condition: 馬場状態 (条件別統計用)
            place: 開催場 (条件別統計用)

        Returns:
            統計値の辞書
        """
        df = self.load_race_results()

        # 対象馬の race_date より前のレースを抽出
        past = df[
            (df["horse_id"] == horse_id) &
            (df["race_date"] < race_date)
        ].copy()

        stats: Dict[str, Any] = {}

        # Global stats
        stats["h_n_starts"] = len(past)
        if len(past) == 0:
            stats["h_n_wins"] = 0
            stats["h_n_in3"] = 0
            stats["h_win_rate"] = None
            stats["h_in3_rate"] = None
            stats["h_avg_finish"] = None
            stats["h_std_finish"] = None
            stats["h_best_finish"] = None
            stats["h_worst_finish"] = None
            stats["h_is_first_run"] = 1
        else:
            stats["h_n_wins"] = (past["finish_order"] == 1).sum()
            stats["h_n_in3"] = (past["finish_order"] <= 3).sum()
            stats["h_win_rate"] = stats["h_n_wins"] / len(past)
            stats["h_in3_rate"] = stats["h_n_in3"] / len(past)
            stats["h_avg_finish"] = past["finish_order"].mean()
            stats["h_std_finish"] = past["finish_order"].std() if len(past) > 1 else None
            stats["h_best_finish"] = past["finish_order"].min()
            stats["h_worst_finish"] = past["finish_order"].max()
            stats["h_is_first_run"] = 0

        # Distance category stats
        if distance_cat is not None and len(past) > 0:
            past_dist = past[past["distance_cat"] == distance_cat]
            stats["h_n_starts_dist"] = len(past_dist)
            if len(past_dist) > 0:
                stats["h_win_rate_dist"] = (past_dist["finish_order"] == 1).sum() / len(past_dist)
                stats["h_in3_rate_dist"] = (past_dist["finish_order"] <= 3).sum() / len(past_dist)
                stats["h_avg_finish_dist"] = past_dist["finish_order"].mean()
                stats["h_avg_last3f_dist"] = past_dist["last_3f"].mean()
            else:
                stats["h_win_rate_dist"] = None
                stats["h_in3_rate_dist"] = None
                stats["h_avg_finish_dist"] = None
                stats["h_avg_last3f_dist"] = None
        else:
            stats["h_n_starts_dist"] = 0
            stats["h_win_rate_dist"] = None
            stats["h_in3_rate_dist"] = None
            stats["h_avg_finish_dist"] = None
            stats["h_avg_last3f_dist"] = None

        # Surface stats
        if surface is not None and len(past) > 0:
            past_surface = past[past["surface"] == surface]
            stats["h_n_starts_surface"] = len(past_surface)
            if len(past_surface) > 0:
                stats["h_win_rate_surface"] = (past_surface["finish_order"] == 1).sum() / len(past_surface)
                stats["h_in3_rate_surface"] = (past_surface["finish_order"] <= 3).sum() / len(past_surface)
            else:
                stats["h_win_rate_surface"] = None
                stats["h_in3_rate_surface"] = None
        else:
            stats["h_n_starts_surface"] = 0
            stats["h_win_rate_surface"] = None
            stats["h_in3_rate_surface"] = None

        # Track condition stats
        if track_condition is not None and len(past) > 0:
            past_track = past[past["track_condition"] == track_condition]
            stats["h_n_starts_track"] = len(past_track)
            if len(past_track) > 0:
                stats["h_win_rate_track"] = (past_track["finish_order"] == 1).sum() / len(past_track)
                stats["h_in3_rate_track"] = (past_track["finish_order"] <= 3).sum() / len(past_track)
            else:
                stats["h_win_rate_track"] = None
                stats["h_in3_rate_track"] = None
        else:
            stats["h_n_starts_track"] = 0
            stats["h_win_rate_track"] = None
            stats["h_in3_rate_track"] = None

        # Course stats
        if place is not None and len(past) > 0:
            past_course = past[past["place"] == place]
            stats["h_n_starts_course"] = len(past_course)
            if len(past_course) > 0:
                stats["h_win_rate_course"] = (past_course["finish_order"] == 1).sum() / len(past_course)
                stats["h_in3_rate_course"] = (past_course["finish_order"] <= 3).sum() / len(past_course)
            else:
                stats["h_win_rate_course"] = None
                stats["h_in3_rate_course"] = None
        else:
            stats["h_n_starts_course"] = 0
            stats["h_win_rate_course"] = None
            stats["h_in3_rate_course"] = None

        # Recent form (last 3/5 races)
        if len(past) > 0:
            past_sorted = past.sort_values("race_date", ascending=False)

            # Days since last run
            last_race_date = past_sorted.iloc[0]["race_date"]
            try:
                days_diff = (datetime.strptime(race_date, "%Y-%m-%d") -
                            datetime.strptime(last_race_date, "%Y-%m-%d")).days
                stats["h_days_since_last"] = days_diff
            except:
                stats["h_days_since_last"] = None

            # Last 3 races
            recent3 = past_sorted.head(3)
            stats["h_recent3_avg_finish"] = recent3["finish_order"].mean()
            stats["h_recent3_best_finish"] = recent3["finish_order"].min()
            stats["h_recent3_avg_last3f"] = recent3["last_3f"].mean()

            # Last 5 races
            recent5 = past_sorted.head(5)
            stats["h_recent5_avg_finish"] = recent5["finish_order"].mean()
            stats["h_recent5_win_rate"] = (recent5["finish_order"] == 1).sum() / len(recent5)
            stats["h_recent5_in3_rate"] = (recent5["finish_order"] <= 3).sum() / len(recent5)

            # Weight stats (全て過去レースのみから算出 - pre-race safe)
            weight_values = past["body_weight"].dropna()
            if len(weight_values) > 0:
                stats["h_avg_body_weight"] = weight_values.mean()
            else:
                stats["h_avg_body_weight"] = None

            # --- 体重の "前日安全版" 特徴量 (pre-race safe) ---
            # 直近出走の馬体重 & 増減
            recent_with_weight = past_sorted[past_sorted["body_weight"].notna()]
            if len(recent_with_weight) > 0:
                stats["h_last_body_weight"] = recent_with_weight.iloc[0]["body_weight"]
                stats["h_last_body_weight_diff"] = recent_with_weight.iloc[0]["body_weight_diff"]
            else:
                stats["h_last_body_weight"] = None
                stats["h_last_body_weight_diff"] = None

            # 直近3走の体重統計
            recent3_weight = recent_with_weight.head(3)["body_weight"]
            if len(recent3_weight) >= 1:
                stats["h_recent3_avg_body_weight"] = recent3_weight.mean()
                if len(recent3_weight) >= 2:
                    stats["h_recent3_std_body_weight"] = recent3_weight.std()
                else:
                    stats["h_recent3_std_body_weight"] = None
                # トレンド: (最新 - 最古) / 走数  (負なら減量傾向)
                if len(recent3_weight) >= 2:
                    newest = recent3_weight.iloc[0]
                    oldest = recent3_weight.iloc[-1]
                    stats["h_recent3_body_weight_trend"] = (newest - oldest) / (len(recent3_weight) - 1)
                else:
                    stats["h_recent3_body_weight_trend"] = None
            else:
                stats["h_recent3_avg_body_weight"] = None
                stats["h_recent3_std_body_weight"] = None
                stats["h_recent3_body_weight_trend"] = None

            # 体重z-score: (直近体重 - 全平均) / 全標準偏差
            if (stats.get("h_last_body_weight") is not None and
                stats.get("h_avg_body_weight") is not None and
                len(weight_values) >= 2):
                weight_std = weight_values.std()
                if weight_std and weight_std > 0:
                    stats["h_body_weight_z"] = (
                        stats["h_last_body_weight"] - stats["h_avg_body_weight"]
                    ) / weight_std
                else:
                    stats["h_body_weight_z"] = None
            else:
                stats["h_body_weight_z"] = None

            # Last 3F stats
            last3f_values = past["last_3f"].dropna()
            if len(last3f_values) > 0:
                stats["h_avg_last3f"] = last3f_values.mean()
                stats["h_best_last3f"] = last3f_values.min()
            else:
                stats["h_avg_last3f"] = None
                stats["h_best_last3f"] = None

            # Prize stats
            prize_values = past["prize_money"].dropna()
            if len(prize_values) > 0:
                stats["h_total_prize"] = prize_values.sum()
                stats["h_avg_prize"] = prize_values.mean()
                stats["h_max_prize"] = prize_values.max()

                recent3_prize = past_sorted.head(3)["prize_money"].dropna()
                stats["h_recent3_total_prize"] = recent3_prize.sum()

                if stats["h_avg_prize"] > 0:
                    stats["h_prize_momentum"] = stats["h_recent3_total_prize"] / (3 * stats["h_avg_prize"])
                else:
                    stats["h_prize_momentum"] = None
            else:
                stats["h_total_prize"] = 0
                stats["h_avg_prize"] = None
                stats["h_max_prize"] = None
                stats["h_recent3_total_prize"] = 0
                stats["h_prize_momentum"] = None
        else:
            stats["h_days_since_last"] = None
            stats["h_recent3_avg_finish"] = None
            stats["h_recent3_best_finish"] = None
            stats["h_recent3_avg_last3f"] = None
            stats["h_recent5_avg_finish"] = None
            stats["h_recent5_win_rate"] = None
            stats["h_recent5_in3_rate"] = None
            stats["h_avg_body_weight"] = None
            # 体重の "前日安全版" 特徴量 (pre-race safe) - 初出走時は全てNone
            stats["h_last_body_weight"] = None
            stats["h_last_body_weight_diff"] = None
            stats["h_recent3_avg_body_weight"] = None
            stats["h_recent3_std_body_weight"] = None
            stats["h_recent3_body_weight_trend"] = None
            stats["h_body_weight_z"] = None
            stats["h_avg_last3f"] = None
            stats["h_best_last3f"] = None
            stats["h_total_prize"] = 0
            stats["h_avg_prize"] = None
            stats["h_max_prize"] = None
            stats["h_recent3_total_prize"] = 0
            stats["h_prize_momentum"] = None

        return stats

    # =========================================================================
    # Jockey As-Of Stats
    # =========================================================================

    def compute_jockey_asof_stats(
        self,
        jockey_id: str,
        race_date: str,
        distance_cat: Optional[int] = None,
        surface: Optional[str] = None,
        place: Optional[str] = None,
        horse_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        指定した騎手の as-of 統計を計算

        Args:
            jockey_id: 騎手ID
            race_date: 基準日
            distance_cat: 距離カテゴリ
            surface: 馬場
            place: 開催場
            horse_id: 馬ID (コンビ成績用)

        Returns:
            統計値の辞書
        """
        df = self.load_race_results()

        past = df[
            (df["jockey_id"] == jockey_id) &
            (df["race_date"] < race_date)
        ].copy()

        stats: Dict[str, Any] = {}

        # Global stats
        stats["j_n_starts"] = len(past)
        if len(past) > 0:
            stats["j_n_wins"] = (past["finish_order"] == 1).sum()
            stats["j_win_rate"] = stats["j_n_wins"] / len(past)
            stats["j_in3_rate"] = (past["finish_order"] <= 3).sum() / len(past)
        else:
            stats["j_n_wins"] = 0
            stats["j_win_rate"] = None
            stats["j_in3_rate"] = None

        # Distance category stats
        if distance_cat is not None and len(past) > 0:
            past_dist = past[past["distance_cat"] == distance_cat]
            stats["j_n_starts_dist"] = len(past_dist)
            if len(past_dist) > 0:
                stats["j_win_rate_dist"] = (past_dist["finish_order"] == 1).sum() / len(past_dist)
                stats["j_in3_rate_dist"] = (past_dist["finish_order"] <= 3).sum() / len(past_dist)
            else:
                stats["j_win_rate_dist"] = None
                stats["j_in3_rate_dist"] = None
        else:
            stats["j_n_starts_dist"] = 0
            stats["j_win_rate_dist"] = None
            stats["j_in3_rate_dist"] = None

        # Surface stats
        if surface is not None and len(past) > 0:
            past_surface = past[past["surface"] == surface]
            if len(past_surface) > 0:
                stats["j_win_rate_surface"] = (past_surface["finish_order"] == 1).sum() / len(past_surface)
                stats["j_in3_rate_surface"] = (past_surface["finish_order"] <= 3).sum() / len(past_surface)
            else:
                stats["j_win_rate_surface"] = None
                stats["j_in3_rate_surface"] = None
        else:
            stats["j_win_rate_surface"] = None
            stats["j_in3_rate_surface"] = None

        # Course stats
        if place is not None and len(past) > 0:
            past_course = past[past["place"] == place]
            if len(past_course) > 0:
                stats["j_win_rate_course"] = (past_course["finish_order"] == 1).sum() / len(past_course)
                stats["j_in3_rate_course"] = (past_course["finish_order"] <= 3).sum() / len(past_course)
            else:
                stats["j_win_rate_course"] = None
                stats["j_in3_rate_course"] = None
        else:
            stats["j_win_rate_course"] = None
            stats["j_in3_rate_course"] = None

        # Recent 30 days stats
        try:
            race_dt = datetime.strptime(race_date, "%Y-%m-%d")
            date_30d_ago = (race_dt - timedelta(days=30)).strftime("%Y-%m-%d")

            past_30d = past[past["race_date"] >= date_30d_ago]
            stats["j_recent30d_n_rides"] = len(past_30d)
            if len(past_30d) > 0:
                stats["j_recent30d_win_rate"] = (past_30d["finish_order"] == 1).sum() / len(past_30d)
                stats["j_recent30d_in3_rate"] = (past_30d["finish_order"] <= 3).sum() / len(past_30d)
            else:
                stats["j_recent30d_win_rate"] = None
                stats["j_recent30d_in3_rate"] = None
        except:
            stats["j_recent30d_n_rides"] = 0
            stats["j_recent30d_win_rate"] = None
            stats["j_recent30d_in3_rate"] = None

        # Jockey-Horse combo stats
        if horse_id is not None and len(past) > 0:
            past_combo = past[past["horse_id"] == horse_id]
            stats["jh_n_combos"] = len(past_combo)
            if len(past_combo) > 0:
                stats["jh_combo_win_rate"] = (past_combo["finish_order"] == 1).sum() / len(past_combo)
                stats["jh_combo_in3_rate"] = (past_combo["finish_order"] <= 3).sum() / len(past_combo)
            else:
                stats["jh_combo_win_rate"] = None
                stats["jh_combo_in3_rate"] = None
        else:
            stats["jh_n_combos"] = 0
            stats["jh_combo_win_rate"] = None
            stats["jh_combo_in3_rate"] = None

        return stats

    # =========================================================================
    # Trainer As-Of Stats
    # =========================================================================

    def compute_trainer_asof_stats(
        self,
        trainer_id: str,
        race_date: str,
        distance_cat: Optional[int] = None,
        surface: Optional[str] = None,
        place: Optional[str] = None,
        jockey_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        指定した調教師の as-of 統計を計算

        Args:
            trainer_id: 調教師ID
            race_date: 基準日
            distance_cat: 距離カテゴリ
            surface: 馬場
            place: 開催場
            jockey_id: 騎手ID (コンビ成績用)

        Returns:
            統計値の辞書
        """
        df = self.load_race_results()

        past = df[
            (df["trainer_id"] == trainer_id) &
            (df["race_date"] < race_date)
        ].copy()

        stats: Dict[str, Any] = {}

        # Global stats
        stats["t_n_starts"] = len(past)
        if len(past) > 0:
            stats["t_n_wins"] = (past["finish_order"] == 1).sum()
            stats["t_win_rate"] = stats["t_n_wins"] / len(past)
            stats["t_in3_rate"] = (past["finish_order"] <= 3).sum() / len(past)
        else:
            stats["t_n_wins"] = 0
            stats["t_win_rate"] = None
            stats["t_in3_rate"] = None

        # Distance category stats
        if distance_cat is not None and len(past) > 0:
            past_dist = past[past["distance_cat"] == distance_cat]
            stats["t_n_starts_dist"] = len(past_dist)
            if len(past_dist) > 0:
                stats["t_win_rate_dist"] = (past_dist["finish_order"] == 1).sum() / len(past_dist)
                stats["t_in3_rate_dist"] = (past_dist["finish_order"] <= 3).sum() / len(past_dist)
            else:
                stats["t_win_rate_dist"] = None
                stats["t_in3_rate_dist"] = None
        else:
            stats["t_n_starts_dist"] = 0
            stats["t_win_rate_dist"] = None
            stats["t_in3_rate_dist"] = None

        # Surface stats
        if surface is not None and len(past) > 0:
            past_surface = past[past["surface"] == surface]
            if len(past_surface) > 0:
                stats["t_win_rate_surface"] = (past_surface["finish_order"] == 1).sum() / len(past_surface)
                stats["t_in3_rate_surface"] = (past_surface["finish_order"] <= 3).sum() / len(past_surface)
            else:
                stats["t_win_rate_surface"] = None
                stats["t_in3_rate_surface"] = None
        else:
            stats["t_win_rate_surface"] = None
            stats["t_in3_rate_surface"] = None

        # Course stats
        if place is not None and len(past) > 0:
            past_course = past[past["place"] == place]
            if len(past_course) > 0:
                stats["t_win_rate_course"] = (past_course["finish_order"] == 1).sum() / len(past_course)
                stats["t_in3_rate_course"] = (past_course["finish_order"] <= 3).sum() / len(past_course)
            else:
                stats["t_win_rate_course"] = None
                stats["t_in3_rate_course"] = None
        else:
            stats["t_win_rate_course"] = None
            stats["t_in3_rate_course"] = None

        # Recent 30 days stats
        try:
            race_dt = datetime.strptime(race_date, "%Y-%m-%d")
            date_30d_ago = (race_dt - timedelta(days=30)).strftime("%Y-%m-%d")

            past_30d = past[past["race_date"] >= date_30d_ago]
            stats["t_recent30d_n_entries"] = len(past_30d)
            if len(past_30d) > 0:
                stats["t_recent30d_win_rate"] = (past_30d["finish_order"] == 1).sum() / len(past_30d)
                stats["t_recent30d_in3_rate"] = (past_30d["finish_order"] <= 3).sum() / len(past_30d)
            else:
                stats["t_recent30d_win_rate"] = None
                stats["t_recent30d_in3_rate"] = None
        except:
            stats["t_recent30d_n_entries"] = 0
            stats["t_recent30d_win_rate"] = None
            stats["t_recent30d_in3_rate"] = None

        # Trainer-Jockey combo stats
        if jockey_id is not None and len(past) > 0:
            past_combo = past[past["jockey_id"] == jockey_id]
            stats["tj_n_combos"] = len(past_combo)
            if len(past_combo) > 0:
                stats["tj_combo_win_rate"] = (past_combo["finish_order"] == 1).sum() / len(past_combo)
                stats["tj_combo_in3_rate"] = (past_combo["finish_order"] <= 3).sum() / len(past_combo)
            else:
                stats["tj_combo_win_rate"] = None
                stats["tj_combo_in3_rate"] = None
        else:
            stats["tj_n_combos"] = 0
            stats["tj_combo_win_rate"] = None
            stats["tj_combo_in3_rate"] = None

        return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_horse_asof_stats(
    conn: sqlite3.Connection,
    horse_id: str,
    race_date: str,
    **kwargs,
) -> Dict[str, Any]:
    """馬の as-of 統計を計算するコンビニエンス関数"""
    agg = AsOfAggregator(conn)
    return agg.compute_horse_asof_stats(horse_id, race_date, **kwargs)


def compute_jockey_asof_stats(
    conn: sqlite3.Connection,
    jockey_id: str,
    race_date: str,
    **kwargs,
) -> Dict[str, Any]:
    """騎手の as-of 統計を計算するコンビニエンス関数"""
    agg = AsOfAggregator(conn)
    return agg.compute_jockey_asof_stats(jockey_id, race_date, **kwargs)


def compute_trainer_asof_stats(
    conn: sqlite3.Connection,
    trainer_id: str,
    race_date: str,
    **kwargs,
) -> Dict[str, Any]:
    """調教師の as-of 統計を計算するコンビニエンス関数"""
    agg = AsOfAggregator(conn)
    return agg.compute_trainer_asof_stats(trainer_id, race_date, **kwargs)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    print("=" * 60)
    print("AsOfAggregator Test")
    print("=" * 60)

    # テスト用のダミーデータでの動作確認は省略
    print("\nTo test, connect to an actual database and call:")
    print("  agg = AsOfAggregator(conn)")
    print("  stats = agg.compute_horse_asof_stats('horse_id', '2024-01-01')")
    print("=" * 60)
