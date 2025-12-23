# -*- coding: utf-8 -*-
"""
feature_builder_v4.py - FeaturePack v1 特徴量生成エンジン

200+ 特徴量を生成するメインモジュール。
全ての統計は as-of (race_date より前) で計算し、未来情報リークを防止。

【処理フロー】
1. 対象レースの取得 (年/月フィルタ可能)
2. 各レースについて:
   a. レース基本情報を抽出 (base_race)
   b. 各出走馬について:
      - 馬の過去成績 (horse_form)
      - ペース・位置取り (pace_position)
      - クラス・賞金 (class_prize)
      - 騎手統計 (jockey)
      - 調教師統計 (trainer)
      - 血統ハッシュ (pedigree)
3. feature_table_v4 に UPSERT

【パフォーマンス】
- race_results を一括ロードしてキャッシュ
- バッチ処理でクエリ数を最小化
- 進捗ログで状況を可視化
"""

import logging
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from .feature_table_v4 import (
    create_feature_table_v4,
    get_feature_v4_columns,
)
from .asof_aggregator import (
    AsOfAggregator,
    map_distance_to_cat,
    map_class_to_id,
    PLACE_MAP,
    SURFACE_MAP,
    TRACK_CONDITION_MAP,
    GRADE_MAP,
    SEX_MAP,
    TURN_MAP,
    INOUT_MAP,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pedigree Hashing
# =============================================================================

def hash_pedigree_sire_dam(
    sire_name: Optional[str],
    dam_name: Optional[str],
    broodmare_sire_name: Optional[str],
    n_dims: int = 512,
) -> np.ndarray:
    """
    父×母×母父 の組み合わせを Feature Hashing で n_dims 次元に変換

    Args:
        sire_name: 父馬名
        dam_name: 母馬名
        broodmare_sire_name: 母父馬名
        n_dims: 出力次元数

    Returns:
        np.ndarray of shape (n_dims,)
    """
    result = np.zeros(n_dims, dtype=np.float32)

    # 各要素をハッシュ
    names = [
        ("sire", sire_name),
        ("dam", dam_name),
        ("bms", broodmare_sire_name),
    ]

    for prefix, name in names:
        if name is None or pd.isna(name):
            continue

        key = f"{prefix}:{name}"
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        idx = h % n_dims
        sign = 1 if (h // n_dims) % 2 == 0 else -1
        result[idx] += sign * 1.0

    # L2 正規化
    norm = np.linalg.norm(result)
    if norm > 0:
        result /= norm

    return result


def hash_ancestor_frequency(
    ancestors: List[Tuple[str, int]],
    n_dims: int = 128,
) -> np.ndarray:
    """
    5代血統の祖先頻度を Feature Hashing で n_dims 次元に変換

    Args:
        ancestors: [(ancestor_name, generation), ...] のリスト
        n_dims: 出力次元数

    Returns:
        np.ndarray of shape (n_dims,)
    """
    result = np.zeros(n_dims, dtype=np.float32)

    if not ancestors:
        return result

    for ancestor_name, generation in ancestors:
        if ancestor_name is None or pd.isna(ancestor_name):
            continue

        # 世代が若いほど重み大
        weight = 1.0 / (generation ** 0.5)

        key = f"anc:{ancestor_name}"
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        idx = h % n_dims
        sign = 1 if (h // n_dims) % 2 == 0 else -1
        result[idx] += sign * weight

    # L2 正規化
    norm = np.linalg.norm(result)
    if norm > 0:
        result /= norm

    return result


# =============================================================================
# Feature Builder Class
# =============================================================================

class FeatureBuilderV4:
    """
    FeaturePack v1 特徴量生成クラス

    【使用例】
        builder = FeatureBuilderV4(conn)
        builder.build_all(
            start_year=2021,
            end_year=2024,
            include_pedigree=True,
        )
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.asof = AsOfAggregator(conn)
        self._pedigree_cache: Dict[str, List[Tuple[str, int]]] = {}

    # =========================================================================
    # Pedigree Loading
    # =========================================================================

    def load_pedigree(self, horse_id: str) -> Dict[str, Any]:
        """
        馬の血統情報をロード

        Returns:
            {
                "sire_name": str,
                "dam_name": str,
                "broodmare_sire_name": str,
                "ancestors": [(name, generation), ...]
            }
        """
        result = {
            "sire_name": None,
            "dam_name": None,
            "broodmare_sire_name": None,
            "ancestors": [],
        }

        # horses テーブルから基本血統
        try:
            cursor = self.conn.execute("""
                SELECT sire_name, dam_name, broodmare_sire_name
                FROM horses
                WHERE horse_id = ?
            """, (horse_id,))
            row = cursor.fetchone()
            if row:
                result["sire_name"] = row[0]
                result["dam_name"] = row[1]
                result["broodmare_sire_name"] = row[2]
        except Exception as e:
            logger.debug("Failed to load basic pedigree for %s: %s", horse_id, e)

        # horse_pedigree テーブルから詳細血統
        if horse_id in self._pedigree_cache:
            result["ancestors"] = self._pedigree_cache[horse_id]
        else:
            try:
                cursor = self.conn.execute("""
                    SELECT ancestor_name, generation
                    FROM horse_pedigree
                    WHERE horse_id = ?
                """, (horse_id,))
                ancestors = [(row[0], row[1]) for row in cursor.fetchall()]
                self._pedigree_cache[horse_id] = ancestors
                result["ancestors"] = ancestors
            except Exception as e:
                logger.debug("Failed to load pedigree for %s: %s", horse_id, e)

        return result

    # =========================================================================
    # Corner Position Parsing
    # =========================================================================

    def parse_passing_order(
        self,
        passing_order: Optional[str],
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        通過順位をパース

        Args:
            passing_order: "1-2-3-4" 形式の文字列

        Returns:
            (corner1, corner2, corner3, corner4)
        """
        if passing_order is None or pd.isna(passing_order):
            return (None, None, None, None)

        try:
            parts = str(passing_order).split("-")
            positions = []
            for i in range(4):
                if i < len(parts):
                    pos = parts[i].strip()
                    positions.append(int(pos) if pos.isdigit() else None)
                else:
                    positions.append(None)
            return tuple(positions)
        except:
            return (None, None, None, None)

    # =========================================================================
    # Build Features for One Race
    # =========================================================================

    def build_features_for_race(
        self,
        race_id: str,
        include_pedigree: bool = True,
        include_market: bool = False,
    ) -> pd.DataFrame:
        """
        1レース分の特徴量を構築

        Args:
            race_id: レースID
            include_pedigree: 血統ハッシュを含めるか
            include_market: 市場情報を含めるか

        Returns:
            DataFrame with one row per horse
        """
        # レース情報を取得
        cursor = self.conn.execute("""
            SELECT
                r.race_id,
                r.date,
                r.place,
                r.course_type,
                r.distance,
                r.track_condition,
                r.race_class,
                r.grade,
                r.race_no,
                r.course_turn,
                r.course_inout,
                r.head_count
            FROM races r
            WHERE r.race_id = ?
        """, (race_id,))
        race_row = cursor.fetchone()

        if race_row is None:
            logger.warning("Race not found: %s", race_id)
            return pd.DataFrame()

        race_date = race_row[1]
        if race_date is None:
            logger.warning("Race date is NULL for %s", race_id)
            return pd.DataFrame()

        # 日付を正規化
        race_date_norm = self.asof._normalize_date(race_date)
        if race_date_norm is None:
            logger.warning("Failed to parse race date for %s: %s", race_id, race_date)
            return pd.DataFrame()

        # レース属性
        place = race_row[2]
        surface = race_row[3]
        distance = race_row[4]
        track_condition = race_row[5]
        race_class = race_row[6]
        grade = race_row[7]
        race_no = race_row[8]
        course_turn = race_row[9]
        course_inout = race_row[10]
        head_count = race_row[11]

        # エンコーディング
        place_id = PLACE_MAP.get(place) if place else None
        surface_id = SURFACE_MAP.get(surface) if surface else None
        distance_cat = map_distance_to_cat(distance)
        track_condition_id = TRACK_CONDITION_MAP.get(track_condition) if track_condition else None
        race_class_id = map_class_to_id(race_class)
        grade_id = GRADE_MAP.get(grade) if grade else None
        course_turn_id = TURN_MAP.get(course_turn) if course_turn else None
        course_inout_id = INOUT_MAP.get(course_inout) if course_inout else None

        # 日付要素
        try:
            race_dt = datetime.strptime(race_date_norm, "%Y-%m-%d")
            race_year = race_dt.year
            race_month = race_dt.month
            race_day_of_week = race_dt.weekday()
        except:
            race_year = None
            race_month = None
            race_day_of_week = None

        # 出走馬を取得
        cursor = self.conn.execute("""
            SELECT
                horse_id,
                finish_order,
                last_3f,
                body_weight,
                body_weight_diff,
                passing_order,
                win_odds,
                popularity,
                prize_money,
                jockey_id,
                trainer_id,
                sex,
                age,
                weight,
                frame_no,
                horse_no
            FROM race_results
            WHERE race_id = ?
        """, (race_id,))
        entries = cursor.fetchall()

        if not entries:
            logger.warning("No entries found for race %s", race_id)
            return pd.DataFrame()

        field_size = head_count or len(entries)

        features_list = []

        for entry in entries:
            horse_id = entry[0]
            finish_order = entry[1]
            last_3f = entry[2]
            body_weight = entry[3]
            body_weight_diff = entry[4]
            passing_order = entry[5]
            win_odds = entry[6]
            popularity = entry[7]
            prize_money = entry[8]
            jockey_id = entry[9]
            trainer_id = entry[10]
            sex = entry[11]
            age = entry[12]
            weight_carried = entry[13]
            frame_no = entry[14]
            horse_no = entry[15]

            # ターゲット変数
            target_win = 1 if finish_order == 1 else 0 if finish_order else None
            target_in3 = 1 if finish_order and finish_order <= 3 else 0 if finish_order else None
            target_quinella = 1 if finish_order and finish_order <= 2 else 0 if finish_order else None

            # 基本特徴量
            features = {
                "race_id": race_id,
                "horse_id": horse_id,
                "race_date": race_date_norm,
                "target_win": target_win,
                "target_in3": target_in3,
                "target_quinella": target_quinella,

                # base_race
                "place": place,
                "place_id": place_id,
                "surface": surface,
                "surface_id": surface_id,
                "distance": distance,
                "distance_cat": distance_cat,
                "track_condition": track_condition,
                "track_condition_id": track_condition_id,
                "course_turn": course_turn,
                "course_turn_id": course_turn_id,
                "course_inout": course_inout,
                "course_inout_id": course_inout_id,
                "race_year": race_year,
                "race_month": race_month,
                "race_day_of_week": race_day_of_week,
                "race_no": race_no,
                "grade": grade,
                "grade_id": grade_id,
                "race_class": race_class,
                "race_class_id": race_class_id,
                "field_size": field_size,
                "waku": frame_no,
                "umaban": horse_no,
                "umaban_norm": horse_no / field_size if horse_no and field_size else None,
                "sex": sex,
                "sex_id": SEX_MAP.get(sex) if sex else None,
                "age": age,
                "weight_carried": weight_carried,
            }

            # 馬の as-of 統計
            horse_stats = self.asof.compute_horse_asof_stats(
                horse_id=horse_id,
                race_date=race_date_norm,
                distance_cat=distance_cat,
                surface=surface,
                track_condition=track_condition,
                place=place,
            )
            features.update(horse_stats)

            # 馬体重特徴量
            features["h_body_weight"] = body_weight
            features["h_body_weight_diff"] = body_weight_diff
            if body_weight and horse_stats.get("h_avg_body_weight"):
                features["h_body_weight_dev"] = body_weight - horse_stats["h_avg_body_weight"]
            else:
                features["h_body_weight_dev"] = None

            # 通過順位
            c1, c2, c3, c4 = self.parse_passing_order(passing_order)
            features["h_avg_corner1_pos"] = None  # 過去平均は別途計算必要
            features["h_avg_corner2_pos"] = None
            features["h_avg_corner3_pos"] = None
            features["h_avg_corner4_pos"] = None

            # ペース関連 (将来拡張用)
            features["h_avg_early_pos"] = None
            features["h_avg_late_pos"] = None
            features["h_pos_change_tendency"] = None
            features["h_recent3_last3f_rank"] = None
            features["h_hlap_early_ratio"] = None
            features["h_hlap_mid_ratio"] = None
            features["h_hlap_late_ratio"] = None
            features["h_hlap_finish_ratio"] = None
            features["h_running_style_id"] = None
            features["h_running_style_entropy"] = None

            # クラス関連
            features["h_highest_class_id"] = None  # 要計算
            features["h_class_progression"] = None
            features["h_n_class_wins"] = None
            features["h_n_class_in3"] = None
            features["h_class_win_rate"] = None
            features["h_n_g1_runs"] = None
            features["h_n_g2_runs"] = None
            features["h_n_g3_runs"] = None
            features["h_n_op_runs"] = None
            features["h_graded_in3_rate"] = None

            # 騎手の as-of 統計
            if jockey_id:
                jockey_stats = self.asof.compute_jockey_asof_stats(
                    jockey_id=jockey_id,
                    race_date=race_date_norm,
                    distance_cat=distance_cat,
                    surface=surface,
                    place=place,
                    horse_id=horse_id,
                )
                features.update(jockey_stats)
            else:
                # デフォルト値
                for key in ["j_n_starts", "j_n_wins", "j_win_rate", "j_in3_rate",
                           "j_n_starts_dist", "j_win_rate_dist", "j_in3_rate_dist",
                           "j_win_rate_surface", "j_in3_rate_surface",
                           "j_win_rate_course", "j_in3_rate_course",
                           "j_recent30d_win_rate", "j_recent30d_in3_rate", "j_recent30d_n_rides",
                           "jh_n_combos", "jh_combo_win_rate", "jh_combo_in3_rate"]:
                    features[key] = None

            # 調教師の as-of 統計
            if trainer_id:
                trainer_stats = self.asof.compute_trainer_asof_stats(
                    trainer_id=trainer_id,
                    race_date=race_date_norm,
                    distance_cat=distance_cat,
                    surface=surface,
                    place=place,
                    jockey_id=jockey_id,
                )
                features.update(trainer_stats)
            else:
                # デフォルト値
                for key in ["t_n_starts", "t_n_wins", "t_win_rate", "t_in3_rate",
                           "t_n_starts_dist", "t_win_rate_dist", "t_in3_rate_dist",
                           "t_win_rate_surface", "t_in3_rate_surface",
                           "t_win_rate_course", "t_in3_rate_course",
                           "t_recent30d_win_rate", "t_recent30d_in3_rate", "t_recent30d_n_entries",
                           "tj_n_combos", "tj_combo_win_rate", "tj_combo_in3_rate"]:
                    features[key] = None

            # 血統ハッシュ
            if include_pedigree:
                ped_info = self.load_pedigree(horse_id)

                # Sire-Dam Hash (512 dims)
                ped_hash = hash_pedigree_sire_dam(
                    ped_info["sire_name"],
                    ped_info["dam_name"],
                    ped_info["broodmare_sire_name"],
                    n_dims=512,
                )
                for i in range(512):
                    features[f"ped_hash_{i:03d}"] = ped_hash[i]

                # Ancestor Frequency Hash (128 dims)
                anc_hash = hash_ancestor_frequency(
                    ped_info["ancestors"],
                    n_dims=128,
                )
                for i in range(128):
                    features[f"anc_hash_{i:03d}"] = anc_hash[i]

            # 市場情報 (オプション)
            if include_market:
                features["market_win_odds"] = win_odds
                features["market_popularity"] = popularity
                features["market_odds_rank"] = None  # フィールド内順位は別途計算

            features_list.append(features)

        return pd.DataFrame(features_list)

    # =========================================================================
    # Build All Features
    # =========================================================================

    def build_all(
        self,
        start_year: int = 2021,
        end_year: int = 2024,
        include_pedigree: bool = True,
        include_market: bool = False,
        batch_size: int = 100,
    ) -> int:
        """
        全レースの特徴量を構築して feature_table_v4 に保存

        Args:
            start_year: 開始年
            end_year: 終了年
            include_pedigree: 血統ハッシュを含めるか
            include_market: 市場情報を含めるか
            batch_size: バッチサイズ

        Returns:
            挿入した行数
        """
        logger.info("=" * 80)
        logger.info("Building FeaturePack v1 (feature_table_v4)")
        logger.info("  Years: %d - %d", start_year, end_year)
        logger.info("  Include pedigree: %s", include_pedigree)
        logger.info("  Include market: %s", include_market)
        logger.info("=" * 80)

        # テーブル作成
        create_feature_table_v4(self.conn, include_pedigree_hash=include_pedigree)

        # 対象レース取得
        cursor = self.conn.execute("""
            SELECT DISTINCT race_id
            FROM races
            WHERE CAST(SUBSTR(race_id, 1, 4) AS INTEGER) BETWEEN ? AND ?
            ORDER BY race_id
        """, (start_year, end_year))
        race_ids = [row[0] for row in cursor.fetchall()]

        logger.info("Found %d races to process", len(race_ids))

        total_rows = 0
        batch_dfs = []

        for i, race_id in enumerate(race_ids, 1):
            try:
                df = self.build_features_for_race(
                    race_id=race_id,
                    include_pedigree=include_pedigree,
                    include_market=include_market,
                )

                if len(df) > 0:
                    batch_dfs.append(df)

                # バッチ保存
                if len(batch_dfs) >= batch_size:
                    batch_df = pd.concat(batch_dfs, ignore_index=True)
                    rows = self._insert_batch(batch_df)
                    total_rows += rows
                    batch_dfs = []

                    logger.info(
                        "Progress: %d/%d races (%.1f%%), %d rows",
                        i, len(race_ids), 100 * i / len(race_ids), total_rows
                    )

            except Exception as e:
                logger.error("Failed to process race %s: %s", race_id, e)
                continue

        # 残りのバッチを保存
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            rows = self._insert_batch(batch_df)
            total_rows += rows

        logger.info("=" * 80)
        logger.info("Completed: %d rows inserted into feature_table_v4", total_rows)
        logger.info("=" * 80)

        return total_rows

    def _insert_batch(self, df: pd.DataFrame) -> int:
        """バッチをテーブルに挿入"""
        if df is None or len(df) == 0:
            return 0

        # カラム名を取得
        columns = list(df.columns)

        # NaN を None に変換
        df = df.where(pd.notnull(df), None)

        # INSERT OR REPLACE
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"""
            INSERT OR REPLACE INTO feature_table_v4 ({", ".join(columns)})
            VALUES ({placeholders})
        """

        cursor = self.conn.cursor()
        rows = df.to_dict("records")

        for row in rows:
            values = [self._safe_value(row.get(col)) for col in columns]
            cursor.execute(sql, values)

        self.conn.commit()
        return len(rows)

    @staticmethod
    def _safe_value(v):
        """値を SQLite 用に変換"""
        if v is None:
            return None
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v


# =============================================================================
# Convenience Functions
# =============================================================================

def build_feature_table_v4(
    conn: sqlite3.Connection,
    start_year: int = 2021,
    end_year: int = 2024,
    include_pedigree: bool = True,
    include_market: bool = False,
) -> int:
    """
    feature_table_v4 を構築するコンビニエンス関数

    Args:
        conn: SQLite 接続
        start_year: 開始年
        end_year: 終了年
        include_pedigree: 血統ハッシュを含めるか
        include_market: 市場情報を含めるか

    Returns:
        挿入した行数
    """
    builder = FeatureBuilderV4(conn)
    return builder.build_all(
        start_year=start_year,
        end_year=end_year,
        include_pedigree=include_pedigree,
        include_market=include_market,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Build feature_table_v4")
    parser.add_argument("--db", default="netkeiba.db", help="Database path")
    parser.add_argument("--start-year", type=int, default=2021, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--no-pedigree", action="store_true", help="Exclude pedigree hash")
    parser.add_argument("--include-market", action="store_true", help="Include market features")

    args = parser.parse_args()

    if not Path(args.db).exists():
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    try:
        rows = build_feature_table_v4(
            conn=conn,
            start_year=args.start_year,
            end_year=args.end_year,
            include_pedigree=not args.no_pedigree,
            include_market=args.include_market,
        )
        logger.info("Inserted %d rows", rows)
    finally:
        conn.close()
