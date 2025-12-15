# -*- coding: utf-8 -*-
"""
feature_builder.py (v3.3.1 - Bug Fix Release)

競馬予想AI用の特徴量テーブル構築モジュール

v3.3.1の修正:
- NameError修正: load_base_tables()内でraces変数を定義
- IndexError修正: horse_past_runsに存在しないrace_idをemptyチェックで回避
- race_attrs_cacheに登録されないレースは安全にスキップ
- ログの改善（スキップ数を表示）

v3.3の改善:
- パフォーマンス最適化（race_attrs_cache）
- 型ヒントの完全化
- ログの改善（絵文字削除）
- 馬体重差分の計算追加

v3.2の改善:
- racesテーブルのsurface/distance/track_conditionがNULLの場合、
  horse_past_runsから自動補完する機能を追加

設計方針:
- DBスキーマの差異に強い実装
- 未来情報リーク対策を徹底維持
- 説明可能性を重視した豊富な特徴量
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


def has_required_columns(df: pd.DataFrame, columns: List[str]) -> bool:
    """DataFrame が指定したカラムをすべて持っているかを判定するヘルパー"""

    return isinstance(df, pd.DataFrame) and set(columns).issubset(df.columns)


def normalize_race_date_value(value: Any) -> Optional[str]:
    """日付っぽい値を "%Y/%m/%d" 形式の文字列に正規化する"""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, datetime):
        return value.strftime("%Y/%m/%d")

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y/%m/%d")

    try:
        value_str = str(value).strip()
        if not value_str:
            return None

        # デフォルトはスラッシュ区切り、ハイフン区切りも許容
        value_str = value_str.replace("-", "/")
        parsed = datetime.strptime(value_str, "%Y/%m/%d")
        return parsed.strftime("%Y/%m/%d")
    except Exception:
        try:
            parsed = pd.to_datetime(value)
            if pd.isna(parsed):
                return None
            return parsed.strftime("%Y/%m/%d")
        except Exception:
            return None


# ==============================================================================
# race_id から日付を抽出
# ==============================================================================

def extract_date_from_race_id(race_id: str) -> Optional[str]:
    """
    race_id から日付を抽出
    
    Args:
        race_id: レースID（例: "202301010101"）
    
    Returns:
        日付文字列（例: "2023/01/01"）、抽出失敗時はNone
    """
    if not race_id or len(race_id) < 8:
        return None
    
    try:
        year = race_id[0:4]
        month = race_id[4:6]
        day = race_id[6:8]
        
        datetime.strptime(f"{year}/{month}/{day}", "%Y/%m/%d")
        return f"{year}/{month}/{day}"
    except Exception as e:
        logger.warning("Failed to extract date from race_id=%s: %s", race_id, e)
        return None


# ==============================================================================
# カテゴリID化ヘルパー
# ==============================================================================

def encode_track_condition(condition: Optional[str]) -> Optional[int]:
    """馬場状態をカテゴリIDに変換（0:良, 1:稍, 2:重, 3:不）"""
    if condition is None or pd.isna(condition):
        return None
    
    mapping = {
        "良": 0, "稍": 1, "稍重": 1,
        "重": 2, "不": 3, "不良": 3,
    }
    return mapping.get(str(condition).strip(), None)


def encode_surface(surface: Optional[str]) -> Optional[int]:
    """馬場をカテゴリIDに変換（0:芝, 1:ダ, 2:障害）"""
    if surface is None or pd.isna(surface):
        return None
    
    surface_str = str(surface).strip()
    if "障" in surface_str:
        return 2
    elif "ダ" in surface_str:
        return 1
    elif "芝" in surface_str:
        return 0
    return None


# ==============================================================================
# 距離カテゴリマッピング
# ==============================================================================

def map_distance_to_cat(distance: Optional[int], surface: Optional[str] = None) -> Optional[int]:
    """距離を距離カテゴリに変換（障害レースは9000番台）"""
    if distance is None or pd.isna(distance):
        return None
    
    distance = int(distance)
    
    if surface and "障" in str(surface):
        if distance < 3000:
            return 9000
        elif distance < 3500:
            return 9300
        else:
            return 9600
    
    # 平地レース
    if distance == 1000:
        return 1000
    elif 1100 <= distance <= 1200:
        return 1200
    elif 1300 <= distance <= 1400:
        return 1400
    elif 1500 <= distance <= 1600:
        return 1600
    elif 1700 <= distance <= 1800:
        return 1800
    elif 1900 <= distance <= 2000:
        return 2000
    elif 2100 <= distance <= 2200:
        return 2200
    elif 2300 <= distance <= 2400:
        return 2400
    elif distance == 2500:
        return 2500
    elif distance >= 2600:
        return 3000
    else:
        logger.warning("Unexpected distance: %d", distance)
        return None


# ==============================================================================
# ベーステーブルのロード（v3.3.1 - バグ修正版）
# ==============================================================================

def load_base_tables(conn: sqlite3.Connection) -> Dict[str, Any]:
    """必要なベーステーブルをロード + race_attrs_cacheを構築（現行DBスキーマ対応版）

    Returns:
        tables: 以下のキーを含む辞書
            - "races": racesテーブル
            - "race_results": race_resultsテーブル
            - "horse_past_runs": horse_past_runsテーブル（なければ空DF）
            - "race_attrs_cache": race_id -> {surface, distance, track_condition} の辞書
            - "lap_times": レースラップ
            - "horse_laps": 馬ごとのラップ
    """
    logger.info("Loading base tables (modern schema compatible)...")

    tables: Dict[str, Any] = {}

    # ========================================
    # races テーブル（name / date / place を吸収）
    # ========================================
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(races)")
        races_columns = [row[1] for row in cur.fetchall()]

        select_exprs: list[str] = []

        # race_id は必須
        if "race_id" not in races_columns:
            raise RuntimeError("races table must have race_id column")
        select_exprs.append("race_id")

        # レース名: race_name または name を race_name に寄せる
        if "race_name" in races_columns:
            select_exprs.append("race_name")
        elif "name" in races_columns:
            select_exprs.append("name AS race_name")
        else:
            logger.warning("races table has no race_name/name column; race_name will be None")

        # 日付: race_date or date
        if "race_date" in races_columns:
            select_exprs.append("race_date")
        elif "date" in races_columns:
            select_exprs.append("date AS race_date")

        # コース: course or place
        if "course" in races_columns:
            select_exprs.append("course")
        elif "place" in races_columns:
            select_exprs.append("place AS course")

        # その他はあればそのまま
        for col in ["course_type", "distance", "track_condition", "race_class"]:
            if col in races_columns:
                select_exprs.append(col)

        select_sql = "SELECT " + ", ".join(select_exprs) + " FROM races"
        races_df = pd.read_sql_query(select_sql, conn)

        # 足りない列は明示的に追加しておく
        for col in ["race_name", "race_date", "course", "course_type", "distance", "track_condition", "race_class"]:
            if col not in races_df.columns:
                races_df[col] = None

        tables["races"] = races_df
        logger.info(
            "Loaded races: %d rows (columns: %s)",
            len(races_df),
            list(races_df.columns),
        )
    except Exception as e:
        logger.error("Failed to load races table: %s", e)
        tables["races"] = pd.DataFrame(
            columns=[
                "race_id",
                "race_name",
                "race_date",
                "course",
                "course_type",
                "distance",
                "track_condition",
                "race_class",
            ]
        )

    # ========================================
    # race_results テーブル（finish_order → position）
    # ========================================
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(race_results)")
        results_columns = [row[1] for row in cur.fetchall()]

        select_exprs: list[str] = []

        # race_id / horse_id / horse_name は必須
        for col in ["race_id", "horse_id", "horse_name"]:
            if col in results_columns:
                select_exprs.append(col)
            else:
                raise RuntimeError(f"race_results table must have {col} column")

        # 着順: position または finish_order を position に寄せる
        if "position" in results_columns:
            select_exprs.append("position")
        elif "finish_order" in results_columns:
            select_exprs.append("finish_order AS position")
        else:
            logger.warning("race_results has no position/finish_order; position will be NULL")

        optional_columns = [
            "race_date",
            "frame_no",
            "horse_no",
            "win_odds",
            "popularity",
            "body_weight",
            "jockey_name",
            "trainer_name",
            "race_class",
        ]
        for col in optional_columns:
            if col in results_columns:
                select_exprs.append(col)

        select_sql = "SELECT " + ", ".join(select_exprs) + " FROM race_results"
        results_df = pd.read_sql_query(select_sql, conn)

        if "position" not in results_df.columns:
            results_df["position"] = None

        tables["race_results"] = results_df
        logger.info(
            "Loaded race_results: %d rows (columns: %s)",
            len(results_df),
            list(results_df.columns),
        )
    except Exception as e:
        logger.error("Failed to load race_results table: %s", e)
        tables["race_results"] = pd.DataFrame(
            columns=["race_id", "horse_id", "horse_name", "position"]
        )

    # ========================================
    # horse_past_runs テーブル（無ければ空でOK）
    # ========================================
    try:
        tables["horse_past_runs"] = pd.read_sql_query(
            """
            SELECT 
                horse_id, race_id, race_date, place, weather, race_number, race_name,
                num_head, waku, umaban, odds, popularity, finish, jockey, weight_carried,
                surface, distance, track_condition, time, time_seconds, time_diff,
                pace_front_3f, pace_back_3f, passing, last_3f, horse_weight,
                horse_weight_diff, winner_name, prize
            FROM horse_past_runs
            """,
            conn,
        )
        logger.info("Loaded horse_past_runs: %d rows", len(tables["horse_past_runs"]))
    except Exception as e:
        logger.warning("Failed to load horse_past_runs table (may not exist yet): %s", e)
        tables["horse_past_runs"] = pd.DataFrame()

    # ========================================
    # lap_times / horse_laps
    # ========================================
    try:
        tables["lap_times"] = pd.read_sql_query(
            "SELECT race_id, lap_index, distance_m, time_sec FROM lap_times",
            conn,
        )
        logger.info("Loaded lap_times: %d rows", len(tables["lap_times"]))
    except Exception as e:
        logger.warning("Failed to load lap_times table: %s", e)
        tables["lap_times"] = pd.DataFrame(
            columns=["race_id", "lap_index", "distance_m", "time_sec"]
        )

    try:
        tables["horse_laps"] = pd.read_sql_query(
            "SELECT race_id, horse_id, section_m, time_sec, position FROM horse_laps",
            conn,
        )
        logger.info("Loaded horse_laps: %d rows", len(tables["horse_laps"]))
    except Exception as e:
        logger.warning("Failed to load horse_laps table: %s", e)
        tables["horse_laps"] = pd.DataFrame(
            columns=["race_id", "horse_id", "section_m", "time_sec", "position"]
        )

    # ========================================
    # race_attrs_cache 構築
    #   horse_past_runs があれば従来ロジック
    #   無ければ races から直接 surface/distance/track_condition を使う
    # ========================================
    logger.info("Building race_attrs_cache...")

    race_attrs_cache: Dict[str, Dict[str, Optional[Any]]] = {}

    races_df = tables["races"]
    horse_past = tables["horse_past_runs"]

    if not horse_past.empty:
        skipped_count = 0
        for race_id in races_df["race_id"].unique():
            race_past = horse_past[horse_past["race_id"] == race_id]
            if race_past.empty:
                skipped_count += 1
                continue

            race_data = race_past.iloc[0]
            race_attrs_cache[race_id] = {
                "surface": race_data["surface"] if pd.notna(race_data["surface"]) else None,
                "distance": int(race_data["distance"]) if pd.notna(race_data["distance"]) else None,
                "track_condition": race_data["track_condition"] if pd.notna(race_data["track_condition"]) else None,
            }

        logger.info(
            "Built race_attrs_cache from horse_past_runs: %d races cached, %d races skipped",
            len(race_attrs_cache),
            skipped_count,
        )
    else:
        skipped_count = 0
        for _, row in races_df.iterrows():
            race_id = row["race_id"]
            surface = row.get("course_type", None)
            distance = row.get("distance", None)
            track_cond = row.get("track_condition", None)

            if pd.isna(surface):
                surface = None
            if pd.isna(distance):
                distance = None
            else:
                try:
                    distance = int(distance)
                except Exception:
                    distance = None
            if pd.isna(track_cond):
                track_cond = None

            if surface is None and distance is None and track_cond is None:
                skipped_count += 1
                continue

            race_attrs_cache[race_id] = {
                "surface": surface,
                "distance": distance,
                "track_condition": track_cond,
            }

        logger.info(
            "Built race_attrs_cache from races table: %d races cached, %d races skipped",
            len(race_attrs_cache),
            skipped_count,
        )

    tables["race_attrs_cache"] = race_attrs_cache

    return tables


# ==============================================================================
# 補完ロジック
# ==============================================================================

def infer_course_from_past_runs(
    race_id: str,
    horse_ids: List[str],
    horse_past_runs: pd.DataFrame,
) -> Optional[str]:
    """
    horse_past_runs から course（競馬場）を推定
    
    注意: これはレース自体の開催場所を取得するもので、未来情報リークではない
    """
    if horse_past_runs is None or horse_past_runs.empty:
        return None

    if not has_required_columns(horse_past_runs, ["horse_id", "race_id", "place"]):
        return None

    past_places = horse_past_runs[
        (horse_past_runs["horse_id"].isin(horse_ids)) &
        (horse_past_runs["race_id"] == race_id)
    ]["place"].dropna()
    
    if len(past_places) > 0:
        return past_places.iloc[0] if len(past_places) > 0 else None
    return None


def infer_race_class_from_results(
    race_id: str,
    race_results: pd.DataFrame,
) -> Optional[str]:
    """race_results から race_class を推定"""
    if "race_class" in race_results.columns:
        race_class_values = race_results[
            race_results["race_id"] == race_id
        ]["race_class"].dropna()
        
        if len(race_class_values) > 0:
            return race_class_values.iloc[0]
    return None


def get_race_attributes_from_cache(
    race_id: str,
    race_attrs_cache: Dict[str, Dict[str, Optional[Any]]],
) -> Dict[str, Optional[Any]]:
    """
    キャッシュから surface / distance / track_condition を取得
    
    v3.3: パフォーマンス最適化のため、事前構築したキャッシュを使用
    v3.3.1: キャッシュに存在しない場合のデフォルト値を返す
    """
    return race_attrs_cache.get(race_id, {
        "surface": None,
        "distance": None,
        "track_condition": None,
    })


# ==============================================================================
# 特徴量計算関数
# ==============================================================================

def compute_global_stats(
    past_runs: pd.DataFrame,
    target_date: str,
) -> Dict[str, Optional[float]]:
    """グローバル能力指標を計算"""
    if len(past_runs) == 0:
        return {
            "n_starts_total": 0,
            "win_rate_total": 0.0,
            "in3_rate_total": 0.0,
            "avg_finish_total": None,
            "std_finish_total": None,
        }
    
    n_starts = len(past_runs)
    wins = (past_runs["finish"] == 1).sum()
    in3 = (past_runs["finish"] <= 3).sum()
    
    finish_values = past_runs["finish"].dropna()
    avg_finish = finish_values.mean() if len(finish_values) > 0 else None
    std_finish = finish_values.std() if len(finish_values) > 1 else None
    
    return {
        "n_starts_total": n_starts,
        "win_rate_total": wins / n_starts if n_starts > 0 else 0.0,
        "in3_rate_total": in3 / n_starts if n_starts > 0 else 0.0,
        "avg_finish_total": avg_finish,
        "std_finish_total": std_finish,
    }


def compute_distance_cat_stats(
    past_runs: pd.DataFrame,
    distance_cat: int,
) -> Dict[str, Optional[float]]:
    """距離カテゴリ別能力指標を計算"""
    if len(past_runs) == 0:
        return {
            "n_starts_dist_cat": 0,
            "win_rate_dist_cat": 0.0,
            "in3_rate_dist_cat": 0.0,
            "avg_finish_dist_cat": None,
            "avg_last3f_dist_cat": None,
        }
    
    n_starts = len(past_runs)
    wins = (past_runs["finish"] == 1).sum()
    in3 = (past_runs["finish"] <= 3).sum()
    
    finish_values = past_runs["finish"].dropna()
    avg_finish = finish_values.mean() if len(finish_values) > 0 else None
    
    last3f_values = past_runs["last_3f"].dropna()
    avg_last3f = last3f_values.mean() if len(last3f_values) > 0 else None
    
    return {
        "n_starts_dist_cat": n_starts,
        "win_rate_dist_cat": wins / n_starts if n_starts > 0 else 0.0,
        "in3_rate_dist_cat": in3 / n_starts if n_starts > 0 else 0.0,
        "avg_finish_dist_cat": avg_finish,
        "avg_last3f_dist_cat": avg_last3f,
    }


def compute_recent_form(
    past_runs: pd.DataFrame,
    target_date: datetime,
    n_recent: int = 3,
) -> Dict[str, Optional[float]]:
    """直近フォームを計算"""
    if len(past_runs) == 0:
        return {
            "days_since_last_run": None,
            "recent_avg_finish_3": None,
            "recent_best_finish_3": None,
            "recent_avg_last3f_3": None,
        }
    
    past_runs = past_runs.sort_values("race_date", ascending=False)
    
    last_race_date_str = past_runs.iloc[0]["race_date"]
    try:
        last_race_dt = datetime.strptime(last_race_date_str, "%Y/%m/%d")
        days_since_last = (target_date - last_race_dt).days
    except Exception as e:
        logger.warning("Failed to parse last_race_date: %s, error: %s", last_race_date_str, e)
        days_since_last = None
    
    recent = past_runs.head(n_recent)
    
    finish_values = recent["finish"].dropna()
    avg_finish = finish_values.mean() if len(finish_values) > 0 else None
    best_finish = finish_values.min() if len(finish_values) > 0 else None
    
    last3f_values = recent["last_3f"].dropna()
    avg_last3f = last3f_values.mean() if len(last3f_values) > 0 else None
    
    return {
        "days_since_last_run": days_since_last,
        "recent_avg_finish_3": avg_finish,
        "recent_best_finish_3": best_finish,
        "recent_avg_last3f_3": avg_last3f,
    }


def compute_track_condition_stats(
    past_runs: pd.DataFrame,
    track_condition: str,
) -> Dict[str, float]:
    """馬場状態別能力指標を計算"""
    if len(past_runs) == 0:
        return {
            "n_starts_track_condition": 0,
            "win_rate_track_condition": 0.0,
        }
    
    n_starts = len(past_runs)
    wins = (past_runs["finish"] == 1).sum()
    
    return {
        "n_starts_track_condition": n_starts,
        "win_rate_track_condition": wins / n_starts if n_starts > 0 else 0.0,
    }


def compute_course_stats(
    past_runs: pd.DataFrame,
    course: str,
) -> Dict[str, float]:
    """コース別能力指標を計算"""
    if len(past_runs) == 0:
        return {
            "n_starts_course": 0,
            "win_rate_course": 0.0,
        }
    
    n_starts = len(past_runs)
    wins = (past_runs["finish"] == 1).sum()
    
    return {
        "n_starts_course": n_starts,
        "win_rate_course": wins / n_starts if n_starts > 0 else 0.0,
    }


def compute_horse_weight_stats(
    past_runs: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    """馬体重関連の統計を計算"""
    if len(past_runs) == 0:
        return {"avg_horse_weight": None}
    
    weight_values = past_runs["horse_weight"].dropna()
    avg_weight = weight_values.mean() if len(weight_values) > 0 else None
    
    return {"avg_horse_weight": avg_weight}


def parse_body_weight(body_weight: Any) -> Optional[int]:
    """
    body_weightをパース（"502(-4)"のような文字列に対応）
    
    v3.3: 馬体重差分計算のために追加
    """
    if body_weight is None or pd.isna(body_weight):
        return None
    
    try:
        if isinstance(body_weight, str):
            match = re.match(r'(\d+)', body_weight)
            if match:
                return int(match.group(1))
        else:
            return int(body_weight)
    except:
        return None
    
    return None

def compute_lap_ratio_features_for_race(
    race_id: str,
    horse_laps: pd.DataFrame,
    lap_times: pd.DataFrame,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    hlap_* ラップ比特徴量は現行バージョンでは未使用のため、空を返す。
    """
    return {}


# ==============================================================================
# レース単位での特徴量構築（v3.3.1）
# ==============================================================================

def build_features_for_race(
    race_id: str,
    race_date_str: str,
    tables: Dict[str, Any],
) -> pd.DataFrame:
    """
    1レース分の特徴量を構築
    
    v3.3.1: race_attrs_cacheに存在しないrace_idに対する安全性を向上
    """
    logger.debug("Building features for race_id=%s, race_date=%s", race_id, race_date_str)
    
    # 日付をdatetimeに変換
    try:
        race_date_norm = normalize_race_date_value(race_date_str)
        race_date = datetime.strptime(race_date_norm, "%Y/%m/%d") if race_date_norm else None
    except Exception as e:
        race_date = None

    if race_date is None:
        logger.warning("Failed to parse race_date: %s, using default", race_date_str)
        race_date = datetime(2000, 1, 1)
    
    # レース情報を取得
    race_info = tables["races"][tables["races"]["race_id"] == race_id]
    if len(race_info) == 0:
        logger.warning("Race not found in races table: race_id=%s", race_id)
        race_info = pd.Series({
            "race_id": race_id,
            "race_name": None,
            "course": None,
            "course_type": None,
            "distance": None,
            "track_condition": None,
            "race_class": None,
        })
    else:
        race_info = race_info.iloc[0]
    
    # レース結果を取得
    race_results = tables["race_results"][tables["race_results"]["race_id"] == race_id]
    if len(race_results) == 0:
        logger.warning("No results found for race_id=%s", race_id)
        return pd.DataFrame()
    
    # courseを補完
    course = race_info.get("course", None)
    if course is None or pd.isna(course):
        horse_ids = race_results["horse_id"].tolist()
        horse_past_runs = tables.get("horse_past_runs", pd.DataFrame())
        if has_required_columns(horse_past_runs, ["horse_id", "race_id", "place"]) and not horse_past_runs.empty:
            course = infer_course_from_past_runs(race_id, horse_ids, horse_past_runs)
            if course:
                logger.debug("Inferred course: %s", course)
    
    # surface / distance / track_condition を取得
    surface = race_info.get("course_type", None)
    distance = race_info.get("distance", None)
    track_condition = race_info.get("track_condition", None)
    
    # ★ v3.3.1: キャッシュから補完（安全なget）
    補完_needed = (surface is None or pd.isna(surface)) or \
                   (distance is None or pd.isna(distance)) or \
                   (track_condition is None or pd.isna(track_condition))
    
    if 補完_needed:
        logger.info("Inferring race attributes from cache for race_id=%s", race_id)
        inferred_attrs = get_race_attributes_from_cache(race_id, tables["race_attrs_cache"])
        
        if surface is None or pd.isna(surface):
            surface = inferred_attrs["surface"]
            if surface:
                logger.info("  [OK] Inferred surface: %s", surface)
            else:
                logger.warning("  [WARN] Could not infer surface for race_id=%s", race_id)
        
        if distance is None or pd.isna(distance):
            distance = inferred_attrs["distance"]
            if distance:
                logger.info("  [OK] Inferred distance: %d", distance)
            else:
                logger.warning("  [WARN] Could not infer distance for race_id=%s", race_id)
        
        if track_condition is None or pd.isna(track_condition):
            track_condition = inferred_attrs["track_condition"]
            if track_condition:
                logger.info("  [OK] Inferred track_condition: %s", track_condition)
            else:
                logger.warning("  [WARN] Could not infer track_condition for race_id=%s", race_id)
    
    # race_classを補完
    race_class = race_info.get("race_class", None)
    if race_class is None or pd.isna(race_class):
        race_class = infer_race_class_from_results(race_id, tables["race_results"])
    
    # 距離カテゴリ
    distance_cat = map_distance_to_cat(distance, surface)
    
    # カテゴリID化
    surface_id = encode_surface(surface)
    track_condition_id = encode_track_condition(track_condition)
    
    # レース側特徴量
    field_size = len(race_results)
    race_year = race_date.year
    race_month = race_date.month

    # 各馬の特徴量を構築
    features_list = []

    horse_past_runs = tables.get("horse_past_runs", pd.DataFrame())
    past_run_required_cols = [
        "horse_id",
        "race_date",
        "distance",
        "surface",
        "track_condition",
        "place",
        "finish",
        "last_3f",
        "horse_weight",
    ]
    has_past_run_data = (
        not getattr(horse_past_runs, "empty", True)
        and has_required_columns(horse_past_runs, past_run_required_cols)
    )

    for _, row in race_results.iterrows():
        horse_id = row["horse_id"]

        # ターゲット変数
        target_win = 1 if row["position"] == 1 else 0
        target_in3 = 1 if row["position"] <= 3 else 0

        win_odds = row.get("win_odds", None)
        if target_in3 == 1 and win_odds is not None and win_odds >= 10.0:
            target_value = 1
        else:
            target_value = 0

        if has_past_run_data:
            horse_past_all = horse_past_runs[
                horse_past_runs["horse_id"] == horse_id
            ].copy()

            horse_past_all["race_date_dt"] = pd.to_datetime(
                horse_past_all["race_date"],
                format="%Y/%m/%d",
                errors="coerce",
            )

            # ★ datetime比較で未来情報を完全排除
            horse_past = horse_past_all[
                horse_past_all["race_date_dt"] < race_date
            ].copy()

            # 新馬フラグ
            is_first_run = 1 if len(horse_past) == 0 else 0

            # 距離カテゴリを追加
            horse_past["distance_cat"] = horse_past.apply(
                lambda x: map_distance_to_cat(x["distance"], x["surface"]),
                axis=1
            )
        else:
            horse_past = pd.DataFrame()
            is_first_run = None

        # 各種統計
        global_stats = compute_global_stats(horse_past, race_date_str)

        horse_past_dist = (
            horse_past[horse_past["distance_cat"] == distance_cat]
            if "distance_cat" in horse_past.columns
            else horse_past
        )
        dist_cat_stats = compute_distance_cat_stats(horse_past_dist, distance_cat)

        recent_form = compute_recent_form(horse_past, race_date, n_recent=3)

        if track_condition:
            horse_past_track = (
                horse_past[horse_past["track_condition"] == track_condition]
                if "track_condition" in horse_past.columns
                else horse_past
            )
            track_stats = compute_track_condition_stats(horse_past_track, track_condition)
        else:
            track_stats = {
                "n_starts_track_condition": 0,
                "win_rate_track_condition": 0.0,
            }

        if course:
            horse_past_course = (
                horse_past[horse_past["place"] == course]
                if "place" in horse_past.columns
                else horse_past
            )
            course_stats = compute_course_stats(horse_past_course, course)
        else:
            course_stats = {
                "n_starts_course": 0,
                "win_rate_course": 0.0,
            }

        weight_stats = compute_horse_weight_stats(horse_past)

        # ★ v3.3: 馬体重差分を計算
        current_weight = parse_body_weight(row.get("body_weight", None))
        avg_weight = weight_stats.get("avg_horse_weight", None)

        weight_diff = None
        if current_weight is not None and avg_weight is not None:
            weight_diff = current_weight - avg_weight
        
        # 特徴量を統合
        features = {
            "race_id": race_id,
            "horse_id": horse_id,
            "target_win": target_win,
            "target_in3": target_in3,
            "target_value": target_value,
            "course": course,
            "surface": surface,
            "surface_id": surface_id,
            "distance": distance,
            "distance_cat": distance_cat,
            "track_condition": track_condition,
            "track_condition_id": track_condition_id,
            "field_size": field_size,
            "race_class": race_class,
            "race_year": race_year,
            "race_month": race_month,
            "waku": row.get("frame_no", None),
            "umaban": row.get("horse_no", None),
            "horse_weight": current_weight,
            "horse_weight_diff": weight_diff,
            "is_first_run": is_first_run,
            **global_stats,
            **dist_cat_stats,
            **recent_form,
            **track_stats,
            **course_stats,
            **weight_stats,
        }

        features_list.append(features)
    
    df_features = pd.DataFrame(features_list)
    logger.debug("Built features for %d horses in race_id=%s", len(df_features), race_id)
    
    return df_features


# ==============================================================================
# 全レースの特徴量構築（v3.3.1）
# ==============================================================================

def build_feature_table(
    conn: sqlite3.Connection,
    target_race_ids: Optional[List[str]] = None,
) -> int:
    """
    feature_table を構築
    
    v3.3.1: race_attrs_cacheに存在しないrace_idは安全にスキップ
    """
    from .sqlite_store_feature import create_table_feature, insert_feature_rows
    
    logger.info("=" * 80)
    logger.info("Building feature_table (v3.3.1)...")
    logger.info("=" * 80)
    
    create_table_feature(conn)
    
    tables = load_base_tables(conn)
    
    if target_race_ids is None:
        race_ids = tables["race_results"]["race_id"].unique()
        logger.info("Target: all races (%d races)", len(race_ids))
    else:
        race_ids = target_race_ids
        logger.info("Target: %d races", len(race_ids))
    
    # ★ v3.3.1: race_attrs_cacheに存在しないrace_idを除外（空なら全件実行）
    race_attrs_cache = tables["race_attrs_cache"]
    if race_attrs_cache:
        available_race_ids = [rid for rid in race_ids if rid in race_attrs_cache]
        skipped_race_ids = [rid for rid in race_ids if rid not in race_attrs_cache]

        if skipped_race_ids:
            logger.warning(
                "Skipping %d races (not in race_attrs_cache): %s",
                len(skipped_race_ids),
                skipped_race_ids[:5] if len(skipped_race_ids) > 5 else skipped_race_ids
            )
    else:
        available_race_ids = list(race_ids)
        skipped_race_ids = []
        logger.info("race_attrs_cache is empty; processing all %d races", len(available_race_ids))

    if not available_race_ids:
        logger.warning("No races available after cache filtering; falling back to all races")
        available_race_ids = list(race_ids)

    logger.info("Processing %d races", len(available_race_ids))
    
    # race_date を補完
    race_date_map = {}
    
    for race_id in available_race_ids:
        race_date = None
        
        # 優先度1: races
        if "race_date" in tables["races"].columns:
            race_rows = tables["races"][tables["races"]["race_id"] == race_id]
            if len(race_rows) > 0:
                race_date = race_rows.iloc[0]["race_date"]
                if race_date is not None and not pd.isna(race_date):
                    logger.debug("Using race_date from races: %s", race_date)

        # 優先度2: race_results
        if race_date is None or pd.isna(race_date):
            if "race_date" in tables["race_results"].columns:
                result_rows = tables["race_results"][tables["race_results"]["race_id"] == race_id]
                if len(result_rows) > 0:
                    race_date = result_rows.iloc[0]["race_date"]
                    if race_date is not None and not pd.isna(race_date):
                        logger.debug("Using race_date from race_results: %s", race_date)

        # 優先度3: horse_past_runs
        if race_date is None or pd.isna(race_date):
            horse_past_runs = tables.get("horse_past_runs", pd.DataFrame())
            if has_required_columns(horse_past_runs, ["race_id", "race_date"]):
                past_rows = horse_past_runs[horse_past_runs["race_id"] == race_id]
                if len(past_rows) > 0:
                    race_date = past_rows.iloc[0]["race_date"]
                    if race_date is not None and not pd.isna(race_date):
                        logger.debug("Using race_date from horse_past_runs: %s", race_date)

        # 優先度4: race_idから抽出
        if race_date is None or pd.isna(race_date):
            race_date = extract_date_from_race_id(race_id)
            if race_date:
                logger.info("Extracted race_date from race_id=%s: %s", race_id, race_date)

        race_date_norm = normalize_race_date_value(race_date)

        if race_date_norm is not None:
            race_date_map[race_id] = race_date_norm
        else:
            logger.warning("Could not determine race_date for race_id=%s, skipping", race_id)
    
    # 各レースの特徴量を構築
    total_rows = 0
    for i, race_id in enumerate(available_race_ids, 1):
        if race_id not in race_date_map:
            continue
        
        race_date = race_date_map[race_id]
        
        try:
            df_features = build_features_for_race(race_id, race_date, tables)
            if len(df_features) > 0:
                insert_feature_rows(conn, df_features)
                total_rows += len(df_features)
                
                if i % 100 == 0:
                    logger.info("Progress: %d / %d races processed, %d rows inserted", i, len(available_race_ids), total_rows)
        except Exception as e:
            logger.error("Failed to build features for race_id=%s: %s", race_id, e, exc_info=True)
    
    logger.info("=" * 80)
    logger.info("Feature table built: %d rows", total_rows)
    logger.info("=" * 80)
    
    return total_rows
