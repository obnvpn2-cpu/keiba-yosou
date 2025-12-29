#!/usr/bin/env python3
"""
generate_pre_race_materials.py - Pre-race Materials Generator

前日運用向けの予測材料を生成するスクリプト。
指定された日付のレース予測をJSON形式で出力する。

【目的】
- 前日までに確定している情報のみを使用して予測
- 当日スマホで判断材料として使用可能なJSON生成

【出力形式】
- summary_YYYY-MM-DD.json: 開催日まとめ
- race_<race_id>.json: レースごとの詳細

Usage:
    python scripts/generate_pre_race_materials.py \
        --db netkeiba.db \
        --date 2024-05-12 \
        --out artifacts/pre_race/2024-05-12
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_v4.train_eval_v4 import (
    load_feature_data,
    get_train_feature_columns,
    log_feature_selection_summary,
)
from src.features_v4.feature_table_v4 import get_feature_v4_columns

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading
# =============================================================================

def load_booster(model_path: str) -> "lgb.Booster":
    """
    LightGBM Boosterをロードする（model_str フォールバック付き）
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # まず model_file でロードを試みる
    try:
        model = lgb.Booster(model_file=str(path))
        logger.info("Loaded model via model_file: %s", path)
        return model
    except Exception as e:
        logger.warning("lgb.Booster(model_file=...) failed (%s), trying model_str fallback...", e)

    # フォールバック: Python read → model_str
    try:
        model_str = path.read_text(encoding="utf-8")
        model = lgb.Booster(model_str=model_str)
        logger.info("Loaded model via model_str fallback: %s", path)
        return model
    except Exception as e2:
        raise RuntimeError(
            f"Failed to load model from {model_path}. "
            f"model_str error: {e2}"
        ) from e2


def load_feature_columns(models_dir: str, target: str) -> List[str]:
    """
    特徴量カラムをロード（v4 → legacy フォールバック）
    """
    v4_path = os.path.join(models_dir, f"feature_columns_{target}_v4.json")
    legacy_path = os.path.join(models_dir, f"feature_columns_{target}.json")

    if os.path.exists(v4_path):
        with open(v4_path, "r") as f:
            return json.load(f)
    elif os.path.exists(legacy_path):
        with open(legacy_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(
            f"Feature columns file not found. Tried:\n  - {v4_path}\n  - {legacy_path}"
        )


def load_exclude_features(filepath: str) -> Set[str]:
    """
    除外特徴量リストを読み込む
    """
    exclude_set = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            exclude_set.add(line)
    return exclude_set


# =============================================================================
# Data Loading
# =============================================================================

def get_races_for_date(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    """
    指定日のレース一覧を取得
    """
    sql = """
    SELECT
        race_id,
        date,
        place,
        race_no,
        name,
        grade,
        race_class,
        course_type,
        distance,
        track_condition,
        head_count,
        start_time
    FROM races
    WHERE date = ?
    ORDER BY race_id
    """
    return pd.read_sql_query(sql, conn, params=[date])


def get_race_entries(conn: sqlite3.Connection, race_id: str) -> pd.DataFrame:
    """
    指定レースの出走馬一覧を取得
    """
    sql = """
    SELECT
        horse_id,
        horse_no,
        horse_name,
        jockey_name,
        trainer_name,
        sex,
        age
    FROM race_results
    WHERE race_id = ?
    ORDER BY horse_no
    """
    return pd.read_sql_query(sql, conn, params=[race_id])


def get_feature_data_for_date(
    conn: sqlite3.Connection,
    date: str,
    include_pedigree: bool = True,
) -> pd.DataFrame:
    """
    指定日の特徴量データを取得
    """
    sql = """
    SELECT *
    FROM feature_table_v4
    WHERE race_date = ?
    ORDER BY race_id, umaban
    """
    return pd.read_sql_query(sql, conn, params=[date])


# =============================================================================
# Prediction
# =============================================================================

def predict_race(
    model: "lgb.Booster",
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """
    モデルで予測実行
    """
    # 特徴量の準備
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].fillna(-999)

    # 足りないカラムは警告
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        logger.warning("Missing %d feature columns in data: %s",
                       len(missing_cols), list(missing_cols)[:5])

    # 予測
    probs = model.predict(X)
    return probs


# =============================================================================
# Output Generation
# =============================================================================

def generate_race_json(
    race_info: Dict[str, Any],
    entries_df: pd.DataFrame,
    predictions: Dict[str, Dict[str, float]],
    feature_version: str = "v4",
) -> Dict[str, Any]:
    """
    レースごとのJSON出力を生成
    """
    race_id = race_info["race_id"]

    # 馬ごとのエントリー
    entries = []
    for _, row in entries_df.iterrows():
        horse_id = row["horse_id"]
        pred = predictions.get(horse_id, {})

        entry = {
            "horse_id": horse_id,
            "umaban": int(row["horse_no"]) if pd.notna(row["horse_no"]) else None,
            "name": row["horse_name"],
            "jockey": row.get("jockey_name"),
            "trainer": row.get("trainer_name"),
            "sex": row.get("sex"),
            "age": int(row["age"]) if pd.notna(row.get("age")) else None,
            "p_win": pred.get("p_win"),
            "p_in3": pred.get("p_in3"),
            "rank_win": pred.get("rank_win"),
            "rank_in3": pred.get("rank_in3"),
        }
        entries.append(entry)

    # rank でソート
    entries.sort(key=lambda x: x.get("rank_win") or 999)

    return {
        "race_id": race_id,
        "date": race_info.get("date"),
        "place": race_info.get("place"),
        "race_no": race_info.get("race_no"),
        "name": race_info.get("name"),
        "grade": race_info.get("grade"),
        "race_class": race_info.get("race_class"),
        "course": race_info.get("course_type"),
        "distance": race_info.get("distance"),
        "track_condition": race_info.get("track_condition"),
        "head_count": race_info.get("head_count"),
        "start_time": race_info.get("start_time"),
        "entries": entries,
        "features_version": feature_version,
        "mode": "pre_race",
        "generated_at": datetime.now().isoformat(),
    }


def generate_summary_json(
    date: str,
    races: List[Dict[str, Any]],
    n_features: int,
    models_used: List[str],
) -> Dict[str, Any]:
    """
    開催日サマリーのJSON出力を生成
    """
    return {
        "date": date,
        "n_races": len(races),
        "races": races,
        "n_features_used": n_features,
        "models_used": models_used,
        "mode": "pre_race",
        "features_version": "v4",
        "generated_at": datetime.now().isoformat(),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-race prediction materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate materials for a specific date
  python scripts/generate_pre_race_materials.py \\
      --db netkeiba.db \\
      --date 2024-05-12 \\
      --out artifacts/pre_race/2024-05-12

  # Use specific models directory
  python scripts/generate_pre_race_materials.py \\
      --db netkeiba.db \\
      --date 2024-05-12 \\
      --out artifacts/pre_race/2024-05-12 \\
      --models-dir models/pre_race

  # Only generate target_win predictions
  python scripts/generate_pre_race_materials.py \\
      --db netkeiba.db \\
      --date 2024-05-12 \\
      --include-targets target_win
"""
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB (e.g., netkeiba.db)",
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Target date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models (default: models)",
    )
    parser.add_argument(
        "--include-targets",
        type=str,
        default="target_win,target_in3",
        help="Comma-separated list of targets to include (default: target_win,target_in3)",
    )
    parser.add_argument(
        "--no-pedigree",
        action="store_true",
        help="Exclude pedigree features",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if not HAS_LIGHTGBM:
        logger.error("lightgbm is not installed")
        sys.exit(1)

    # Validate database path
    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Parse targets
    targets = [t.strip() for t in args.include_targets.split(",")]

    # Load pre_race exclude features
    pre_race_exclude_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "exclude_features", "pre_race.txt"
    )
    if os.path.exists(pre_race_exclude_path):
        exclude_set = load_exclude_features(pre_race_exclude_path)
        logger.info("Loaded %d exclude features from: %s", len(exclude_set), pre_race_exclude_path)
    else:
        logger.warning("Pre-race exclude file not found: %s", pre_race_exclude_path)
        exclude_set = set()

    logger.info("=" * 70)
    logger.info("Pre-race Materials Generator")
    logger.info("=" * 70)
    logger.info("  Database:    %s", db_path)
    logger.info("  Date:        %s", args.date)
    logger.info("  Output:      %s", args.out)
    logger.info("  Models:      %s", args.models_dir)
    logger.info("  Targets:     %s", targets)
    logger.info("  Exclude:     %d features", len(exclude_set))
    logger.info("=" * 70)

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        # Load models for each target
        models = {}
        feature_cols_map = {}
        models_used = []

        for target in targets:
            model_path = os.path.join(args.models_dir, f"lgbm_{target}_v4.txt")

            # Try fallback paths
            if not os.path.exists(model_path):
                fallback_path = os.path.join(args.models_dir, f"lgbm_{target}.txt")
                if os.path.exists(fallback_path):
                    model_path = fallback_path

            if not os.path.exists(model_path):
                logger.warning("Model not found for %s: %s", target, model_path)
                continue

            logger.info("Loading model for %s: %s", target, model_path)
            models[target] = load_booster(model_path)
            models_used.append(model_path)

            # Load feature columns
            feature_cols = load_feature_columns(args.models_dir, target)

            # Apply pre_race exclusion
            candidate_features = feature_cols
            feature_cols = [c for c in candidate_features if c not in exclude_set]

            # Log feature selection
            log_feature_selection_summary(
                mode="pre_race",
                target_col=target,
                candidate_features=candidate_features,
                exclude_set=exclude_set,
                final_features=feature_cols,
                output_dir=str(out_dir),
            )

            feature_cols_map[target] = feature_cols

        if not models:
            logger.error("No models could be loaded")
            sys.exit(1)

        # Get races for the date
        races_df = get_races_for_date(conn, args.date)
        logger.info("Found %d races for date %s", len(races_df), args.date)

        if len(races_df) == 0:
            logger.warning("No races found for date %s", args.date)
            # Still generate empty summary
            summary = generate_summary_json(
                date=args.date,
                races=[],
                n_features=0,
                models_used=models_used,
            )
            summary_path = out_dir / f"summary_{args.date}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info("Saved empty summary to: %s", summary_path)
            return

        # Get feature data for the date
        logger.info("Loading feature data...")
        feature_df = get_feature_data_for_date(conn, args.date, include_pedigree=not args.no_pedigree)
        logger.info("Loaded %d entries from feature_table_v4", len(feature_df))

        if len(feature_df) == 0:
            logger.error("No feature data found for date %s. Run build_feature_table_v4.py first.", args.date)
            sys.exit(1)

        # Process each race
        race_summaries = []
        n_features_used = 0

        for _, race_row in races_df.iterrows():
            race_id = race_row["race_id"]
            logger.info("Processing race: %s (%s R%d)", race_id, race_row["place"], race_row["race_no"])

            # Get entries
            entries_df = get_race_entries(conn, race_id)
            if len(entries_df) == 0:
                logger.warning("No entries found for race %s", race_id)
                continue

            # Get feature data for this race
            race_feature_df = feature_df[feature_df["race_id"] == race_id]
            if len(race_feature_df) == 0:
                logger.warning("No feature data for race %s", race_id)
                continue

            # Predict for each target
            predictions = {}
            for horse_id in race_feature_df["horse_id"].unique():
                predictions[horse_id] = {}

            for target, model in models.items():
                feature_cols = feature_cols_map[target]
                n_features_used = max(n_features_used, len(feature_cols))

                probs = predict_race(model, race_feature_df, feature_cols)

                # Map predictions to horse_id
                for i, (_, row) in enumerate(race_feature_df.iterrows()):
                    horse_id = row["horse_id"]
                    prob = float(probs[i]) if i < len(probs) else 0.0

                    if target == "target_win":
                        predictions[horse_id]["p_win"] = round(prob, 4)
                    elif target == "target_in3":
                        predictions[horse_id]["p_in3"] = round(prob, 4)

            # Calculate ranks
            if "target_win" in models:
                sorted_by_win = sorted(predictions.items(), key=lambda x: x[1].get("p_win", 0), reverse=True)
                for rank, (horse_id, _) in enumerate(sorted_by_win, 1):
                    predictions[horse_id]["rank_win"] = rank

            if "target_in3" in models:
                sorted_by_in3 = sorted(predictions.items(), key=lambda x: x[1].get("p_in3", 0), reverse=True)
                for rank, (horse_id, _) in enumerate(sorted_by_in3, 1):
                    predictions[horse_id]["rank_in3"] = rank

            # Generate race JSON
            race_info = race_row.to_dict()
            race_json = generate_race_json(race_info, entries_df, predictions)

            # Save race JSON
            race_path = out_dir / f"race_{race_id}.json"
            with open(race_path, "w", encoding="utf-8") as f:
                json.dump(race_json, f, ensure_ascii=False, indent=2)
            logger.info("  Saved: %s", race_path.name)

            # Add to summary
            race_summaries.append({
                "race_id": race_id,
                "place": race_row["place"],
                "race_no": int(race_row["race_no"]) if pd.notna(race_row["race_no"]) else None,
                "name": race_row["name"],
                "grade": race_row["grade"],
                "distance": race_row["distance"],
                "n_entries": len(entries_df),
                "top3_win": [
                    predictions.get(h, {}).get("p_win")
                    for h in [e["horse_id"] for e in race_json["entries"][:3]]
                ],
            })

        # Generate summary JSON
        summary = generate_summary_json(
            date=args.date,
            races=race_summaries,
            n_features=n_features_used,
            models_used=models_used,
        )
        summary_path = out_dir / f"summary_{args.date}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Generation Complete")
        logger.info("=" * 70)
        logger.info("  Date:           %s", args.date)
        logger.info("  Races:          %d", len(race_summaries))
        logger.info("  Output files:   %d", len(race_summaries) + 1)
        logger.info("  Features used:  %d (pre_race feature set)", n_features_used)
        logger.info("  Models used:    %s", models_used)
        logger.info("  Summary:        %s", summary_path)
        logger.info("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
