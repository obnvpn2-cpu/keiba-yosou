# src/models/predict_lgbm.py
import argparse
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import pandas as pd

from .model_utils import (
    load_features_for_races,
    load_feature_columns,
)


def load_lgbm_model(model_dir: str, target_col: str) -> lgb.Booster:
    model_path = Path(model_dir) / f"lgbm_{target_col}.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    return model


def predict_for_races(
    db_path: str,
    model_dir: str,
    race_ids: List[str],
) -> Optional[pd.DataFrame]:
    """
    指定 race_ids について、target_win / target_in3 の予測確率を出す
    """
    if not race_ids:
        raise ValueError("race_ids must not be empty")

    # feature_table から対象レースのデータ取得
    df = load_features_for_races(db_path, race_ids)

    if df.empty:
        print("No rows found for given race_ids.")
        return None

    # race_id, horse_id を控えておく（出力用）
    base_cols = ["race_id", "horse_id"]
    if "horse_name" in df.columns:
        base_cols.append("horse_name")

    results = df[base_cols].copy()

    for target_col in ["target_win", "target_in3"]:
        print(f"\n=== Predicting for {target_col} ===")
        feature_cols = load_feature_columns(model_dir, target_col)
        model = load_lgbm_model(model_dir, target_col)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in feature_table: {missing}")

        X = df[feature_cols]
        proba = model.predict(X, num_iteration=model.best_iteration)
        results[f"pred_{target_col}"] = proba

    # ソート：レースごとに target_win の予測値が高い順
    results = results.sort_values(["race_id", "pred_target_win"], ascending=[True, False])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=str,
        default="data/keiba.db",
        help="Path to SQLite DB",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models",
        help="Directory of trained models",
    )
    parser.add_argument(
        "--race-ids",
        type=str,
        nargs="+",
        required=True,
        help="Race IDs to predict (space separated)",
    )
    args = parser.parse_args()

    df_pred = predict_for_races(
        db_path=args.db,
        model_dir=args.models,
        race_ids=args.race_ids,
    )

    if df_pred is None or df_pred.empty:
        print("No prediction results.")
        return

    # コンソール出力（とりあえず top5）
    for rid in df_pred["race_id"].unique():
        print(f"\n=== Race {rid} ===")
        df_r = df_pred[df_pred["race_id"] == rid].copy()
        cols_to_show = [c for c in ["horse_id", "horse_name", "pred_target_win", "pred_target_in3"] if c in df_r.columns]
        print(
            df_r[cols_to_show]
            .sort_values("pred_target_win", ascending=False)
            .head(5)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
