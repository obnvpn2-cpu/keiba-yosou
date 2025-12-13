# src/scenario/runner.py

from __future__ import annotations

import sqlite3
from typing import Dict, Any, Tuple

import pandas as pd

# NOTE:
# - この相対インポートは、
#   - プロジェクトルートから `python -m scripts.run_scenario_prediction` と実行する場合
#   - PYTHONPATH=src で `from scenario.runner import ...` と REPL から使う場合
#   の両方で動くようにしてある。
from ..models.predict_lgbm import predict_for_races
from .spec import ScenarioSpec
from .adjuster import ScenarioAdjuster, create_adjuster_with_custom_config
from .score import ScenarioScore


def load_horse_context(
    db_path: str,
    race_id: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    race_results から「馬ごとのコンテキスト」を作る。

    Returns:
        horse_features:
            {
                horse_id: {
                    "frame_no": int | None,
                    "run_style": str | None,
                    "jockey_name": str | None,
                },
                ...
            }
        horse_names:
            { horse_id: horse_name }
    """
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT
            horse_id,
            horse_name,
            frame_no,
            jockey_name
        FROM race_results
        WHERE race_id = ?
        """,
        conn,
        params=(race_id,),
    )

    conn.close()

    horse_features: Dict[str, Dict[str, Any]] = {}
    horse_names: Dict[str, str] = {}

    for _, row in df.iterrows():
        horse_id = str(row["horse_id"])
        horse_names[horse_id] = row.get("horse_name") or ""

        frame_no = row.get("frame_no")
        try:
            frame_no_int = int(frame_no) if pd.notna(frame_no) else None
        except Exception:
            frame_no_int = None

        horse_features[horse_id] = {
            "frame_no": frame_no_int,
            "run_style": None,  # TODO: あとで「逃げ/先行/差し/追込」を入れる
            "jockey_name": row.get("jockey_name") or None,
        }

    return horse_features, horse_names


def run_scenario_for_race(
    db_path: str,
    models_dir: str,
    spec: ScenarioSpec,
) -> ScenarioScore:
    """
    1レース × 1シナリオを実行して ScenarioScore を返す共通関数。

    - DB から特徴量の元となる情報を取得
    - LightGBM ベースモデルで勝率 / 3着内率を予測
    - ScenarioAdjuster でシナリオ補正

    CLI やノートブックから共通で使う想定。
    """
    race_id = spec.race_id

    # 1) ベース予測を取得
    df_pred = predict_for_races(
        db_path=db_path,
        model_dir=models_dir,
        race_ids=[race_id],
    )

    if df_pred.empty:
        raise ValueError(f"No predictions found for race_id={race_id}")

    df_pred_race = df_pred[df_pred["race_id"] == race_id].copy()

    base_predictions: Dict[str, Dict[str, float]] = {}
    for _, row in df_pred_race.iterrows():
        horse_id = str(row["horse_id"])
        base_predictions[horse_id] = {
            "win": float(row["pred_target_win"]),
            "in3": float(row["pred_target_in3"]),
        }

    # 2) DB から馬の情報（枠順・騎手名など）を取得
    horse_features, horse_names = load_horse_context(db_path, race_id)

    # 3) ScenarioAdjuster で補正
    adjuster: ScenarioAdjuster = create_adjuster_with_custom_config()
    scenario_score = adjuster.adjust(
        spec=spec,
        base_predictions=base_predictions,
        horse_features=horse_features,
        horse_names=horse_names,
    )

    return scenario_score
