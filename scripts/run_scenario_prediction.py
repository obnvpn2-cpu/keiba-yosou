# scripts/run_scenario_prediction.py

"""
ベースモデル（LightGBM）の予測値 ＋ シナリオ補正レイヤをつなぐ
簡易ブリッジスクリプト。

使い方（例）:
    python -m scripts.run_scenario_prediction \
      --db data/keiba.db \
      --models models \
      --race-id 202306030212 \
      --scenario-id slow_inner \
      --pace S \
      --track-condition 良 \
      --bias 内 \
      --front-runner-ids 2019110042 2018104765
"""

from __future__ import annotations

import argparse

from src.scenario import (
    ScenarioSpec,
    RaceContext,
    TrackMoisture,
    MoistureReading,
)
from src.scenario.runner import run_scenario_for_race


# =============================================================================
#  シナリオ定義のユーティリティ
# =============================================================================

def build_scenario_spec_from_args(args: argparse.Namespace) -> ScenarioSpec:
    """
    CLI の引数から ScenarioSpec を組み立てる。
    moisture / cushion は今は任意。
    """
    race_ctx = RaceContext(
        race_id=args.race_id,
        race_name=args.race_name,
        course=args.course,
        surface=args.surface,
        distance=args.distance,
    )

    # moisture（含水率）はとりあえず全部 None にしておく。
    moisture = TrackMoisture(
        turf=MoistureReading(goal=None, corner_4=None),
        dirt=MoistureReading(goal=None, corner_4=None),
    )

    spec = ScenarioSpec(
        scenario_id=args.scenario_id,
        race_context=race_ctx,
        pace=args.pace,
        track_condition=args.track_condition,
        bias=args.bias,
        cushion_value=args.cushion_value,
        moisture=moisture,
        front_runner_ids=[str(x) for x in (args.front_runner_ids or [])],
        weak_horse_ids=[str(x) for x in (args.weak_horse_ids or [])],
        stalker_ids=[str(x) for x in (args.stalker_ids or [])],
        closer_ids=[str(x) for x in (args.closer_ids or [])],
        notes=args.notes or "",
    )
    return spec


# =============================================================================
#  メイン処理
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ベース予測＋シナリオ補正をまとめて実行するスクリプト",
    )

    # 必須: DB, モデルディレクトリ, レースID
    parser.add_argument("--db", required=True, help="SQLite DB path (e.g. data/keiba.db)")
    parser.add_argument("--models", required=True, help="Model directory (e.g. models)")
    parser.add_argument("--race-id", required=True, help="Target race_id")

    # シナリオ ID
    parser.add_argument("--scenario-id", required=True, help="Scenario ID (e.g. slow_inner)")

    # コアシナリオ
    parser.add_argument("--pace", required=True, choices=["S", "M", "H"], help="想定ペース (S/M/H)")
    parser.add_argument(
        "--track-condition",
        required=True,
        choices=["良", "稍重", "重", "不良"],
        help="馬場状態",
    )
    parser.add_argument(
        "--bias",
        required=True,
        choices=["内", "外", "フラット"],
        help="バイアス (内/外/フラット)",
    )

    # レースコンテキスト（任意）
    parser.add_argument("--race-name", default="", help="レース名（任意）")
    parser.add_argument("--course", default="", help="コース名（例: 東京、中山）")
    parser.add_argument("--surface", default="turf", choices=["turf", "dirt"], help="サーフェス")
    parser.add_argument("--distance", type=int, default=0, help="距離 (m)")

    # 馬場関連（任意）
    parser.add_argument("--cushion-value", type=float, default=None, help="当日朝のクッション値")

    # 脚質・状態に関する主観情報（任意・複数指定可）
    parser.add_argument(
        "--front-runner-ids",
        nargs="*",
        help="逃げ想定の horse_id リスト",
    )
    parser.add_argument(
        "--weak-horse-ids",
        nargs="*",
        help="状態不安・信頼度低い horse_id リスト",
    )
    parser.add_argument(
        "--stalker-ids",
        nargs="*",
        help="先行〜好位勢の horse_id リスト",
    )
    parser.add_argument(
        "--closer-ids",
        nargs="*",
        help="差し・追込の horse_id リスト",
    )

    # メモ
    parser.add_argument("--notes", default="", help="シナリオの補足メモ")

    args = parser.parse_args()

    # 1) ScenarioSpec を組み立て
    spec = build_scenario_spec_from_args(args)

    # 2) 共通エンジンで 1 シナリオ分を実行
    scenario_score = run_scenario_for_race(
        db_path=args.db,
        models_dir=args.models,
        spec=spec,
    )

    # 3) 結果の表示（とりあえずコンソールに出す）
    print(f"\n=== Scenario [{scenario_score.scenario_id}] for race [{scenario_score.race_id}] ===")
    print(f"pace={spec.pace}, track_condition={spec.track_condition}, bias={spec.bias}")
    if spec.notes:
        print(f"notes: {spec.notes}")

    print("\n--- Top horses by adj_win ---")
    top_horses = scenario_score.top_horses_by_adj_win(n=10)
    for h in top_horses:
        name = h.horse_name or h.horse_id
        reasons = [r.name for r in h.adjustment_reasons]
        print(
            f"{name} ({h.horse_id}): "
            f"base_win={h.base_win:.3f} -> adj_win={h.adj_win:.3f} "
            f"(delta={h.win_delta:+.3f}) reasons={reasons}"
        )

    # 4) LLM 用のコンテキストも吐いておく（デバッグ用に print）
    llm_ctx = scenario_score.to_llm_context()
    print("\n--- LLM context (debug) ---")
    print(llm_ctx)


if __name__ == "__main__":
    main()
