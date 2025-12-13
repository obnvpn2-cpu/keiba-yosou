# scripts/run_compare_scenarios.py

"""
1レースに対して
- ベースシナリオ
- シナリオA
- シナリオB
をまとめて流して、シナリオ比較用の JSON コンテキストを吐き出すスクリプト。

使い方（例）:

    python -m scripts.run_compare_scenarios \
      --db data/keiba.db \
      --models models \
      --race-id 202306030212 \
      --race-name "安田記念(仮)" \
      --course "東京" \
      --surface turf \
      --distance 1600 \
      \
      --base-scenario-id base \
      --base-pace M \
      --base-track-condition 良 \
      --base-bias フラット \
      \
      --scenario-a-id slow_inner \
      --scenario-a-pace S \
      --scenario-a-bias 内 \
      --a-front-runner-ids 2019110042 \
      --a-notes "逃げ馬1頭でスロー＋内有利想定" \
      \
      --scenario-b-id high_outer \
      --scenario-b-pace H \
      --scenario-b-bias 外 \
      --b-notes "ハイペース＋外差し有利想定"

※ A/B の track-condition を省略した場合は、ベースシナリオと同じ値が使われます。
"""

from __future__ import annotations

import argparse
import json

from src.scenario import (
    ScenarioSpec,
    RaceContext,
    TrackMoisture,
    MoistureReading,
)
from src.scenario.runner import run_scenario_for_race
from src.scenario.compare import build_scenario_comparison_context


def build_race_context(args: argparse.Namespace) -> RaceContext:
    """レース共通のコンテキストを組み立てる。"""
    return RaceContext(
        race_id=args.race_id,
        race_name=args.race_name,
        course=args.course,
        surface=args.surface,
        distance=args.distance,
    )


def build_base_spec(args: argparse.Namespace, race_ctx: RaceContext) -> ScenarioSpec:
    """ベースシナリオ用の ScenarioSpec を組み立てる。"""
    moisture = TrackMoisture(
        turf=MoistureReading(goal=None, corner_4=None),
        dirt=MoistureReading(goal=None, corner_4=None),
    )

    spec = ScenarioSpec(
        scenario_id=args.base_scenario_id,
        race_context=race_ctx,
        pace=args.base_pace,
        track_condition=args.base_track_condition,
        bias=args.base_bias,
        cushion_value=args.base_cushion_value,
        moisture=moisture,
        # ベースは基本的に「補正なし」想定なので人間シグナルは空にしておく
        front_runner_ids=[],
        stalker_ids=[],
        closer_ids=[],
        weak_horse_ids=[],
        notes=args.base_notes or "",
    )
    return spec


def build_scenario_a_spec(args: argparse.Namespace, race_ctx: RaceContext) -> ScenarioSpec:
    """シナリオA用の ScenarioSpec を組み立てる。"""
    moisture = TrackMoisture(
        turf=MoistureReading(goal=None, corner_4=None),
        dirt=MoistureReading(goal=None, corner_4=None),
    )

    track_condition = args.scenario_a_track_condition or args.base_track_condition

    spec = ScenarioSpec(
        scenario_id=args.scenario_a_id,
        race_context=race_ctx,
        pace=args.scenario_a_pace,
        track_condition=track_condition,
        bias=args.scenario_a_bias,
        cushion_value=args.scenario_a_cushion_value,
        moisture=moisture,
        front_runner_ids=[str(x) for x in (args.a_front_runner_ids or [])],
        stalker_ids=[str(x) for x in (args.a_stalker_ids or [])],
        closer_ids=[str(x) for x in (args.a_closer_ids or [])],
        weak_horse_ids=[str(x) for x in (args.a_weak_horse_ids or [])],
        notes=args.a_notes or "",
    )
    return spec


def build_scenario_b_spec(args: argparse.Namespace, race_ctx: RaceContext) -> ScenarioSpec:
    """シナリオB用の ScenarioSpec を組み立てる。"""
    moisture = TrackMoisture(
        turf=MoistureReading(goal=None, corner_4=None),
        dirt=MoistureReading(goal=None, corner_4=None),
    )

    track_condition = args.scenario_b_track_condition or args.base_track_condition

    spec = ScenarioSpec(
        scenario_id=args.scenario_b_id,
        race_context=race_ctx,
        pace=args.scenario_b_pace,
        track_condition=track_condition,
        bias=args.scenario_b_bias,
        cushion_value=args.scenario_b_cushion_value,
        moisture=moisture,
        front_runner_ids=[str(x) for x in (args.b_front_runner_ids or [])],
        stalker_ids=[str(x) for x in (args.b_stalker_ids or [])],
        closer_ids=[str(x) for x in (args.b_closer_ids or [])],
        weak_horse_ids=[str(x) for x in (args.b_weak_horse_ids or [])],
        notes=args.b_notes or "",
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ベース＋2シナリオをまとめて実行し、比較用コンテキストを出力するスクリプト",
    )

    # 必須: DB, モデルディレクトリ, レースID
    parser.add_argument("--db", required=True, help="SQLite DB path (e.g. data/keiba.db)")
    parser.add_argument("--models", required=True, help="Model directory (e.g. models)")
    parser.add_argument("--race-id", required=True, help="Target race_id")

    # レースコンテキスト（任意）
    parser.add_argument("--race-name", default="", help="レース名（任意）")
    parser.add_argument("--course", default="", help="コース名（例: 東京、中山）")
    parser.add_argument(
        "--surface",
        default="turf",
        choices=["turf", "dirt"],
        help="サーフェス (turf/dirt)",
    )
    parser.add_argument("--distance", type=int, default=0, help="距離 (m)")

    # -------------------------------------------------------------------------
    # ベースシナリオ
    # -------------------------------------------------------------------------
    parser.add_argument("--base-scenario-id", default="base", help="ベースシナリオの ID")
    parser.add_argument(
        "--base-pace",
        default="M",
        choices=["S", "M", "H"],
        help="ベースシナリオの想定ペース (S/M/H)",
    )
    parser.add_argument(
        "--base-track-condition",
        default="良",
        choices=["良", "稍重", "重", "不良"],
        help="ベースシナリオの馬場状態",
    )
    parser.add_argument(
        "--base-bias",
        default="フラット",
        choices=["内", "外", "フラット"],
        help="ベースシナリオのバイアス",
    )
    parser.add_argument(
        "--base-cushion-value",
        type=float,
        default=None,
        help="ベースシナリオのクッション値（任意）",
    )
    parser.add_argument("--base-notes", default="", help="ベースシナリオのメモ（任意）")

    # -------------------------------------------------------------------------
    # シナリオA
    # -------------------------------------------------------------------------
    parser.add_argument("--scenario-a-id", required=True, help="シナリオAの ID")
    parser.add_argument(
        "--scenario-a-pace",
        required=True,
        choices=["S", "M", "H"],
        help="シナリオAの想定ペース (S/M/H)",
    )
    parser.add_argument(
        "--scenario-a-track-condition",
        choices=["良", "稍重", "重", "不良"],
        default=None,
        help="シナリオAの馬場状態（省略時はベースと同じ）",
    )
    parser.add_argument(
        "--scenario-a-bias",
        required=True,
        choices=["内", "外", "フラット"],
        help="シナリオAのバイアス",
    )
    parser.add_argument(
        "--scenario-a-cushion-value",
        type=float,
        default=None,
        help="シナリオAのクッション値（任意）",
    )
    parser.add_argument("--a-notes", default="", help="シナリオAのメモ（任意）")

    parser.add_argument(
        "--a-front-runner-ids",
        nargs="*",
        help="シナリオAで逃げ想定の horse_id リスト",
    )
    parser.add_argument(
        "--a-stalker-ids",
        nargs="*",
        help="シナリオAで先行〜好位勢の horse_id リスト",
    )
    parser.add_argument(
        "--a-closer-ids",
        nargs="*",
        help="シナリオAで差し・追込の horse_id リスト",
    )
    parser.add_argument(
        "--a-weak-horse-ids",
        nargs="*",
        help="シナリオAで状態不安・信頼度低の horse_id リスト",
    )

    # -------------------------------------------------------------------------
    # シナリオB
    # -------------------------------------------------------------------------
    parser.add_argument("--scenario-b-id", required=True, help="シナリオBの ID")
    parser.add_argument(
        "--scenario-b-pace",
        required=True,
        choices=["S", "M", "H"],
        help="シナリオBの想定ペース (S/M/H)",
    )
    parser.add_argument(
        "--scenario-b-track-condition",
        choices=["良", "稍重", "重", "不良"],
        default=None,
        help="シナリオBの馬場状態（省略時はベースと同じ）",
    )
    parser.add_argument(
        "--scenario-b-bias",
        required=True,
        choices=["内", "外", "フラット"],
        help="シナリオBのバイアス",
    )
    parser.add_argument(
        "--scenario-b-cushion-value",
        type=float,
        default=None,
        help="シナリオBのクッション値（任意）",
    )
    parser.add_argument("--b-notes", default="", help="シナリオBのメモ（任意）")

    parser.add_argument(
        "--b-front-runner-ids",
        nargs="*",
        help="シナリオBで逃げ想定の horse_id リスト",
    )
    parser.add_argument(
        "--b-stalker-ids",
        nargs="*",
        help="シナリオBで先行〜好位勢の horse_id リスト",
    )
    parser.add_argument(
        "--b-closer-ids",
        nargs="*",
        help="シナリオBで差し・追込の horse_id リスト",
    )
    parser.add_argument(
        "--b-weak-horse-ids",
        nargs="*",
        help="シナリオBで状態不安・信頼度低の horse_id リスト",
    )

    args = parser.parse_args()

    # 1) 共通の RaceContext を作る
    race_ctx = build_race_context(args)

    # 2) 各シナリオの Spec を組み立てる
    base_spec = build_base_spec(args, race_ctx)
    scenario_a_spec = build_scenario_a_spec(args, race_ctx)
    scenario_b_spec = build_scenario_b_spec(args, race_ctx)

    # 3) それぞれ run_scenario_for_race を回す
    base_score = run_scenario_for_race(
        db_path=args.db,
        models_dir=args.models,
        spec=base_spec,
    )
    scenario_a_score = run_scenario_for_race(
        db_path=args.db,
        models_dir=args.models,
        spec=scenario_a_spec,
    )
    scenario_b_score = run_scenario_for_race(
        db_path=args.db,
        models_dir=args.models,
        spec=scenario_b_spec,
    )

    # 4) 比較コンテキストを構築
    ctx = build_scenario_comparison_context(
        base_score=base_score,
        scenario_a_score=scenario_a_score,
        scenario_b_score=scenario_b_score,
        top_n=10,
        max_diff_horses=10,
    )

    # 5) 軽くサマリだけ標準出力に出してから、JSON を吐く
    print(
        f"\n=== race_id={base_score.race_id} "
        f"base=[{base_score.scenario_id}] "
        f"A=[{scenario_a_score.scenario_id}] "
        f"B=[{scenario_b_score.scenario_id}] ==="
    )
    print(f"base: pace={base_spec.pace}, bias={base_spec.bias}, tc={base_spec.track_condition}")
    print(f"A   : pace={scenario_a_spec.pace}, bias={scenario_a_spec.bias}, tc={scenario_a_spec.track_condition}")
    print(f"B   : pace={scenario_b_spec.pace}, bias={scenario_b_spec.bias}, tc={scenario_b_spec.track_condition}")

    print("\n--- JSON (LLMに投げる用コンテキスト) ---")
    print(json.dumps(ctx, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
