# scripts/run_scenario_ui.py

"""
Scenario UI runner

使い方のイメージ：

python -m scripts.run_scenario_ui \
  --db data/keiba.db \
  --models models \
  --race-id 202306050811 \
  --scenario-id slow_inner \
  --pace S \
  --track-condition 良 \
  --bias 内 \
  --front-runner-ids 2019110042 \
  --notes "逃げ馬1頭でスロー濃厚、内伸びバイアス想定" \
  --top-n 10 \
  --pretty

"""

import argparse
import json
from typing import List, Optional

from src.scenario import RaceContext, ScenarioSpec  # run_scenario_prediction.py と同じスタイル
from src.scenario.runner import run_scenario_for_race
from src.scenario.ui import build_scenario_ui_context


# -----------------------------
# ユーティリティ
# -----------------------------
def _parse_id_list(raw: Optional[str]) -> List[str]:
    """カンマ区切りのID文字列を ['id1', 'id2', ...] に変換"""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_scenario_spec_from_args(args: argparse.Namespace) -> ScenarioSpec:
    """
    run_scenario_prediction.py とほぼ同じロジックで ScenarioSpec を組み立てる。
    （moisture 周りは省略。必要になったらここに足す）
    """
    race_ctx = RaceContext(
        race_id=args.race_id,
        race_name=args.race_name,
        course=args.course,
        surface=args.surface,
        distance=args.distance,
        distance_cat=None,
        race_date=args.race_date,
        race_class=args.race_class,
    )

    spec = ScenarioSpec(
        scenario_id=args.scenario_id,
        race_context=race_ctx,
        pace=args.pace,
        track_condition=args.track_condition,
        bias=args.bias,
        cushion_value=args.cushion_value,
        front_runner_ids=_parse_id_list(args.front_runner_ids),
        stalker_ids=_parse_id_list(args.stalker_ids),
        closer_ids=_parse_id_list(args.closer_ids),
        weak_horse_ids=_parse_id_list(args.weak_horse_ids),
        notes=args.notes,
    )
    return spec


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scenario-adjusted prediction and output UI JSON context."
    )

    # モデル・DB 周り
    parser.add_argument(
        "--db",
        dest="db_path",
        required=True,
        help="Path to SQLite DB (e.g., data/keiba.db)",
    )
    parser.add_argument(
        "--models",
        dest="models_dir",
        required=True,
        help="Directory where trained models are stored",
    )

    # レース情報（最低 race-id があれば動く。残りは任意）
    parser.add_argument("--race-id", required=True, help="Race ID (e.g., 202306050811)")
    parser.add_argument("--race-name", default=None, help="Optional race name")
    parser.add_argument("--course", default=None, help="Course name (e.g., 中山)")
    parser.add_argument("--surface", default=None, help="Surface type (e.g., turf)")
    parser.add_argument(
        "--distance",
        type=int,
        default=None,
        help="Race distance in meters (e.g., 2500)",
    )
    parser.add_argument("--race-date", default=None, help="Race date (string)")
    parser.add_argument("--race-class", default=None, help="Race class (e.g., G1)")

    # シナリオ定義
    parser.add_argument("--scenario-id", default="default", help="Scenario ID label")
    parser.add_argument(
        "--pace",
        choices=["S", "M", "H"],
        required=True,
        help="Pace scenario: S (slow), M (middle), H (high)",
    )
    parser.add_argument(
        "--track-condition",
        dest="track_condition",
        choices=["良", "稍重", "重", "不良"],
        required=True,
        help="Track condition (Japanese labels)",
    )
    parser.add_argument(
        "--bias",
        choices=["内", "外", "フラット"],
        required=True,
        help="Track bias",
    )
    parser.add_argument(
        "--cushion-value",
        dest="cushion_value",
        type=float,
        default=None,
        help="Optional cushion value (float)",
    )

    # 脚質指定系（任意）
    parser.add_argument(
        "--front-runner-ids",
        type=str,
        default="",
        help="Comma separated horse IDs that you want to force as escape/front-runner",
    )
    parser.add_argument(
        "--stalker-ids",
        type=str,
        default="",
        help="Comma separated stalker IDs",
    )
    parser.add_argument(
        "--closer-ids",
        type=str,
        default="",
        help="Comma separated closer IDs",
    )
    parser.add_argument(
        "--weak-horse-ids",
        type=str,
        default="",
        help="Comma separated IDs you want to down-weight as weak horses",
    )

    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form notes about scenario (goes into scenario.notes)",
    )

    # UI 用オプション
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Limit number of horses in 'horses' list (None = all)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="adj_win",
        help="Sort key for horses section (e.g., 'adj_win', 'base_win')",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indent=2",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ScenarioSpec を構築
    spec = build_scenario_spec_from_args(args)

    # シナリオ補正済みの ScenarioScore を取得
    score = run_scenario_for_race(
        args.db_path,
        args.models_dir,
        spec,
    )

    # UI 向け JSON コンテキストを構築
    ui_ctx = build_scenario_ui_context(
        score,
        sort_by=args.sort_by,
        top_n=args.top_n,
        include_summary=True,
    )

    if args.pretty:
        print(json.dumps(ui_ctx, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(ui_ctx, ensure_ascii=False))


if __name__ == "__main__":
    main()
