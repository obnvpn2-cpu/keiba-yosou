"""
scenario - シナリオ補正レイヤ

人間が定義するレースシナリオを受け取り、
ベースモデル（LightGBM）の予測値に対して補正を適用するモジュール。

Usage:
    from scenario import ScenarioSpec, ScenarioScore, ScenarioAdjuster, RaceContext
    
    # レースコンテキストの作成
    race_ctx = RaceContext(
        race_id="202305021211",
        race_name="日本ダービー",
        course="東京",
        surface="turf",
        distance=2400,
    )
    
    # シナリオ定義
    spec = ScenarioSpec(
        scenario_id="slow_inner",
        race_context=race_ctx,
        pace="S",
        track_condition="良",
        bias="内",
        front_runner_ids=["2019104567"],
        notes="逃げ馬が1頭しかいないのでスロー濃厚",
    )
    
    # 補正適用
    adjuster = ScenarioAdjuster()
    score = adjuster.adjust(
        spec=spec,
        base_predictions={"horse_1": {"win": 0.12, "in3": 0.35}},
    )
    
    # 結果参照
    for h in score.top_horses_by_adj_win(5):
        print(f"{h.horse_id}: {h.base_win:.3f} -> {h.adj_win:.3f}")
"""

from .types import (
    PaceType,
    TrackConditionType,
    BiasType,
    SurfaceType,
    DistanceCategoryType,
    JRACourseType,
)

from .spec import (
    MoistureReading,
    TrackMoisture,
    RaceContext,
    ScenarioSpec,
)

from .score import (
    AdjustmentReason,
    BodyWeightContext,
    ScenarioHorseScore,
    ScenarioScore,
)

from .adjuster import (
    AdjustmentConfig,
    ScenarioAdjuster,
    create_adjuster_with_custom_config,
)

from .ui import (
    build_scenario_ui_context,
    get_pace_label_ja,
    get_bias_label_ja,
)

__all__ = [
    # types
    "PaceType",
    "TrackConditionType",
    "BiasType",
    "SurfaceType",
    "DistanceCategoryType",
    "JRACourseType",
    # spec
    "MoistureReading",
    "TrackMoisture",
    "RaceContext",
    "ScenarioSpec",
    # score
    "AdjustmentReason",
    "BodyWeightContext",
    "ScenarioHorseScore",
    "ScenarioScore",
    # adjuster
    "AdjustmentConfig",
    "ScenarioAdjuster",
    "create_adjuster_with_custom_config",
    # ui
    "build_scenario_ui_context",
    "get_pace_label_ja",
    "get_bias_label_ja",
]

__version__ = "1.0.0"
