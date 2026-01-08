# -*- coding: utf-8 -*-
"""
src/features_v4 - FeaturePack v1 (200+ features)

Road 3: リーク防止を徹底した特徴量エンジニアリングモジュール

【設計原則】
1. 未来情報リーク防止: 全ての集計は as-of (race_date より前) で実施
2. 同日オッズ・人気不使用: 当該レースの win_odds/popularity は使用禁止
3. 高速処理: Python ループ禁止、SQL ベースのベクトル演算を活用
4. 再現性: 同じデータに対して同じ結果を保証

【特徴量グループ】
- base_race: 基本レース情報 (~25 columns)
- horse_form: 馬の過去成績・フォーム (~40 columns)
- pace_position: ペース・位置取り (~20 columns)
- class_prize: クラス・賞金 (~15 columns)
- jockey_trainer: 騎手・調教師の as-of 成績 (~40 columns)
- pedigree: 血統ハッシュ (512 + 128 = 640 columns)

合計: 200+ 特徴量

【使用禁止データ】
- 当該レースの win_odds, popularity
- 馬のキャリア通算成績 (masters テーブルの career_* フィールド)
- race_date 以降の情報

【使用可能データ】
- race_date より前の race_results からの集計
- horse_pedigree テーブル (血統は静的情報)
- horses/jockeys/trainers の静的属性 (名前、所属など)
"""

from .feature_table_v4 import (
    CREATE_FEATURE_TABLE_V4,
    create_feature_table_v4,
    get_feature_v4_columns,
)

from .asof_aggregator import (
    AsOfAggregator,
    compute_horse_asof_stats,
    compute_jockey_asof_stats,
    compute_trainer_asof_stats,
    map_distance_to_cat,
    map_class_to_id,
    PLACE_MAP,
    SURFACE_MAP,
    TRACK_CONDITION_MAP,
    GRADE_MAP,
    SEX_MAP,
)

from .quality_report import (
    MasterQualityReporter,
    QualityReport,
    generate_quality_report,
)

from .feature_builder_v4 import (
    FeatureBuilderV4,
    build_feature_table_v4,
    hash_pedigree_sire_dam,
    hash_ancestor_frequency,
)

from .train_eval_v4 import (
    TrainConfig,
    EvalResult,
    ROIResult,
    RankingResult,
    ROIStrategyResult,
    ROIEvalResult,
    SelectiveBetResult,
    ROISweepResult,
    train_model,
    evaluate_model,
    evaluate_ranking,
    evaluate_roi_strategy,
    evaluate_roi_all_strategies,
    analyze_roi,
    run_full_pipeline,
    run_roi_sweep,
    generate_model_top1_prob_threshold_bets,
    generate_model_top1_gap_threshold_bets,
    load_odds_snapshots_for_races,
    save_roi_sweep_artifacts,
    save_roi_sweep_flat_artifacts,
    DEFAULT_PROB_THRESHOLDS,
    DEFAULT_GAP_THRESHOLDS,
)

from .diagnostics import (
    DiagnosticsReport,
    FeatureImportanceResult,
    PermutationImportanceResult,
    FeatureGroupImportanceResult,
    SegmentPerformanceResult,
    compute_lgbm_importance,
    compute_permutation_importance,
    compute_group_importance,
    compute_segment_performance,
    run_diagnostics,
    save_diagnostics,
    get_feature_group,
    FEATURE_GROUP_PREFIXES,
    SEGMENT_DEFINITIONS,
)

# Explain runner (F-3: bridge name resolution for UI)
from .explain_runner import (
    FeatureExplanation,
    ExplainResult,
    resolve_feature_name,
    build_feature_explanation,
    generate_explain_result,
    run_explain,
)

# Bridge layer for v3 feature migration (optional)
try:
    from .bridge_v3_features import (
        BridgeV3Features,
        BridgeConfig,
        BridgeResult,
        MigrationCandidate,
        FeatureSafetyInfo,
        BRIDGE_PREFIX,
        SAFETY_SKIP,
        SAFETY_WARN,
        apply_bridge_to_dataframe,
        get_bridge_feature_columns,
        list_available_v3_features,
        load_bridge_feature_map,
        get_original_feature_name,
        get_feature_safety_for_bridge,
    )
    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False

__all__ = [
    # DDL
    "CREATE_FEATURE_TABLE_V4",
    "create_feature_table_v4",
    "get_feature_v4_columns",
    # Aggregators
    "AsOfAggregator",
    "compute_horse_asof_stats",
    "compute_jockey_asof_stats",
    "compute_trainer_asof_stats",
    # Encodings
    "map_distance_to_cat",
    "map_class_to_id",
    "PLACE_MAP",
    "SURFACE_MAP",
    "TRACK_CONDITION_MAP",
    "GRADE_MAP",
    "SEX_MAP",
    # Quality Report
    "MasterQualityReporter",
    "QualityReport",
    "generate_quality_report",
    # Feature Builder
    "FeatureBuilderV4",
    "build_feature_table_v4",
    "hash_pedigree_sire_dam",
    "hash_ancestor_frequency",
    # Train/Eval
    "TrainConfig",
    "EvalResult",
    "ROIResult",
    "RankingResult",
    "ROIStrategyResult",
    "ROIEvalResult",
    "SelectiveBetResult",
    "ROISweepResult",
    "train_model",
    "evaluate_model",
    "evaluate_ranking",
    "evaluate_roi_strategy",
    "evaluate_roi_all_strategies",
    "analyze_roi",
    "run_full_pipeline",
    "run_roi_sweep",
    "generate_model_top1_prob_threshold_bets",
    "generate_model_top1_gap_threshold_bets",
    "load_odds_snapshots_for_races",
    "save_roi_sweep_artifacts",
    "save_roi_sweep_flat_artifacts",
    "DEFAULT_PROB_THRESHOLDS",
    "DEFAULT_GAP_THRESHOLDS",
    # Diagnostics
    "DiagnosticsReport",
    "FeatureImportanceResult",
    "PermutationImportanceResult",
    "FeatureGroupImportanceResult",
    "SegmentPerformanceResult",
    "compute_lgbm_importance",
    "compute_permutation_importance",
    "compute_group_importance",
    "compute_segment_performance",
    "run_diagnostics",
    "save_diagnostics",
    "get_feature_group",
    "FEATURE_GROUP_PREFIXES",
    "SEGMENT_DEFINITIONS",
    # Explain runner (F-3)
    "FeatureExplanation",
    "ExplainResult",
    "resolve_feature_name",
    "build_feature_explanation",
    "generate_explain_result",
    "run_explain",
    # Bridge (v3 feature migration)
    "HAS_BRIDGE",
]

# Add bridge exports if available
if HAS_BRIDGE:
    __all__.extend([
        "BridgeV3Features",
        "BridgeConfig",
        "BridgeResult",
        "MigrationCandidate",
        "FeatureSafetyInfo",
        "BRIDGE_PREFIX",
        "SAFETY_SKIP",
        "SAFETY_WARN",
        "apply_bridge_to_dataframe",
        "get_bridge_feature_columns",
        "list_available_v3_features",
        "load_bridge_feature_map",
        "get_original_feature_name",
        "get_feature_safety_for_bridge",
    ])
