# -*- coding: utf-8 -*-
"""
explain_runner.py - Feature Explanation with Bridge Name Resolution

Step F-3: v4_bridge_* を "元のv3特徴量名" に解決して説明を生成するモジュール。

【機能】
1. bridge feature map を使った名前解決
2. display_name, bridged_name, origin, safety 情報の付与
3. JSON形式での explain 出力

【使用例】
    python -m src.features_v4.explain_runner --model-dir models/ --target target_win

    # 特定の特徴量リストを解説
    python -m src.features_v4.explain_runner --features "v4_bridge_hr_test,h_recent3_avg_finish"
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Bridge utilities
from .bridge_v3_features import (
    BRIDGE_PREFIX,
    load_bridge_feature_map,
    get_original_feature_name,
    get_feature_safety_for_bridge,
)


# =============================================================================
# JSON Sanitization
# =============================================================================


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize object for JSON serialization.
    Converts NaN/Infinity floats to None.
    """
    import math

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif obj is None or isinstance(obj, (str, int, bool)):
        return obj
    else:
        # Handle pandas NaN or other types
        try:
            import pandas as pd
            if pd.isna(obj):
                return None
        except (ImportError, TypeError):
            pass
        return obj


# =============================================================================
# Data Classes
# =============================================================================


# =============================================================================
# Feature Description Dictionaries
# =============================================================================

# Feature description mapping (v4 native features)
FEATURE_DESC_MAP: Dict[str, str] = {
    # Race attributes
    "surface_id": "芝/ダート",
    "distance": "距離(m)",
    "distance_cat": "距離カテゴリ",
    "track_condition_id": "馬場状態",
    "field_size": "出走頭数",
    "race_year": "開催年",
    "race_month": "開催月",
    "place_id": "開催場ID",
    "race_class": "レースクラス",
    "race_grade": "レースグレード",
    # Horse attributes
    "waku": "枠番",
    "umaban": "馬番",
    "horse_weight": "馬体重",
    "horse_weight_diff": "馬体重増減",
    "sex_id": "性別",
    "age": "馬齢",
    "is_blinker": "ブリンカー着用",
    # Historical stats - total
    "h_starts_total": "馬・総出走数",
    "h_win_rate_total": "馬・総合勝率",
    "h_in3_rate_total": "馬・総合複勝率",
    "h_avg_finish_total": "馬・平均着順",
    "h_std_finish_total": "馬・着順標準偏差",
    # Recent form
    "h_recent3_avg_finish": "馬・直近3走平均着順",
    "h_recent3_best_finish": "馬・直近3走最高着順",
    "h_recent3_avg_last3f": "馬・直近3走平均上がり3F",
    "h_recent5_avg_finish": "馬・直近5走平均着順",
    "h_recent5_best_finish": "馬・直近5走最高着順",
    "h_days_since_last": "馬・前走からの日数",
    # Distance category stats
    "h_starts_dist_cat": "馬・距離別出走数",
    "h_win_rate_dist_cat": "馬・距離別勝率",
    "h_in3_rate_dist_cat": "馬・距離別複勝率",
    "h_avg_finish_dist_cat": "馬・距離別平均着順",
    "h_avg_last3f_dist_cat": "馬・距離別平均上がり3F",
    # Track condition stats
    "h_starts_track_cond": "馬・馬場状態別出走数",
    "h_win_rate_track_cond": "馬・馬場状態別勝率",
    "h_in3_rate_track_cond": "馬・馬場状態別複勝率",
    # Course stats
    "h_starts_course": "馬・コース別出走数",
    "h_win_rate_course": "馬・コース別勝率",
    "h_in3_rate_course": "馬・コース別複勝率",
    # Surface stats
    "h_starts_surface": "馬・芝ダ別出走数",
    "h_win_rate_surface": "馬・芝ダ別勝率",
    "h_in3_rate_surface": "馬・芝ダ別複勝率",
    # Jockey stats
    "j_win_rate_total": "騎手・総合勝率",
    "j_in3_rate_total": "騎手・総合複勝率",
    "j_starts_total": "騎手・総出走数",
    "j_win_rate_recent30d": "騎手・直近30日勝率",
    "j_in3_rate_recent30d": "騎手・直近30日複勝率",
    # Trainer stats
    "t_win_rate_total": "調教師・総合勝率",
    "t_in3_rate_total": "調教師・総合複勝率",
    "t_starts_total": "調教師・総出走数",
    "t_win_rate_recent30d": "調教師・直近30日勝率",
    "t_in3_rate_recent30d": "調教師・直近30日複勝率",
    # Jockey-Horse combo
    "jh_win_rate": "騎手×馬・勝率",
    "jh_in3_rate": "騎手×馬・複勝率",
    "jh_starts": "騎手×馬・騎乗回数",
    # Trainer-Jockey combo
    "tj_win_rate": "調教師×騎手・勝率",
    "tj_in3_rate": "調教師×騎手・複勝率",
    "tj_starts": "調教師×騎手・コンビ回数",
    # Pedigree
    "sire_win_rate": "父馬勝率",
    "sire_in3_rate": "父馬複勝率",
    "bms_win_rate": "母父勝率",
    "bms_in3_rate": "母父複勝率",
    "sire_distance_apt": "父馬・距離適性",
    "bms_distance_apt": "母父・距離適性",
    # Weight features
    "avg_horse_weight": "馬・平均馬体重",
    # First run
    "is_first_run": "初出走フラグ",
}

# Bridged feature description mapping (v3 features, keyed by original name without prefix)
BRIDGED_DESC_MAP: Dict[str, str] = {
    # ax8 jockey/trainer asof features
    "ax8_jockey_in3_rate_total_asof": "騎手複勝率(時点)",
    "ax8_jockey_win_rate_total_asof": "騎手勝率(時点)",
    "ax8_trainer_in3_rate_total_asof": "調教師複勝率(時点)",
    "ax8_trainer_win_rate_total_asof": "調教師勝率(時点)",
    "ax8_jockey_starts_total_asof": "騎手出走数(時点)",
    "ax8_trainer_starts_total_asof": "調教師出走数(時点)",
    # hr (horse recent) features
    "hr_test": "馬過去成績(テスト)",
    "hr_avg_finish": "馬・過去平均着順",
    "hr_win_rate": "馬・過去勝率",
    "hr_in3_rate": "馬・過去複勝率",
    # Legacy v3 features
    "win_rate_total": "総合勝率",
    "in3_rate_total": "総合複勝率",
    "n_starts_total": "総出走数",
    "avg_finish_total": "平均着順",
    "std_finish_total": "着順標準偏差",
    "win_rate_dist_cat": "距離別勝率",
    "in3_rate_dist_cat": "距離別複勝率",
    "n_starts_dist_cat": "距離別出走数",
    "avg_finish_dist_cat": "距離別平均着順",
    "avg_last3f_dist_cat": "距離別平均上がり3F",
    "win_rate_track_condition": "馬場状態別勝率",
    "n_starts_track_condition": "馬場状態別出走数",
    "win_rate_course": "コース別勝率",
    "n_starts_course": "コース別出走数",
    "recent_avg_finish_3": "直近3走平均着順",
    "recent_best_finish_3": "直近3走最高着順",
    "recent_avg_last3f_3": "直近3走平均上がり3F",
    "days_since_last_run": "前走からの日数",
    # Lap features
    "hlap_overall_vs_race": "ラップ・全体vs平均",
    "hlap_early_vs_race": "ラップ・前半vs平均",
    "hlap_mid_vs_race": "ラップ・中盤vs平均",
    "hlap_late_vs_race": "ラップ・後半vs平均",
    "hlap_last600_vs_race": "ラップ・上がり3Fvs平均",
}


# =============================================================================
# Automatic Description Inference
# =============================================================================

# Prefix → Subject mapping
_PREFIX_MAP: Dict[str, str] = {
    "h_": "馬",
    "j_": "騎手",
    "t_": "調教師",
    "jh_": "騎手×馬",
    "tj_": "調教師×騎手",
    "race_": "レース",
    "sire_": "父馬",
    "bms_": "母父",
    "hlap_": "ラップ",
}

# Stat keywords → Japanese
_STAT_MAP: Dict[str, str] = {
    "win_rate": "勝率",
    "in3_rate": "複勝率",
    "avg_finish": "平均着順",
    "best_finish": "最高着順",
    "worst_finish": "最低着順",
    "std_finish": "着順標準偏差",
    "avg_last3f": "平均上がり3F",
    "starts": "出走数",
    "days_since": "経過日数",
}

# Condition suffixes → Japanese
_SUFFIX_MAP: Dict[str, str] = {
    "_total": "・総合",
    "_dist_cat": "・距離別",
    "_dist": "・距離別",
    "_course": "・コース別",
    "_surface": "・芝ダ別",
    "_track_cond": "・馬場状態別",
    "_track_condition": "・馬場状態別",
    "_place": "・開催場別",
    "_recent30d": "・直近30日",
    "_recent3": "・直近3走",
    "_recent5": "・直近5走",
    "_asof": "(時点)",
}

# Window patterns
_WINDOW_PATTERNS: Dict[str, str] = {
    "recent3": "直近3走",
    "recent5": "直近5走",
    "recent10": "直近10走",
    "recent30d": "直近30日",
    "recent60d": "直近60日",
    "recent90d": "直近90日",
}


def infer_desc_from_name(
    feature_name: str,
    display_name: str,
    origin: str,
    safety_label: str = "unknown",
) -> str:
    """
    Infer a human-readable description from feature naming conventions.

    Args:
        feature_name: The actual column name (may have v4_bridge_ prefix)
        display_name: The display name (original v3 name for bridged features)
        origin: "v4_native" or "v3_bridged"
        safety_label: Feature safety classification

    Returns:
        Inferred description string, or empty string if inference fails
    """
    import re

    # Use display_name for inference (it's the "real" name)
    name = display_name.lower()

    parts = []

    # 1) Detect subject prefix
    subject = ""
    for prefix, subj in _PREFIX_MAP.items():
        if name.startswith(prefix):
            subject = subj
            name = name[len(prefix):]
            break

    # 2) Detect window patterns (recent3, recent5, etc.)
    window = ""
    for pattern, desc in _WINDOW_PATTERNS.items():
        if pattern in name:
            window = desc
            name = name.replace(pattern, "")
            break

    # 3) Detect condition suffixes
    condition = ""
    for suffix, cond in _SUFFIX_MAP.items():
        if name.endswith(suffix):
            condition = cond
            name = name[: -len(suffix)]
            break

    # 4) Detect stat type
    stat = ""
    for key, val in _STAT_MAP.items():
        if key in name:
            stat = val
            name = name.replace(key, "")
            break

    # 5) Build description
    if subject:
        parts.append(subject)

    if window:
        parts.append(window)

    if stat:
        parts.append(stat)
    elif name.strip("_"):
        # Use remaining name as fallback
        remaining = name.strip("_").replace("_", " ")
        if remaining:
            parts.append(remaining)

    if condition:
        parts.append(condition)

    # Join parts
    if parts:
        # Clean up formatting: subject・stat・condition
        result = "・".join(parts)
        # Fix double separators
        result = re.sub(r"・+", "・", result)
        result = result.strip("・")
        return result

    return ""


def get_feature_desc(
    feature_name: str,
    display_name: str,
    origin: str,
    safety_label: str = "unknown",
) -> str:
    """
    Get human-readable description for a feature.

    Priority:
    1. Exact match in FEATURE_DESC_MAP (v4 native dict)
    2. Exact match in BRIDGED_DESC_MAP (v3 bridged dict) - for bridged features
    3. display_name match in FEATURE_DESC_MAP
    4. Auto-infer from naming conventions (infer_desc_from_name)
    5. Empty string as final fallback

    Args:
        feature_name: The actual feature column name
        display_name: The display name (original name for bridged features)
        origin: "v4_native" or "v3_bridged"
        safety_label: Feature safety classification (for inference context)

    Returns:
        Description string, or empty string if not found
    """
    # 1) Try feature_name in v4 native map
    if feature_name in FEATURE_DESC_MAP:
        return FEATURE_DESC_MAP[feature_name]

    # 2) For bridged features, try display_name in bridged map
    if origin == "v3_bridged" and display_name in BRIDGED_DESC_MAP:
        return BRIDGED_DESC_MAP[display_name]

    # 3) Try display_name in v4 map as fallback
    if display_name in FEATURE_DESC_MAP:
        return FEATURE_DESC_MAP[display_name]

    # 4) Auto-infer from naming conventions
    inferred = infer_desc_from_name(feature_name, display_name, origin, safety_label)
    if inferred:
        return inferred

    # 5) Final fallback: empty string
    return ""


@dataclass
class FeatureExplanation:
    """Single feature explanation with bridge name resolution"""
    feature_name: str  # Actual column name in the model
    display_name: str  # Human-readable name (original v3 name if bridged)
    origin: str  # "v4_native" or "v3_bridged"
    safety_label: str  # "safe", "warn", "unsafe", "unknown"
    safety_notes: str  # Human-readable safety notes
    importance_gain: float = 0.0  # Feature importance (gain)
    importance_split: int = 0  # Feature importance (split)
    contribution: float = 0.0  # SHAP or other contribution (if available)
    desc: str = ""  # Human-readable description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ExplainResult:
    """Complete explanation result for a prediction"""
    target: str
    n_features: int
    n_bridged: int
    n_native: int
    features: List[FeatureExplanation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # MVP schema extensions
    schema_version: str = "1.0"
    generated_at: str = ""
    model_version: str = "v4"

    def __post_init__(self):
        """Set generated_at if not provided"""
        if not self.generated_at:
            from datetime import datetime
            self.generated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "target": self.target,
            "n_features": self.n_features,
            "n_bridged": self.n_bridged,
            "n_native": self.n_native,
            "features": [f.to_dict() for f in self.features],
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> None:
        """Save to JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(self.to_dict()), f, indent=2, ensure_ascii=True)


# =============================================================================
# Core Functions
# =============================================================================


def resolve_feature_name(
    feature_name: str,
    bridge_map_data: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    """
    Resolve feature name to display name with safety info

    Args:
        feature_name: Feature column name (may or may not have bridge prefix)
        bridge_map_data: Loaded bridge feature map data

    Returns:
        (display_name, origin, safety_label, safety_notes)
    """
    feature_map = bridge_map_data.get("feature_map", {})

    # Get original name (for display)
    if feature_name.startswith(BRIDGE_PREFIX):
        display_name = get_original_feature_name(feature_name, feature_map)
        origin = "v3_bridged"
    else:
        display_name = feature_name
        origin = "v4_native"

    # Get safety info
    safety_label, safety_notes, _ = get_feature_safety_for_bridge(
        feature_name, bridge_map_data
    )

    return display_name, origin, safety_label, safety_notes


def build_feature_explanation(
    feature_name: str,
    bridge_map_data: Dict[str, Any],
    importance_gain: float = 0.0,
    importance_split: int = 0,
    contribution: float = 0.0,
) -> FeatureExplanation:
    """
    Build a single feature explanation

    Args:
        feature_name: Feature column name
        bridge_map_data: Loaded bridge feature map data
        importance_gain: Feature importance (gain)
        importance_split: Feature importance (split)
        contribution: SHAP or other contribution

    Returns:
        FeatureExplanation object
    """
    display_name, origin, safety_label, safety_notes = resolve_feature_name(
        feature_name, bridge_map_data
    )
    desc = get_feature_desc(feature_name, display_name, origin, safety_label)

    return FeatureExplanation(
        feature_name=feature_name,
        display_name=display_name,
        origin=origin,
        safety_label=safety_label,
        safety_notes=safety_notes,
        importance_gain=importance_gain,
        importance_split=importance_split,
        contribution=contribution,
        desc=desc,
    )


def generate_explain_result(
    feature_names: List[str],
    target: str,
    bridge_map_data: Dict[str, Any],
    importance_df: Optional["pd.DataFrame"] = None,
    exclude_warn: bool = False,
    model_version: str = "v4",
) -> ExplainResult:
    """
    Generate complete explanation result for features

    Args:
        feature_names: List of feature column names
        target: Target column name
        bridge_map_data: Loaded bridge feature map data
        importance_df: DataFrame with feature importance (optional)
        exclude_warn: If True, exclude warn features from output
        model_version: Model version string (e.g. "v4")

    Returns:
        ExplainResult object
    """
    import pandas as pd

    features = []
    n_bridged = 0
    n_native = 0

    # Build importance lookup
    importance_lookup = {}
    if importance_df is not None:
        for _, row in importance_df.iterrows():
            feat = row.get("feature", "")
            # Support both column naming conventions: gain/split and importance_gain/importance_split
            gain_val = row.get("importance_gain", row.get("gain", 0.0))
            split_val = row.get("importance_split", row.get("split", 0))
            importance_lookup[feat] = {
                "gain": gain_val if gain_val is not None else 0.0,
                "split": int(split_val) if split_val is not None else 0,
            }

    for feature_name in feature_names:
        display_name, origin, safety_label, safety_notes = resolve_feature_name(
            feature_name, bridge_map_data
        )

        # Skip warn features if requested
        if exclude_warn and safety_label == "warn":
            continue

        # Get importance if available
        imp = importance_lookup.get(feature_name, {})
        importance_gain = imp.get("gain", 0.0)
        importance_split = imp.get("split", 0)

        # Get description
        desc = get_feature_desc(feature_name, display_name, origin, safety_label)

        explanation = FeatureExplanation(
            feature_name=feature_name,
            display_name=display_name,
            origin=origin,
            safety_label=safety_label,
            safety_notes=safety_notes,
            importance_gain=importance_gain,
            importance_split=importance_split,
            desc=desc,
        )

        features.append(explanation)

        if origin == "v3_bridged":
            n_bridged += 1
        else:
            n_native += 1

    # Sort by importance: gain descending, then split descending
    features.sort(key=lambda x: (x.importance_gain, x.importance_split), reverse=True)

    return ExplainResult(
        target=target,
        n_features=len(features),
        n_bridged=n_bridged,
        n_native=n_native,
        features=features,
        metadata={
            "exclude_warn": exclude_warn,
            "bridge_map_loaded": bool(bridge_map_data),
        },
        model_version=model_version,
    )


def load_importance_from_csv(csv_path: Path) -> "pd.DataFrame":
    """Load feature importance from CSV file"""
    import pandas as pd

    if not csv_path.exists():
        logger.warning(f"Importance CSV not found: {csv_path}")
        return pd.DataFrame()

    return pd.read_csv(csv_path, encoding="utf-8")


def load_feature_columns_from_json(json_path: Path) -> List[str]:
    """Load feature columns from JSON file"""
    if not json_path.exists():
        logger.warning(f"Feature columns JSON not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# CLI Runner
# =============================================================================


def run_explain(
    model_dir: Path,
    target: str = "target_win",
    version: str = "v4",
    exclude_warn: bool = False,
    output_path: Optional[Path] = None,
) -> ExplainResult:
    """
    Run feature explanation generation

    Args:
        model_dir: Directory containing model artifacts
        target: Target column name
        version: Model version
        exclude_warn: If True, exclude warn features
        output_path: Optional path to save explain JSON

    Returns:
        ExplainResult object
    """
    logger.info("=" * 60)
    logger.info("Feature Explanation Generator")
    logger.info("=" * 60)

    # Load bridge feature map
    bridge_map_path = model_dir / f"bridge_feature_map_{target}.json"
    bridge_map_data = load_bridge_feature_map(bridge_map_path)

    if bridge_map_data:
        logger.info(f"Loaded bridge feature map: {bridge_map_path}")
    else:
        logger.warning(f"No bridge feature map found at {bridge_map_path}")
        bridge_map_data = {}

    # Load feature columns
    feature_cols_path = model_dir / f"feature_columns_{target}_{version}.json"
    feature_names = load_feature_columns_from_json(feature_cols_path)

    if not feature_names:
        # Fallback: try without version suffix
        feature_cols_path = model_dir / f"feature_columns_{target}.json"
        feature_names = load_feature_columns_from_json(feature_cols_path)

    if not feature_names:
        logger.warning("No feature columns found, using bridge map features")
        feature_names = list(bridge_map_data.get("feature_map", {}).keys())

    logger.info(f"Loaded {len(feature_names)} feature columns")

    # Load feature importance
    importance_path = model_dir / f"feature_importance_{target}_{version}.csv"
    importance_df = load_importance_from_csv(importance_path)

    if importance_df.empty:
        # Fallback: try without version suffix
        importance_path = model_dir / f"feature_importance_{target}.csv"
        importance_df = load_importance_from_csv(importance_path)

    if not importance_df.empty:
        logger.info(f"Loaded feature importance from {importance_path}")

    # Generate explanation
    result = generate_explain_result(
        feature_names=feature_names,
        target=target,
        bridge_map_data=bridge_map_data,
        importance_df=importance_df if not importance_df.empty else None,
        exclude_warn=exclude_warn,
    )

    logger.info("=" * 60)
    logger.info(f"Explanation generated:")
    logger.info(f"  Total features: {result.n_features}")
    logger.info(f"  Native (v4): {result.n_native}")
    logger.info(f"  Bridged (v3): {result.n_bridged}")
    logger.info("=" * 60)

    # Print top features
    logger.info("Top 10 features by importance:")
    for i, feat in enumerate(result.features[:10]):
        origin_tag = "[v3]" if feat.origin == "v3_bridged" else "[v4]"
        logger.info(
            f"  {i+1}. {feat.display_name} {origin_tag} "
            f"(gain={feat.importance_gain:.2f}, safety={feat.safety_label})"
        )

    # Save if output path provided
    if output_path:
        result.to_json(output_path)
        logger.info(f"Saved explain JSON to: {output_path}")

    return result


def generate_explain_from_pipeline(
    feature_cols: List[str],
    target: str,
    output_dir: Path,
    bridge_map_path: Optional[Path] = None,
    importance_csv_path: Optional[Path] = None,
    model_version: str = "v4",
) -> Optional[ExplainResult]:
    """
    Generate explain JSON from pipeline context (called by train_eval_v4)

    This is a convenience wrapper for run_full_pipeline integration.
    Errors are logged but not raised (explain failure shouldn't affect training).

    Args:
        feature_cols: List of feature column names used in model
        target: Target column name
        output_dir: Output directory for explain JSON
        bridge_map_path: Path to bridge feature map JSON (optional)
        importance_csv_path: Path to feature importance CSV (optional)
        model_version: Model version string (e.g. "v4")

    Returns:
        ExplainResult if successful, None if failed
    """
    import pandas as pd

    try:
        logger.info("-" * 40)
        logger.info("Generating Explain JSON (F-3)")
        logger.info("-" * 40)

        # Load bridge map if available
        bridge_map_data = {}
        if bridge_map_path and bridge_map_path.exists():
            bridge_map_data = load_bridge_feature_map(bridge_map_path)
            logger.info(f"Loaded bridge map: {bridge_map_path}")
        else:
            logger.info("No bridge map available (v4 native features only)")

        # Load importance if available
        importance_df = None
        if importance_csv_path and importance_csv_path.exists():
            importance_df = pd.read_csv(importance_csv_path, encoding="utf-8")
            logger.info(f"Loaded importance: {importance_csv_path}")

        # Generate explain result
        result = generate_explain_result(
            feature_names=feature_cols,
            target=target,
            bridge_map_data=bridge_map_data,
            importance_df=importance_df,
            exclude_warn=False,
            model_version=model_version,
        )

        # Save to output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"explain_{target}_{model_version}.json"
        result.to_json(output_path)

        logger.info(f"Explain JSON generated: {output_path}")
        logger.info(f"  Features: {result.n_features} (native: {result.n_native}, bridged: {result.n_bridged})")

        return result

    except Exception as e:
        logger.warning(f"Explain generation failed (non-fatal): {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate feature explanation with bridge name resolution"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model artifacts",
    )
    parser.add_argument(
        "--target",
        default="target_win",
        help="Target column name",
    )
    parser.add_argument(
        "--version",
        default="v4",
        help="Model version",
    )
    parser.add_argument(
        "--exclude-warn",
        action="store_true",
        help="Exclude warn features from output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for explain JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Default output path
    output_path = args.output
    if output_path is None:
        output_path = args.model_dir / f"explain_{args.target}_{args.version}.json"

    result = run_explain(
        model_dir=args.model_dir,
        target=args.target,
        version=args.version,
        exclude_warn=args.exclude_warn,
        output_path=output_path,
    )

    print(f"\nGenerated explanation with {result.n_features} features")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
