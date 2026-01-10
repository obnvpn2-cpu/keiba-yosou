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
    # Horse attributes
    "waku": "枠番",
    "umaban": "馬番",
    "horse_weight": "馬体重",
    "horse_weight_diff": "馬体重増減",
    "sex_id": "性別",
    "age": "馬齢",
    # Historical stats
    "h_starts_total": "総出走数",
    "h_win_rate_total": "総合勝率",
    "h_in3_rate_total": "総合複勝率",
    "h_avg_finish_total": "平均着順",
    "h_recent3_avg_finish": "直近3走平均着順",
    "h_recent3_best_finish": "直近3走最高着順",
    "h_recent3_avg_last3f": "直近3走平均上がり3F",
    "h_days_since_last": "前走からの日数",
    # Distance category stats
    "h_starts_dist_cat": "距離別出走数",
    "h_win_rate_dist_cat": "距離別勝率",
    "h_in3_rate_dist_cat": "距離別複勝率",
    # Track condition stats
    "h_starts_track_cond": "馬場別出走数",
    "h_win_rate_track_cond": "馬場別勝率",
    # Course stats
    "h_starts_course": "コース別出走数",
    "h_win_rate_course": "コース別勝率",
    # Pedigree
    "sire_win_rate": "父馬勝率",
    "sire_in3_rate": "父馬複勝率",
    "bms_win_rate": "母父勝率",
    "bms_in3_rate": "母父複勝率",
}

# Bridged feature description mapping (v3 features, keyed by original name without prefix)
BRIDGED_DESC_MAP: Dict[str, str] = {
    "ax8_jockey_in3_rate_total_asof": "騎手複勝率(asof)",
    "ax8_jockey_win_rate_total_asof": "騎手勝率(asof)",
    "ax8_trainer_in3_rate_total_asof": "調教師複勝率(asof)",
    "ax8_trainer_win_rate_total_asof": "調教師勝率(asof)",
    "hr_test": "馬過去成績(テスト)",
}


def get_feature_desc(feature_name: str, display_name: str, origin: str) -> str:
    """
    Get human-readable description for a feature.

    Args:
        feature_name: The actual feature column name
        display_name: The display name (original name for bridged features)
        origin: "v4_native" or "v3_bridged"

    Returns:
        Description string, or empty string if not found
    """
    # Try v4 native map first
    if feature_name in FEATURE_DESC_MAP:
        return FEATURE_DESC_MAP[feature_name]

    # For bridged features, try display_name in bridged map
    if origin == "v3_bridged" and display_name in BRIDGED_DESC_MAP:
        return BRIDGED_DESC_MAP[display_name]

    # Try display_name in v4 map as fallback
    if display_name in FEATURE_DESC_MAP:
        return FEATURE_DESC_MAP[display_name]

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
            json.dump(sanitize_for_json(self.to_dict()), f, indent=2, ensure_ascii=False)


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
    desc = get_feature_desc(feature_name, display_name, origin)

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
        desc = get_feature_desc(feature_name, display_name, origin)

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
