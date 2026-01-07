#!/usr/bin/env python3
"""
report_featurepacks.py - Feature Comparison Report Generator (Step E)

全バージョン横断の特徴量比較レポートを生成するスクリプト。
audit_featurepacks.py の出力を読み込み、比較用の成果物を生成する。

【入力】
- artifacts/feature_audit/{version}/{mode}/{target}/
  - used_features.txt
  - feature_inventory.json
  - importance_gain.csv
  - importance_split.csv
  - group_importance.csv
  - summary.json

【出力】
- artifacts/feature_audit/report/
  - index.md (人間が読むサマリー)
  - feature_compare.csv (横持ち比較の主テーブル)
  - feature_compare.json (同内容)
  - top_features_{target}_{version}.csv (各versionの上位N)
  - common_top_features_{target}.csv (共通上位)
  - diff_v4_vs_v3_{target}.csv (v4で伸びた/増えた)
  - safety_summary_{target}.csv (unsafe/warnの件数と上位)

Usage:
    # 基本実行
    python scripts/report_featurepacks.py

    # 入力ディレクトリを指定
    python scripts/report_featurepacks.py --input artifacts/feature_audit

    # 出力ディレクトリを指定
    python scripts/report_featurepacks.py --output artifacts/feature_audit/report

    # dry-run（生成ファイル一覧のみ）
    python scripts/report_featurepacks.py --dry-run
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_audit.safety import classify_feature_safety

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "artifacts" / "feature_audit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "feature_audit" / "report"

DEFAULT_VERSIONS = ["v4", "v3", "v2", "v1"]
DEFAULT_TARGETS = ["target_win", "target_in3"]
DEFAULT_MODE = "pre_race"

TOP_N_FEATURES = 30  # Top N features to highlight


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureData:
    """Single feature data across versions"""
    feature: str
    safety_label: str = "safe"
    safety_notes: str = ""
    # Per-version data: {version: {importance_gain, importance_split, rank, ...}}
    version_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VersionAuditData:
    """Audit data for a single version/target combination"""
    version: str
    target: str
    mode: str
    features: List[str] = field(default_factory=list)
    importance_gain: Dict[str, float] = field(default_factory=dict)
    importance_split: Dict[str, float] = field(default_factory=dict)
    group_importance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    feature_inventory: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Data Loading
# =============================================================================

def load_audit_data(
    input_dir: Path,
    versions: List[str],
    targets: List[str],
    mode: str = DEFAULT_MODE,
    models_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, VersionAuditData]]:
    """
    Load audit data from artifacts directory with fallback to models directory.

    Args:
        input_dir: Audit artifacts directory (artifacts/feature_audit/)
        versions: List of versions to load
        targets: List of targets to load
        mode: Mode (pre_race, etc.)
        models_dir: Models directory for fallback loading

    Returns:
        {version: {target: VersionAuditData}}
    """
    result: Dict[str, Dict[str, VersionAuditData]] = {}

    for version in versions:
        result[version] = {}
        for target in targets:
            audit_dir = input_dir / version / mode / target
            data = VersionAuditData(version=version, target=target, mode=mode)

            if audit_dir.exists():
                # Load from audit artifacts
                _load_from_audit_artifacts(audit_dir, data)
            else:
                logger.warning(f"Audit directory not found: {audit_dir}")

            # Fallback: load from models directory if importance is missing
            if models_dir and not data.importance_gain:
                logger.info(f"Trying fallback load from models/ for {version}/{target}")
                _load_from_models_fallback(models_dir, version, target, data)

            # Only add if we have some data
            if data.features or data.importance_gain:
                result[version][target] = data
                logger.info(
                    f"Loaded {version}/{target}: "
                    f"{len(data.features)} features, "
                    f"{len(data.importance_gain)} importance entries"
                )

    return result


def _load_from_audit_artifacts(audit_dir: Path, data: VersionAuditData) -> None:
    """Load data from audit artifacts directory."""
    # Load used_features.txt
    features_file = audit_dir / "used_features.txt"
    if features_file.exists():
        with open(features_file, "r", encoding="utf-8") as f:
            data.features = [line.strip() for line in f if line.strip()]

    # Load importance_gain.csv
    gain_file = audit_dir / "importance_gain.csv"
    if gain_file.exists():
        data.importance_gain = _load_importance_csv(gain_file)

    # Load importance_split.csv
    split_file = audit_dir / "importance_split.csv"
    if split_file.exists():
        data.importance_split = _load_importance_csv(split_file)

    # Load group_importance.csv
    group_file = audit_dir / "group_importance.csv"
    if group_file.exists():
        data.group_importance = _load_group_importance_csv(group_file)

    # Load summary.json
    summary_file = audit_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            data.summary = json.load(f)

    # Load feature_inventory.json
    inventory_file = audit_dir / "feature_inventory.json"
    if inventory_file.exists():
        with open(inventory_file, "r", encoding="utf-8") as f:
            data.feature_inventory = json.load(f)


def _load_from_models_fallback(
    models_dir: Path,
    version: str,
    target: str,
    data: VersionAuditData,
) -> None:
    """
    Fallback: load directly from models directory.

    Tries versioned files first, then version-less files.
    """
    # Try versioned files first, then version-less
    suffixes = [f"_{version}", ""]

    # Load feature columns
    if not data.features:
        for suffix in suffixes:
            json_path = models_dir / f"feature_columns_{target}{suffix}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data.features = json.load(f)
                    logger.info(f"Loaded features from: {json_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {json_path}: {e}")

    # Load importance from CSV
    if not data.importance_gain:
        for suffix in suffixes:
            csv_path = models_dir / f"feature_importance_{target}{suffix}.csv"
            if csv_path.exists():
                try:
                    data.importance_gain = _load_importance_csv_flexible(csv_path)
                    logger.info(f"Loaded importance from: {csv_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {csv_path}: {e}")

    # Fallback: load from model file directly using lightgbm
    if not data.importance_gain:
        for suffix in suffixes:
            model_path = models_dir / f"lgbm_{target}{suffix}.txt"
            if model_path.exists():
                try:
                    gain, split = _load_importance_from_model(model_path)
                    if gain:
                        data.importance_gain = gain
                        data.importance_split = split
                        # Use model's features as feature list if not loaded
                        if not data.features:
                            data.features = list(gain.keys())
                        logger.info(f"Loaded importance from model: {model_path}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load model {model_path}: {e}")


def _load_importance_from_model(model_path: Path) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Load importance directly from LightGBM model file."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not available for model loading")
        return {}, {}

    try:
        # Try loading model file
        try:
            model = lgb.Booster(model_file=str(model_path))
        except Exception:
            # Fallback: read as string
            model_str = model_path.read_text(encoding="utf-8")
            model = lgb.Booster(model_str=model_str)

        feature_names = model.feature_name()
        gain_values = model.feature_importance(importance_type="gain")
        split_values = model.feature_importance(importance_type="split")

        gain_dict = dict(zip(feature_names, gain_values.tolist()))
        split_dict = dict(zip(feature_names, split_values.astype(int).tolist()))

        return gain_dict, split_dict
    except Exception as e:
        logger.warning(f"Failed to load importance from model: {e}")
        return {}, {}


def _load_importance_csv_flexible(path: Path) -> Dict[str, float]:
    """Load importance CSV with flexible column names (gain/importance/split)."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = row.get("feature", "")
            # Try different column names
            importance = None
            for col in ["gain", "importance", "split"]:
                if col in row and row[col]:
                    try:
                        importance = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
            if feature and importance is not None:
                result[feature] = importance
    return result


def _load_importance_csv(path: Path) -> Dict[str, float]:
    """Load importance CSV file."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = row.get("feature", "")
            importance = float(row.get("importance", 0))
            if feature:
                result[feature] = importance
    return result


def _load_group_importance_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load group importance CSV file."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row.get("group", "")
            if group:
                result[group] = {
                    "gain_sum": float(row.get("importance_gain_sum", 0)),
                    "split_sum": int(row.get("importance_split_sum", 0)),
                    "n_features": int(row.get("n_features", 0)),
                }
    return result


# =============================================================================
# Report Generation
# =============================================================================

def generate_feature_compare(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    target: str,
    versions: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate feature comparison data for a target.

    Returns list of dicts with:
    - feature
    - safety_label
    - safety_notes
    - For each version:
      - {version}_rank
      - {version}_gain
      - {version}_gain_norm (0-1 normalized)
      - {version}_cumulative
    """
    # Collect all features across versions
    all_features: Set[str] = set()
    for version in versions:
        if version in audit_data and target in audit_data[version]:
            all_features.update(audit_data[version][target].features)

    # Build comparison data
    rows = []
    for feature in sorted(all_features):
        row: Dict[str, Any] = {
            "feature": feature,
        }

        # Get safety label
        safety_label, safety_notes = classify_feature_safety(feature)
        row["safety_label"] = safety_label
        row["safety_notes"] = safety_notes

        # Get prefix/group
        if "_" in feature:
            row["group"] = feature.split("_")[0] + "_"
        else:
            row["group"] = "other"

        # Per-version data
        for version in versions:
            prefix = version

            if version not in audit_data or target not in audit_data[version]:
                row[f"{prefix}_rank"] = None
                row[f"{prefix}_gain"] = None
                row[f"{prefix}_gain_norm"] = None
                row[f"{prefix}_cumulative"] = None
                row[f"{prefix}_present"] = False
                continue

            vdata = audit_data[version][target]
            gain_dict = vdata.importance_gain

            if feature in gain_dict:
                row[f"{prefix}_present"] = True
                row[f"{prefix}_gain"] = gain_dict[feature]

                # Calculate rank
                sorted_features = sorted(gain_dict.items(), key=lambda x: -x[1])
                for i, (f, _) in enumerate(sorted_features, 1):
                    if f == feature:
                        row[f"{prefix}_rank"] = i
                        break

                # Normalized gain (0-1)
                max_gain = max(gain_dict.values()) if gain_dict else 1
                row[f"{prefix}_gain_norm"] = gain_dict[feature] / max_gain if max_gain > 0 else 0

                # Cumulative importance
                total_gain = sum(gain_dict.values())
                sorted_gains = sorted(gain_dict.values(), reverse=True)
                cumulative = 0.0
                for g in sorted_gains:
                    cumulative += g
                    if g == gain_dict[feature]:
                        break
                row[f"{prefix}_cumulative"] = cumulative / total_gain if total_gain > 0 else 0
            else:
                row[f"{prefix}_present"] = False
                row[f"{prefix}_rank"] = None
                row[f"{prefix}_gain"] = None
                row[f"{prefix}_gain_norm"] = None
                row[f"{prefix}_cumulative"] = None

        rows.append(row)

    # Sort by v4 rank (if available), then by feature name
    def sort_key(r):
        v4_rank = r.get("v4_rank")
        if v4_rank is None:
            return (1, 9999, r["feature"])
        return (0, v4_rank, r["feature"])

    rows.sort(key=sort_key)

    return rows


def generate_top_features(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    target: str,
    version: str,
    top_n: int = TOP_N_FEATURES,
) -> List[Dict[str, Any]]:
    """Generate top N features for a specific version/target."""
    if version not in audit_data or target not in audit_data[version]:
        return []

    vdata = audit_data[version][target]
    gain_dict = vdata.importance_gain

    if not gain_dict:
        return []

    sorted_features = sorted(gain_dict.items(), key=lambda x: -x[1])[:top_n]
    total_gain = sum(gain_dict.values())

    rows = []
    cumulative = 0.0
    for rank, (feature, gain) in enumerate(sorted_features, 1):
        cumulative += gain
        safety_label, safety_notes = classify_feature_safety(feature)

        rows.append({
            "rank": rank,
            "feature": feature,
            "importance_gain": gain,
            "importance_gain_norm": gain / max(gain_dict.values()) if gain_dict else 0,
            "cumulative_pct": cumulative / total_gain if total_gain > 0 else 0,
            "safety_label": safety_label,
            "safety_notes": safety_notes,
            "group": feature.split("_")[0] + "_" if "_" in feature else "other",
        })

    return rows


def generate_common_top_features(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    target: str,
    versions: List[str],
    top_n: int = TOP_N_FEATURES,
) -> List[Dict[str, Any]]:
    """Generate features that are in top N across all versions."""
    # Get top N features for each version
    top_sets: Dict[str, Set[str]] = {}
    top_ranks: Dict[str, Dict[str, int]] = {}  # version -> feature -> rank

    for version in versions:
        if version not in audit_data or target not in audit_data[version]:
            continue

        vdata = audit_data[version][target]
        gain_dict = vdata.importance_gain

        if not gain_dict:
            continue

        sorted_features = sorted(gain_dict.items(), key=lambda x: -x[1])[:top_n]
        top_sets[version] = set(f for f, _ in sorted_features)
        top_ranks[version] = {f: i + 1 for i, (f, _) in enumerate(sorted_features)}

    if not top_sets:
        return []

    # Find common features (in at least 2 versions)
    all_top_features = set()
    for features in top_sets.values():
        all_top_features.update(features)

    rows = []
    for feature in all_top_features:
        present_in = [v for v in versions if v in top_sets and feature in top_sets[v]]

        if len(present_in) < 2:
            continue

        safety_label, safety_notes = classify_feature_safety(feature)
        row = {
            "feature": feature,
            "n_versions": len(present_in),
            "versions": ",".join(present_in),
            "safety_label": safety_label,
            "safety_notes": safety_notes,
        }

        # Add rank for each version
        for version in versions:
            if version in top_ranks and feature in top_ranks[version]:
                row[f"{version}_rank"] = top_ranks[version][feature]
            else:
                row[f"{version}_rank"] = None

        rows.append(row)

    # Sort by number of versions (descending), then by average rank
    def sort_key(r):
        ranks = [r.get(f"{v}_rank", 9999) or 9999 for v in versions]
        avg_rank = sum(ranks) / len(ranks)
        return (-r["n_versions"], avg_rank)

    rows.sort(key=sort_key)

    return rows


def generate_diff_v4_vs_v3(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    target: str,
) -> List[Dict[str, Any]]:
    """Generate features that improved or are new in v4 vs v3."""
    if "v4" not in audit_data or target not in audit_data["v4"]:
        return []
    if "v3" not in audit_data or target not in audit_data["v3"]:
        return []

    v4_data = audit_data["v4"][target]
    v3_data = audit_data["v3"][target]

    v4_gain = v4_data.importance_gain
    v3_gain = v3_data.importance_gain

    v4_features = set(v4_data.features)
    v3_features = set(v3_data.features)

    rows = []

    # New in v4
    new_in_v4 = v4_features - v3_features
    for feature in new_in_v4:
        safety_label, safety_notes = classify_feature_safety(feature)
        rows.append({
            "feature": feature,
            "change_type": "new_in_v4",
            "v4_gain": v4_gain.get(feature, 0),
            "v3_gain": None,
            "gain_diff": v4_gain.get(feature, 0),
            "v4_rank": _get_rank(v4_gain, feature),
            "v3_rank": None,
            "rank_diff": None,
            "safety_label": safety_label,
            "safety_notes": safety_notes,
        })

    # Removed in v4 (was in v3)
    removed_in_v4 = v3_features - v4_features
    for feature in removed_in_v4:
        safety_label, safety_notes = classify_feature_safety(feature)
        rows.append({
            "feature": feature,
            "change_type": "removed_in_v4",
            "v4_gain": None,
            "v3_gain": v3_gain.get(feature, 0),
            "gain_diff": -v3_gain.get(feature, 0),
            "v4_rank": None,
            "v3_rank": _get_rank(v3_gain, feature),
            "rank_diff": None,
            "safety_label": safety_label,
            "safety_notes": safety_notes,
        })

    # Changed importance
    common_features = v4_features & v3_features
    for feature in common_features:
        v4_g = v4_gain.get(feature, 0)
        v3_g = v3_gain.get(feature, 0)
        v4_r = _get_rank(v4_gain, feature)
        v3_r = _get_rank(v3_gain, feature)

        gain_diff = v4_g - v3_g
        rank_diff = (v3_r - v4_r) if (v4_r and v3_r) else None  # Positive = improved

        # Only include if there's significant change
        if abs(gain_diff) < 0.01 and (rank_diff is None or abs(rank_diff) < 5):
            continue

        safety_label, safety_notes = classify_feature_safety(feature)
        rows.append({
            "feature": feature,
            "change_type": "changed",
            "v4_gain": v4_g,
            "v3_gain": v3_g,
            "gain_diff": gain_diff,
            "v4_rank": v4_r,
            "v3_rank": v3_r,
            "rank_diff": rank_diff,
            "safety_label": safety_label,
            "safety_notes": safety_notes,
        })

    # Sort by absolute gain diff descending
    rows.sort(key=lambda x: -abs(x.get("gain_diff") or 0))

    return rows


def generate_safety_summary(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    target: str,
    versions: List[str],
    top_n: int = TOP_N_FEATURES,
) -> List[Dict[str, Any]]:
    """Generate safety summary for each version."""
    rows = []

    for version in versions:
        if version not in audit_data or target not in audit_data[version]:
            continue

        vdata = audit_data[version][target]
        gain_dict = vdata.importance_gain

        # Count safety labels
        n_unsafe = 0
        n_warn = 0
        n_safe = 0
        unsafe_in_top = []
        warn_in_top = []

        sorted_features = sorted(gain_dict.items(), key=lambda x: -x[1])

        for rank, (feature, gain) in enumerate(sorted_features, 1):
            label, _ = classify_feature_safety(feature)
            if label == "unsafe":
                n_unsafe += 1
                if rank <= top_n:
                    unsafe_in_top.append(f"{feature} (rank={rank})")
            elif label == "warn":
                n_warn += 1
                if rank <= top_n:
                    warn_in_top.append(f"{feature} (rank={rank})")
            else:
                n_safe += 1

        rows.append({
            "version": version,
            "n_features": len(gain_dict),
            "n_unsafe": n_unsafe,
            "n_warn": n_warn,
            "n_safe": n_safe,
            "unsafe_pct": n_unsafe / len(gain_dict) * 100 if gain_dict else 0,
            "warn_pct": n_warn / len(gain_dict) * 100 if gain_dict else 0,
            "unsafe_in_top_n": len(unsafe_in_top),
            "warn_in_top_n": len(warn_in_top),
            "unsafe_features_in_top": "; ".join(unsafe_in_top[:5]),
            "warn_features_in_top": "; ".join(warn_in_top[:5]),
        })

    return rows


# =============================================================================
# Step F-0: Migration Candidates
# =============================================================================

def generate_migration_candidates(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    target: str,
    source_version: str = "v3",
    target_version: str = "v4",
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    """
    Generate migration candidates: features in source's top-K but not in target.

    Args:
        audit_data: Audit data
        target: Target column (target_win, target_in3)
        source_version: Source version to pull candidates from
        target_version: Target version that should receive features
        top_k: Top K features from source to consider

    Returns:
        List of candidate dicts with suggested actions based on safety
    """
    if source_version not in audit_data or target not in audit_data[source_version]:
        return []
    if target_version not in audit_data or target not in audit_data[target_version]:
        # Still return candidates but mark target as missing
        pass

    source_data = audit_data[source_version][target]
    source_gain = source_data.importance_gain

    # Get target features (may be empty if target version is missing)
    target_features = set()
    if target_version in audit_data and target in audit_data[target_version]:
        target_features = set(audit_data[target_version][target].features)

    # Get top-K from source
    sorted_source = sorted(source_gain.items(), key=lambda x: -x[1])[:top_k]

    rows = []
    for rank, (feature, gain) in enumerate(sorted_source, 1):
        # Only include features NOT in target
        if feature in target_features:
            continue

        safety_label, safety_notes = classify_feature_safety(feature)

        # Get group/prefix
        if "_" in feature:
            group = feature.split("_")[0] + "_"
        else:
            group = "other"

        # Determine suggested action based on safety
        if safety_label == "unsafe":
            suggested_action = "skip"
            blocked_reason = f"unsafe: {safety_notes}"
        elif safety_label == "warn":
            suggested_action = "needs_review"
            blocked_reason = f"warn: {safety_notes}"
        else:
            suggested_action = "port"
            blocked_reason = ""

        rows.append({
            "feature": feature,
            f"{source_version}_gain": gain,
            f"{source_version}_rank": rank,
            "safety_label": safety_label,
            "safety_notes": safety_notes,
            "reason_tag": group,
            "blocked_reason": blocked_reason,
            "suggested_action": suggested_action,
        })

    # Sort by source rank
    rows.sort(key=lambda x: x.get(f"{source_version}_rank", 999))

    return rows


def generate_candidate_summary(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    targets: List[str],
    source_version: str = "v3",
    target_version: str = "v4",
    top_k: int = 30,
) -> str:
    """
    Generate candidate summary markdown content.

    Returns:
        Markdown string with summary of migration candidates
    """
    lines = []
    lines.append("# Migration Candidate Summary (Step F-0)")
    lines.append("")
    lines.append(f"Source: {source_version} → Target: {target_version}")
    lines.append(f"Candidates: Top {top_k} features in {source_version} not present in {target_version}")
    lines.append("")

    for target in targets:
        lines.append(f"## {target}")
        lines.append("")

        candidates = generate_migration_candidates(
            audit_data, target, source_version, target_version, top_k
        )

        if not candidates:
            lines.append("No migration candidates found.")
            lines.append("")
            continue

        # Count by action
        n_port = sum(1 for c in candidates if c["suggested_action"] == "port")
        n_review = sum(1 for c in candidates if c["suggested_action"] == "needs_review")
        n_skip = sum(1 for c in candidates if c["suggested_action"] == "skip")

        lines.append("### Summary")
        lines.append("")
        lines.append(f"- **Port** (safe): {n_port}")
        lines.append(f"- **Needs Review** (warn): {n_review}")
        lines.append(f"- **Skip** (unsafe): {n_skip}")
        lines.append("")

        # Show port candidates
        port_candidates = [c for c in candidates if c["suggested_action"] == "port"]
        if port_candidates:
            lines.append("### Port Candidates (Safe)")
            lines.append("")
            lines.append(f"| Rank | Feature | {source_version} Gain | Group |")
            lines.append("|------|---------|-------|-------|")
            for c in port_candidates[:15]:
                lines.append(
                    f"| {c[f'{source_version}_rank']} | {c['feature']} | "
                    f"{c[f'{source_version}_gain']:.2f} | {c['reason_tag']} |"
                )
            lines.append("")

        # Show review candidates (warn)
        review_candidates = [c for c in candidates if c["suggested_action"] == "needs_review"]
        if review_candidates:
            lines.append("### Needs Review (Warn)")
            lines.append("")
            lines.append(f"| Rank | Feature | {source_version} Gain | Reason |")
            lines.append("|------|---------|-------|--------|")
            for c in review_candidates[:10]:
                lines.append(
                    f"| {c[f'{source_version}_rank']} | {c['feature']} | "
                    f"{c[f'{source_version}_gain']:.2f} | {c['blocked_reason']} |"
                )
            lines.append("")

        # Show skip candidates (unsafe)
        skip_candidates = [c for c in candidates if c["suggested_action"] == "skip"]
        if skip_candidates:
            lines.append("### Blocked (Unsafe)")
            lines.append("")
            lines.append(f"| Rank | Feature | Reason |")
            lines.append("|------|---------|--------|")
            for c in skip_candidates[:5]:
                lines.append(
                    f"| {c[f'{source_version}_rank']} | {c['feature']} | {c['blocked_reason']} |"
                )
            lines.append("")

    return "\n".join(lines)


def _get_rank(gain_dict: Dict[str, float], feature: str) -> Optional[int]:
    """Get rank of feature in gain dict."""
    if feature not in gain_dict:
        return None
    sorted_features = sorted(gain_dict.items(), key=lambda x: -x[1])
    for i, (f, _) in enumerate(sorted_features, 1):
        if f == feature:
            return i
    return None


# =============================================================================
# Report Writers
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """Write rows to CSV file."""
    if not rows:
        logger.warning(f"No data to write to {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {path}")


def write_json(path: Path, data: Any) -> None:
    """Write data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote JSON to {path}")


def generate_index_md(
    audit_data: Dict[str, Dict[str, VersionAuditData]],
    targets: List[str],
    versions: List[str],
    output_dir: Path,
) -> str:
    """Generate index.md content."""
    lines = []

    lines.append("# Feature Comparison Report (Step E)")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("| Version | Target | Features | Unsafe | Warn | Safe |")
    lines.append("|---------|--------|----------|--------|------|------|")

    for version in versions:
        for target in targets:
            if version not in audit_data or target not in audit_data[version]:
                lines.append(f"| {version} | {target} | - | - | - | - |")
                continue

            vdata = audit_data[version][target]
            summary = vdata.summary

            n_feat = summary.get("n_features", len(vdata.features))
            n_unsafe = summary.get("n_unsafe", 0)
            n_warn = summary.get("n_warn", 0)
            n_safe = summary.get("n_safe", 0)

            lines.append(f"| {version} | {target} | {n_feat} | {n_unsafe} | {n_warn} | {n_safe} |")

    lines.append("")

    # V4 Top Features
    lines.append("## V4 Top Features")
    lines.append("")

    for target in targets:
        lines.append(f"### {target}")
        lines.append("")

        top_features = generate_top_features(audit_data, target, "v4", 10)
        if top_features:
            lines.append("| Rank | Feature | Gain | Cumulative | Safety |")
            lines.append("|------|---------|------|------------|--------|")
            for row in top_features:
                gain = row.get("importance_gain", 0)
                cum = row.get("cumulative_pct", 0) * 100
                safety = row.get("safety_label", "safe")
                safety_icon = "⚠️" if safety == "unsafe" else ("⚡" if safety == "warn" else "✅")
                lines.append(f"| {row['rank']} | {row['feature']} | {gain:.2f} | {cum:.1f}% | {safety_icon} {safety} |")
        else:
            lines.append("No data available.")

        lines.append("")

    # V4 vs V3 Changes
    lines.append("## V4 vs V3 Changes")
    lines.append("")

    for target in targets:
        lines.append(f"### {target}")
        lines.append("")

        diff_data = generate_diff_v4_vs_v3(audit_data, target)
        if diff_data:
            new_features = [r for r in diff_data if r["change_type"] == "new_in_v4"]
            removed_features = [r for r in diff_data if r["change_type"] == "removed_in_v4"]
            changed_features = [r for r in diff_data if r["change_type"] == "changed"]

            lines.append(f"- **New in V4**: {len(new_features)} features")
            lines.append(f"- **Removed in V4**: {len(removed_features)} features")
            lines.append(f"- **Changed**: {len(changed_features)} features with significant importance change")
            lines.append("")

            # Show top new features
            if new_features:
                lines.append("#### New Features in V4 (Top 10 by gain)")
                lines.append("")
                lines.append("| Feature | V4 Gain | V4 Rank | Safety |")
                lines.append("|---------|---------|---------|--------|")
                for row in sorted(new_features, key=lambda x: -(x.get("v4_gain") or 0))[:10]:
                    safety = row.get("safety_label", "safe")
                    safety_icon = "⚠️" if safety == "unsafe" else ("⚡" if safety == "warn" else "✅")
                    lines.append(f"| {row['feature']} | {row.get('v4_gain', 0):.2f} | {row.get('v4_rank', '-')} | {safety_icon} |")
                lines.append("")

            # Show removed features (migration candidates)
            if removed_features:
                important_removed = [r for r in removed_features if (r.get("v3_rank") or 999) <= 50]
                if important_removed:
                    lines.append("#### Migration Candidates (Important in V3, not in V4)")
                    lines.append("")
                    lines.append("| Feature | V3 Gain | V3 Rank | Safety |")
                    lines.append("|---------|---------|---------|--------|")
                    for row in sorted(important_removed, key=lambda x: x.get("v3_rank") or 999)[:10]:
                        safety = row.get("safety_label", "safe")
                        safety_icon = "⚠️" if safety == "unsafe" else ("⚡" if safety == "warn" else "✅")
                        lines.append(f"| {row['feature']} | {row.get('v3_gain', 0):.2f} | {row.get('v3_rank', '-')} | {safety_icon} |")
                    lines.append("")
        else:
            lines.append("No data available.")
            lines.append("")

    # Safety Warnings
    lines.append("## Safety Warnings")
    lines.append("")

    for target in targets:
        lines.append(f"### {target}")
        lines.append("")

        safety_data = generate_safety_summary(audit_data, target, versions)
        if safety_data:
            has_warnings = False
            for row in safety_data:
                if row.get("unsafe_in_top_n", 0) > 0:
                    has_warnings = True
                    lines.append(f"**⚠️ {row['version']}**: {row['unsafe_in_top_n']} unsafe features in top {TOP_N_FEATURES}")
                    if row.get("unsafe_features_in_top"):
                        lines.append(f"  - {row['unsafe_features_in_top']}")
                    lines.append("")

                if row.get("warn_in_top_n", 0) > 0:
                    lines.append(f"**⚡ {row['version']}**: {row['warn_in_top_n']} warn features in top {TOP_N_FEATURES}")
                    if row.get("warn_features_in_top"):
                        lines.append(f"  - {row['warn_features_in_top']}")
                    lines.append("")

            if not has_warnings:
                lines.append("No unsafe features in top rankings. ✅")
                lines.append("")
        else:
            lines.append("No data available.")
            lines.append("")

    # Common Top Features
    lines.append("## Common Top Features Across Versions")
    lines.append("")

    for target in targets:
        lines.append(f"### {target}")
        lines.append("")

        common_data = generate_common_top_features(audit_data, target, versions, 30)
        if common_data:
            lines.append("Features appearing in top 30 across multiple versions:")
            lines.append("")
            lines.append("| Feature | Versions | V4 Rank | V3 Rank | V2 Rank | V1 Rank | Safety |")
            lines.append("|---------|----------|---------|---------|---------|---------|--------|")
            for row in common_data[:15]:
                safety = row.get("safety_label", "safe")
                safety_icon = "⚠️" if safety == "unsafe" else ("⚡" if safety == "warn" else "✅")
                ranks = [str(row.get(f"{v}_rank", "-") or "-") for v in versions]
                lines.append(f"| {row['feature']} | {row['n_versions']} | {' | '.join(ranks)} | {safety_icon} |")
            lines.append("")
        else:
            lines.append("No common features found.")
            lines.append("")

    # Next Steps
    lines.append("## Suggested Next Steps")
    lines.append("")
    lines.append("Based on this analysis, consider the following actions:")
    lines.append("")
    lines.append("1. **Review unsafe features**: If any unsafe features appear in top rankings, investigate and consider removal")
    lines.append("2. **Migration candidates**: Review features that were important in legacy versions but missing in V4")
    lines.append("3. **Feature engineering**: Focus on improving features that are consistently important across versions")
    lines.append("4. **Safety audit**: Run detailed safety audit on warn features to determine if they're truly safe for pre-race prediction")
    lines.append("")

    # Generated Files
    lines.append("## Generated Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `feature_compare.csv` | Full cross-version comparison table |")
    lines.append("| `feature_compare.json` | Same data in JSON format |")

    for target in targets:
        lines.append(f"| `top_features_{target}_*.csv` | Top features per version |")
        lines.append(f"| `common_top_features_{target}.csv` | Features common across versions |")
        lines.append(f"| `diff_v4_vs_v3_{target}.csv` | V4 vs V3 changes |")
        lines.append(f"| `safety_summary_{target}.csv` | Safety label summary |")

    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Logic
# =============================================================================

def run_report(
    input_dir: Path,
    output_dir: Path,
    versions: List[str],
    targets: List[str],
    mode: str = DEFAULT_MODE,
    dry_run: bool = False,
    models_dir: Optional[Path] = None,
) -> int:
    """
    Generate comparison reports.

    Args:
        input_dir: Audit artifacts directory
        output_dir: Output directory for reports
        versions: List of versions to include
        targets: List of targets to include
        mode: Mode (pre_race, etc.)
        dry_run: Only show what would be generated
        models_dir: Models directory for fallback loading

    Returns:
        Exit code (0=success, 1=error)
    """
    print(f"Feature Comparison Report Generator (Step E)")
    print(f"=" * 60)
    print(f"  Input:    {input_dir}")
    print(f"  Output:   {output_dir}")
    print(f"  Models:   {models_dir or '(not specified)'}")
    print(f"  Versions: {', '.join(versions)}")
    print(f"  Targets:  {', '.join(targets)}")
    print(f"  Mode:     {mode}")
    print(f"  Dry run:  {dry_run}")
    print()

    # Load audit data with fallback to models directory
    print("Loading audit data...")
    audit_data = load_audit_data(input_dir, versions, targets, mode, models_dir)

    # Check if we have any data
    has_data = False
    for version in versions:
        for target in targets:
            if version in audit_data and target in audit_data[version]:
                has_data = True
                break

    if not has_data:
        print("ERROR: No audit data found. Run audit_featurepacks.py first.")
        return 1

    if dry_run:
        print("\nDry run - files that would be generated:")
        print(f"  {output_dir / 'index.md'}")
        print(f"  {output_dir / 'feature_compare.csv'}")
        print(f"  {output_dir / 'feature_compare.json'}")
        print(f"  {output_dir / 'candidate_summary.md'}")
        for target in targets:
            for version in versions:
                print(f"  {output_dir / f'top_features_{target}_{version}.csv'}")
            print(f"  {output_dir / f'common_top_features_{target}.csv'}")
            print(f"  {output_dir / f'diff_v4_vs_v3_{target}.csv'}")
            print(f"  {output_dir / f'safety_summary_{target}.csv'}")
            print(f"  {output_dir / f'migration_candidates_{target}.csv'}")
        return 0

    # Generate reports
    print("\nGenerating reports...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature comparison (per target)
    all_compare_data = {}
    for target in targets:
        compare_data = generate_feature_compare(audit_data, target, versions)
        all_compare_data[target] = compare_data

        # Write CSV
        if compare_data:
            # Define column order
            base_cols = ["feature", "safety_label", "safety_notes", "group"]
            version_cols = []
            for v in versions:
                version_cols.extend([
                    f"{v}_present", f"{v}_rank", f"{v}_gain",
                    f"{v}_gain_norm", f"{v}_cumulative"
                ])
            write_csv(
                output_dir / f"feature_compare_{target}.csv",
                compare_data,
                base_cols + version_cols
            )

    # Combined feature compare (JSON)
    write_json(output_dir / "feature_compare.json", all_compare_data)

    # Top features per version
    for target in targets:
        for version in versions:
            top_data = generate_top_features(audit_data, target, version)
            if top_data:
                write_csv(
                    output_dir / f"top_features_{target}_{version}.csv",
                    top_data
                )

    # Common top features
    for target in targets:
        common_data = generate_common_top_features(audit_data, target, versions)
        if common_data:
            write_csv(
                output_dir / f"common_top_features_{target}.csv",
                common_data
            )

    # Diff V4 vs V3
    for target in targets:
        diff_data = generate_diff_v4_vs_v3(audit_data, target)
        if diff_data:
            write_csv(
                output_dir / f"diff_v4_vs_v3_{target}.csv",
                diff_data
            )

    # Safety summary
    for target in targets:
        safety_data = generate_safety_summary(audit_data, target, versions)
        if safety_data:
            write_csv(
                output_dir / f"safety_summary_{target}.csv",
                safety_data
            )

    # Migration candidates (Step F-0)
    for target in targets:
        candidates = generate_migration_candidates(audit_data, target, "v3", "v4", 30)
        if candidates:
            # Define column order for CSV
            fieldnames = [
                "feature", "v3_gain", "v3_rank", "safety_label", "safety_notes",
                "reason_tag", "blocked_reason", "suggested_action"
            ]
            write_csv(
                output_dir / f"migration_candidates_{target}.csv",
                candidates,
                fieldnames
            )

    # Candidate summary markdown (Step F-0)
    candidate_md = generate_candidate_summary(audit_data, targets, "v3", "v4", 30)
    candidate_path = output_dir / "candidate_summary.md"
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write(candidate_md)
    print(f"Wrote candidate_summary.md to {candidate_path}")

    # Index markdown
    index_md = generate_index_md(audit_data, targets, versions, output_dir)
    index_path = output_dir / "index.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_md)
    print(f"Wrote index.md to {index_path}")

    print(f"\nReport generation complete!")
    print(f"Output directory: {output_dir}")

    return 0


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature Comparison Report Generator (Step E)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python scripts/report_featurepacks.py

  # Custom input/output directories
  python scripts/report_featurepacks.py --input artifacts/feature_audit --output reports/

  # Dry run
  python scripts/report_featurepacks.py --dry-run
"""
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Input directory (audit output) (default: {DEFAULT_INPUT_DIR.relative_to(PROJECT_ROOT).as_posix()})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix()})",
    )
    parser.add_argument(
        "--versions",
        type=str,
        default=",".join(DEFAULT_VERSIONS),
        help=f"Comma-separated list of versions (default: {','.join(DEFAULT_VERSIONS)})",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(DEFAULT_TARGETS),
        help=f"Comma-separated list of targets (default: {','.join(DEFAULT_TARGETS)})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help=f"Mode (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Models directory for fallback loading (default: models/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be generated without writing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse lists
    versions = args.versions.split(",") if args.versions else DEFAULT_VERSIONS
    targets = args.targets.split(",") if args.targets else DEFAULT_TARGETS

    # Parse models_dir
    models_dir = Path(args.models).absolute() if args.models else None

    # Run
    exit_code = run_report(
        input_dir=Path(args.input).absolute(),
        output_dir=Path(args.output).absolute(),
        versions=versions,
        targets=targets,
        mode=args.mode,
        dry_run=args.dry_run,
        models_dir=models_dir,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
