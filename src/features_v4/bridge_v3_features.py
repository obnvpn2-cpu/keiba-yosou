# -*- coding: utf-8 -*-
"""
bridge_v3_features.py - Bridge Layer for v3 Feature Migration to v4

v3 の有望特徴量を v4 パイプラインで試験するための bridge 層。
直接 src/features_v4/ に混ぜ込むのではなく、overlay として提供する。

【設計原則】
1. v4 のコア特徴量は変更しない
2. v3 からの移植特徴量は "v4_bridge_" prefix を付与
3. ON/OFF が簡単に切り替え可能
4. 移植候補は migration_candidates.csv から読み込み

【使用方法】
1. 移植候補CSVを用意: artifacts/feature_audit/report/migration_candidates_{target}.csv
2. bridge を有効化:
   - train_eval_v4.py --bridge-v3
   - または Python から:
     from src.features_v4.bridge_v3_features import BridgeV3Features
     bridge = BridgeV3Features(conn)
     bridged_df = bridge.apply_bridge(v4_df, target="target_win")

【安全性】
- unsafe / warn 分類は migration_candidates.csv の safety_label を参照
- unsafe 特徴量は自動的にスキップ
- warn 特徴量は含めるが、ログに警告を出力
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DEFAULT_CANDIDATES_DIR = PROJECT_ROOT / "artifacts" / "feature_audit" / "report"

# Bridge feature prefix
BRIDGE_PREFIX = "v4_bridge_"

# Safety labels
SAFETY_SKIP = {"unsafe"}  # These are never included
SAFETY_WARN = {"warn"}    # These are included but with warning


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MigrationCandidate:
    """Single migration candidate from v3 to v4"""
    feature: str
    v3_gain: float
    v3_rank: int
    safety_label: str
    safety_notes: str
    reason_tag: str
    blocked_reason: str
    suggested_action: str


@dataclass
class BridgeConfig:
    """Configuration for bridge layer"""
    # Source version (default: v3)
    source_version: str = "v3"

    # Maximum number of features to bridge
    max_features: int = 20

    # Include features with these safety labels
    include_warn: bool = True  # Include "warn" features (with logging)

    # Feature list override (if provided, use these instead of candidates file)
    explicit_features: List[str] = field(default_factory=list)

    # Candidates CSV path override
    candidates_path: Optional[Path] = None


@dataclass
class FeatureSafetyInfo:
    """Safety information for a single feature"""
    original_name: str
    bridged_name: str
    safety_label: str  # "safe", "warn", "unsafe"
    safety_notes: str
    origin: str = "v3_bridged"  # "v4_native" or "v3_bridged"


@dataclass
class BridgeResult:
    """Result of applying bridge (F-2 enhanced)"""
    # Counts
    n_features_requested: int
    n_features_applied: int
    n_features_skipped_unsafe: int
    n_features_skipped_missing: int
    n_features_skipped_warn: int  # Renamed for clarity
    n_features_warn: int  # warn features that WERE included

    # Feature lists (bridged names)
    applied_features: List[str]
    skipped_features: List[str]
    warn_features: List[str]

    # F-2: Enhanced tracking
    # Mapping: bridged_name -> original_name
    feature_map: Dict[str, str] = field(default_factory=dict)

    # Mapping: bridged_name -> FeatureSafetyInfo
    safety_info: Dict[str, FeatureSafetyInfo] = field(default_factory=dict)

    # Detailed exclusion lists (original names)
    excluded_unsafe: List[str] = field(default_factory=list)
    excluded_warn: List[str] = field(default_factory=list)
    excluded_missing: List[str] = field(default_factory=list)

    # Included features with both names: [(original_name, bridged_name), ...]
    included_features: List[Tuple[str, str]] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for JSON serialization"""
        return {
            "n_features_applied": self.n_features_applied,
            "n_features_skipped_unsafe": self.n_features_skipped_unsafe,
            "n_features_skipped_warn": self.n_features_skipped_warn,
            "n_features_skipped_missing": self.n_features_skipped_missing,
            "n_features_warn_included": self.n_features_warn,
            "applied_features": self.applied_features,
            "excluded_unsafe": self.excluded_unsafe,
            "excluded_warn": self.excluded_warn,
            "excluded_missing": self.excluded_missing,
            "feature_map": self.feature_map,
        }

    def log_summary(self, logger_func=None) -> None:
        """Log a formatted summary of bridge results"""
        if logger_func is None:
            logger_func = logger.info

        logger_func("=" * 60)
        logger_func("Bridge Summary (v3 -> v4)")
        logger_func("=" * 60)
        logger_func(f"  Included:        {self.n_features_applied}")
        logger_func(f"    - safe:        {self.n_features_applied - self.n_features_warn}")
        logger_func(f"    - warn:        {self.n_features_warn}")
        logger_func(f"  Excluded:")
        logger_func(f"    - unsafe:      {self.n_features_skipped_unsafe}")
        logger_func(f"    - warn:        {self.n_features_skipped_warn}")
        logger_func(f"    - missing:     {self.n_features_skipped_missing}")

        if self.excluded_unsafe:
            logger_func("  Unsafe features excluded:")
            for f in self.excluded_unsafe[:5]:  # Show first 5
                logger_func(f"    - {f}")
            if len(self.excluded_unsafe) > 5:
                logger_func(f"    ... and {len(self.excluded_unsafe) - 5} more")

        if self.excluded_warn:
            logger_func("  Warn features excluded (--bridge-no-warn):")
            for f in self.excluded_warn[:5]:
                logger_func(f"    - {f}")
            if len(self.excluded_warn) > 5:
                logger_func(f"    ... and {len(self.excluded_warn) - 5} more")

        if self.warn_features:
            logger_func("  Warn features INCLUDED (use --bridge-no-warn to exclude):")
            for f in self.warn_features[:5]:
                logger_func(f"    - {f}")
            if len(self.warn_features) > 5:
                logger_func(f"    ... and {len(self.warn_features) - 5} more")

        logger_func("=" * 60)


# =============================================================================
# Bridge Implementation
# =============================================================================

class BridgeV3Features:
    """
    Bridge layer to inject v3 features into v4 pipeline

    Usage:
        bridge = BridgeV3Features(conn)
        bridged_df = bridge.apply_bridge(v4_df, target="target_win")
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: Optional[BridgeConfig] = None,
    ):
        """
        Initialize bridge

        Args:
            conn: SQLite connection (must have feature_table_v3)
            config: Bridge configuration
        """
        self.conn = conn
        self.config = config or BridgeConfig()
        self._v3_table_exists = None
        self._candidates_cache: Dict[str, List[MigrationCandidate]] = {}

    def _check_v3_table(self) -> bool:
        """Check if feature_table_v3 exists"""
        if self._v3_table_exists is None:
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_table_v3'"
            )
            self._v3_table_exists = cursor.fetchone() is not None
        return self._v3_table_exists

    def load_candidates(
        self,
        target: str,
        candidates_path: Optional[Path] = None,
    ) -> List[MigrationCandidate]:
        """
        Load migration candidates from CSV

        Args:
            target: Target column (target_win, target_in3)
            candidates_path: Override path to candidates CSV

        Returns:
            List of MigrationCandidate
        """
        # Check cache
        cache_key = f"{target}:{candidates_path}"
        if cache_key in self._candidates_cache:
            return self._candidates_cache[cache_key]

        # Determine path
        if candidates_path is None:
            candidates_path = self.config.candidates_path
        if candidates_path is None:
            candidates_path = DEFAULT_CANDIDATES_DIR / f"migration_candidates_{target}.csv"

        candidates = []

        if not candidates_path.exists():
            logger.warning(f"Candidates file not found: {candidates_path}")
            return candidates

        try:
            df = pd.read_csv(candidates_path)
            for _, row in df.iterrows():
                candidate = MigrationCandidate(
                    feature=row.get("feature", ""),
                    v3_gain=float(row.get("v3_gain", 0)),
                    v3_rank=int(row.get("v3_rank", 0)),
                    safety_label=row.get("safety_label", "safe"),
                    safety_notes=row.get("safety_notes", ""),
                    reason_tag=row.get("reason_tag", ""),
                    blocked_reason=row.get("blocked_reason", ""),
                    suggested_action=row.get("suggested_action", "skip"),
                )
                candidates.append(candidate)

            logger.info(f"Loaded {len(candidates)} migration candidates from {candidates_path}")
            self._candidates_cache[cache_key] = candidates

        except Exception as e:
            logger.error(f"Failed to load candidates from {candidates_path}: {e}")

        return candidates

    def get_bridge_features(
        self,
        target: str,
        candidates_path: Optional[Path] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, MigrationCandidate]]:
        """
        Get list of features to bridge, with safety filtering (F-2 enhanced)

        Args:
            target: Target column
            candidates_path: Override path to candidates CSV

        Returns:
            (features_to_bridge, excluded_unsafe, excluded_warn, warn_included, candidate_map)
            - features_to_bridge: features that will be bridged
            - excluded_unsafe: features excluded due to unsafe label (ALWAYS excluded)
            - excluded_warn: features excluded due to warn label (when include_warn=False)
            - warn_included: warn features that are included (when include_warn=True)
            - candidate_map: feature -> MigrationCandidate for tracking safety info
        """
        # If explicit features are provided, use them (no safety info available)
        if self.config.explicit_features:
            return self.config.explicit_features, [], [], [], {}

        # Load candidates
        candidates = self.load_candidates(target, candidates_path)

        features_to_bridge = []
        excluded_unsafe = []
        excluded_warn = []
        warn_included = []
        candidate_map: Dict[str, MigrationCandidate] = {}

        for candidate in candidates:
            feature = candidate.feature

            # F-2 SAFETY GATE: unsafe is ALWAYS excluded (no config option)
            # This is a hard guarantee - unsafe features can NEVER be included
            if candidate.safety_label in SAFETY_SKIP:
                excluded_unsafe.append(feature)
                logger.debug(f"[SAFETY] Excluded unsafe feature: {feature}")
                continue

            # suggested_action == "skip" also triggers exclusion (usually unsafe)
            if candidate.suggested_action == "skip":
                excluded_unsafe.append(feature)
                logger.debug(f"[SAFETY] Excluded by suggested_action=skip: {feature}")
                continue

            # F-2 SAFETY GATE: warn features controlled by include_warn
            if candidate.safety_label in SAFETY_WARN:
                if self.config.include_warn:
                    warn_included.append(feature)
                    features_to_bridge.append(feature)
                    candidate_map[feature] = candidate
                    logger.debug(f"[SAFETY] Including warn feature: {feature}")
                else:
                    excluded_warn.append(feature)
                    logger.debug(f"[SAFETY] Excluded warn feature (--bridge-no-warn): {feature}")
                continue

            # Safe feature - include it
            features_to_bridge.append(feature)
            candidate_map[feature] = candidate

            # Check max features limit
            if len(features_to_bridge) >= self.config.max_features:
                logger.info(f"Reached max_features limit ({self.config.max_features})")
                break

        return features_to_bridge, excluded_unsafe, excluded_warn, warn_included, candidate_map

    def load_v3_features(
        self,
        race_ids: List[str],
        features: List[str],
    ) -> pd.DataFrame:
        """
        Load specific features from feature_table_v3

        Args:
            race_ids: List of race_ids to load
            features: List of feature column names

        Returns:
            DataFrame with race_id, horse_id, and requested features
        """
        if not self._check_v3_table():
            logger.error("feature_table_v3 not found in database")
            return pd.DataFrame()

        if not features:
            return pd.DataFrame()

        # Check which columns exist
        cursor = self.conn.execute("PRAGMA table_info(feature_table_v3)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        valid_features = [f for f in features if f in existing_cols]
        missing_features = [f for f in features if f not in existing_cols]

        if missing_features:
            logger.warning(f"Features not found in v3 table: {missing_features}")

        if not valid_features:
            logger.warning("No valid features to load from v3")
            return pd.DataFrame()

        # Build query
        cols = ["race_id", "horse_id"] + valid_features
        placeholders = ",".join(["?"] * len(race_ids))

        sql = f"""
            SELECT {", ".join(cols)}
            FROM feature_table_v3
            WHERE race_id IN ({placeholders})
        """

        df = pd.read_sql_query(sql, self.conn, params=race_ids)
        logger.info(f"Loaded {len(df)} rows with {len(valid_features)} v3 features")

        return df

    def apply_bridge(
        self,
        v4_df: pd.DataFrame,
        target: str = "target_win",
        candidates_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, BridgeResult]:
        """
        Apply bridge: add v3 features to v4 DataFrame (F-2 enhanced)

        Args:
            v4_df: Original v4 feature DataFrame
            target: Target column for loading candidates
            candidates_path: Override path to candidates CSV

        Returns:
            (bridged_df, BridgeResult)
        """
        # Get features to bridge with enhanced safety tracking
        (
            features_to_bridge,
            excluded_unsafe,
            excluded_warn,
            warn_included,
            candidate_map,
        ) = self.get_bridge_features(target, candidates_path)

        # Initialize result with F-2 enhanced tracking
        result = BridgeResult(
            n_features_requested=len(features_to_bridge) + len(excluded_unsafe) + len(excluded_warn),
            n_features_applied=0,
            n_features_skipped_unsafe=len(excluded_unsafe),
            n_features_skipped_missing=0,
            n_features_skipped_warn=len(excluded_warn),
            n_features_warn=len(warn_included),
            applied_features=[],
            skipped_features=[],
            warn_features=[],
            feature_map={},
            safety_info={},
            excluded_unsafe=list(excluded_unsafe),
            excluded_warn=list(excluded_warn),
            excluded_missing=[],
            included_features=[],
        )

        if not features_to_bridge:
            logger.warning("No features to bridge after safety filtering")
            return v4_df, result

        # Log warnings for warn features
        for feat in warn_included:
            logger.warning(f"Including 'warn' safety feature: {feat}")

        # Get race_ids from v4_df
        if "race_id" not in v4_df.columns:
            logger.error("v4_df must have 'race_id' column")
            return v4_df, result

        race_ids = v4_df["race_id"].unique().tolist()

        # Load v3 features
        v3_df = self.load_v3_features(race_ids, features_to_bridge)

        if v3_df.empty:
            logger.warning("No v3 features loaded")
            return v4_df, result

        # Rename v3 features with bridge prefix
        rename_map = {f: f"{BRIDGE_PREFIX}{f}" for f in features_to_bridge if f in v3_df.columns}
        v3_df = v3_df.rename(columns=rename_map)

        # Track missing features
        loaded_features = [f for f in features_to_bridge if f in rename_map]
        missing_features = [f for f in features_to_bridge if f not in rename_map]
        result.n_features_skipped_missing = len(missing_features)
        result.excluded_missing = list(missing_features)
        result.skipped_features = list(excluded_unsafe) + list(excluded_warn) + missing_features

        # Merge with v4_df
        bridge_cols = ["race_id", "horse_id"] + list(rename_map.values())
        v3_subset = v3_df[bridge_cols]

        bridged_df = v4_df.merge(
            v3_subset,
            on=["race_id", "horse_id"],
            how="left",
        )

        # Update result with F-2 enhanced tracking
        result.n_features_applied = len(loaded_features)
        result.applied_features = [f"{BRIDGE_PREFIX}{f}" for f in loaded_features]
        result.warn_features = [f"{BRIDGE_PREFIX}{f}" for f in warn_included if f in loaded_features]

        # Build feature map and safety info
        for original_name in loaded_features:
            bridged_name = f"{BRIDGE_PREFIX}{original_name}"
            result.feature_map[bridged_name] = original_name
            result.included_features.append((original_name, bridged_name))

            # Get safety info from candidate map
            if original_name in candidate_map:
                candidate = candidate_map[original_name]
                result.safety_info[bridged_name] = FeatureSafetyInfo(
                    original_name=original_name,
                    bridged_name=bridged_name,
                    safety_label=candidate.safety_label,
                    safety_notes=candidate.safety_notes,
                    origin="v3_bridged",
                )
            else:
                # Explicit feature or unknown - assume safe
                result.safety_info[bridged_name] = FeatureSafetyInfo(
                    original_name=original_name,
                    bridged_name=bridged_name,
                    safety_label="safe",
                    safety_notes="(explicit or unknown)",
                    origin="v3_bridged",
                )

        # Log summary
        result.log_summary()

        return bridged_df, result

    def save_feature_map(
        self,
        result: BridgeResult,
        output_dir: Path,
        target: str,
    ) -> Path:
        """
        Save bridge feature map to JSON file (F-2 explainability)

        Args:
            result: BridgeResult from apply_bridge
            output_dir: Directory to save the map
            target: Target column name

        Returns:
            Path to saved file
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"bridge_feature_map_{target}.json"

        # Build comprehensive feature map
        feature_map_data = {
            "metadata": {
                "source_version": self.config.source_version,
                "target": target,
                "n_features_applied": result.n_features_applied,
                "n_features_warn": result.n_features_warn,
                "n_excluded_unsafe": result.n_features_skipped_unsafe,
                "n_excluded_warn": result.n_features_skipped_warn,
                "n_excluded_missing": result.n_features_skipped_missing,
            },
            "feature_map": result.feature_map,  # bridged_name -> original_name
            "features": {},
        }

        # Add detailed info for each feature
        for bridged_name, safety_info in result.safety_info.items():
            feature_map_data["features"][bridged_name] = {
                "original_name": safety_info.original_name,
                "display_name": safety_info.original_name,  # For UI display
                "bridged_name": safety_info.bridged_name,
                "safety_label": safety_info.safety_label,
                "safety_notes": safety_info.safety_notes,
                "origin": safety_info.origin,
            }

        # Add excluded features for reference
        feature_map_data["excluded"] = {
            "unsafe": result.excluded_unsafe,
            "warn": result.excluded_warn,
            "missing": result.excluded_missing,
        }

        # Write with UTF-8 encoding (Windows compatible)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(feature_map_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved bridge feature map to: {output_path}")
        return output_path


# =============================================================================
# Utility Functions
# =============================================================================

def get_bridge_feature_columns(
    conn: sqlite3.Connection,
    target: str = "target_win",
    config: Optional[BridgeConfig] = None,
) -> List[str]:
    """
    Get list of bridge feature column names (with prefix)

    Args:
        conn: SQLite connection
        target: Target column
        config: Bridge configuration

    Returns:
        List of feature column names with BRIDGE_PREFIX
    """
    bridge = BridgeV3Features(conn, config)
    features, _, _, _, _ = bridge.get_bridge_features(target)
    return [f"{BRIDGE_PREFIX}{f}" for f in features]


def load_bridge_feature_map(map_path: Path) -> Dict[str, Any]:
    """
    Load bridge feature map from JSON file

    Args:
        map_path: Path to bridge_feature_map_{target}.json

    Returns:
        Feature map data dictionary
    """
    import json

    if not map_path.exists():
        logger.warning(f"Bridge feature map not found: {map_path}")
        return {}

    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_original_feature_name(bridged_name: str, feature_map: Dict[str, str]) -> str:
    """
    Get original v3 feature name from bridged name

    Args:
        bridged_name: Bridged feature name (v4_bridge_*)
        feature_map: Mapping from bridged_name -> original_name

    Returns:
        Original feature name, or bridged_name if not found
    """
    if bridged_name in feature_map:
        return feature_map[bridged_name]

    # Fallback: strip prefix
    if bridged_name.startswith(BRIDGE_PREFIX):
        return bridged_name[len(BRIDGE_PREFIX):]

    return bridged_name


def get_feature_safety_for_bridge(
    feature_name: str,
    bridge_map_data: Dict[str, Any],
) -> Tuple[str, str, str]:
    """
    Get safety information for a feature (works for both bridged and native)

    Args:
        feature_name: Feature name (may or may not have bridge prefix)
        bridge_map_data: Loaded bridge feature map data

    Returns:
        (safety_label, safety_notes, origin)
        - safety_label: "safe", "warn", "unsafe", or "unknown"
        - safety_notes: Human-readable notes
        - origin: "v4_native" or "v3_bridged"
    """
    features = bridge_map_data.get("features", {})

    if feature_name in features:
        info = features[feature_name]
        return (
            info.get("safety_label", "unknown"),
            info.get("safety_notes", ""),
            info.get("origin", "v3_bridged"),
        )

    # Not in bridge map - check if it's a v4 native feature
    if not feature_name.startswith(BRIDGE_PREFIX):
        return ("safe", "v4 native feature", "v4_native")

    # Unknown bridged feature
    return ("unknown", "Not found in bridge map", "v3_bridged")


def apply_bridge_to_dataframe(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    target: str = "target_win",
    config: Optional[BridgeConfig] = None,
) -> Tuple[pd.DataFrame, BridgeResult]:
    """
    Convenience function to apply bridge to a DataFrame

    Args:
        conn: SQLite connection
        df: Input DataFrame (must have race_id, horse_id)
        target: Target column
        config: Bridge configuration

    Returns:
        (bridged_df, BridgeResult)
    """
    bridge = BridgeV3Features(conn, config)
    return bridge.apply_bridge(df, target)


def list_available_v3_features(conn: sqlite3.Connection) -> List[str]:
    """
    List all available features in feature_table_v3

    Args:
        conn: SQLite connection

    Returns:
        List of feature column names
    """
    cursor = conn.execute("PRAGMA table_info(feature_table_v3)")
    columns = [row[1] for row in cursor.fetchall()]

    # Exclude identity and target columns
    exclude = {"race_id", "horse_id", "horse_no", "race_date",
               "target_win", "target_in3", "target_value"}

    return [c for c in columns if c not in exclude]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Bridge v3 features to v4 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available v3 features
  python -m src.features_v4.bridge_v3_features --db netkeiba.db --list-v3

  # Show migration candidates for target_win
  python -m src.features_v4.bridge_v3_features --db netkeiba.db --show-candidates target_win

  # Show bridge features that would be applied
  python -m src.features_v4.bridge_v3_features --db netkeiba.db --show-bridge target_win
""",
    )
    parser.add_argument("--db", default="netkeiba.db", help="Database path")
    parser.add_argument("--list-v3", action="store_true", help="List v3 features")
    parser.add_argument("--show-candidates", type=str, help="Show candidates for target")
    parser.add_argument("--show-bridge", type=str, help="Show bridge features for target")
    parser.add_argument("--max-features", type=int, default=20, help="Max features to bridge")
    parser.add_argument("--include-warn", action="store_true", default=True,
                       help="Include warn-safety features")
    parser.add_argument("--no-warn", action="store_true", help="Exclude warn-safety features")

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    try:
        if args.list_v3:
            features = list_available_v3_features(conn)
            print(f"\nAvailable v3 features ({len(features)}):")
            for f in sorted(features):
                print(f"  {f}")

        elif args.show_candidates:
            config = BridgeConfig(
                max_features=args.max_features,
                include_warn=not args.no_warn,
            )
            bridge = BridgeV3Features(conn, config)
            candidates = bridge.load_candidates(args.show_candidates)

            print(f"\nMigration candidates for {args.show_candidates} ({len(candidates)}):")
            print("-" * 80)
            for c in candidates:
                status = "SKIP" if c.suggested_action == "skip" else c.safety_label.upper()
                print(f"  [{status:6s}] {c.feature:40s} (rank={c.v3_rank}, gain={c.v3_gain:.2f})")

        elif args.show_bridge:
            config = BridgeConfig(
                max_features=args.max_features,
                include_warn=not args.no_warn,
            )
            bridge = BridgeV3Features(conn, config)
            features, skipped, warn = bridge.get_bridge_features(args.show_bridge)

            print(f"\nBridge features for {args.show_bridge}:")
            print("-" * 80)
            print(f"Features to bridge ({len(features)}):")
            for f in features:
                marker = " [WARN]" if f in warn else ""
                print(f"  {BRIDGE_PREFIX}{f}{marker}")

            if skipped:
                print(f"\nSkipped features ({len(skipped)}):")
                for f in skipped:
                    print(f"  {f}")

        else:
            parser.print_help()

    finally:
        conn.close()
