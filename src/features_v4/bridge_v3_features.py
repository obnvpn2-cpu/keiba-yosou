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
class BridgeResult:
    """Result of applying bridge"""
    n_features_requested: int
    n_features_applied: int
    n_features_skipped_unsafe: int
    n_features_skipped_missing: int
    n_features_warn: int
    applied_features: List[str]
    skipped_features: List[str]
    warn_features: List[str]


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
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Get list of features to bridge, with safety filtering

        Args:
            target: Target column
            candidates_path: Override path to candidates CSV

        Returns:
            (features_to_bridge, skipped_unsafe, warn_features)
        """
        # If explicit features are provided, use them
        if self.config.explicit_features:
            return self.config.explicit_features, [], []

        # Load candidates
        candidates = self.load_candidates(target, candidates_path)

        features_to_bridge = []
        skipped_unsafe = []
        warn_features = []

        for candidate in candidates:
            # Check suggested action
            if candidate.suggested_action == "skip":
                skipped_unsafe.append(candidate.feature)
                continue

            # Check safety label
            if candidate.safety_label in SAFETY_SKIP:
                skipped_unsafe.append(candidate.feature)
                continue

            if candidate.safety_label in SAFETY_WARN:
                if self.config.include_warn:
                    warn_features.append(candidate.feature)
                    features_to_bridge.append(candidate.feature)
                else:
                    skipped_unsafe.append(candidate.feature)
                continue

            # Safe feature
            features_to_bridge.append(candidate.feature)

            # Check max features limit
            if len(features_to_bridge) >= self.config.max_features:
                break

        return features_to_bridge, skipped_unsafe, warn_features

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
        Apply bridge: add v3 features to v4 DataFrame

        Args:
            v4_df: Original v4 feature DataFrame
            target: Target column for loading candidates
            candidates_path: Override path to candidates CSV

        Returns:
            (bridged_df, BridgeResult)
        """
        # Get features to bridge
        features_to_bridge, skipped_unsafe, warn_features = self.get_bridge_features(
            target, candidates_path
        )

        result = BridgeResult(
            n_features_requested=len(features_to_bridge) + len(skipped_unsafe),
            n_features_applied=0,
            n_features_skipped_unsafe=len(skipped_unsafe),
            n_features_skipped_missing=0,
            n_features_warn=len(warn_features),
            applied_features=[],
            skipped_features=list(skipped_unsafe),
            warn_features=list(warn_features),
        )

        if not features_to_bridge:
            logger.warning("No features to bridge after safety filtering")
            return v4_df, result

        # Log warnings for warn features
        for feat in warn_features:
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
        result.skipped_features.extend(missing_features)

        # Merge with v4_df
        bridge_cols = ["race_id", "horse_id"] + list(rename_map.values())
        v3_subset = v3_df[bridge_cols]

        bridged_df = v4_df.merge(
            v3_subset,
            on=["race_id", "horse_id"],
            how="left",
        )

        # Update result
        result.n_features_applied = len(loaded_features)
        result.applied_features = [f"{BRIDGE_PREFIX}{f}" for f in loaded_features]

        logger.info(
            f"Bridge applied: {result.n_features_applied} features added, "
            f"{result.n_features_skipped_unsafe} unsafe skipped, "
            f"{result.n_features_skipped_missing} missing skipped"
        )

        return bridged_df, result


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
    features, _, _ = bridge.get_bridge_features(target)
    return [f"{BRIDGE_PREFIX}{f}" for f in features]


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
