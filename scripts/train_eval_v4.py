#!/usr/bin/env python3
"""
train_eval_v4.py - CLI for FeaturePack v1 Training/Evaluation Pipeline

LightGBM モデルの学習・評価・ROI 分析パイプライン。

【時系列分割モード】
1. year_based (デフォルト):
   - train: 2021-01 ~ 2023-12
   - val:   2023-10 ~ 2023-12 (train末尾を検証に流用)
   - test:  2024-01 ~ 現在

2. date_based:
   - train_end, val_end を明示指定
   - 例: --split-mode date_based --train-end 2023-06-30 --val-end 2023-12-31

【出力ファイル】
- models/lgbm_target_win_v4.txt: LightGBM モデル
- models/feature_columns_target_win_v4.json: 特徴量リスト
- models/feature_importance_target_win_v4.csv: 特徴量重要度
- models/results_target_win_v4.json: 評価結果

Usage:
    python scripts/train_eval_v4.py --db netkeiba.db
    python scripts/train_eval_v4.py --db netkeiba.db --target target_in3
    python scripts/train_eval_v4.py --db netkeiba.db --split-mode date_based --train-end 2023-06-30
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_v4 import TrainConfig, run_full_pipeline
from src.features_v4.train_eval_v4 import (
    load_feature_data,
    split_time_series,
    get_train_feature_columns,
    train_model,
    log_feature_selection_summary,
)
from src.features_v4.diagnostics import run_diagnostics, save_diagnostics
from src.feature_audit.safety import classify_feature_safety

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None


logger = logging.getLogger(__name__)


def load_feature_columns_with_fallback(
    output_dir: str,
    target: str,
) -> tuple[list[str], str]:
    """
    feature_columns JSONファイルを読み込む（v4 → legacy フォールバック付き）

    Args:
        output_dir: 出力ディレクトリ
        target: ターゲットカラム名

    Returns:
        tuple of (feature_cols, used_path)
    """
    # v4 パスを優先
    v4_path = os.path.join(output_dir, f"feature_columns_{target}_v4.json")
    legacy_path = os.path.join(output_dir, f"feature_columns_{target}.json")

    if os.path.exists(v4_path):
        logger.info("Loading feature columns from: %s", v4_path)
        with open(v4_path, "r") as f:
            return json.load(f), v4_path
    elif os.path.exists(legacy_path):
        logger.info("v4 feature_columns not found, using legacy: %s", legacy_path)
        with open(legacy_path, "r") as f:
            return json.load(f), legacy_path
    else:
        raise FileNotFoundError(
            f"Feature columns file not found. Tried:\n  - {v4_path}\n  - {legacy_path}"
        )


def load_exclude_features(filepath: str) -> set[str]:
    """
    除外特徴量リストを読み込む

    ファイル形式: 1行1特徴量名、# で始まる行はコメント

    Args:
        filepath: 除外特徴量ファイルパス

    Returns:
        除外特徴量名のセット
    """
    exclude_set = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # コメント行と空行をスキップ
            if not line or line.startswith("#"):
                continue
            exclude_set.add(line)
    logger.info("Loaded %d features to exclude from: %s", len(exclude_set), filepath)
    return exclude_set


def apply_feature_exclusion(
    feature_cols: list[str],
    exclude_set: set[str],
) -> list[str]:
    """
    特徴量リストから除外特徴量を除去

    Args:
        feature_cols: 元の特徴量リスト
        exclude_set: 除外する特徴量名のセット

    Returns:
        フィルタ後の特徴量リスト
    """
    filtered = [c for c in feature_cols if c not in exclude_set]
    n_excluded = len(feature_cols) - len(filtered)
    if n_excluded > 0:
        logger.info("Excluded %d features, remaining: %d", n_excluded, len(filtered))
    return filtered


def load_booster(model_path: str) -> "lgb.Booster":
    """
    LightGBM Boosterをロードする（model_str フォールバック付き）

    Windows の日本語パス環境では lgb.Booster(model_file=...) が失敗する場合がある。
    その場合、Python の file read で内容を読み込み、model_str 経由でロードする。

    Args:
        model_path: モデルファイルのパス

    Returns:
        lgb.Booster インスタンス

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # まず model_file でロードを試みる
    model_file_error = None
    try:
        model = lgb.Booster(model_file=str(path))
        logger.info("Loaded model via model_file: %s", path)
        return model
    except Exception as e:
        model_file_error = e
        logger.warning("lgb.Booster(model_file=...) failed (%s), trying model_str fallback...", e)

    # フォールバック: Python read → model_str
    try:
        model_str = path.read_text(encoding="utf-8")
        model = lgb.Booster(model_str=model_str)
        logger.info("Loaded model via model_str fallback: %s", path)
        return model
    except Exception as e2:
        raise RuntimeError(
            f"Failed to load model from {model_path}. "
            f"model_file error: {model_file_error}, model_str error: {e2}"
        ) from e2


# =============================================================================
# Feature Table Merging (--feature-sources all)
# =============================================================================

# Columns to exclude from legacy tables (IDs, targets, leak-prone)
LEGACY_EXCLUDE_PATTERNS = {
    "race_id", "horse_id", "horse_name", "jockey_id", "jockey_name",
    "trainer_id", "trainer_name", "owner_id", "owner_name", "sire_id", "bms_id",
    "race_date", "race_name", "place", "course",
    "target_win", "target_in3", "target_quinella", "target_value",
    "finish_order", "finish_status", "time_sec", "time_str", "margin",
    "win_odds", "popularity", "payout", "odds", "result",
    "label", "y_true", "y_pred",
}


def get_table_numeric_columns(
    conn: sqlite3.Connection,
    table_name: str,
    exclude_patterns: Set[str],
) -> List[str]:
    """
    テーブルから数値型カラム名を取得（除外パターン適用済み）

    Args:
        conn: SQLite接続
        table_name: テーブル名
        exclude_patterns: 除外するカラム名のパターン（部分一致）

    Returns:
        数値型カラム名のリスト
    """
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall()

    numeric_types = {"INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"}
    numeric_cols = []

    for row in rows:
        col_name = row[1]
        col_type = row[2].upper() if row[2] else ""

        # 除外パターンチェック（完全一致または部分一致）
        skip = False
        for pattern in exclude_patterns:
            if pattern in col_name.lower() or col_name.lower() == pattern:
                skip = True
                break
        if skip:
            continue

        # 数値型のみ
        if any(nt in col_type for nt in numeric_types):
            numeric_cols.append(col_name)

    return numeric_cols


def filter_columns_by_safety(
    columns: List[str],
    include_warn: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    カラムを安全性分類でフィルタする

    Args:
        columns: カラム名リスト
        include_warn: warnラベルを含めるか

    Returns:
        (safe_cols, warn_cols, unsafe_cols) のタプル
    """
    safe_cols = []
    warn_cols = []
    unsafe_cols = []

    for col in columns:
        label, _ = classify_feature_safety(col)
        if label == "unsafe":
            unsafe_cols.append(col)
        elif label == "warn":
            warn_cols.append(col)
        else:
            safe_cols.append(col)

    return safe_cols, warn_cols, unsafe_cols


def merge_all_feature_tables(
    conn: sqlite3.Connection,
    base_df: pd.DataFrame,
    include_warn: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    v4をベースに、feature_table/v2/v3をLEFT JOINで合流

    Args:
        conn: SQLite接続
        base_df: v4のDataFrame（race_id, horse_id含む）
        include_warn: warn特徴量を含めるか

    Returns:
        (merged_df, added_columns) のタプル
    """
    added_columns = []

    # テーブル存在チェック用関数
    def table_exists(name: str) -> bool:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,)
        )
        return cursor.fetchone() is not None

    # 各テーブルからカラムを取得してマージ
    table_configs = [
        ("feature_table", "legacy__base__"),
        ("feature_table_v2", "legacy__v2__"),
        ("feature_table_v3", "legacy__v3__"),
    ]

    for table_name, prefix in table_configs:
        if not table_exists(table_name):
            logger.debug(f"Table {table_name} not found, skipping")
            continue

        # 数値カラムを取得
        numeric_cols = get_table_numeric_columns(conn, table_name, LEGACY_EXCLUDE_PATTERNS)
        if not numeric_cols:
            logger.debug(f"No numeric columns found in {table_name}")
            continue

        # 安全性フィルタ適用
        safe_cols, warn_cols, unsafe_cols = filter_columns_by_safety(numeric_cols, include_warn)

        if unsafe_cols:
            logger.debug(f"Excluding {len(unsafe_cols)} unsafe columns from {table_name}")

        # 採用するカラム
        use_cols = safe_cols
        if include_warn:
            use_cols = safe_cols + warn_cols

        if not use_cols:
            logger.debug(f"No columns to use from {table_name} after safety filter")
            continue

        # SQLでデータ取得（race_id, horse_idで結合用）
        cols_sql = ", ".join([f'"{c}"' for c in use_cols])
        sql = f'SELECT race_id, horse_id, {cols_sql} FROM {table_name}'
        try:
            legacy_df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.warning(f"Failed to load {table_name}: {e}")
            continue

        # カラム名にプレフィックスを付与
        rename_map = {c: f"{prefix}{c}" for c in use_cols}
        legacy_df = legacy_df.rename(columns=rename_map)

        # LEFT JOIN
        new_cols = [f"{prefix}{c}" for c in use_cols]
        base_df = base_df.merge(
            legacy_df,
            on=["race_id", "horse_id"],
            how="left",
            suffixes=("", "_dup"),
        )

        # 重複カラム削除
        dup_cols = [c for c in base_df.columns if c.endswith("_dup")]
        if dup_cols:
            base_df = base_df.drop(columns=dup_cols)

        added_columns.extend(new_cols)
        logger.info(f"Merged {len(new_cols)} columns from {table_name} (prefix: {prefix})")

    return base_df, added_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/Eval/ROI Pipeline for FeaturePack v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with year-based split (train=2021-2023, test=2024)
  python scripts/train_eval_v4.py --db netkeiba.db

  # Train for top-3 finish prediction
  python scripts/train_eval_v4.py --db netkeiba.db --target target_in3

  # Use date-based split
  python scripts/train_eval_v4.py --db netkeiba.db --split-mode date_based --train-end 2023-06-30 --val-end 2023-12-31

  # Train without pedigree features
  python scripts/train_eval_v4.py --db netkeiba.db --no-pedigree

  # Include market features (for ROI analysis)
  python scripts/train_eval_v4.py --db netkeiba.db --include-market

  # Train with feature diagnostics
  python scripts/train_eval_v4.py --db netkeiba.db --feature-diagnostics

  # Run diagnostics on existing model (skip training)
  python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only

  # Fast diagnostics (skip permutation importance)
  python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only --no-permutation

  # Use legacy model with fallback (auto-detects feature_columns_target_win.json)
  python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only --model-path models/lgbm_target_win.txt

  # Exclude specific features via file
  python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only --exclude-features-file exclude_features.txt

  # Pre-race mode (前日運用): excludes race-day features (h_body_weight, h_body_weight_diff, h_body_weight_dev)
  python scripts/train_eval_v4.py --db netkeiba.db --mode pre_race --feature-diagnostics

  # Pre-race mode with diagnostics-only
  python scripts/train_eval_v4.py --db netkeiba.db --mode pre_race --diagnostics-only --model-path models/lgbm_target_win_v4_prerace.txt
"""
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB (e.g., netkeiba.db)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "pre_race"],
        default="default",
        help=(
            "Operation mode: 'default' uses all features, "
            "'pre_race' excludes race-day features (h_body_weight, h_body_weight_diff, "
            "h_body_weight_dev, market_*) for pre-day operation"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models",
        help="Output directory for models (default: models)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target_win",
        choices=["target_win", "target_in3", "target_quinella"],
        help="Target column (default: target_win)",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["year_based", "date_based"],
        default="year_based",
        help="Split mode: year_based (train=2021-2023, test=2024) or date_based",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2023-12-31",
        help="Train end date for date_based mode (default: 2023-12-31)",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2023-12-31",
        help="Validation end date for date_based mode (default: 2023-12-31)",
    )
    parser.add_argument(
        "--no-pedigree",
        action="store_true",
        help="Exclude pedigree hash features",
    )
    parser.add_argument(
        "--include-market",
        action="store_true",
        help="Include market features (win_odds, popularity) for ROI analysis",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=63,
        help="Number of leaves (default: 63)",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=1000,
        help="Number of boosting rounds (default: 1000)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # ROI Sweep options
    parser.add_argument(
        "--roi-sweep",
        action="store_true",
        help="Run ROI sweep with selective betting strategies (prob/gap thresholds)",
    )
    parser.add_argument(
        "--roi-sweep-prob",
        type=str,
        default=None,
        help="Comma-separated probability thresholds for sweep (default: 0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.18,0.20)",
    )
    parser.add_argument(
        "--roi-sweep-gap",
        type=str,
        default=None,
        help="Comma-separated gap thresholds for sweep (default: 0.005,0.01,0.015,0.02,0.03,0.05,0.08)",
    )

    # Pre-day cutoff options (前日締め運用)
    parser.add_argument(
        "--decision-cutoff",
        type=str,
        default=None,
        help="Decision cutoff time for market evaluation (ISO format, e.g., 2024-12-27T21:00:00)",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable using odds_snapshots for popularity (use race_results instead)",
    )

    # Feature Diagnostics options
    parser.add_argument(
        "--feature-diagnostics",
        action="store_true",
        help="Run feature diagnostics (LightGBM importance, permutation importance, segment performance)",
    )
    parser.add_argument(
        "--perm-top-n",
        type=int,
        default=50,
        help="Number of top features to evaluate for permutation importance (default: 50)",
    )
    parser.add_argument(
        "--perm-n-repeats",
        type=int,
        default=3,
        help="Number of repeats for permutation importance (default: 3)",
    )
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="Skip permutation importance (faster, only LightGBM importance)",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Only run diagnostics on existing model (skip training)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to existing model file for --diagnostics-only mode",
    )
    parser.add_argument(
        "--exclude-features-file",
        type=str,
        default=None,
        help="Path to file with feature names to exclude (1 per line, # for comments)",
    )

    # Feature Sources options (all tables merge)
    parser.add_argument(
        "--feature-sources",
        type=str,
        choices=["v4", "all"],
        default="v4",
        help=(
            "Feature sources: 'v4' uses feature_table_v4 only (default), "
            "'all' merges feature_table, v2, v3 via LEFT JOIN with v4 as base"
        ),
    )
    parser.add_argument(
        "--include-warn",
        action="store_true",
        help="Include features with 'warn' safety label (default: exclude)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Validate database path
    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("FeaturePack v1 Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Database: {db_path}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Split mode: {args.split_mode}")
    logger.info(f"Include pedigree: {not args.no_pedigree}")
    logger.info(f"Include market: {args.include_market}")
    logger.info(f"ROI sweep: {args.roi_sweep}")
    logger.info(f"Decision cutoff: {args.decision_cutoff}")
    logger.info(f"Use snapshots: {not args.no_snapshots}")
    logger.info(f"Feature diagnostics: {args.feature_diagnostics}")
    if args.feature_diagnostics:
        logger.info(f"  Permutation: {not args.no_permutation}")
        logger.info(f"  Perm top-n: {args.perm_top_n}")
        logger.info(f"  Perm repeats: {args.perm_n_repeats}")
    if args.diagnostics_only:
        logger.info(f"Diagnostics only mode: {args.diagnostics_only}")
        logger.info(f"Model path: {args.model_path}")
    if args.exclude_features_file:
        logger.info(f"Exclude features file: {args.exclude_features_file}")
    logger.info(f"Feature sources: {args.feature_sources}")
    if args.feature_sources == "all":
        logger.info(f"  Include warn: {args.include_warn}")

    # Build exclude features set from mode and/or file
    exclude_features_set = set()

    # pre_race mode: load config/exclude_features/pre_race.txt
    if args.mode == "pre_race":
        pre_race_exclude_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "exclude_features", "pre_race.txt"
        )
        if os.path.exists(pre_race_exclude_path):
            pre_race_excludes = load_exclude_features(pre_race_exclude_path)
            exclude_features_set.update(pre_race_excludes)
            logger.info(f"Pre-race mode: loaded {len(pre_race_excludes)} exclude features")
        else:
            logger.warning(f"Pre-race exclude file not found: {pre_race_exclude_path}")

    # Additional excludes from user-specified file
    if args.exclude_features_file:
        user_excludes = load_exclude_features(args.exclude_features_file)
        exclude_features_set.update(user_excludes)

    if exclude_features_set:
        logger.info(f"Total features to exclude: {len(exclude_features_set)}")

    # Create output directory
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create config
    config = TrainConfig(
        target_col=args.target,
        include_pedigree=not args.no_pedigree,
        include_market=args.include_market,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        num_boost_round=args.num_boost_round,
    )

    # Parse ROI sweep thresholds
    roi_sweep_prob = None
    roi_sweep_gap = None
    if args.roi_sweep_prob:
        roi_sweep_prob = [float(x.strip()) for x in args.roi_sweep_prob.split(",")]
    if args.roi_sweep_gap:
        roi_sweep_gap = [float(x.strip()) for x in args.roi_sweep_gap.split(",")]

    conn = sqlite3.connect(db_path)
    try:
        # Check for diagnostics-only mode
        if args.diagnostics_only:
            if not HAS_LIGHTGBM:
                logger.error("lightgbm is not installed, cannot run diagnostics")
                sys.exit(1)

            if not args.model_path:
                # Try default path
                default_model_path = os.path.join(output_dir, f"lgbm_{args.target}_v4.txt")
                if os.path.exists(default_model_path):
                    args.model_path = default_model_path
                else:
                    logger.error("--model-path required for --diagnostics-only mode")
                    sys.exit(1)

            logger.info("Loading existing model: %s", args.model_path)
            model = load_booster(args.model_path)

            # Load feature columns with fallback (candidate features)
            candidate_features, used_cols_path = load_feature_columns_with_fallback(
                output_dir, args.target
            )

            # Apply feature exclusion (from mode and/or file)
            feature_cols = [c for c in candidate_features if c not in exclude_features_set]

            # Log feature selection summary
            log_feature_selection_summary(
                mode=args.mode,
                target_col=args.target,
                candidate_features=candidate_features,
                exclude_set=exclude_features_set,
                final_features=feature_cols,
                output_dir=output_dir,
            )

            # Load data
            logger.info("Loading feature data...")
            df = load_feature_data(
                conn,
                include_pedigree=not args.no_pedigree,
                include_market=args.include_market,
            )

            # Merge legacy feature tables if requested
            if args.feature_sources == "all":
                logger.info("Merging legacy feature tables (feature_sources=all)...")
                df, added_cols = merge_all_feature_tables(
                    conn, df, include_warn=args.include_warn
                )
                if added_cols:
                    # Add merged columns to feature list (exclude those already excluded)
                    new_features = [c for c in added_cols if c not in exclude_features_set]
                    feature_cols = feature_cols + new_features
                    logger.info(f"Added {len(new_features)} legacy features to diagnostics")

            # Split data
            train_df, val_df, test_df = split_time_series(
                df, args.train_end, args.val_end, args.split_mode
            )

            # Run diagnostics
            logger.info("")
            logger.info("Running Feature Diagnostics (diagnostics-only mode)...")
            diag_report = run_diagnostics(
                model=model,
                df=test_df,
                feature_cols=feature_cols,
                target_col=args.target,
                dataset_name="test",
                compute_perm=not args.no_permutation,
                perm_top_n=args.perm_top_n,
                perm_n_repeats=args.perm_n_repeats,
            )

            # Use suffix for output files when feature_sources=all
            output_target = args.target
            if args.feature_sources == "all":
                output_target = f"{args.target}_all"
                logger.info(f"Using output suffix for 'all' mode: {output_target}")

            save_diagnostics(diag_report, output_dir, output_target)
            logger.info("Diagnostics complete!")
            return

        # Normal training flow
        # Note: --feature-sources all is only supported in --diagnostics-only mode
        if args.feature_sources == "all":
            logger.warning(
                "--feature-sources all is currently only supported with --diagnostics-only mode. "
                "Training will use v4 features only."
            )

        results = run_full_pipeline(
            conn=conn,
            config=config,
            output_dir=output_dir,
            train_end=args.train_end,
            val_end=args.val_end,
            split_mode=args.split_mode,
            roi_sweep=args.roi_sweep,
            roi_sweep_prob=roi_sweep_prob,
            roi_sweep_gap=roi_sweep_gap,
            decision_cutoff=args.decision_cutoff,
            use_snapshots=not args.no_snapshots,
            exclude_features=exclude_features_set if exclude_features_set else None,
            mode=args.mode,
        )

        # Print results summary
        logger.info("=" * 70)
        logger.info("Results Summary")
        logger.info("=" * 70)

        if "test_result" in results:
            test = results["test_result"]
            logger.info(f"Test AUC:      {test['auc']:.4f}")
            logger.info(f"Test LogLoss:  {test['logloss']:.4f}")
            logger.info(f"Test Accuracy: {test['accuracy']:.4f}")
            logger.info(f"Test Samples:  {test['n_samples']:,}")

        if results.get("roi_result"):
            roi = results["roi_result"]
            logger.info("")
            logger.info("ROI Analysis:")
            logger.info(f"  Bets:     {roi['n_bets']:,}")
            logger.info(f"  Wins:     {roi['n_wins']:,}")
            logger.info(f"  Hit Rate: {roi['hit_rate']:.2%}")
            logger.info(f"  ROI:      {roi['roi']:.2%}")

        # 非シリアライズ可能なオブジェクトを抽出（diagnostics 用）
        trained_model = results.pop("_model", None)
        trained_feature_cols = results.pop("_feature_cols", None)
        trained_test_df = results.pop("_test_df", None)

        logger.info("=" * 70)
        print(json.dumps(results, indent=2, ensure_ascii=False))

        # Run Feature Diagnostics if requested
        if args.feature_diagnostics:
            if not HAS_LIGHTGBM:
                logger.error("lightgbm is not installed, cannot run diagnostics")
            elif trained_model is None:
                logger.error("No model available for diagnostics (training may have failed)")
            else:
                logger.info("")
                logger.info("=" * 70)
                logger.info("Running Feature Diagnostics (using in-memory model)...")
                logger.info("=" * 70)

                # in-memory model を使用（ディスクから再ロードしない）
                model = trained_model
                feature_cols = trained_feature_cols
                test_df = trained_test_df

                # Note: exclude_features は学習時に既に適用済み
                # trained_feature_cols には除外後の特徴量のみ含まれる

                # Run diagnostics on test data
                diag_report = run_diagnostics(
                    model=model,
                    df=test_df,
                    feature_cols=feature_cols,
                    target_col=args.target,
                    dataset_name="test",
                    compute_perm=not args.no_permutation,
                    perm_top_n=args.perm_top_n,
                    perm_n_repeats=args.perm_n_repeats,
                )
                save_diagnostics(diag_report, output_dir, args.target)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
