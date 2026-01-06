#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_legacy.py - Unified training script for legacy feature tables (v3/v2/v1)

Trains LightGBM models for target_win and target_in3 using legacy feature tables.
This script is designed for the Feature Audit System (Step D-2).

Usage:
    # Train using v3 table
    python scripts/train_eval_legacy.py --db netkeiba.db --version v3

    # Train using v2 table
    python scripts/train_eval_legacy.py --db netkeiba.db --version v2

    # Train using v1 table (original feature_table)
    python scripts/train_eval_legacy.py --db netkeiba.db --version v1

    # Custom output directory
    python scripts/train_eval_legacy.py --db netkeiba.db --version v3 --out models/legacy/
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ========================================
# Version-to-Table Mapping
# ========================================

VERSION_TABLE_MAP = {
    "v3": "feature_table_v3",
    "v2": "feature_table_v2",
    "v1": "feature_table",
}

# ========================================
# Prohibited Columns (Pre-race Safety)
# ========================================

PROHIBITED_EXACT = {
    "race_id",
    "horse_id",
    "horse_no",
    "target_win",
    "target_in3",
    "target_value",
    "finish_order",
    "finish_position",
    "paid_places",
    "payout_count",
    "should_have_payout",
    "fukusho_payout",
    # Pre-race mode: exclude track condition (scenario layer handles it)
    "track_condition",
    "track_condition_id",
    # Pre-race mode: exclude same-day weight info
    "horse_weight",
    "horse_weight_diff",
}

PROHIBITED_PATTERNS = (
    "target",
    "payout",
    "paid_",
    "should_have",
)


def _is_prohibited(col: str) -> bool:
    """Check if column is prohibited as a feature."""
    col_lower = col.lower()
    if col_lower in PROHIBITED_EXACT:
        return True
    for pattern in PROHIBITED_PATTERNS:
        if pattern in col_lower:
            return True
    return False


# ========================================
# Data Loading
# ========================================


def check_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get column names from a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def load_dataset(
    conn: sqlite3.Connection,
    table_name: str,
    year_min: int = 2021,
    year_max: int = 2024,
) -> pd.DataFrame:
    """
    Load dataset from specified feature table.

    Args:
        conn: Database connection
        table_name: Name of the feature table
        year_min: Minimum year to include
        year_max: Maximum year to include

    Returns:
        DataFrame with features and targets
    """
    # Get available columns
    all_columns = get_table_columns(conn, table_name)
    logger.info(f"Table {table_name} has {len(all_columns)} columns")

    # Build query
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE race_year BETWEEN {year_min} AND {year_max}
    """

    df = pd.read_sql_query(query, conn)
    logger.info(f"Loaded {len(df):,} rows from {table_name}")

    # Log column categories
    hr_cols = [c for c in df.columns if c.startswith("hr_")]
    ax_cols = [c for c in df.columns if c.startswith("ax")]
    rr_cols = [c for c in df.columns if c.startswith("rr_")]

    logger.info(f"  - hr_* columns: {len(hr_cols)}")
    logger.info(f"  - ax* columns: {len(ax_cols)}")
    logger.info(f"  - rr_* columns: {len(rr_cols)}")

    return df


def split_by_year(
    df: pd.DataFrame,
    train_years: Tuple[int, int] = (2021, 2023),
    test_year: int = 2024,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by year: train on historical years, test on recent year.

    Args:
        df: Full dataset
        train_years: (min_year, max_year) for training
        test_year: Year for testing

    Returns:
        (df_train, df_test)
    """
    df_train = df[
        (df["race_year"] >= train_years[0]) &
        (df["race_year"] <= train_years[1])
    ].copy()
    df_test = df[df["race_year"] == test_year].copy()

    train_races = df_train["race_id"].nunique()
    test_races = df_test["race_id"].nunique()

    logger.info(f"Train: {len(df_train):,} rows, {train_races:,} races ({train_years[0]}-{train_years[1]})")
    logger.info(f"Test:  {len(df_test):,} rows, {test_races:,} races ({test_year})")

    return df_train, df_test


# ========================================
# Feature Matrix Building
# ========================================


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "target_win",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build feature matrix from dataframe.

    Args:
        df: Input dataframe
        target_col: Target column name

    Returns:
        (X, y, feature_cols)
    """
    feature_cols = []
    excluded_exact = []
    excluded_pattern = []

    for col in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Check prohibitions
        col_lower = col.lower()
        if col_lower in PROHIBITED_EXACT:
            excluded_exact.append(col)
            continue

        is_pattern_match = False
        for pattern in PROHIBITED_PATTERNS:
            if pattern in col_lower:
                excluded_pattern.append(col)
                is_pattern_match = True
                break

        if is_pattern_match:
            continue

        feature_cols.append(col)

    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Excluded (exact): {len(excluded_exact)} - {excluded_exact[:10]}...")
    logger.info(f"Excluded (pattern): {len(excluded_pattern)}")

    # Verify no leakage
    leaked = [c for c in feature_cols if _is_prohibited(c)]
    if leaked:
        raise RuntimeError(f"Prohibited columns in features: {leaked}")

    X = df[feature_cols].fillna(0)
    y = df[target_col].astype(int).values

    return X, y, feature_cols


# ========================================
# Model Training
# ========================================


def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 42,
) -> lgb.Booster:
    """
    Train LightGBM binary classification model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        seed: Random seed

    Returns:
        Trained LightGBM Booster
    """
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'max_depth': 6,
        'seed': seed,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.log_evaluation(period=100)]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=True))

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks,
        )
    else:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=callbacks,
        )

    return model


# ========================================
# Model Saving
# ========================================


def save_model_artifacts(
    model: lgb.Booster,
    feature_cols: List[str],
    out_dir: Path,
    target: str,
    version: str,
) -> Dict[str, Path]:
    """
    Save model and related artifacts.

    Args:
        model: Trained model
        feature_cols: Feature column names
        out_dir: Output directory
        target: Target name (target_win or target_in3)
        version: Version name (v3, v2, v1)

    Returns:
        Dictionary of saved file paths
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    # Model filename pattern: lgbm_target_win_v3.txt
    base_name = f"lgbm_{target}_{version}"

    # 1. Save model (text format)
    model_path = out_dir / f"{base_name}.txt"
    model.save_model(str(model_path))
    saved_files["model"] = model_path
    logger.info(f"Saved model: {model_path}")

    # 2. Save feature columns (JSON)
    features_json = out_dir / f"feature_columns_{target}_{version}.json"
    with open(features_json, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)
    saved_files["feature_columns"] = features_json
    logger.info(f"Saved feature columns: {features_json}")

    # 3. Save feature importance (CSV)
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'gain': importance_gain,
        'split': importance_split.astype(int),
    }).sort_values('gain', ascending=False)

    importance_csv = out_dir / f"feature_importance_{target}_{version}.csv"
    importance_df.to_csv(importance_csv, index=False)
    saved_files["importance"] = importance_csv
    logger.info(f"Saved importance: {importance_csv}")

    # Log top 10 features
    logger.info(f"Top 10 features by gain for {target}:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['gain']:.2f}")

    return saved_files


# ========================================
# Evaluation
# ========================================


def evaluate_model(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    target: str,
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        df_test: Test dataframe (for race-level evaluation)
        target: Target name

    Returns:
        Dictionary of metrics
    """
    y_pred_prob = model.predict(X_test)
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    metrics = {}

    # Global metrics
    metrics["accuracy"] = accuracy_score(y_test, y_pred_binary)

    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["log_loss"] = log_loss(y_test, y_pred_prob)
    except ValueError:
        metrics["log_loss"] = float("nan")

    metrics["brier_score"] = brier_score_loss(y_test, y_pred_prob)

    logger.info("=" * 60)
    logger.info(f"Evaluation Results for {target}")
    logger.info("=" * 60)
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info(f"Log Loss:    {metrics['log_loss']:.4f}")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")

    # Race-level ranking evaluation
    df_eval = df_test.copy()
    df_eval["pred_prob"] = y_pred_prob

    ranking_results = []
    for race_id, race_df in df_eval.groupby("race_id"):
        if race_df.empty:
            continue

        race_sorted = race_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        race_sorted["rank"] = range(1, len(race_sorted) + 1)

        positive_horses = race_sorted[race_sorted[target] == 1]
        if len(positive_horses) > 0:
            best_rank = positive_horses["rank"].min()
            ranking_results.append(best_rank)

    if ranking_results:
        results = np.array(ranking_results)
        metrics["hit_top1"] = (results == 1).mean()
        metrics["hit_top3"] = (results <= 3).mean()
        metrics["avg_rank"] = results.mean()

        logger.info(f"Top1 Hit Rate: {metrics['hit_top1']:.4f}")
        logger.info(f"Top3 Hit Rate: {metrics['hit_top3']:.4f}")
        logger.info(f"Avg Best Rank: {metrics['avg_rank']:.2f}")

    return metrics


# ========================================
# Main
# ========================================


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM models for legacy feature tables (v3/v2/v1)"
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["v3", "v2", "v1"],
        help="Feature table version to use"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models",
        help="Output directory for models (default: models)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["target_win", "target_in3"],
        help="Target columns to train (default: target_win target_in3)"
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs=2,
        default=[2021, 2023],
        help="Train year range (default: 2021 2023)"
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2024,
        help="Test year (default: 2024)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check table existence without training"
    )

    args = parser.parse_args()

    # Resolve paths
    db_path = Path(args.db)
    out_dir = Path(args.out)

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Get table name
    table_name = VERSION_TABLE_MAP[args.version]
    logger.info(f"Version: {args.version} -> Table: {table_name}")

    # Connect to database
    conn = sqlite3.connect(str(db_path))

    # Check table exists
    if not check_table_exists(conn, table_name):
        logger.error(f"Table {table_name} not found in database")
        conn.close()
        sys.exit(1)

    logger.info(f"Table {table_name} found")

    if args.dry_run:
        columns = get_table_columns(conn, table_name)
        logger.info(f"Columns ({len(columns)}): {columns[:20]}...")
        conn.close()
        logger.info("Dry run complete")
        return

    # Load dataset
    df = load_dataset(
        conn,
        table_name,
        year_min=args.train_years[0],
        year_max=args.test_year,
    )

    # Split data
    df_train, df_test = split_by_year(
        df,
        train_years=tuple(args.train_years),
        test_year=args.test_year,
    )

    # Train for each target
    all_results = {}

    for target in args.targets:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Training model for {target}")
        logger.info("=" * 60)

        if target not in df.columns:
            logger.warning(f"Target column {target} not found, skipping")
            continue

        # Build feature matrix
        X_train, y_train, feature_cols = build_feature_matrix(df_train, target)
        X_test, y_test, _ = build_feature_matrix(df_test, target)

        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Positive rate (train): {y_train.mean():.4f}")
        logger.info(f"Positive rate (test): {y_test.mean():.4f}")

        # Train model
        model = train_lgbm_model(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            seed=args.seed,
        )

        # Save model
        saved_files = save_model_artifacts(
            model, feature_cols, out_dir, target, args.version
        )

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, df_test, target)
        metrics["saved_files"] = {k: str(v) for k, v in saved_files.items()}

        all_results[target] = metrics

    conn.close()

    # Save summary
    summary_path = out_dir / f"training_summary_{args.version}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": args.version,
            "table": table_name,
            "train_years": args.train_years,
            "test_year": args.test_year,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved training summary: {summary_path}")

    logger.info("")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
