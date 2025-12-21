"""
LightGBM baseline model training and evaluation script.

Mirrors train_eval_logistic.py but uses LightGBM for in3 (top-3 finish) classification.
Produces the same evaluation metrics for direct comparison with logistic regression baseline.

Usage:
    python scripts/train_eval_lgbm.py --db netkeiba.db
    python scripts/train_eval_lgbm.py --db netkeiba.db --add-rr --rr-kind both
    python scripts/train_eval_lgbm.py --db netkeiba.db --debug-features
"""
import argparse
import json
import logging
import os
import sqlite3
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ========================================
# Prohibited Columns (Leak Prevention)
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
    # Scope separation: track condition handled by scenario layer, not base model
    "track_condition",
    "track_condition_id",
    # Early-preview (前日) only: exclude same-day weight info
    "horse_weight",
    "horse_weight_diff",
}

# Columns that must never be used as features if their names contain these substrings
PROHIBITED_PATTERNS = (
    "target",
    "payout",
    "paid_",
    "should_have",
)

DEFAULT_RR_COLS = [
    "hr_career_in3_rate",
    "hr_career_win_rate",
    "hr_career_avg_finish",
    "hr_career_avg_last3f",
    "hr_recent5_avg_finish",
    "hr_recent5_best_finish",
    "hr_recent5_avg_last3f",
    "hr_recent5_big_loss_count",
    "hr_recent3_avg_win_odds",
    "hr_recent3_finish_trend",
    "hr_days_since_prev",
]

# Columns where lower is better; rankings/z-scores will be inverted to make larger = better
RR_SMALLER_IS_BETTER = {
    "hr_career_avg_finish",
    "hr_career_avg_last3f",
    "hr_recent5_avg_finish",
    "hr_recent5_best_finish",
    "hr_recent5_avg_last3f",
    "hr_recent5_big_loss_count",
    "hr_recent3_avg_win_odds",
    "hr_days_since_prev",
}


# ========================================
# Data Loading
# ========================================


def load_dataset(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load dataset from feature_table_v3/v2/v1 (prefers v3 > v2 > v1)."""
    hr_cols_in_table: List[str] = []
    ax_cols_in_table: List[str] = []

    # Check which feature table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'feature_table%'")
    available_tables = [row[0] for row in cursor.fetchall()]

    if 'feature_table_v3' in available_tables:
        feature_table_name = 'feature_table_v3'
        print(f"[INFO] Using feature_table_v3")

        # Check columns in feature_table_v3
        cursor.execute(f"PRAGMA table_info({feature_table_name})")
        table_columns = [row[1] for row in cursor.fetchall()]
        hr_cols_in_table = [c for c in table_columns if c.startswith("hr_")]
        ax_cols_in_table = [c for c in table_columns if c.startswith("ax")]
        print(f"[INFO] hr_* columns in {feature_table_name} table: {len(hr_cols_in_table)}")
        print(f"[INFO] ax*_ columns in {feature_table_name} table: {len(ax_cols_in_table)}")
        if ax_cols_in_table:
            print(f"[INFO] ax*_ columns: {ax_cols_in_table}")

    elif 'feature_table_v2' in available_tables:
        feature_table_name = 'feature_table_v2'
        print(f"[INFO] Using feature_table_v2")

        # Check columns in feature_table_v2
        cursor.execute(f"PRAGMA table_info({feature_table_name})")
        table_columns = [row[1] for row in cursor.fetchall()]
        hr_cols_in_table = [c for c in table_columns if c.startswith("hr_")]
        print(f"[INFO] hr_* columns in {feature_table_name} table: {len(hr_cols_in_table)}")
        if hr_cols_in_table:
            print(f"[INFO] hr_* columns: {hr_cols_in_table}")
        else:
            print(f"[WARN] No hr_* columns found in {feature_table_name} table!")
            print(f"[WARN] This might indicate wrong database file or table needs rebuild")

    elif 'feature_table' in available_tables:
        feature_table_name = 'feature_table'
        print(f"[WARN] feature_table_v2 not found, falling back to feature_table")
        print(f"[WARN] This means hr_* features will NOT be available")
    else:
        raise RuntimeError(
            "Neither feature_table_v3 nor feature_table_v2 nor feature_table found in database.\n"
            "Please run:\n"
            "  python scripts/build_feature_table_v3.py --db <your_db_path>\n"
            "or check your database path."
        )

    # Build column list: base columns + hr_* columns
    base_cols = [
        "race_id", "horse_id", "target_win", "target_in3", "target_value",
        "course", "surface", "surface_id", "distance", "distance_cat",
        "track_condition", "track_condition_id", "field_size", "race_class",
        "race_year", "race_month", "waku", "umaban", "horse_weight",
        "horse_weight_diff", "is_first_run", "n_starts_total",
        "win_rate_total", "in3_rate_total", "avg_finish_total",
        "std_finish_total", "n_starts_dist_cat", "win_rate_dist_cat",
        "in3_rate_dist_cat", "avg_finish_dist_cat", "avg_last3f_dist_cat",
        "days_since_last_run", "recent_avg_finish_3", "recent_best_finish_3",
        "recent_avg_last3f_3", "n_starts_track_condition",
        "win_rate_track_condition", "n_starts_course", "win_rate_course",
        "avg_horse_weight"
    ]

    # Add hr_* and ax*_ columns if available
    select_cols = base_cols + hr_cols_in_table + ax_cols_in_table
    select_cols_sql = ",\n                ".join(select_cols)

    query = f"""
        WITH f AS (
            SELECT
                {select_cols_sql}
            FROM {feature_table_name}
            -- ここでは年で絞らない（2021〜2024 を全部持ってくる）
        ),
        rr AS (
            SELECT
                race_id,
                horse_id,
                horse_no,
                finish_order
            FROM race_results
        ),
        fuku AS (
            SELECT
                race_id,
                combination AS horse_no_str,
                payout AS fukusho_payout
            FROM payouts
            WHERE bet_type = '複勝'
        )
        SELECT
            f.*,
            rr.horse_no,
            rr.finish_order,
            fuku.fukusho_payout
        FROM f
        JOIN rr
          ON f.race_id = rr.race_id
         AND f.horse_id = rr.horse_id
        LEFT JOIN fuku
          ON f.race_id = fuku.race_id
         AND CAST(rr.horse_no AS TEXT) = fuku.horse_no_str
        WHERE f.race_year BETWEEN 2021 AND 2024
    """

    # Debug: Log query
    print(f"[DEBUG] SQL query:")
    print(query)
    print()

    df = pd.read_sql_query(query, conn)
    print(f"[INFO] loaded dataset rows: {len(df):,}")
    print(f"[DEBUG] Total columns in df: {len(df.columns)}")
    print(f"[DEBUG] All df columns: {df.columns.tolist()}")

    # Check for hr_* columns
    hr_cols_in_df = [c for c in df.columns if c.startswith("hr_")]
    print(f"[INFO] hr_* columns in dataset: {len(hr_cols_in_df)}")
    if hr_cols_in_df:
        print(f"[INFO] hr_* column names: {hr_cols_in_df}")
    else:
        print(f"[WARN] No hr_* columns found in dataset despite being in table!")

    missing_fuku = df["fukusho_payout"].isna().sum()
    print(f"[INFO] fukusho_payout missing rows: {missing_fuku:,}")
    return df


def split_by_race(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by year: 2021-2023 train, 2024 test."""
    df_train = df[(df["race_year"] >= 2021) & (df["race_year"] <= 2023)].copy()
    df_test = df[df["race_year"] == 2024].copy()

    train_race_ids = df_train["race_id"].nunique()
    test_race_ids = df_test["race_id"].nunique()
    total_races = df["race_id"].nunique()

    print(f"[INFO] races: total={total_races}, train={train_race_ids}, test={test_race_ids}")
    print(f"[INFO] rows: train={len(df_train):,}, test={len(df_test):,}")
    return df_train, df_test


# ========================================
# Prohibited Column Checking
# ========================================


def _prohibited_reason(col: str) -> Tuple[bool, str]:
    """
    Check if column is prohibited as a feature.

    Returns:
        (is_prohibited, reason) where reason is 'exact' or 'pattern:<pattern>'
    """
    normalized = col.lower()
    if normalized in PROHIBITED_EXACT:
        return True, "exact"

    for pattern in PROHIBITED_PATTERNS:
        if pattern in normalized:
            return True, f"pattern:{pattern}"

    return False, ""


def _log_feature_overview(feature_cols: List[str], hr_feature_cols: List[str]) -> None:
    print("=" * 60)
    print("[INFO] Feature columns summary")
    print("=" * 60)
    print(f"[INFO] feature cols count: {len(feature_cols)}")

    head_preview = feature_cols[:30]
    tail_preview = feature_cols[-30:] if len(feature_cols) > 30 else []
    print(f"[INFO] feature cols (head 30): {head_preview}")
    if tail_preview:
        print(f"[INFO] feature cols (tail 30): {tail_preview}")
    print(f"[INFO] hr_* feature cols ({len(hr_feature_cols)}): {hr_feature_cols}")
    print("=" * 60)

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    features_path = os.path.join(artifacts_dir, "features_used.txt")
    try:
        with open(features_path, "w", encoding="utf-8") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        print(f"[INFO] Saved full feature list to: {features_path}")
    except OSError as e:
        print(f"[WARN] Failed to write feature list to {features_path}: {e}")


def _log_prohibited_summary(
    prohibited_exact_hits: List[str],
    prohibited_pattern_hits: List[str],
    detected_in_features: List[str],
) -> None:
    print("=" * 60)
    print("[INFO] Prohibited column check")
    print("=" * 60)
    print(f"[INFO] Excluded by exact match ({len(prohibited_exact_hits)}): {sorted(set(prohibited_exact_hits))}")
    print(f"[INFO] Excluded by pattern match ({len(prohibited_pattern_hits)}): {sorted(set(prohibited_pattern_hits))}")
    if detected_in_features:
        print(f"[ERROR] Prohibited columns detected in feature set: {sorted(set(detected_in_features))}")
    else:
        print("[INFO] No prohibited columns present in feature set.")
    print("=" * 60)


def _log_rr_summary(
    enabled: bool,
    rr_kind: str,
    created_cols: List[str],
    used_base_cols: List[str],
) -> None:
    print("=" * 60)
    print("[INFO] Relative (rr_) feature generation")
    print("=" * 60)
    print(f"[INFO] add_rr enabled: {enabled}")
    print(f"[INFO] rr_kind: {rr_kind}")
    print(f"[INFO] rr base cols used ({len(used_base_cols)}): {used_base_cols}")
    print(f"[INFO] rr created cols count: {len(created_cols)}")
    if created_cols:
        head = created_cols[:15]
        tail = created_cols[-15:] if len(created_cols) > 15 else []
        print(f"[INFO] rr created cols (head): {head}")
        if tail:
            print(f"[INFO] rr created cols (tail): {tail}")
    print("=" * 60)


# ========================================
# Race-Relative Feature Generation
# ========================================


def _add_rr_features(
    df: pd.DataFrame,
    rr_kind: str,
    rr_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Add race-relative (rr_) features computed per race_id.

    Returns:
        df_with_rr, created_rr_cols, used_base_cols
    """
    df = df.copy()
    created_cols: List[str] = []
    used_base_cols: List[str] = []

    valid_rr_cols = []
    for col in rr_cols:
        if col not in df.columns:
            print(f"[WARN] rr base column not found and skipped: {col}")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"[WARN] rr base column is not numeric and skipped: {col}")
            continue
        valid_rr_cols.append(col)

    if not valid_rr_cols:
        print("[WARN] No valid rr base columns found; skipping rr feature generation.")
        return df, created_cols, used_base_cols

    grouped = df.groupby("race_id")

    for col in valid_rr_cols:
        series = df[col]
        used_base_cols.append(col)

        if rr_kind in ("rank_pct", "both"):
            rank = grouped[col].rank(pct=True, method="average")
            if col in RR_SMALLER_IS_BETTER:
                rank = 1 - rank
            new_col = f"rr_rank_pct_{col}"
            df[new_col] = rank
            created_cols.append(new_col)

        if rr_kind in ("zscore", "both"):
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std")
            z = (series - mean) / std.replace(0, np.nan)
            z = z.fillna(0)
            if col in RR_SMALLER_IS_BETTER:
                z = -z
            new_col = f"rr_zscore_{col}"
            df[new_col] = z
            created_cols.append(new_col)

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    rr_path = os.path.join(artifacts_dir, "rr_cols_used.txt")
    try:
        with open(rr_path, "w", encoding="utf-8") as f:
            for col in created_cols:
                f.write(f"{col}\n")
        print(f"[INFO] Saved rr feature list to: {rr_path}")
    except OSError as e:
        print(f"[WARN] Failed to write rr feature list to {rr_path}: {e}")

    return df, created_cols, used_base_cols


# ========================================
# Feature Matrix Building
# ========================================


def build_feature_matrix(
    df: pd.DataFrame,
    debug_features: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build feature matrix from dataframe.

    Returns:
        X (DataFrame), y (ndarray), feature_cols (list)
    """
    feature_cols: List[str] = []
    prohibited_exact_hits: List[str] = []
    prohibited_pattern_hits: List[str] = []

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        is_prohibited, reason = _prohibited_reason(col)
        if is_prohibited:
            if reason == "exact":
                prohibited_exact_hits.append(col)
            elif reason.startswith("pattern:"):
                prohibited_pattern_hits.append(col)
            continue

        feature_cols.append(col)

    # Count hr_* features
    hr_feature_cols = [c for c in feature_cols if c.startswith("hr_")]

    leaking_cols = [c for c in feature_cols if _prohibited_reason(c)[0]]
    if leaking_cols:
        _log_prohibited_summary(prohibited_exact_hits, prohibited_pattern_hits, leaking_cols)
        raise RuntimeError(
            "Prohibited columns detected in feature set (possible data leak): "
            f"{sorted(leaking_cols)}"
        )

    _log_prohibited_summary(prohibited_exact_hits, prohibited_pattern_hits, leaking_cols)
    if debug_features:
        _log_feature_overview(feature_cols, hr_feature_cols)

    X = df[feature_cols].fillna(0)
    y = df["target_in3"].astype(int).values

    return X, y, feature_cols


# ========================================
# Model Training (LightGBM)
# ========================================


def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
) -> lgb.Booster:
    """
    Train LightGBM model for binary classification.

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
        'seed': 42,
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


def save_model(
    model: lgb.Booster,
    feature_cols: List[str],
    model_dir: str = "models",
    model_name: str = "lgbm_in3",
) -> None:
    """Save model and feature columns to disk."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model (text format)
    model_path = os.path.join(model_dir, f"{model_name}.txt")
    model.save_model(model_path)
    print(f"[INFO] Saved model to: {model_path}")

    # Save feature columns
    features_path = os.path.join(model_dir, f"{model_name}_features.txt")
    with open(features_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"[INFO] Saved feature columns to: {features_path}")

    # Save feature columns as JSON (for easier programmatic loading)
    features_json_path = os.path.join(model_dir, f"{model_name}_features.json")
    with open(features_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved feature columns (JSON) to: {features_json_path}")

    # Save feature importance
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    importance_path = os.path.join(model_dir, f"{model_name}_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"[INFO] Saved feature importance to: {importance_path}")

    # Log top 20 features
    print("\n[INFO] Top 20 important features (gain):")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # ========================================
    # Prefix-based gain summary (ax1-ax9, rr, other)
    # ========================================
    try:
        def get_prefix_group(feature_name: str) -> str:
            """Classify feature into ax1-ax9, rr, or other."""
            for i in range(1, 10):
                if feature_name.startswith(f"ax{i}_"):
                    return f"ax{i}"
            if feature_name.startswith("rr_"):
                return "rr"
            return "other"

        importance_df["prefix"] = importance_df["feature"].apply(get_prefix_group)
        prefix_summary = importance_df.groupby("prefix")["importance"].sum().reset_index()
        prefix_summary.columns = ["prefix", "gain_sum"]
        total_gain = prefix_summary["gain_sum"].sum()
        prefix_summary["share_pct"] = 100.0 * prefix_summary["gain_sum"] / total_gain if total_gain > 0 else 0.0
        prefix_summary = prefix_summary.sort_values("gain_sum", ascending=False)

        print("\n[INFO] Gain summary by prefix (ax1-ax9 / rr / other):")
        print(f"  {'prefix':<8} {'gain_sum':>14} {'share':>8}")
        print(f"  {'-'*8} {'-'*14} {'-'*8}")
        for _, row in prefix_summary.iterrows():
            print(f"  {row['prefix']:<8} {row['gain_sum']:>14.0f} {row['share_pct']:>7.2f}%")
        print(f"  {'-'*8} {'-'*14} {'-'*8}")
        print(f"  {'TOTAL':<8} {total_gain:>14.0f} {'100.00':>7}%")

        # Save to artifacts/feature_importance_gain.csv (fail-soft)
        try:
            artifacts_dir = "artifacts"
            os.makedirs(artifacts_dir, exist_ok=True)
            artifacts_importance_path = os.path.join(artifacts_dir, "feature_importance_gain.csv")
            importance_df[["feature", "importance"]].to_csv(artifacts_importance_path, index=False)
            print(f"[INFO] Saved feature importance to: {artifacts_importance_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save to artifacts: {e}")

    except Exception as e:
        print(f"[WARNING] Failed to compute prefix-based gain summary: {e}")


# ========================================
# Evaluation Functions
# ========================================


def evaluate_global_metrics(
    df: pd.DataFrame,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
) -> Dict[str, float]:
    """
    グローバル指標の計算：accuracy, ROC-AUC, log loss, Brier score
    """
    y_true = df[y_true_col].astype(int).values
    y_pred_prob = df[y_pred_col].values
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred_binary)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = float("nan")

    try:
        logloss = log_loss(y_true, y_pred_prob)
    except ValueError:
        logloss = float("nan")

    brier = brier_score_loss(y_true, y_pred_prob)

    logger.info("=" * 60)
    logger.info("Global Metrics (全体指標)")
    logger.info("=" * 60)
    logger.info(f"Accuracy:     {accuracy:.4f}")
    logger.info(f"ROC-AUC:      {roc_auc:.4f}")
    logger.info(f"Log Loss:     {logloss:.4f}")
    logger.info(f"Brier Score:  {brier:.4f}")
    logger.info("")

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "brier_score": brier,
    }


def evaluate_ranking(
    df: pd.DataFrame,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
) -> Dict[str, float]:
    """
    レース単位のランキング評価
    各レースで in3 の馬のうち、最も予測確率が高い馬の順位を評価
    """
    results = []

    for race_id, race_df in df.groupby("race_id"):
        if race_df.empty:
            continue

        # 予測確率でソート
        race_sorted = race_df.sort_values(y_pred_col, ascending=False).reset_index(drop=True)
        race_sorted["rank"] = range(1, len(race_sorted) + 1)

        # in3 の馬を取得
        in3_horses = race_sorted[race_sorted[y_true_col] == 1]

        if len(in3_horses) == 0:
            # in3 の馬が1頭もいないレース（稀だが安全のため）
            continue

        # in3 馬の中で最も予測確率が高い馬の順位
        best_in3_rank = in3_horses["rank"].min()
        results.append(best_in3_rank)

    if not results:
        logger.warning("No races with in3 horses found!")
        return {}

    results = np.array(results)
    hit_top1 = (results == 1).mean()
    hit_top2 = (results <= 2).mean()
    hit_top3 = (results <= 3).mean()
    avg_rank = results.mean()

    logger.info("=" * 60)
    logger.info("Ranking Evaluation (ランキング評価)")
    logger.info("=" * 60)
    logger.info(f"テストレース数:        {len(results):,}")
    logger.info(f"Top1 カバー率:         {hit_top1:.3f}")
    logger.info(f"Top2 カバー率:         {hit_top2:.3f}")
    logger.info(f"Top3 カバー率:         {hit_top3:.3f}")
    logger.info(f"平均 rank (best_in3):  {avg_rank:.2f}")
    logger.info("")

    return {
        "n_races": len(results),
        "hit_top1_rate": hit_top1,
        "hit_top2_rate": hit_top2,
        "hit_top3_rate": hit_top3,
        "avg_rank_best_in3": avg_rank,
    }


def evaluate_strategy_top1_all(
    df: pd.DataFrame,
    bet_amount: int = 100,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
    payout_col: str = "fukusho_payout",
    exclude_missing_payout: bool = True,
) -> Dict[str, float]:
    """
    戦略A：全レースで予測確率最大の馬に複勝ベット

    Args:
        exclude_missing_payout: True の場合、should_have_payout (JRAルール) なのに payout が NULL のレースを除外
    """
    # paid_places の計算（JRAルール）
    def calculate_paid_places(field_size):
        if field_size >= 8:
            return 3
        elif field_size >= 5:
            return 2
        else:
            return 0

    # 除外前の計算
    n_races_before = 0
    n_hits_before = 0
    total_return_before = 0.0
    excluded_races = []

    for race_id, race_df in df.groupby("race_id"):
        if race_df.empty:
            continue

        n_races_before += 1
        best_idx = race_df[y_pred_col].idxmax()
        chosen = race_df.loc[best_idx]

        payout = chosen.get(payout_col)
        is_in3 = int(chosen[y_true_col]) == 1

        # JRAルールに基づく除外判定
        if exclude_missing_payout:
            field_size = chosen.get("field_size", len(race_df))
            paid_places = calculate_paid_places(field_size)
            finish_order = chosen.get("finish_order", 999)

            # should_have_payout = 払戻対象着順なのに
            should_have_payout = (finish_order <= paid_places) and (paid_places > 0)

            # 真の異常：should_have_payout なのに payout が NULL
            if should_have_payout and pd.isna(payout):
                excluded_races.append(race_id)

        if pd.notna(payout):
            total_return_before += float(payout)
            if is_in3:
                n_hits_before += 1

    # 除外後の計算
    n_races_after = 0
    n_hits_after = 0
    total_return_after = 0.0

    for race_id, race_df in df.groupby("race_id"):
        if race_df.empty:
            continue

        # 除外対象のレースをスキップ
        if exclude_missing_payout and race_id in excluded_races:
            continue

        n_races_after += 1
        best_idx = race_df[y_pred_col].idxmax()
        chosen = race_df.loc[best_idx]

        payout = chosen.get(payout_col)
        if pd.notna(payout):
            total_return_after += float(payout)
            if int(chosen[y_true_col]) == 1:
                n_hits_after += 1

    # 除外統計
    n_excluded_races = len(excluded_races)
    n_excluded_rows = df[df["race_id"].isin(excluded_races)].shape[0] if n_excluded_races > 0 else 0

    # メトリクス計算
    if exclude_missing_payout and n_excluded_races > 0:
        # 除外後の結果を使用
        n_races = n_races_after
        n_hits = n_hits_after
        total_return = total_return_after
    else:
        # 除外前の結果を使用
        n_races = n_races_before
        n_hits = n_hits_before
        total_return = total_return_before

    total_bet = n_races * bet_amount
    hit_rate = n_hits / n_races if n_races > 0 else 0.0
    roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Strategy: Top1 All Races (全レース top1 買い)")
    logger.info("=" * 60)

    if exclude_missing_payout and n_excluded_races > 0:
        logger.info(f"  Excluded races (should have payout but NULL, JRA rules): {n_excluded_races:,} races ({n_excluded_rows:,} rows)")
        logger.info("")
        logger.info("Results (AFTER exclusion):")

    logger.info(f"ベットレース数:  {n_races:,}")
    logger.info(f"的中数:          {n_hits:,}")
    logger.info(f"的中率:          {hit_rate:.3f}")
    logger.info(f"総投資額:        {total_bet:,} 円")
    logger.info(f"総払戻:          {int(total_return):,} 円")
    logger.info(f"回収率:          {roi:.1f} %")

    # 除外の影響を表示
    if exclude_missing_payout and n_excluded_races > 0:
        total_bet_before = n_races_before * bet_amount
        hit_rate_before = n_hits_before / n_races_before if n_races_before > 0 else 0.0
        roi_before = (total_return_before / total_bet_before * 100) if total_bet_before > 0 else 0.0

        logger.info("")
        logger.info("Impact of exclusion:")
        logger.info(f"  Races:    {n_races_before:,} -> {n_races:,} (delta {n_races - n_races_before:+,})")
        logger.info(f"  Hits:     {n_hits_before:,} -> {n_hits:,} (delta {n_hits - n_hits_before:+,})")
        logger.info(f"  Hit rate: {hit_rate_before:.3f} -> {hit_rate:.3f} (delta {hit_rate - hit_rate_before:+.3f})")
        logger.info(f"  ROI:      {roi_before:.1f}% -> {roi:.1f}% (delta {roi - roi_before:+.1f}%)")

    logger.info("")

    return {
        "n_races": n_races,
        "n_hits": n_hits,
        "hit_rate": hit_rate,
        "total_bet": total_bet,
        "total_return": total_return,
        "roi": roi,
        "n_excluded_races": n_excluded_races if exclude_missing_payout else 0,
        "n_excluded_rows": n_excluded_rows if exclude_missing_payout else 0,
    }


def evaluate_strategy_top1_thresholds(
    df: pd.DataFrame,
    thresholds: List[float],
    bet_amount: int = 100,
    y_pred_col: str = "pred_in3_prob",
    payout_col: str = "fukusho_payout",
    y_true_col: str = "target_in3",
    exclude_missing_payout: bool = True,
) -> List[Dict[str, float]]:
    """
    閾値付き戦略：予測確率が閾値以上の場合のみベット

    Args:
        exclude_missing_payout: True の場合、should_have_payout (JRAルール) なのに payout が NULL のレースを除外
    """
    # paid_places の計算（JRAルール）
    def calculate_paid_places(field_size):
        if field_size >= 8:
            return 3
        elif field_size >= 5:
            return 2
        else:
            return 0

    # 除外対象のレースを事前に特定
    excluded_races = []
    if exclude_missing_payout:
        for race_id, race_df in df.groupby("race_id"):
            if race_df.empty:
                continue

            best_idx = race_df[y_pred_col].idxmax()
            chosen = race_df.loc[best_idx]

            payout = chosen.get(payout_col)

            # JRAルールに基づく除外判定
            field_size = chosen.get("field_size", len(race_df))
            paid_places = calculate_paid_places(field_size)
            finish_order = chosen.get("finish_order", 999)

            # should_have_payout = 払戻対象着順なのに
            should_have_payout = (finish_order <= paid_places) and (paid_places > 0)

            # 真の異常：should_have_payout なのに payout が NULL
            if should_have_payout and pd.isna(payout):
                excluded_races.append(race_id)

    results = []

    for thr in thresholds:
        n_bet_races = 0
        n_hits = 0
        total_return = 0.0
        pred_probs = []

        for race_id, race_df in df.groupby("race_id"):
            if race_df.empty:
                continue

            # 除外対象のレースをスキップ
            if exclude_missing_payout and race_id in excluded_races:
                continue

            best_idx = race_df[y_pred_col].idxmax()
            chosen = race_df.loc[best_idx]

            # 閾値チェック
            if chosen[y_pred_col] < thr:
                continue

            n_bet_races += 1
            pred_probs.append(chosen[y_pred_col])

            payout = chosen.get(payout_col)
            if pd.notna(payout):
                total_return += float(payout)
                if int(chosen[y_true_col]) == 1:
                    n_hits += 1

        total_bet = n_bet_races * bet_amount
        hit_rate = n_hits / n_bet_races if n_bet_races > 0 else 0.0
        roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0
        avg_pred = np.mean(pred_probs) if pred_probs else 0.0

        results.append({
            "threshold": thr,
            "n_bet_races": n_bet_races,
            "n_hits": n_hits,
            "hit_rate": hit_rate,
            "total_bet": total_bet,
            "total_return": total_return,
            "roi": roi,
            "avg_pred_prob": avg_pred,
        })

    logger.info("=" * 60)
    logger.info("Strategy: Top1 with Thresholds (閾値付き戦略)")
    logger.info("=" * 60)

    if exclude_missing_payout and len(excluded_races) > 0:
        logger.info(f"  Excluded races (should have payout but NULL, JRA rules): {len(excluded_races):,} races")
        logger.info("")

    logger.info(f"{'Threshold':>10} {'Bets':>6} {'Hits':>6} {'Hit%':>7} {'ROI%':>8} {'AvgProb':>8}")
    logger.info("-" * 60)

    for r in results:
        logger.info(
            f"{r['threshold']:>10.2f} {r['n_bet_races']:>6} {r['n_hits']:>6} "
            f"{r['hit_rate']:>7.3f} {r['roi']:>8.1f} {r['avg_pred_prob']:>8.3f}"
        )
    logger.info("")

    return results


def evaluate_calibration(
    df: pd.DataFrame,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
    n_bins: int = 10,
) -> Dict:
    """
    キャリブレーション評価：予測確率と実際の的中率の一致度
    """
    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values

    # ビン番号を計算
    bins = np.floor(y_pred * n_bins).clip(0, n_bins - 1).astype(int)

    bin_stats = []
    total_n = len(y_pred)
    ece_sum = 0.0

    for bin_idx in range(n_bins):
        mask = (bins == bin_idx)
        n = mask.sum()

        if n == 0:
            continue

        avg_pred = y_pred[mask].mean()
        emp_rate = y_true[mask].mean()
        diff = avg_pred - emp_rate

        bin_stats.append({
            "bin": bin_idx,
            "n": n,
            "avg_pred": avg_pred,
            "emp_rate": emp_rate,
            "diff": diff,
        })

        ece_sum += (n / total_n) * abs(diff)

    logger.info("=" * 60)
    logger.info("Calibration Evaluation (キャリブレーション評価)")
    logger.info("=" * 60)
    logger.info(f"{'Bin':>4} {'N':>8} {'AvgPred':>10} {'EmpRate':>10} {'Diff':>10}")
    logger.info("-" * 60)

    for stat in bin_stats:
        logger.info(
            f"{stat['bin']:>4} {stat['n']:>8} {stat['avg_pred']:>10.4f} "
            f"{stat['emp_rate']:>10.4f} {stat['diff']:>10.4f}"
        )

    logger.info("-" * 60)
    logger.info(f"ECE (Expected Calibration Error): {ece_sum:.4f}")
    logger.info("")

    return {
        "bins": bin_stats,
        "ece": ece_sum,
    }


def evaluate_debug(df: pd.DataFrame, payout_col: str = "fukusho_payout") -> Dict:
    """
    デバッグ用チェック：DB 側の欠損状況を確認
    """
    n_total = len(df)
    n_missing_payout = df[payout_col].isna().sum()

    # レースごとの複勝払戻件数をチェック
    payout_counts = []
    for race_id, race_df in df.groupby("race_id"):
        n_payout = race_df[payout_col].notna().sum()
        payout_counts.append(n_payout)

    payout_counts = np.array(payout_counts)
    avg_payout_per_race = payout_counts.mean()
    min_payout_per_race = payout_counts.min()
    max_payout_per_race = payout_counts.max()

    logger.info("=" * 60)
    logger.info("Debug Information (デバッグ情報)")
    logger.info("=" * 60)
    logger.info(f"Total rows:                {n_total:,}")
    logger.info(f"Missing fukusho_payout:    {n_missing_payout:,} ({n_missing_payout/n_total*100:.1f}%)")
    logger.info(f"Races:                     {len(payout_counts):,}")
    logger.info(f"Avg payout per race:       {avg_payout_per_race:.1f}")
    logger.info(f"Min payout per race:       {min_payout_per_race}")
    logger.info(f"Max payout per race:       {max_payout_per_race}")

    # 複勝払戻が3件未満のレースを警告
    races_with_few_payouts = (payout_counts < 3).sum()
    if races_with_few_payouts > 0:
        logger.warning(f"Races with < 3 payouts:    {races_with_few_payouts:,}")

    logger.info("")

    return {
        "n_total": n_total,
        "n_missing_payout": n_missing_payout,
        "n_races": len(payout_counts),
        "avg_payout_per_race": avg_payout_per_race,
        "races_with_few_payouts": races_with_few_payouts,
    }


def evaluate(
    df_test: pd.DataFrame,
    model: lgb.Booster,
    X_test: pd.DataFrame,
    exclude_missing_payout: bool = True,
) -> None:
    """
    総合評価関数：各種評価メトリクスを実行

    Args:
        exclude_missing_payout: True の場合、戦略評価で payout 不明レースを除外
    """
    proba = model.predict(X_test)

    df_test = df_test.copy()
    df_test["pred_in3_prob"] = proba

    # 1. グローバル指標
    evaluate_global_metrics(df_test)

    # 2. ランキング評価
    evaluate_ranking(df_test)

    # 3. 戦略A：全レースで top1 買い
    evaluate_strategy_top1_all(df_test, exclude_missing_payout=exclude_missing_payout)

    # 4. 閾値付き戦略
    thresholds = [0.25, 0.30, 0.35, 0.40]
    evaluate_strategy_top1_thresholds(df_test, thresholds=thresholds, exclude_missing_payout=exclude_missing_payout)

    # 5. キャリブレーション評価
    evaluate_calibration(df_test, n_bins=10)

    # 6. デバッグ情報
    evaluate_debug(df_test)


# ========================================
# Main
# ========================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate LightGBM baseline model on feature_table"
    )
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)",
    )
    parser.add_argument(
        "--exclude-missing-payout",
        action="store_true",
        default=True,
        help="Exclude races where target_in3==1 but payout is NULL from strategy evaluation (default: True)",
    )
    parser.add_argument(
        "--no-exclude-missing-payout",
        dest="exclude_missing_payout",
        action="store_false",
        help="Do NOT exclude races with missing payout (opposite of --exclude-missing-payout)",
    )
    parser.add_argument(
        "--debug-features",
        action="store_true",
        help="Output feature column summary and prohibited column checks before training",
    )
    parser.add_argument(
        "--add-rr",
        action="store_true",
        help="Add race-relative (rr_) features computed within each race_id",
    )
    parser.add_argument(
        "--rr-kind",
        choices=["rank_pct", "zscore", "both"],
        default="rank_pct",
        help="Type of rr_ features to add (default: rank_pct)",
    )
    parser.add_argument(
        "--rr-cols",
        type=str,
        help="Comma-separated list of base columns to use for rr_ features (default set will be used if omitted)",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save the trained model (default: models)",
    )
    parser.add_argument(
        "--model-name",
        default="lgbm_in3",
        help="Model name prefix for saved files (default: lgbm_in3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the trained model",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    print(f"[INFO] Database path: {db_path}")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = load_dataset(conn)

        if args.add_rr:
            if args.rr_cols:
                rr_cols = [c.strip() for c in args.rr_cols.split(",") if c.strip()]
            else:
                rr_cols = DEFAULT_RR_COLS

            df, rr_created_cols, rr_base_cols_used = _add_rr_features(df, args.rr_kind, rr_cols)
            if args.debug_features:
                _log_rr_summary(True, args.rr_kind, rr_created_cols, rr_base_cols_used)
        elif args.debug_features:
            _log_rr_summary(False, args.rr_kind, [], [])

        df_train, df_test = split_by_race(df)

        if df_train.empty or df_test.empty:
            raise RuntimeError("train or test dataset is empty; check data availability")

        X_train, y_train, feature_cols_train = build_feature_matrix(df_train, debug_features=args.debug_features)
        X_test, y_test, feature_cols_test = build_feature_matrix(df_test, debug_features=args.debug_features)

        # Ensure feature columns match
        if feature_cols_train != feature_cols_test:
            logger.warning("Feature columns differ between train and test! Using train columns.")
            # Align test features to train
            for col in feature_cols_train:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[feature_cols_train]

        print(f"\n[INFO] Training LightGBM model...")
        print(f"[INFO] Train samples: {len(X_train):,}, features: {len(feature_cols_train)}")
        print(f"[INFO] Test samples:  {len(X_test):,}")

        # Train model (use 2023 data as validation for early stopping)
        df_train_2021_2022 = df_train[df_train["race_year"] <= 2022]
        df_val_2023 = df_train[df_train["race_year"] == 2023]

        if len(df_val_2023) > 0:
            X_train_sub, y_train_sub, _ = build_feature_matrix(df_train_2021_2022, debug_features=False)
            X_val, y_val, _ = build_feature_matrix(df_val_2023, debug_features=False)

            # Align columns
            for col in feature_cols_train:
                if col not in X_train_sub.columns:
                    X_train_sub[col] = 0
                if col not in X_val.columns:
                    X_val[col] = 0
            X_train_sub = X_train_sub[feature_cols_train]
            X_val = X_val[feature_cols_train]

            print(f"[INFO] Using 2023 data as validation set for early stopping")
            print(f"[INFO] Train (2021-2022): {len(X_train_sub):,} samples")
            print(f"[INFO] Valid (2023):      {len(X_val):,} samples")

            model = train_lgbm_model(X_train_sub, y_train_sub, X_val, y_val)
        else:
            model = train_lgbm_model(X_train, y_train)

        print(f"\n[INFO] Model training completed.")
        print(f"[INFO] Best iteration: {model.best_iteration}")

        # Save model
        if not args.no_save:
            save_model(model, feature_cols_train, args.model_dir, args.model_name)

        # Evaluate
        print(f"\n[INFO] Evaluating on test set (2024)...")
        evaluate(df_test, model, X_test, exclude_missing_payout=args.exclude_missing_payout)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
