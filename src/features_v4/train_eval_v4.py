# -*- coding: utf-8 -*-
"""
train_eval_v4.py - Train/Eval/ROI Pipeline for FeaturePack v1

LightGBM モデルの学習・評価・ROI 分析パイプライン。

【主要機能】
1. 時系列分割 (train/val/test)
2. LightGBM 学習
3. 評価メトリクス (AUC, LogLoss, Calibration)
4. ROI 分析 (バックテスト)
5. 特徴量重要度

【時系列分割モード】
1. year_based (デフォルト):
   - train: 2021-01 ~ 2023-12
   - val:   2023-10 ~ 2023-12 (train末尾を検証に流用)
   - test:  2024-01 ~ 現在

2. date_based:
   - train_end, val_end を明示指定
   - 例: train_end="2023-06-30", val_end="2023-12-31"

【リーク防止】
- 時系列分割を厳守 (未来データで学習しない)
- market_* 特徴量は学習時に除外可能
"""

import logging
import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .feature_table_v4 import get_feature_v4_columns

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainConfig:
    """学習設定"""
    target_col: str = "target_win"
    include_pedigree: bool = True
    include_market: bool = False
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = 8
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    lambda_l2: float = 0.1
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    min_child_samples: int = 20


@dataclass
class EvalResult:
    """評価結果"""
    dataset: str  # "val" or "test"
    auc: float
    logloss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    n_positive: int


@dataclass
class ROIResult:
    """ROI 分析結果"""
    n_bets: int
    n_wins: int
    hit_rate: float
    total_bet: float
    total_return: float
    roi: float
    avg_odds: float
    max_odds: float


# =============================================================================
# Data Loading
# =============================================================================

def load_feature_data(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_pedigree: bool = True,
    include_market: bool = False,
) -> pd.DataFrame:
    """
    feature_table_v4 からデータをロード

    Args:
        conn: SQLite 接続
        start_date: 開始日 (YYYY-MM-DD)
        end_date: 終了日 (YYYY-MM-DD)
        include_pedigree: 血統ハッシュを含めるか
        include_market: 市場情報を含めるか

    Returns:
        DataFrame
    """
    sql = "SELECT * FROM feature_table_v4"
    params = []

    if start_date or end_date:
        conditions = []
        if start_date:
            conditions.append("race_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("race_date <= ?")
            params.append(end_date)
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY race_date, race_id"

    df = pd.read_sql_query(sql, conn, params=params)

    # 不要なカラムを除外
    feature_cols = get_feature_v4_columns(
        include_pedigree=include_pedigree,
        include_market=include_market,
    )

    # identity + targets + features を残す
    keep_cols = ["race_id", "horse_id", "race_date",
                 "target_win", "target_in3", "target_quinella"]
    keep_cols.extend(feature_cols)
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


def split_time_series(
    df: pd.DataFrame,
    train_end: str = "2023-12-31",
    val_end: str = "2023-12-31",
    split_mode: str = "year_based",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    時系列分割

    Args:
        df: 全データ
        train_end: 学習データの終了日
        val_end: 検証データの終了日
        split_mode: 分割モード
            - "year_based": train=2021-2023, val=2023後半, test=2024
            - "date_based": train_end, val_end を明示指定

    Returns:
        (train_df, val_df, test_df)
    """
    if split_mode == "year_based":
        # Year-based split: train=2021-2023, test=2024
        # val は train の末尾 (2023-10-01~2023-12-31) を流用
        train_end = "2023-12-31"
        val_start = "2023-10-01"
        test_start = "2024-01-01"

        train_df = df[df["race_date"] <= train_end].copy()
        val_df = df[(df["race_date"] >= val_start) & (df["race_date"] <= train_end)].copy()
        test_df = df[df["race_date"] >= test_start].copy()

        logger.info("Time series split (year_based):")
        logger.info("  Train: %d rows (~ %s)", len(train_df), train_end)
        logger.info("  Val:   %d rows (%s ~ %s, subset of train)", len(val_df), val_start, train_end)
        logger.info("  Test:  %d rows (%s ~)", len(test_df), test_start)
    else:
        # Date-based split (original behavior)
        train_df = df[df["race_date"] <= train_end].copy()
        val_df = df[(df["race_date"] > train_end) & (df["race_date"] <= val_end)].copy()
        test_df = df[df["race_date"] > val_end].copy()

        logger.info("Time series split (date_based):")
        logger.info("  Train: %d rows (<= %s)", len(train_df), train_end)
        logger.info("  Val:   %d rows (%s ~ %s)", len(val_df), train_end, val_end)
        logger.info("  Test:  %d rows (> %s)", len(test_df), val_end)

    return train_df, val_df, test_df


# =============================================================================
# Feature Columns
# =============================================================================

def get_train_feature_columns(
    df: pd.DataFrame,
    include_pedigree: bool = True,
    include_market: bool = False,
) -> List[str]:
    """
    学習に使用する特徴量カラムを取得

    除外されるカラム:
    - race_id, horse_id, race_date (identity)
    - target_* (targets)
    - place, surface, track_condition, etc. (text columns)
    """
    # 定義済み特徴量カラムを取得
    feature_cols = get_feature_v4_columns(
        include_pedigree=include_pedigree,
        include_market=include_market,
    )

    # 実際に存在するカラムのみ
    available_cols = [c for c in feature_cols if c in df.columns]

    # object 型を除外
    for c in available_cols[:]:
        if df[c].dtype == "object":
            available_cols.remove(c)

    return available_cols


# =============================================================================
# Training
# =============================================================================

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: TrainConfig,
    output_dir: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """
    LightGBM モデルを学習

    Args:
        train_df: 学習データ
        val_df: 検証データ
        config: 学習設定
        output_dir: モデル保存先 (None の場合は保存しない)

    Returns:
        (model, feature_cols)
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed")

    target_col = config.target_col
    feature_cols = get_train_feature_columns(
        train_df,
        include_pedigree=config.include_pedigree,
        include_market=config.include_market,
    )

    logger.info("Training model for target: %s", target_col)
    logger.info("  Features: %d columns", len(feature_cols))

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # 欠損値を処理
    X_train = X_train.fillna(-999)
    X_val = X_val.fillna(-999)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_child_samples": config.min_child_samples,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": 5,
        "lambda_l2": config.lambda_l2,
        "verbose": -1,
        "force_col_wise": True,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    logger.info("Training completed. Best iteration: %d", model.best_iteration)

    # モデル保存
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # モデル保存
        model_path = out_path / f"lgbm_{target_col}_v4.txt"
        model.save_model(str(model_path))
        logger.info("Saved model to %s", model_path)

        # 特徴量リスト保存
        cols_path = out_path / f"feature_columns_{target_col}_v4.json"
        with open(cols_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        logger.info("Saved feature columns to %s", cols_path)

        # 特徴量重要度保存
        save_feature_importance(model, feature_cols, output_dir, target_col)

    return model, feature_cols


def save_feature_importance(
    model: Any,
    feature_cols: List[str],
    output_dir: str,
    target_col: str,
) -> None:
    """特徴量重要度を保存"""
    importance_gain = model.feature_importance(importance_type="gain")
    importance_split = model.feature_importance(importance_type="split")

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importance_gain,
        "importance_split": importance_split,
    })
    df_imp = df_imp.sort_values("importance_gain", ascending=False)

    out_path = Path(output_dir) / f"feature_importance_{target_col}_v4.csv"
    df_imp.to_csv(out_path, index=False)
    logger.info("Saved feature importance to %s", out_path)

    # Top 20 を表示
    logger.info("Top 20 important features (gain):")
    for i, row in df_imp.head(20).iterrows():
        logger.info("  %s: %.2f", row["feature"], row["importance_gain"])


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    dataset_name: str = "test",
) -> EvalResult:
    """
    モデルを評価

    Args:
        model: 学習済みモデル
        df: 評価データ
        feature_cols: 特徴量カラム
        target_col: ターゲットカラム
        dataset_name: データセット名 ("val" or "test")

    Returns:
        EvalResult
    """
    X = df[feature_cols].fillna(-999)
    y = df[target_col]

    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # メトリクス計算
    try:
        auc = roc_auc_score(y, y_pred_proba)
    except:
        auc = float("nan")

    try:
        logloss = log_loss(y, y_pred_proba)
    except:
        logloss = float("nan")

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    result = EvalResult(
        dataset=dataset_name,
        auc=auc,
        logloss=logloss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        n_samples=len(y),
        n_positive=int(y.sum()),
    )

    logger.info("[%s] Evaluation Results:", dataset_name.upper())
    logger.info("  AUC:       %.4f", result.auc)
    logger.info("  LogLoss:   %.4f", result.logloss)
    logger.info("  Accuracy:  %.4f", result.accuracy)
    logger.info("  Precision: %.4f", result.precision)
    logger.info("  Recall:    %.4f", result.recall)
    logger.info("  F1:        %.4f", result.f1)
    logger.info("  Samples:   %d (positive: %d)", result.n_samples, result.n_positive)

    return result


# =============================================================================
# ROI Analysis (Backtest)
# =============================================================================

def analyze_roi(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_win",
    strategy: str = "top_pick",
    bet_amount: float = 100.0,
) -> ROIResult:
    """
    ROI 分析 (バックテスト)

    Args:
        model: 学習済みモデル
        df: 評価データ (market_win_odds が必要)
        feature_cols: 特徴量カラム
        target_col: ターゲットカラム
        strategy: 戦略 ("top_pick" = 各レースで予測最上位に賭ける)
        bet_amount: 1回の賭け金

    Returns:
        ROIResult
    """
    X = df[feature_cols].fillna(-999)
    y = df[target_col]

    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)

    # 予測確率を追加
    result_df = df[["race_id", "horse_id", target_col]].copy()
    result_df["pred_proba"] = y_pred_proba

    # オッズがあれば取得
    if "market_win_odds" in df.columns:
        result_df["odds"] = df["market_win_odds"]
    else:
        # オッズがない場合はシミュレーション不可
        logger.warning("market_win_odds not found, ROI analysis skipped")
        return ROIResult(
            n_bets=0, n_wins=0, hit_rate=0.0,
            total_bet=0.0, total_return=0.0, roi=0.0,
            avg_odds=0.0, max_odds=0.0,
        )

    # 戦略: 各レースで予測最上位に賭ける
    if strategy == "top_pick":
        bets = []
        for race_id in result_df["race_id"].unique():
            race_df = result_df[result_df["race_id"] == race_id]
            if len(race_df) == 0:
                continue

            # 予測確率最大の馬を選択
            best_idx = race_df["pred_proba"].idxmax()
            best_row = race_df.loc[best_idx]

            bets.append({
                "race_id": race_id,
                "horse_id": best_row["horse_id"],
                "pred_proba": best_row["pred_proba"],
                "odds": best_row["odds"],
                "is_win": best_row[target_col] == 1,
            })

        bets_df = pd.DataFrame(bets)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if len(bets_df) == 0:
        return ROIResult(
            n_bets=0, n_wins=0, hit_rate=0.0,
            total_bet=0.0, total_return=0.0, roi=0.0,
            avg_odds=0.0, max_odds=0.0,
        )

    # 集計
    n_bets = len(bets_df)
    n_wins = bets_df["is_win"].sum()
    hit_rate = n_wins / n_bets if n_bets > 0 else 0.0
    total_bet = n_bets * bet_amount

    # 的中時のリターン
    wins_df = bets_df[bets_df["is_win"]]
    total_return = (wins_df["odds"] * bet_amount).sum() if len(wins_df) > 0 else 0.0

    roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0.0

    avg_odds = wins_df["odds"].mean() if len(wins_df) > 0 else 0.0
    max_odds = wins_df["odds"].max() if len(wins_df) > 0 else 0.0

    result = ROIResult(
        n_bets=n_bets,
        n_wins=int(n_wins),
        hit_rate=hit_rate,
        total_bet=total_bet,
        total_return=total_return,
        roi=roi,
        avg_odds=avg_odds,
        max_odds=max_odds,
    )

    logger.info("ROI Analysis (%s):", strategy)
    logger.info("  Bets:        %d", result.n_bets)
    logger.info("  Wins:        %d", result.n_wins)
    logger.info("  Hit Rate:    %.2f%%", result.hit_rate * 100)
    logger.info("  Total Bet:   %.0f", result.total_bet)
    logger.info("  Total Return:%.0f", result.total_return)
    logger.info("  ROI:         %.2f%%", result.roi * 100)
    logger.info("  Avg Odds:    %.2f", result.avg_odds)

    return result


# =============================================================================
# Full Pipeline
# =============================================================================

def run_full_pipeline(
    conn: sqlite3.Connection,
    config: Optional[TrainConfig] = None,
    output_dir: str = "models",
    train_end: str = "2023-12-31",
    val_end: str = "2023-12-31",
    split_mode: str = "year_based",
) -> Dict[str, Any]:
    """
    完全なパイプラインを実行

    Args:
        conn: SQLite 接続
        config: 学習設定
        output_dir: 出力ディレクトリ
        train_end: 学習データ終了日 (date_based モード用)
        val_end: 検証データ終了日 (date_based モード用)
        split_mode: 分割モード ("year_based" or "date_based")

    Returns:
        結果の辞書
    """
    if config is None:
        config = TrainConfig()

    logger.info("=" * 80)
    logger.info("Running Full Train/Eval/ROI Pipeline")
    logger.info("  Split mode: %s", split_mode)
    logger.info("=" * 80)

    # データロード
    logger.info("Loading data...")
    df = load_feature_data(
        conn,
        include_pedigree=config.include_pedigree,
        include_market=config.include_market,
    )
    logger.info("Loaded %d rows", len(df))

    # 時系列分割
    train_df, val_df, test_df = split_time_series(df, train_end, val_end, split_mode)

    # 学習
    model, feature_cols = train_model(train_df, val_df, config, output_dir)

    # 評価
    val_result = evaluate_model(model, val_df, feature_cols, config.target_col, "val")
    test_result = evaluate_model(model, test_df, feature_cols, config.target_col, "test")

    # ROI 分析 (market_win_odds があれば)
    roi_result = None
    if config.include_market and "market_win_odds" in test_df.columns:
        roi_result = analyze_roi(model, test_df, feature_cols, config.target_col)

    # 結果をまとめる
    results = {
        "config": asdict(config),
        "val_result": asdict(val_result),
        "test_result": asdict(test_result),
        "roi_result": asdict(roi_result) if roi_result else None,
        "n_features": len(feature_cols),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

    # 結果を保存
    if output_dir:
        results_path = Path(output_dir) / f"results_{config.target_col}_v4.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Saved results to %s", results_path)

    logger.info("=" * 80)
    logger.info("Pipeline completed")
    logger.info("=" * 80)

    return results


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

    parser = argparse.ArgumentParser(description="Train/Eval/ROI Pipeline for FeaturePack v1")
    parser.add_argument("--db", default="netkeiba.db", help="Database path")
    parser.add_argument("--output", "-o", default="models", help="Output directory")
    parser.add_argument("--target", default="target_win", help="Target column")
    parser.add_argument(
        "--split-mode",
        choices=["year_based", "date_based"],
        default="year_based",
        help="Split mode: year_based (train=2021-2023, test=2024) or date_based (use --train-end/--val-end)"
    )
    parser.add_argument("--train-end", default="2023-12-31", help="Train end date (for date_based mode)")
    parser.add_argument("--val-end", default="2023-12-31", help="Validation end date (for date_based mode)")
    parser.add_argument("--no-pedigree", action="store_true", help="Exclude pedigree features")
    parser.add_argument("--include-market", action="store_true", help="Include market features")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--num-leaves", type=int, default=63, help="Number of leaves")

    args = parser.parse_args()

    if not Path(args.db).exists():
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    config = TrainConfig(
        target_col=args.target,
        include_pedigree=not args.no_pedigree,
        include_market=args.include_market,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
    )

    conn = sqlite3.connect(args.db)
    try:
        results = run_full_pipeline(
            conn=conn,
            config=config,
            output_dir=args.output,
            train_end=args.train_end,
            val_end=args.val_end,
            split_mode=args.split_mode,
        )
        print(json.dumps(results, indent=2))
    finally:
        conn.close()
