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


@dataclass
class RankingResult:
    """
    レース単位のランキング評価結果

    競馬予測では accuracy は参考にならない（全頭「負け」予測で高accuracyになる）。
    代わりに「各レースで勝ち馬を上位に予測できているか」を評価する。
    """
    dataset: str  # "val" or "test"
    n_races: int  # 評価対象レース数
    n_entries: int  # 評価対象出走数

    # Top-K Hit Rate
    top1_hit_rate: float  # 予測1位が勝ち馬 の割合
    top3_hit_rate: float  # 勝ち馬が予測3位以内 の割合
    top5_hit_rate: float  # 勝ち馬が予測5位以内 の割合

    # MRR (Mean Reciprocal Rank)
    mrr: float  # 1/rank の平均（勝ち馬の予測順位の逆数平均）

    # 勝ち馬の予測順位分布
    winner_rank_histogram: Dict[str, int]  # "1": N, "2": M, ..., "10+": K

    # 平均フィールドサイズ
    avg_field_size: float


@dataclass
class ROIStrategyResult:
    """
    単一戦略のROI評価結果

    単勝 or 複勝 × 戦略ごとの結果を格納。
    """
    strategy: str  # "ModelTop1", "ModelTop3Equal", "PopularityTop1", "RandomTop1"
    bet_type: str  # "単勝" or "複勝"
    n_races: int  # 評価対象レース数
    n_bets: int  # 賭けた回数（Top3なら n_races * 3）
    stake_total_yen: float  # 総賭け金（円）
    return_total_yen: float  # 総払戻（円）
    roi: float  # return / stake
    hit_rate: float  # 的中率（単勝なら勝利率、複勝なら3着内率）
    n_hits: int  # 的中回数
    avg_payout: float  # 的中時の平均払戻（円、100円当たり）


@dataclass
class ROIEvalResult:
    """
    ROI評価の全体結果

    val/test ごとに、複数戦略 × 単勝/複勝 の結果をまとめる。
    """
    dataset: str  # "val" or "test"
    strategies: List[ROIStrategyResult]


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

    df = df[keep_cols]

    # ========================================================================
    # Guard: target カラムの NaN/異常値を除外（保険）
    # feature_builder_v4 で既に除外しているはずだが、二重防御
    # ========================================================================
    target_cols = ["target_win", "target_in3", "target_quinella"]
    for target_col in target_cols:
        if target_col in df.columns:
            # NaN を除外
            n_nan = df[target_col].isna().sum()
            if n_nan > 0:
                logger.warning(
                    "Found %d NaN values in %s, filtering out (this should not happen)",
                    n_nan, target_col
                )
                df = df[df[target_col].notna()]

            # 0/1 以外の値を除外
            valid_mask = df[target_col].isin([0, 1, 0.0, 1.0])
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                logger.warning(
                    "Found %d non-binary values in %s, filtering out",
                    n_invalid, target_col
                )
                df = df[valid_mask]

    return df


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

    # ========================================================================
    # Guard: y_train / y_val の NaN チェック（保険）
    # ========================================================================
    for name, y in [("y_train", y_train), ("y_val", y_val)]:
        n_nan = y.isna().sum()
        if n_nan > 0:
            raise ValueError(
                f"{name} contains {n_nan} NaN values in {target_col}. "
                "This should have been filtered in load_feature_data(). "
                "Check feature_table_v4 for invalid records."
            )
        # 0/1 のみであることを確認
        unique_vals = set(y.unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"{name} contains non-binary values: {unique_vals}. "
                f"Expected only 0 and 1 in {target_col}."
            )

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
    # ========================================================================
    # Guard: target の NaN/異常値チェック（保険）
    # ========================================================================
    y_raw = df[target_col]

    # NaN チェック
    n_nan = y_raw.isna().sum()
    if n_nan > 0:
        logger.error(
            "[%s] Found %d NaN values in %s - this should have been filtered earlier!",
            dataset_name, n_nan, target_col
        )
        # NaN を除外して続行
        valid_idx = y_raw.notna()
        df = df[valid_idx].copy()
        logger.warning("[%s] Filtered out %d NaN rows, continuing with %d rows",
                      dataset_name, n_nan, len(df))

    # 0/1 以外の値チェック
    y = df[target_col]
    invalid_mask = ~y.isin([0, 1, 0.0, 1.0])
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        logger.error(
            "[%s] Found %d non-binary values in %s: %s",
            dataset_name, n_invalid, target_col, y[invalid_mask].unique()[:5]
        )
        # 無効な値を除外
        df = df[~invalid_mask].copy()
        y = df[target_col]

    if len(df) == 0:
        logger.error("[%s] No valid samples after filtering", dataset_name)
        return EvalResult(
            dataset=dataset_name,
            auc=float("nan"),
            logloss=float("nan"),
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            n_samples=0,
            n_positive=0,
        )

    X = df[feature_cols].fillna(-999)

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
# Ranking Evaluation (Per-Race Metrics)
# =============================================================================

def evaluate_ranking(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_win",
    dataset_name: str = "test",
) -> RankingResult:
    """
    レース単位のランキング評価

    競馬予測では accuracy は参考にならない（全頭「負け」予測で高accuracyになる）。
    代わりに「各レースで勝ち馬を上位に予測できているか」を評価する。

    Args:
        model: 学習済みモデル
        df: 評価データ (race_id, target_col が必要)
        feature_cols: 特徴量カラム
        target_col: ターゲットカラム (通常 "target_win")
        dataset_name: データセット名 ("val" or "test")

    Returns:
        RankingResult
    """
    if len(df) == 0:
        logger.warning("[%s] No data for ranking evaluation", dataset_name)
        return RankingResult(
            dataset=dataset_name,
            n_races=0,
            n_entries=0,
            top1_hit_rate=0.0,
            top3_hit_rate=0.0,
            top5_hit_rate=0.0,
            mrr=0.0,
            winner_rank_histogram={},
            avg_field_size=0.0,
        )

    # 予測確率を計算
    X = df[feature_cols].fillna(-999)
    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)

    # 作業用 DataFrame
    result_df = df[["race_id", "horse_id", target_col]].copy()
    result_df["pred_proba"] = y_pred_proba

    # レースごとに評価
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks = []
    winner_ranks = []  # 勝ち馬の予測順位
    field_sizes = []

    for race_id in result_df["race_id"].unique():
        race_df = result_df[result_df["race_id"] == race_id].copy()
        n_horses = len(race_df)
        field_sizes.append(n_horses)

        # 勝ち馬がいるか確認
        winners = race_df[race_df[target_col] == 1]
        if len(winners) == 0:
            # 勝ち馬がいないレースはスキップ（全馬失格など）
            continue

        # 予測確率で降順ランク (1 = 最高予測)
        race_df = race_df.sort_values("pred_proba", ascending=False)
        race_df["pred_rank"] = range(1, n_horses + 1)

        # 勝ち馬の予測順位
        winner_row = race_df[race_df[target_col] == 1].iloc[0]
        winner_rank = winner_row["pred_rank"]
        winner_ranks.append(int(winner_rank))

        # Top-K Hit
        if winner_rank == 1:
            top1_hits += 1
        if winner_rank <= 3:
            top3_hits += 1
        if winner_rank <= 5:
            top5_hits += 1

        # MRR
        reciprocal_ranks.append(1.0 / winner_rank)

    n_races = len(reciprocal_ranks)  # 勝ち馬がいたレース数

    if n_races == 0:
        logger.warning("[%s] No races with winners found", dataset_name)
        return RankingResult(
            dataset=dataset_name,
            n_races=0,
            n_entries=len(df),
            top1_hit_rate=0.0,
            top3_hit_rate=0.0,
            top5_hit_rate=0.0,
            mrr=0.0,
            winner_rank_histogram={},
            avg_field_size=np.mean(field_sizes) if field_sizes else 0.0,
        )

    # メトリクス計算
    top1_hit_rate = top1_hits / n_races
    top3_hit_rate = top3_hits / n_races
    top5_hit_rate = top5_hits / n_races
    mrr = np.mean(reciprocal_ranks)

    # 勝ち馬順位のヒストグラム
    # 1..9 は個別、10以上は "10+" にまとめる
    histogram: Dict[str, int] = {}
    for rank in winner_ranks:
        if rank <= 9:
            key = str(rank)
        else:
            key = "10+"
        histogram[key] = histogram.get(key, 0) + 1

    # ソートして見やすく
    sorted_histogram = {}
    for key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]:
        if key in histogram:
            sorted_histogram[key] = histogram[key]

    result = RankingResult(
        dataset=dataset_name,
        n_races=n_races,
        n_entries=len(df),
        top1_hit_rate=top1_hit_rate,
        top3_hit_rate=top3_hit_rate,
        top5_hit_rate=top5_hit_rate,
        mrr=mrr,
        winner_rank_histogram=sorted_histogram,
        avg_field_size=np.mean(field_sizes) if field_sizes else 0.0,
    )

    logger.info("[%s] Ranking Evaluation Results:", dataset_name.upper())
    logger.info("  Races:         %d", result.n_races)
    logger.info("  Entries:       %d", result.n_entries)
    logger.info("  Top1 Hit Rate: %.2f%% (%d/%d)", result.top1_hit_rate * 100, top1_hits, n_races)
    logger.info("  Top3 Hit Rate: %.2f%% (%d/%d)", result.top3_hit_rate * 100, top3_hits, n_races)
    logger.info("  Top5 Hit Rate: %.2f%% (%d/%d)", result.top5_hit_rate * 100, top5_hits, n_races)
    logger.info("  MRR:           %.4f", result.mrr)
    logger.info("  Avg Field:     %.1f horses", result.avg_field_size)
    logger.info("  Winner Rank Distribution:")
    for rank_key, count in sorted_histogram.items():
        pct = count / n_races * 100
        bar = "█" * int(pct / 5)  # 5%ごとに1ブロック
        logger.info("    Rank %3s: %4d (%5.1f%%) %s", rank_key, count, pct, bar)

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
# Payout-based ROI Evaluation (単勝・複勝)
# =============================================================================

def load_payouts_for_races(
    conn: sqlite3.Connection,
    race_ids: List[str],
    bet_types: List[str] = None,
) -> pd.DataFrame:
    """
    payouts テーブルから指定レースの払戻データをロード

    Args:
        conn: SQLite 接続
        race_ids: レースID リスト
        bet_types: 賭け種別 リスト (デフォルト: ["単勝", "複勝"])

    Returns:
        DataFrame (race_id, bet_type, combination, payout)
    """
    if bet_types is None:
        bet_types = ["単勝", "複勝"]

    if not race_ids:
        return pd.DataFrame(columns=["race_id", "bet_type", "combination", "payout"])

    placeholders = ",".join(["?" for _ in race_ids])
    bet_placeholders = ",".join(["?" for _ in bet_types])

    sql = f"""
        SELECT race_id, bet_type, combination, payout
        FROM payouts
        WHERE race_id IN ({placeholders})
          AND bet_type IN ({bet_placeholders})
    """

    try:
        df = pd.read_sql_query(sql, conn, params=list(race_ids) + bet_types)
        # combination を文字列に統一
        df["combination"] = df["combination"].astype(str).str.strip()
        return df
    except Exception as e:
        logger.warning("Failed to load payouts: %s", e)
        return pd.DataFrame(columns=["race_id", "bet_type", "combination", "payout"])


def load_race_results_for_roi(
    conn: sqlite3.Connection,
    race_ids: List[str],
) -> pd.DataFrame:
    """
    race_results から ROI 評価に必要なデータをロード

    Args:
        conn: SQLite 接続
        race_ids: レースID リスト

    Returns:
        DataFrame (race_id, horse_id, horse_no, finish_order, popularity)
    """
    if not race_ids:
        return pd.DataFrame(columns=["race_id", "horse_id", "horse_no", "finish_order", "popularity"])

    placeholders = ",".join(["?" for _ in race_ids])

    sql = f"""
        SELECT race_id, horse_id, horse_no, finish_order, popularity
        FROM race_results
        WHERE race_id IN ({placeholders})
          AND finish_order IS NOT NULL
          AND horse_no IS NOT NULL
    """

    try:
        df = pd.read_sql_query(sql, conn, params=list(race_ids))
        df["horse_no"] = df["horse_no"].astype(int)
        df["horse_no_str"] = df["horse_no"].astype(str)
        return df
    except Exception as e:
        logger.warning("Failed to load race_results for ROI: %s", e)
        return pd.DataFrame(columns=["race_id", "horse_id", "horse_no", "finish_order", "popularity"])


def evaluate_roi_strategy(
    bets_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    race_results_df: pd.DataFrame,
    bet_type: str,
    strategy_name: str,
    bet_amount: float = 100.0,
) -> ROIStrategyResult:
    """
    単一戦略のROI評価

    Args:
        bets_df: 賭けデータ (race_id, horse_id が必要)
        payouts_df: 払戻データ
        race_results_df: レース結果データ
        bet_type: "単勝" or "複勝"
        strategy_name: 戦略名
        bet_amount: 1賭け当たりの金額（円）

    Returns:
        ROIStrategyResult
    """
    if len(bets_df) == 0:
        return ROIStrategyResult(
            strategy=strategy_name,
            bet_type=bet_type,
            n_races=0,
            n_bets=0,
            stake_total_yen=0.0,
            return_total_yen=0.0,
            roi=0.0,
            hit_rate=0.0,
            n_hits=0,
            avg_payout=0.0,
        )

    # bets_df に horse_no を追加
    bets_with_info = bets_df.merge(
        race_results_df[["race_id", "horse_id", "horse_no_str", "finish_order"]],
        on=["race_id", "horse_id"],
        how="left",
    )

    # 該当する払戻をマージ
    payouts_bet_type = payouts_df[payouts_df["bet_type"] == bet_type].copy()
    bets_with_payout = bets_with_info.merge(
        payouts_bet_type,
        left_on=["race_id", "horse_no_str"],
        right_on=["race_id", "combination"],
        how="left",
    )

    n_races = bets_df["race_id"].nunique()
    n_bets = len(bets_with_payout)
    stake_total = n_bets * bet_amount

    # 的中判定
    if bet_type == "単勝":
        # 単勝: finish_order == 1 かつ payout が存在
        bets_with_payout["is_hit"] = (
            (bets_with_payout["finish_order"] == 1) &
            (bets_with_payout["payout"].notna())
        )
    else:  # 複勝
        # 複勝: finish_order <= 3 かつ payout が存在
        bets_with_payout["is_hit"] = (
            (bets_with_payout["finish_order"] <= 3) &
            (bets_with_payout["payout"].notna())
        )

    hits_df = bets_with_payout[bets_with_payout["is_hit"]]
    n_hits = len(hits_df)
    hit_rate = n_hits / n_bets if n_bets > 0 else 0.0

    # 払戻計算
    if n_hits > 0:
        return_total = (hits_df["payout"] / 100 * bet_amount).sum()
        avg_payout = hits_df["payout"].mean()
    else:
        return_total = 0.0
        avg_payout = 0.0

    roi = return_total / stake_total if stake_total > 0 else 0.0

    return ROIStrategyResult(
        strategy=strategy_name,
        bet_type=bet_type,
        n_races=n_races,
        n_bets=n_bets,
        stake_total_yen=stake_total,
        return_total_yen=return_total,
        roi=roi,
        hit_rate=hit_rate,
        n_hits=n_hits,
        avg_payout=avg_payout,
    )


def generate_model_top1_bets(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    ModelTop1 戦略: 各レースで予測確率最大の1頭を選択

    Args:
        df: 予測対象データ (race_id, horse_id, features)
        model: 学習済みモデル
        feature_cols: 特徴量カラム

    Returns:
        賭け対象 DataFrame (race_id, horse_id)
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["race_id", "horse_id"])

    X = df[feature_cols].fillna(-999)
    pred_proba = model.predict(X, num_iteration=model.best_iteration)

    work_df = df[["race_id", "horse_id"]].copy()
    work_df["pred_proba"] = pred_proba

    # 各レースで pred_proba 最大の馬を選択
    idx = work_df.groupby("race_id")["pred_proba"].idxmax()
    bets_df = work_df.loc[idx, ["race_id", "horse_id"]].reset_index(drop=True)

    return bets_df


def generate_model_top3_bets(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    ModelTop3Equal 戦略: 各レースで予測確率上位3頭を選択

    Args:
        df: 予測対象データ
        model: 学習済みモデル
        feature_cols: 特徴量カラム

    Returns:
        賭け対象 DataFrame (race_id, horse_id)
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["race_id", "horse_id"])

    X = df[feature_cols].fillna(-999)
    pred_proba = model.predict(X, num_iteration=model.best_iteration)

    work_df = df[["race_id", "horse_id"]].copy()
    work_df["pred_proba"] = pred_proba

    # 各レースで上位3頭を選択
    bets_list = []
    for race_id in work_df["race_id"].unique():
        race_df = work_df[work_df["race_id"] == race_id]
        top3 = race_df.nlargest(min(3, len(race_df)), "pred_proba")
        bets_list.append(top3[["race_id", "horse_id"]])

    if bets_list:
        bets_df = pd.concat(bets_list, ignore_index=True)
    else:
        bets_df = pd.DataFrame(columns=["race_id", "horse_id"])

    return bets_df


def generate_popularity_top1_bets(
    race_results_df: pd.DataFrame,
    race_ids: List[str],
) -> pd.DataFrame:
    """
    PopularityTop1 戦略: 各レースで人気1位（popularity=1）を選択

    Args:
        race_results_df: レース結果データ
        race_ids: 対象レースID リスト

    Returns:
        賭け対象 DataFrame (race_id, horse_id)
    """
    if len(race_results_df) == 0:
        return pd.DataFrame(columns=["race_id", "horse_id"])

    # 対象レースのみ
    df = race_results_df[race_results_df["race_id"].isin(race_ids)].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=["race_id", "horse_id"])

    # popularity が最小（1位）の馬を選択
    idx = df.groupby("race_id")["popularity"].idxmin()
    bets_df = df.loc[idx, ["race_id", "horse_id"]].reset_index(drop=True)

    return bets_df


def generate_random_top1_bets(
    df: pd.DataFrame,
    race_ids: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    """
    RandomTop1 戦略: 各レースでランダムに1頭を選択

    Args:
        df: データ (race_id, horse_id)
        race_ids: 対象レースID リスト
        seed: 乱数シード

    Returns:
        賭け対象 DataFrame (race_id, horse_id)
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["race_id", "horse_id"])

    np.random.seed(seed)

    work_df = df[df["race_id"].isin(race_ids)][["race_id", "horse_id"]].copy()

    bets_list = []
    for race_id in work_df["race_id"].unique():
        race_df = work_df[work_df["race_id"] == race_id]
        if len(race_df) > 0:
            chosen = race_df.sample(n=1, random_state=seed)
            bets_list.append(chosen)

    if bets_list:
        bets_df = pd.concat(bets_list, ignore_index=True)
    else:
        bets_df = pd.DataFrame(columns=["race_id", "horse_id"])

    return bets_df


def evaluate_roi_all_strategies(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    dataset_name: str = "test",
    bet_amount: float = 100.0,
) -> ROIEvalResult:
    """
    全戦略のROI評価を実行

    Args:
        conn: SQLite 接続
        df: 評価対象データ (race_id, horse_id, features)
        model: 学習済みモデル
        feature_cols: 特徴量カラム
        dataset_name: データセット名 ("val" or "test")
        bet_amount: 1賭け当たりの金額（円）

    Returns:
        ROIEvalResult
    """
    race_ids = df["race_id"].unique().tolist()

    if len(race_ids) == 0:
        logger.warning("[%s] No races for ROI evaluation", dataset_name)
        return ROIEvalResult(dataset=dataset_name, strategies=[])

    # 払戻データと結果データをロード
    payouts_df = load_payouts_for_races(conn, race_ids)
    race_results_df = load_race_results_for_roi(conn, race_ids)

    if len(payouts_df) == 0:
        logger.warning("[%s] No payout data found for %d races", dataset_name, len(race_ids))
        return ROIEvalResult(dataset=dataset_name, strategies=[])

    logger.info("[%s] ROI Evaluation: %d races, %d payout records",
                dataset_name.upper(), len(race_ids), len(payouts_df))

    strategies_results = []

    # 戦略ごとに賭け対象を生成
    bets_model_top1 = generate_model_top1_bets(df, model, feature_cols)
    bets_model_top3 = generate_model_top3_bets(df, model, feature_cols)
    bets_pop_top1 = generate_popularity_top1_bets(race_results_df, race_ids)
    bets_random = generate_random_top1_bets(df, race_ids)

    bet_types = ["単勝", "複勝"]

    for bet_type in bet_types:
        # ModelTop1
        result = evaluate_roi_strategy(
            bets_model_top1, payouts_df, race_results_df,
            bet_type, "ModelTop1", bet_amount
        )
        strategies_results.append(result)

        # ModelTop3Equal
        result = evaluate_roi_strategy(
            bets_model_top3, payouts_df, race_results_df,
            bet_type, "ModelTop3Equal", bet_amount
        )
        strategies_results.append(result)

        # PopularityTop1
        result = evaluate_roi_strategy(
            bets_pop_top1, payouts_df, race_results_df,
            bet_type, "PopularityTop1", bet_amount
        )
        strategies_results.append(result)

        # RandomTop1
        result = evaluate_roi_strategy(
            bets_random, payouts_df, race_results_df,
            bet_type, "RandomTop1", bet_amount
        )
        strategies_results.append(result)

    # ログ出力
    logger.info("[%s] ROI Evaluation Results:", dataset_name.upper())
    logger.info("-" * 70)
    logger.info("%-15s %-6s %7s %10s %10s %7s %7s %8s",
                "Strategy", "Type", "Races", "Stake", "Return", "ROI", "Hit%", "AvgPay")
    logger.info("-" * 70)

    for r in strategies_results:
        logger.info("%-15s %-6s %7d %10.0f %10.0f %6.1f%% %6.1f%% %8.1f",
                    r.strategy, r.bet_type, r.n_races,
                    r.stake_total_yen, r.return_total_yen,
                    r.roi * 100, r.hit_rate * 100, r.avg_payout)

    logger.info("-" * 70)

    return ROIEvalResult(dataset=dataset_name, strategies=strategies_results)


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

    # ランキング評価 (per-race metrics)
    logger.info("-" * 40)
    logger.info("Ranking Evaluation (per-race metrics)")
    logger.info("-" * 40)
    val_ranking = evaluate_ranking(model, val_df, feature_cols, config.target_col, "val")
    test_ranking = evaluate_ranking(model, test_df, feature_cols, config.target_col, "test")

    # ROI 評価 (payouts テーブルベース、単勝・複勝)
    logger.info("-" * 40)
    logger.info("ROI Evaluation (payout-based, 単勝/複勝)")
    logger.info("-" * 40)
    val_roi = evaluate_roi_all_strategies(conn, val_df, model, feature_cols, "val")
    test_roi = evaluate_roi_all_strategies(conn, test_df, model, feature_cols, "test")

    # 結果をまとめる
    results = {
        "config": asdict(config),
        "val_result": asdict(val_result),
        "test_result": asdict(test_result),
        "val_ranking": asdict(val_ranking),
        "test_ranking": asdict(test_ranking),
        "val_roi": asdict(val_roi),
        "test_roi": asdict(test_roi),
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

    # artifacts/ に保存
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ランキング結果を保存
    ranking_path = artifacts_dir / f"ranking_{config.target_col}_{timestamp}.json"
    ranking_data = {
        "timestamp": timestamp,
        "target_col": config.target_col,
        "val_ranking": asdict(val_ranking),
        "test_ranking": asdict(test_ranking),
    }
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved ranking results to %s", ranking_path)

    # ROI 結果を保存
    roi_path = artifacts_dir / f"roi_{config.target_col}_{timestamp}.json"
    roi_data = {
        "timestamp": timestamp,
        "target_col": config.target_col,
        "val_roi": asdict(val_roi),
        "test_roi": asdict(test_roi),
    }
    with open(roi_path, "w", encoding="utf-8") as f:
        json.dump(roi_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved ROI results to %s", roi_path)

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
