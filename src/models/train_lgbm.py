# src/models/train_lgbm.py (v2.0 - Reviewed & Improved)
"""
LightGBMモデルの学習スクリプト

v2.0の改善:
- パラメータの最適化（正則化追加、過学習対策）
- より詳細なログ出力
- 特徴量重要度の保存
- モデル評価指標の拡充
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score

from .model_utils import (
    load_feature_table,
    split_train_valid,
    save_feature_columns,
)

logger = logging.getLogger(__name__)


def train_one_target(
    db_path: str,
    output_dir: str,
    target_col: str,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = 8,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> Tuple[lgb.Booster, float, float]:
    """
    1つの目的変数(target_col)について LightGBM モデルを学習
    
    v2.0の改善:
    - 正則化パラメータの追加
    - 特徴量重要度の保存
    - より詳細な評価指標
    
    Args:
        db_path: SQLiteデータベースのパス
        output_dir: モデル保存先ディレクトリ
        target_col: ターゲット変数名
        learning_rate: 学習率
        num_leaves: 葉の数
        max_depth: 最大深さ
        num_boost_round: ブースティング回数
        early_stopping_rounds: Early Stopping の rounds
    
    Returns:
        (model, valid_auc, valid_logloss)
    """
    logger.info("=" * 80)
    logger.info("Training model for target: %s", target_col)
    logger.info("=" * 80)

    df = load_feature_table(db_path)
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col} not found in feature_table")

    # 学習/検証 split
    X_train, X_valid, y_train, y_valid, feature_cols = split_train_valid(
        df, target_col
    )

    logger.info("Train size: %d, Valid size: %d", len(X_train), len(X_valid))
    logger.info("Using %d features", len(feature_cols))

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    # ★ v2.0: パラメータの改善（正則化追加）
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,  # v2.0: 深さ制限を追加
        "min_child_samples": 20,  # v2.0: 最小サンプル数を追加
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,  # v2.0: 0.9→0.8に変更
        "bagging_freq": 5,  # v2.0: 1→5に変更（効率化）
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,  # v2.0: L2正則化を追加
        "min_gain_to_split": 0.01,  # v2.0: 分割の最小ゲインを追加
        "verbose": -1,
        "force_col_wise": True,  # v2.0: 高速化
    }

    logger.info("Training parameters:")
    for key, value in params.items():
        logger.info("  %s: %s", key, value)

    # 学習
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    logger.info("Training completed. Best iteration: %d", model.best_iteration)

    # 検証スコア計算
    y_pred_proba = model.predict(X_valid, num_iteration=model.best_iteration)

    # AUC
    try:
        valid_auc = roc_auc_score(y_valid, y_pred_proba)
        logger.info("[%s] Valid AUC: %.4f", target_col, valid_auc)
    except Exception as e:
        logger.warning("[%s] AUC calculation failed: %s", target_col, e)
        valid_auc = float("nan")

    # Logloss
    try:
        valid_logloss = log_loss(y_valid, y_pred_proba)
        logger.info("[%s] Valid Logloss: %.4f", target_col, valid_logloss)
    except Exception as e:
        logger.warning("[%s] Logloss calculation failed: %s", target_col, e)
        valid_logloss = float("nan")

    # ★ v2.0: 追加の評価指標
    try:
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_valid, y_pred_binary)
        precision = precision_score(y_valid, y_pred_binary, zero_division=0)
        recall = recall_score(y_valid, y_pred_binary, zero_division=0)
        
        logger.info("[%s] Valid Accuracy: %.4f", target_col, accuracy)
        logger.info("[%s] Valid Precision: %.4f", target_col, precision)
        logger.info("[%s] Valid Recall: %.4f", target_col, recall)
    except Exception as e:
        logger.warning("[%s] Additional metrics calculation failed: %s", target_col, e)

    # 保存
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # モデル本体 (LightGBM ネイティブ形式)
    model_path_txt = out_dir / f"lgbm_{target_col}.txt"
    model.save_model(str(model_path_txt))
    logger.info("Saved model to %s", model_path_txt)

    # joblib 形式
    model_path_pkl = out_dir / f"lgbm_{target_col}.pkl"
    joblib.dump(model, model_path_pkl)
    logger.info("Saved model (joblib) to %s", model_path_pkl)

    # 使用した特徴量リスト
    save_feature_columns(feature_cols, output_dir, target_col)

    # ★ v2.0: 特徴量重要度を保存
    save_feature_importance(model, feature_cols, output_dir, target_col)

    logger.info("=" * 80)
    
    return model, valid_auc, valid_logloss


def save_feature_importance(
    model: lgb.Booster,
    feature_cols: list,
    output_dir: str,
    target_col: str,
) -> None:
    """
    特徴量重要度をCSVに保存
    
    v2.0: 説明可能性のために追加
    """
    importance_gain = model.feature_importance(importance_type="gain")
    importance_split = model.feature_importance(importance_type="split")
    
    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importance_gain,
        "importance_split": importance_split,
    })
    
    df_importance = df_importance.sort_values("importance_gain", ascending=False)
    
    out_path = Path(output_dir) / f"feature_importance_{target_col}.csv"
    df_importance.to_csv(out_path, index=False)
    
    logger.info("Saved feature importance to %s", out_path)
    
    # Top 10 を表示
    logger.info("Top 10 important features (gain):")
    for i, row in df_importance.head(10).iterrows():
        logger.info(
            "  %2d. %s: %.2f",
            i + 1,
            row["feature"],
            row["importance_gain"]
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=str,
        default="data/keiba.db",
        help="Path to SQLite DB",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=63,
        help="Number of leaves",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum depth",
    )
    args = parser.parse_args()

    db_path = args.db
    out_dir = args.out

    # target_win モデル
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING TARGET_WIN MODEL")
    logger.info("=" * 80)
    train_one_target(
        db_path=db_path,
        output_dir=out_dir,
        target_col="target_win",
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
    )

    # target_in3 モデル
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING TARGET_IN3 MODEL")
    logger.info("=" * 80)
    train_one_target(
        db_path=db_path,
        output_dir=out_dir,
        target_col="target_in3",
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
    )

    logger.info("\n" + "=" * 80)
    logger.info("All models trained successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
