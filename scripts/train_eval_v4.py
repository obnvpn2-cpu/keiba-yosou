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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_v4 import TrainConfig, run_full_pipeline


logger = logging.getLogger(__name__)


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
"""
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB (e.g., netkeiba.db)",
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
    logger.info(f"Target: {args.target}")
    logger.info(f"Split mode: {args.split_mode}")
    logger.info(f"Include pedigree: {not args.no_pedigree}")
    logger.info(f"Include market: {args.include_market}")

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

    conn = sqlite3.connect(db_path)
    try:
        results = run_full_pipeline(
            conn=conn,
            config=config,
            output_dir=output_dir,
            train_end=args.train_end,
            val_end=args.val_end,
            split_mode=args.split_mode,
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

        logger.info("=" * 70)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
