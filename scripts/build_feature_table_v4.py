#!/usr/bin/env python3
"""
build_feature_table_v4.py - CLI for FeaturePack v1 Feature Table Builder

FeaturePack v1 (200+ features) の feature_table_v4 を構築します。

【特徴量グループ】
- base_race: 基本レース情報 (~25 columns)
- horse_form: 馬の過去成績・フォーム (~40 columns)
- pace_position: ペース・位置取り (~20 columns)
- class_prize: クラス・賞金 (~15 columns)
- jockey_trainer: 騎手・調教師の as-of 成績 (~40 columns)
- pedigree: 血統ハッシュ (512 + 128 = 640 columns)

合計: 200+ 特徴量

【リーク防止】
- 全ての統計は as-of (race_date より前) で計算
- 当日のオッズ・人気は含まない

Usage:
    python scripts/build_feature_table_v4.py --db netkeiba.db
    python scripts/build_feature_table_v4.py --db netkeiba.db --start-year 2024 --end-year 2024
    python scripts/build_feature_table_v4.py --db netkeiba.db --no-pedigree
"""

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_v4 import build_feature_table_v4


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature_table_v4 (FeaturePack v1: 200+ features)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build all features for 2021-2024
  python scripts/build_feature_table_v4.py --db netkeiba.db

  # Build for specific year range
  python scripts/build_feature_table_v4.py --db netkeiba.db --start-year 2023 --end-year 2024

  # Build without pedigree hash (faster, smaller table)
  python scripts/build_feature_table_v4.py --db netkeiba.db --no-pedigree

  # Include market features (odds/popularity)
  python scripts/build_feature_table_v4.py --db netkeiba.db --include-market
"""
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB (e.g., netkeiba.db)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year (default: 2021)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year (default: 2024)",
    )
    parser.add_argument(
        "--no-pedigree",
        action="store_true",
        help="Exclude pedigree hash features (512+128 columns)",
    )
    parser.add_argument(
        "--include-market",
        action="store_true",
        help="Include market features (win_odds, popularity)",
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
    logger.info("FeaturePack v1 Feature Builder (feature_table_v4)")
    logger.info("=" * 70)
    logger.info(f"Database: {db_path}")
    logger.info(f"Year range: {args.start_year} - {args.end_year}")
    logger.info(f"Include pedigree: {not args.no_pedigree}")
    logger.info(f"Include market: {args.include_market}")

    conn = sqlite3.connect(db_path)
    try:
        rows = build_feature_table_v4(
            conn=conn,
            start_year=args.start_year,
            end_year=args.end_year,
            include_pedigree=not args.no_pedigree,
            include_market=args.include_market,
        )
        logger.info("=" * 70)
        logger.info(f"Success! Inserted {rows:,} rows into feature_table_v4")
        logger.info("=" * 70)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
