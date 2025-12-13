# -*- coding: utf-8 -*-
"""
run_build_features.py

特徴量テーブル構築の実行スクリプト

使い方:
    # 全レース
    python run_build_features.py
    
    # 特定のレースのみ
    python run_build_features.py --race-ids 202301010101 202301020202
    
    # 最新100レースのみ
    python run_build_features.py --limit 100
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.features.feature_builder import build_feature_table


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Build feature_table for horse racing prediction AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default="data/keiba.db",
        help="SQLite DB path"
    )
    parser.add_argument(
        "--race-ids",
        nargs="+",
        default=None,
        help="Target race IDs (default: all races)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of races to process"
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理"""
    # ログ初期化
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("build_features.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Feature Table Building - Start")
    logger.info("=" * 80)
    logger.info("DB: %s", args.db_path)
    logger.info("Race IDs: %s", args.race_ids if args.race_ids else "all")
    logger.info("Limit: %s", args.limit if args.limit else "none")
    logger.info("=" * 80)
    
    # DB接続
    conn = sqlite3.connect(args.db_path)
    
    try:
        # 対象レースを取得
        if args.race_ids:
            target_race_ids = args.race_ids
        elif args.limit:
            # 最新n件のレースを取得
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT race_id
                FROM race_results
                ORDER BY race_id DESC
                LIMIT ?
                """,
                (args.limit,)
            )
            target_race_ids = [row[0] for row in cur.fetchall()]
            logger.info("Selected %d most recent races", len(target_race_ids))
        else:
            target_race_ids = None
        
        # 特徴量テーブルを構築
        total_rows = build_feature_table(conn, target_race_ids=target_race_ids)
        
        logger.info("=" * 80)
        logger.info("Feature Table Building - Done")
        logger.info("  Total rows: %d", total_rows)
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
