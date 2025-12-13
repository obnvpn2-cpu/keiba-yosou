# -*- coding: utf-8 -*-
"""
ingestion_horse_past.py

馬の過去走成績 (全戦績) を netkeiba から取得して SQLite に保存するスクリプト。

改善点:
- fetcher.fetch() -> fetcher.fetch_soup() に修正
- 統計情報の追加（total, inserted, failed, skipped）
- エラーハンドリングの改善
- ログ出力の詳細化

使い方 (例):
    python -m src.ingestion.ingestion_horse_past --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from typing import Optional, List, Dict
from pathlib import Path

from tqdm import tqdm

from .fetcher import NetkeibaFetcher
from .parser_horse_past import HorsePastRunsParser
from .sqlite_store_horse_past import HorsePastSQLiteStore

logger = logging.getLogger(__name__)


class HorsePastIngestion:
    """
    馬の過去走成績 ingestion のオーケストレータ。
    
    改善点:
    - fetch_soup() メソッドを使用
    - 統計情報の追加
    - エラーハンドリングの改善
    """

    def __init__(
        self,
        db_path: str = "data/keiba.db",
        limit: Optional[int] = None,
        skip_on_error: bool = True,
        fetcher: Optional[NetkeibaFetcher] = None,
    ) -> None:
        self.db_path = db_path
        self.limit = limit
        self.skip_on_error = skip_on_error
        self.fetcher = fetcher or NetkeibaFetcher()
        self.parser = HorsePastRunsParser()
        logger.info("HorsePastIngestion initialized: db=%s", db_path)

    # ------------------------------------------------------------------
    # horse_id のロード
    # ------------------------------------------------------------------
    def _load_target_horse_ids(self) -> List[str]:
        """
        race_results テーブルから horse_id を集約し、
        horse_past_runs にまだ存在しないものだけを返す。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            # race_results から distinct horse_id を取得
            cur.execute(
                """
                SELECT DISTINCT horse_id
                FROM race_results
                WHERE horse_id IS NOT NULL
                  AND horse_id <> ''
                ORDER BY horse_id;
                """
            )
            race_horses = [row["horse_id"] for row in cur.fetchall()]
            logger.info("Loaded %d distinct horse_id from race_results", len(race_horses))

            # 既に horse_past_runs にある horse_id を取得
            try:
                cur.execute("SELECT DISTINCT horse_id FROM horse_past_runs;")
                existed = {row["horse_id"] for row in cur.fetchall()}
                logger.info("Loaded %d existing horse_id from horse_past_runs", len(existed))
            except sqlite3.OperationalError:
                # テーブル未作成の場合は 0 件
                existed = set()
                logger.info("horse_past_runs table does not exist yet (treated as 0 existing)")

            missing = [h for h in race_horses if h not in existed]
            logger.info(
                "Missing horse_past_runs records: total=%d (all=%d, existing=%d)",
                len(missing),
                len(race_horses),
                len(existed),
            )
            return missing
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # main run
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, int]:
        """
        メイン処理
        
        Returns:
            統計情報の辞書
            {
                "total": 処理対象の件数,
                "inserted": 成功した件数,
                "failed": 失敗した件数,
                "skipped": スキップした件数
            }
        """
        horse_ids = self._load_target_horse_ids()
        if self.limit is not None:
            horse_ids = horse_ids[: self.limit]

        if not horse_ids:
            logger.info("No target horse_id to ingest (already up to date?)")
            return {
                "total": 0,
                "inserted": 0,
                "failed": 0,
                "skipped": 0,
            }

        logger.info("Start ingesting %d horse(s) past runs", len(horse_ids))

        stats = {
            "total": len(horse_ids),
            "inserted": 0,
            "failed": 0,
            "skipped": 0,
        }
        failed_horse_ids = []

        with HorsePastSQLiteStore(self.db_path) as store:
            with self.fetcher:
                for horse_id in tqdm(horse_ids, desc="Ingest horse_past_runs", unit="horse"):
                    try:
                        url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
                        
                        # ★★★ 修正ポイント ★★★
                        # fetch() -> fetch_soup() に修正
                        soup = self.fetcher.fetch_soup(url)
                        
                        df = self.parser.parse(soup, horse_id)
                        inserted = store.insert_past_runs(df)
                        
                        stats["inserted"] += 1
                        
                        logger.info(
                            "Ingested horse past runs: horse_id=%s, rows=%d",
                            horse_id,
                            inserted,
                        )
                        
                    except Exception as e:
                        logger.error(
                            "Failed to ingest horse past runs: horse_id=%s, error=%s", 
                            horse_id, 
                            e,
                            exc_info=True
                        )
                        stats["failed"] += 1
                        failed_horse_ids.append(horse_id)
                        
                        if not self.skip_on_error:
                            logger.error("Stopping due to error (skip_on_error=False)")
                            break

        # 統計情報をログ出力
        logger.info("=" * 80)
        logger.info("Ingestion Statistics:")
        logger.info("  Total:    %d", stats["total"])
        logger.info("  Inserted: %d", stats["inserted"])
        logger.info("  Failed:   %d", stats["failed"])
        logger.info("  Skipped:  %d", stats["skipped"])
        logger.info("=" * 80)
        
        if failed_horse_ids:
            logger.warning("Failed horse_ids: %s", ", ".join(failed_horse_ids[:10]))
            if len(failed_horse_ids) > 10:
                logger.warning("... and %d more", len(failed_horse_ids) - 10)

        logger.info("Horse past runs ingestion finished")
        return stats

    # ------------------------------------------------------------------
    # convenience (for REPL)
    # ------------------------------------------------------------------
    def ingest_one(self, horse_id: str) -> int:
        """
        単一 horse_id のみを取得して保存するヘルパー。
        
        Args:
            horse_id: 馬ID
        
        Returns:
            挿入/更新した行数
        """
        with HorsePastSQLiteStore(self.db_path) as store:
            with self.fetcher:
                url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
                soup = self.fetcher.fetch_soup(url)
                df = self.parser.parse(soup, horse_id)
                return store.insert_past_runs(df)


# ==============================================================================
# CLI
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Ingest horse past runs from netkeiba into SQLite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", 
        dest="db_path", 
        default="data/keiba.db", 
        help="SQLite DB path"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="最大頭数 (None = 全て)"
    )
    parser.add_argument(
        "--no-skip-on-error",
        action="store_true",
        help="エラー時にスキップせず例外を投げる (デフォルトはスキップ)",
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理"""
    # ログ初期化
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("ingestion_horse_past.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    args = parse_args()
    skip_on_error = not args.no_skip_on_error

    logger.info("=" * 80)
    logger.info("Horse Past Runs Ingestion - Start")
    logger.info("=" * 80)
    logger.info("DB: %s", args.db_path)
    logger.info("Limit: %s", args.limit)
    logger.info("Skip on error: %s", skip_on_error)
    logger.info("=" * 80)

    # DB ファイルの存在チェック
    db_file = Path(args.db_path)
    if not db_file.exists():
        logger.error("DB file not found: %s", args.db_path)
        logger.error("Please run race_results ingestion first.")
        return

    try:
        ingestion = HorsePastIngestion(
            db_path=args.db_path,
            limit=args.limit,
            skip_on_error=skip_on_error,
        )
        stats = ingestion.run()

        logger.info("=" * 80)
        logger.info("Horse Past Runs Ingestion - Done")
        logger.info("  Inserted: %d / %d", stats["inserted"], stats["total"])
        if stats["failed"] > 0:
            logger.warning("  Failed: %d", stats["failed"])
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
