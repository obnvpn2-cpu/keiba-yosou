# -*- coding: utf-8 -*-
# src/ingestion/ingestion_horse_basic.py
"""
馬の基本情報 ingestion スクリプト。

役割:
- race_results テーブルから horse_id を集計
- horse_basic にまだ無い horse_id を抽出
- 各馬ページ (https://db.netkeiba.com/horse/{horse_id}/) をスクレイピング
- parser_horse_basic でパースして SQLite (horse_basic) に保存

改善点:
- 相対インポートに修正（from .fetcher import）
- エラーハンドリングの改善
- スキップ機能の追加（エラー時に全体を止めない）
- プログレスバーとログの共存
- 詳細な統計情報の出力

使い方（例）:

    # race_results に十分なデータが入っている前提で
    $ python -m src.ingestion.ingestion_horse_basic

    # ある程度テストだけしたい場合（最初の 50 頭だけ）
    $ python -m src.ingestion.ingestion_horse_basic --limit 50
    
    # 別のDBファイルを指定
    $ python -m src.ingestion.ingestion_horse_basic --db data/test.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from typing import List, Set, Optional
from pathlib import Path

from tqdm import tqdm

from .fetcher import NetkeibaFetcher
from .parser_horse_basic import HorseBasicParser, HorseBasicRecord
from .sqlite_store_horse import HorseBasicSQLiteStore

logger = logging.getLogger(__name__)


class HorseBasicIngestion:
    """
    馬の基本情報 ingestion をまとめたクラス
    
    Features:
        - race_results から horse_id を抽出
        - horse_basic との差分を計算
        - 不足している馬情報を自動取得
        - エラー時のスキップ機能
        - 詳細な統計情報
    
    Example:
        >>> with HorseBasicIngestion() as ingestion:
        ...     inserted = ingestion.ingest_missing(limit=10)
        ...     print(f"Inserted: {inserted}")
    """

    def __init__(
        self,
        db_path: str = "data/keiba.db",
        sleep_min: float = 2.0,
        sleep_max: float = 3.5,
        max_retry: int = 3,
        timeout: float = 10.0,
    ) -> None:
        """
        Args:
            db_path: SQLite データベースファイルのパス
            sleep_min: スクレイピング間隔の最小値（秒）
            sleep_max: スクレイピング間隔の最大値（秒）
            max_retry: リトライ回数
            timeout: タイムアウト（秒）
        """
        self.db_path = db_path
        self.fetcher = NetkeibaFetcher(
            sleep_min=sleep_min,
            sleep_max=sleep_max,
            max_retry=max_retry,
            timeout=timeout,
        )
        self.parser = HorseBasicParser()
        logger.info("HorseBasicIngestion initialized: db=%s", db_path)

    # -------------------------- context manager -------------------------
    def __enter__(self) -> "HorseBasicIngestion":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return False

    def close(self) -> None:
        """Fetcher のクローズ"""
        if self.fetcher:
            self.fetcher.close()
            logger.info("Closed fetcher")

    # -------------------------- DB 読み取り ----------------------------
    def _load_distinct_horse_ids_from_race_results(self) -> Set[str]:
        """
        race_results から horse_id を一意に取得
        
        Returns:
            horse_id の集合
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT DISTINCT horse_id
                       FROM race_results
                      WHERE horse_id IS NOT NULL
                        AND horse_id != ''
                """
            )
            rows = cur.fetchall()
            ids = {r[0] for r in rows if r[0]}
            logger.info(
                "Loaded %d distinct horse_id from race_results", len(ids)
            )
            return ids
        finally:
            conn.close()

    def _load_existing_horse_ids(self) -> Set[str]:
        """
        horse_basic に既に存在する horse_id を取得
        
        Returns:
            horse_id の集合
        """
        with HorseBasicSQLiteStore(self.db_path) as store:
            ids = store.get_all_horse_ids()
        logger.info("Loaded %d existing horse_id from horse_basic", len(ids))
        return ids

    def list_missing_horse_ids(self) -> List[str]:
        """
        race_results にあるが horse_basic に無い horse_id を返す。
        
        Returns:
            不足している horse_id のリスト（ソート済み）
        """
        all_ids = self._load_distinct_horse_ids_from_race_results()
        existing = self._load_existing_horse_ids()
        missing = sorted(all_ids - existing)
        logger.info(
            "Missing horse_basic records: total=%d (all=%d, existing=%d)",
            len(missing),
            len(all_ids),
            len(existing),
        )
        return missing

    # -------------------------- 1頭分の取得 ----------------------------
    def fetch_and_parse_one(self, horse_id: str) -> HorseBasicRecord:
        """
        1 頭ぶんの horse ページを取りに行きパースする。
        
        Args:
            horse_id: 馬ID
        
        Returns:
            HorseBasicRecord
        
        Raises:
            Exception: スクレイピングまたはパースに失敗した場合
        """
        url = f"https://db.netkeiba.com/horse/{horse_id}/"
        logger.debug("Fetching horse page: horse_id=%s url=%s", horse_id, url)
        
        # スクレイピング
        soup = self.fetcher.fetch_soup(url)
        
        # パース
        record = self.parser.parse(horse_id, soup)
        
        return record

    # -------------------------- メイン処理 -----------------------------
    def ingest_missing(
        self,
        limit: Optional[int] = None,
        skip_on_error: bool = True,
    ) -> dict:
        """
        不足している horse_basic を埋める。

        Args:
            limit: 取得する最大件数（None の場合は全件）
            skip_on_error: エラー時にスキップするか（False の場合は中断）

        Returns:
            統計情報の辞書
            {
                "total": 処理対象の件数,
                "inserted": 成功した件数,
                "failed": 失敗した件数,
                "skipped": スキップした件数
            }
        """
        missing = self.list_missing_horse_ids()
        
        if limit is not None:
            missing = missing[:limit]
        
        if not missing:
            logger.info("No missing horse_basic records. Nothing to do.")
            return {
                "total": 0,
                "inserted": 0,
                "failed": 0,
                "skipped": 0,
            }

        logger.info("Start ingesting %d horse(s)", len(missing))
        
        stats = {
            "total": len(missing),
            "inserted": 0,
            "failed": 0,
            "skipped": 0,
        }
        
        failed_horse_ids = []

        with HorseBasicSQLiteStore(self.db_path) as store:
            for horse_id in tqdm(missing, desc="Ingest horse_basic", unit="horse"):
                try:
                    record = self.fetch_and_parse_one(horse_id)
                    store.insert_or_update(record)
                    stats["inserted"] += 1
                    
                except Exception as e:
                    logger.error(
                        "Failed to ingest horse_id=%s: %s", 
                        horse_id, 
                        e, 
                        exc_info=True
                    )
                    stats["failed"] += 1
                    failed_horse_ids.append(horse_id)
                    
                    if not skip_on_error:
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
        
        return stats


# ======================================================================
# CLI エントリポイント
# ======================================================================

def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="netkeiba 馬の基本情報 ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/keiba.db",
        help="SQLite DB path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最大取得件数（None なら全件）",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="エラー時に中断する（デフォルトはスキップ）",
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理"""
    # ログ初期化（既存 ingestion 系とフォーマットを合わせる）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("ingestion_horse_basic.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    args = parse_args()
    db_path: str = args.db
    limit: Optional[int] = args.limit
    skip_on_error: bool = not args.no_skip

    logger.info("=" * 80)
    logger.info("Horse Basic Ingestion - Start")
    logger.info("=" * 80)
    logger.info("DB: %s", db_path)
    if limit is not None:
        logger.info("Limit: %d", limit)
    logger.info("Skip on error: %s", skip_on_error)
    logger.info("=" * 80)

    # DB ファイルの存在チェック
    db_file = Path(db_path)
    if not db_file.exists():
        logger.error("DB file not found: %s", db_path)
        logger.error("Please run race_results ingestion first.")
        return

    try:
        with HorseBasicIngestion(db_path=db_path) as ingestion:
            stats = ingestion.ingest_missing(limit=limit, skip_on_error=skip_on_error)
        
        logger.info("=" * 80)
        logger.info("Horse Basic Ingestion - Done")
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
