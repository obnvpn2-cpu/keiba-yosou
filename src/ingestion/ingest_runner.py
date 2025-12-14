"""
netkeiba ingestion パイプライン - メイン実行スクリプト

コマンドライン引数で期間・場を指定し、
レースデータをスクレイプしてSQLiteに保存する。

使用例:
    # 2024年全期間、全JRA場
    python -m ingestion.ingest_runner --start-year 2024 --end-year 2024
    
    # 2024年1-3月、東京・中山のみ
    python -m ingestion.ingest_runner --start-year 2024 --end-year 2024 \\
        --start-mon 1 --end-mon 3 --jyo 05 06
    
    # 特定レースIDのみ
    python -m ingestion.ingest_runner --race-ids 202406050901 202406050902
    
    # ドライラン
    python -m ingestion.ingest_runner --start-year 2024 --end-year 2024 --dry-run
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Optional

from .scraper import get_client, reset_client
from .parser import parse_race_page, parse_horse_laptime_json
from .race_list import fetch_race_ids
from .db import get_database
from .models import JRA_PLACE_CODES

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ingestion(
    start_year: int,
    end_year: int,
    start_mon: int = 1,
    end_mon: int = 12,
    place_codes: Optional[list[str]] = None,
    db_path: str = "netkeiba.db",
    skip_existing: bool = False,
    refresh_horse_laps: bool = True,
    race_ids: Optional[list[str]] = None,
    dry_run: bool = False,
    dump_horse_lap_ids: Optional[set[str]] = None,
) -> dict:
    """
    レースデータのingestionを実行する。
    
    Args:
        start_year: 開始年
        end_year: 終了年
        start_mon: 開始月
        end_mon: 終了月
        place_codes: 場コードのリスト（Noneの場合は全JRA場）
        db_path: SQLiteデータベースのパス
        skip_existing: 既存レースをスキップするかどうか
        race_ids: 直接指定するレースIDリスト（指定時は検索をスキップ）
        dry_run: Trueの場合、DBに保存しない
        dump_horse_lap_ids: 指定レースの horse_laps HTML を debug/ にダンプ
    
    Returns:
        実行統計の辞書
    """
    stats = {
        "total_races": 0,
        "processed": 0,
        "skipped": 0,
        "refreshing_horse_laps": 0,
        "errors": 0,
        "horse_laps_total": 0,
        "start_time": datetime.now(),
    }
    
    # デフォルトは全JRA場
    if place_codes is None:
        place_codes = JRA_PLACE_CODES
    
    client = get_client()
    
    try:
        with get_database(db_path) as db:
            # レースID取得
            if race_ids:
                # 直接指定されたレースID
                target_race_ids = race_ids
                logger.info(f"Using {len(target_race_ids)} directly specified race IDs")
            else:
                # 検索で取得
                logger.info(
                    f"Fetching race IDs: {start_year}/{start_mon} - {end_year}/{end_mon}, "
                    f"places: {place_codes}"
                )
                target_race_ids = fetch_race_ids(
                    client=client,
                    start_year=start_year,
                    start_mon=start_mon,
                    end_year=end_year,
                    end_mon=end_mon,
                    place_codes=place_codes,
                )
            
            stats["total_races"] = len(target_race_ids)
            logger.info(f"Found {len(target_race_ids)} races")
            
            if not target_race_ids:
                logger.warning("No races found")
                return stats
            
            # 既存レースをスキップする場合
            if skip_existing and not dry_run:
                existing_ids = db.get_existing_race_ids(target_race_ids)
                missing_horse_lap_ids = set()
                incomplete_horse_lap_ids = set()
                if refresh_horse_laps:
                    missing_horse_lap_ids = db.get_race_ids_missing_horse_laps(
                        list(existing_ids)
                    )
                    incomplete_horse_lap_ids = db.get_race_ids_incomplete_horse_laps(
                        list(existing_ids)
                    )
                original_count = len(target_race_ids)
                target_race_ids = [
                    rid
                    for rid in target_race_ids
                    if rid not in existing_ids
                    or rid in missing_horse_lap_ids
                    or rid in incomplete_horse_lap_ids
                ]
                skipped = original_count - len(target_race_ids)
                stats["skipped"] = max(skipped, 0)
                stats["refreshing_horse_laps"] = len(
                    missing_horse_lap_ids | incomplete_horse_lap_ids
                )
                if skipped > 0:
                    logger.info(f"Skipping {skipped} existing races")
                if missing_horse_lap_ids or incomplete_horse_lap_ids:
                    logger.info(
                        "Re-fetching horse laps for %d races with missing/partial laps",
                        len(missing_horse_lap_ids | incomplete_horse_lap_ids),
                    )
            
            # 各レースを処理
            for i, race_id in enumerate(target_race_ids, 1):
                try:
                    # 進捗ログ
                    if i % 50 == 0 or i == 1:
                        elapsed = (datetime.now() - stats["start_time"]).total_seconds()
                        rate = i / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {i}/{len(target_race_ids)} "
                            f"({i/len(target_race_ids)*100:.1f}%), "
                            f"rate: {rate:.2f} races/sec"
                        )
                    
                    # レースページ取得
                    logger.debug(f"Fetching race {race_id}")
                    html = client.get_race_page(race_id)
                    
                    # パース
                    data = parse_race_page(html, race_id)
                    
                    # 各馬ラップを AJAX から取得
                    horse_laps = _fetch_and_parse_horse_laps(
                        client, race_id, dump_horse_lap_ids
                    )
                    if horse_laps:
                        data.horse_laps = horse_laps
                        stats["horse_laps_total"] += len(horse_laps)
                        logger.debug(f"[{race_id}] Fetched {len(horse_laps)} horse laps via AJAX")
                    
                    # 保存
                    if not dry_run:
                        db.save_race_data(data)
                        logger.debug(f"Saved race {race_id}")
                    else:
                        logger.debug(f"[DRY-RUN] Would save race {race_id}")
                    
                    stats["processed"] += 1
                    
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Error processing race {race_id}: {e}")
                    continue
            
            # 完了統計
            elapsed = (datetime.now() - stats["start_time"]).total_seconds()
            stats["elapsed_seconds"] = elapsed
            stats["rate"] = stats["processed"] / elapsed if elapsed > 0 else 0
            
            logger.info("=" * 60)
            logger.info("Ingestion completed")
            logger.info(f"  Total races: {stats['total_races']}")
            logger.info(f"  Processed: {stats['processed']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            logger.info(f"  Re-fetch horse laps: {stats['refreshing_horse_laps']}")
            logger.info(f"  Errors: {stats['errors']}")
            logger.info(f"  Horse laps: {stats['horse_laps_total']}")
            logger.info(f"  Elapsed: {elapsed:.1f}s")
            logger.info(f"  Rate: {stats['rate']:.2f} races/sec")
            
            if not dry_run:
                logger.info(f"  DB races: {db.get_race_count()}")
                logger.info(f"  DB results: {db.get_result_count()}")
            
    finally:
        reset_client()
    
    return stats


def _fetch_and_parse_horse_laps(
    client, race_id: str, dump_horse_lap_ids: Optional[set[str]] = None
) -> list:
    """
    AJAXエンドポイントから各馬ラップを取得してパースする。
    
    Args:
        client: NetkeibaClient
        race_id: レースID
    
    Returns:
        HorseLapのリスト
    """
    try:
        json_str = client.fetch_horse_laptime(race_id)
        if not json_str:
            logger.debug(f"[{race_id}] No horse laptime data from AJAX")
            return []

        if dump_horse_lap_ids and race_id in dump_horse_lap_ids:
            debug_dir = Path(__file__).resolve().parent.parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            dump_path = debug_dir / f"horse_laps_{race_id}.html"
            try:
                dump_path.write_text(json_str, encoding="utf-8")
                logger.info(f"[{race_id}] Dumped horse lap HTML to {dump_path}")
            except Exception as dump_err:
                logger.warning(
                    f"[{race_id}] Failed to dump horse lap HTML to {dump_path}: {dump_err}"
                )
        
        horse_laps = parse_horse_laptime_json(json_str, race_id)
        return horse_laps
        
    except Exception as e:
        logger.warning(f"[{race_id}] Failed to fetch/parse horse laps: {e}")
        return []


def main():
    """コマンドラインエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="netkeiba レースデータ ingestion パイプライン"
    )
    
    # 期間指定
    parser.add_argument(
        "--start-year",
        type=int,
        required=False,
        help="開始年（例: 2024）",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        required=False,
        help="終了年（例: 2024）",
    )
    parser.add_argument(
        "--start-mon",
        type=int,
        default=1,
        help="開始月（デフォルト: 1）",
    )
    parser.add_argument(
        "--end-mon",
        type=int,
        default=12,
        help="終了月（デフォルト: 12）",
    )
    
    # 場指定
    parser.add_argument(
        "--jyo",
        nargs="+",
        default=None,
        help="場コード（例: 05 06）。指定しない場合は全JRA場",
    )
    
    # DB設定
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="SQLiteデータベースのパス（デフォルト: netkeiba.db）",
    )
    
    # オプション
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="既にDBにあるレースをスキップ",
    )
    parser.add_argument(
        "--no-refresh-horse-laps",
        action="store_true",
        help="既存レースに horse_laps が無くても再取得しない",
    )
    parser.add_argument(
        "--race-ids",
        nargs="+",
        default=None,
        help="直接レースIDを指定（この場合、期間検索はスキップ）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DBに保存せずログのみ出力",
    )
    parser.add_argument(
        "--dump-horse-laps",
        nargs="+",
        default=None,
        help="指定した race_id の horse_laps HTML を debug/ にダンプ",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細ログを出力",
    )
    
    args = parser.parse_args()
    
    # ログレベル調整
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # requestsのログは抑制
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # 引数バリデーション
    if args.race_ids is None:
        if args.start_year is None or args.end_year is None:
            parser.error("--start-year and --end-year are required when --race-ids is not specified")
    
    # 実行
    try:
        stats = run_ingestion(
            start_year=args.start_year or 2024,
            end_year=args.end_year or 2024,
            start_mon=args.start_mon,
            end_mon=args.end_mon,
            place_codes=args.jyo,
            db_path=args.db,
            skip_existing=args.skip_existing,
            refresh_horse_laps=not args.no_refresh_horse_laps,
            race_ids=args.race_ids,
            dry_run=args.dry_run,
            dump_horse_lap_ids=set(args.dump_horse_laps)
            if args.dump_horse_laps
            else None,
        )
        
        # エラーがあった場合は終了コード1
        if stats["errors"] > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()