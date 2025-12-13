# src/ingestion/run_ingest_netkeiba.py

import argparse
import logging
import random
import re
import sqlite3
import time
from typing import List, Optional


import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from .ingestion_pipeline import IngestionPipeline
from .sqlite_store import RaceResultSQLiteStore


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =============================================================================
# User-Agent プール（Zenn記事準拠）
# =============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 "
    "Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) "
    "Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
]


def get_random_headers() -> dict:
    """netkeiba 用のランダム User-Agent ヘッダ"""
    ua = random.choice(USER_AGENTS)
    return {"User-Agent": ua}


# =============================================================================
# カレンダー → 開催日一覧
# =============================================================================


def get_kaisai_dates(year: int, month: int) -> List[str]:
    """
    netkeiba の開催カレンダーから「開催日（kaisai_date）」一覧を取得する。

    例:
        https://race.netkeiba.com/top/calendar.html?year=2024&month=10
    """
    url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
    headers = get_random_headers()

    logger.info("Fetching calendar: %s", url)
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml")

    kaisai_dates: List[str] = []

    # カレンダー内の <a> に kaisai_date=YYYYMMDD が入っている
    # セレクタは多少変わっても href の正規表現で拾えるようにしておく
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        m = re.search(r"kaisai_date=(\d{8})", href)
        if m:
            kaisai_dates.append(m.group(1))

    kaisai_dates = sorted(set(kaisai_dates))
    logger.info("  Found %d kaisai_dates: %s", len(kaisai_dates), kaisai_dates)
    return kaisai_dates


# =============================================================================
# 開催日 → race_id 一覧
# =============================================================================


def get_race_ids_for_kaisai_date(kaisai_date: str, wait_sec: int = 30) -> List[str]:
    """
    開催日ごとのレース一覧ページから race_id を取得する。

    例:
        https://race.netkeiba.com/top/race_list.html?kaisai_date=20241027
    """
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
    logger.info("Fetching race list: %s", url)

    ua = random.choice(USER_AGENTS)

    chrome_options = Options()
    chrome_options.add_argument(f"--user-agent={ua}")
    # 必要なら headless も追加
    # chrome_options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, wait_sec)

    race_ids: List[str] = []
    try:
        driver.get(url)

        # race_id を含むリンクが出るまで待機（かなり緩めの条件）
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "a[href*='race_id=']")
            )
        )

        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")

        # href に race_id=XXXXXXXXXXXX を含むすべてのリンクから抽出
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            m = re.search(r"race_id=(\d+)", href)
            if m:
                race_ids.append(m.group(1))
    finally:
        driver.quit()

    race_ids = sorted(set(race_ids))
    logger.info(
        "  Found %d race_ids for %s: %s",
        len(race_ids),
        kaisai_date,
        race_ids,
    )
    return race_ids


# =============================================================================
# 既存 DB 内に race_id があるかチェック
# =============================================================================


def race_exists(conn: sqlite3.Connection, race_id: str) -> bool:
    """
    races テーブルに既に存在するかどうかを確認。
    無限に重複 INSERT するのを防ぐため。
    """
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM races WHERE race_id = ? LIMIT 1;", (race_id,))
    return cur.fetchone() is not None


# =============================================================================
# ★ここが「あなたの既存 ingestion 処理」を呼ぶフック★
# =============================================================================


def ingest_one_race(conn: sqlite3.Connection, race_id: str, dry_run: bool = False) -> int:
    """
    単一レースを netkeiba から取得して SQLite に保存する。

    - netkeiba のレース結果ページをスクレイピング
    - parser_race_result で DataFrame に変換
    - RaceResultSQLiteStore を使って races / race_results に UPSERT

    Args:
        conn: 既に開いている SQLite コネクション
        race_id: レースID（12桁）
        dry_run: True のときはスクレイピングまでで DB 書き込みしない

    Returns:
        追加 / 更新した行数（race_results の件数）
    """
    logger = logging.getLogger(__name__)

    logger.info("  Ingesting race_id=%s ...", race_id)

    # DRY RUN オプション（テスト用）
    if dry_run:
        logger.info("  [DRY RUN] ingest_one_race called for race_id=%s (no DB write)", race_id)
        return 0

    # 既に存在するレースはスキップ
    if race_exists(conn, race_id):
        logger.info("  race_id=%s already exists in DB. Skipping.", race_id)
        return 0

    # このコネクションが開いている DB ファイルのパスを取得
    row = conn.execute("PRAGMA database_list;").fetchone()
    db_path = row[2] if row and len(row) >= 3 else "data/keiba.db"
    logger.debug("  DB path detected from connection: %s", db_path)

    # 1. レース結果をスクレイピング
    with IngestionPipeline() as pipeline:
        df = pipeline.scrape_race_result(race_id)

    if df is None or df.empty:
        logger.warning("  No race_result rows scraped for race_id=%s. Skipping DB insert.", race_id)
        return 0

    # 2. SQLite に保存（races / race_results）
    inserted = 0
    with RaceResultSQLiteStore(db_path=db_path) as store:
        inserted = store.insert_race_results(df)

    logger.info("  Saved race_id=%s to SQLite (race_results rows=%d)", race_id, inserted)
    return inserted



# =============================================================================
# メインループ
# =============================================================================


def run_ingest(
    db_path: str,
    year_from: int,
    year_to: int,
    month_from: int = 1,
    month_to: int = 12,
    sleep_sec: float = 2.5,
    max_races: Optional[int] = None,
) -> None:
    """
    指定した year/month の範囲で netkeiba を走査し、
    新しい race_id について ingestion を行う。

    Args:
        db_path: SQLite パス (例: data/keiba.db)
        year_from: 開始年
        year_to: 終了年（含む）
        month_from: 開始月
        month_to: 終了月
        sleep_sec: レースごとのインターバル（サーバー対策で 2〜3秒推奨）
        max_races: 全体での最大レース数（テスト用に制限したいとき）
    """
    conn = sqlite3.connect(db_path)

    total_ingested = 0
    try:
        for year in range(year_from, year_to + 1):
            for month in range(month_from, month_to + 1):
                logger.info("=== Year %d Month %d ===", year, month)

                try:
                    kaisai_dates = get_kaisai_dates(year, month)
                except Exception as e:
                    logger.error(
                        "Failed to fetch kaisai_dates for %d-%02d: %s",
                        year,
                        month,
                        e,
                    )
                    continue

                for kaisai_date in kaisai_dates:
                    try:
                        race_ids = get_race_ids_for_kaisai_date(kaisai_date)
                    except Exception as e:
                        logger.error(
                            "Failed to fetch race_ids for kaisai_date=%s: %s",
                            kaisai_date,
                            e,
                        )
                        continue

                    for race_id in race_ids:
                        if race_exists(conn, race_id):
                            logger.info("  Skip existing race_id=%s", race_id)
                            continue

                        logger.info("  Ingesting race_id=%s ...", race_id)
                        try:
                            ingest_one_race(conn, race_id)
                            conn.commit()
                            total_ingested += 1
                        except Exception as e:
                            logger.exception(
                                "  Failed to ingest race_id=%s: %s",
                                race_id,
                                e,
                            )
                            conn.rollback()

                        if max_races is not None and total_ingested >= max_races:
                            logger.info(
                                "Reached max_races=%d, stopping.", max_races
                            )
                            logger.info(
                                "Total ingested races: %d", total_ingested
                            )
                            return

                        time.sleep(sleep_sec)

        logger.info("Total ingested races: %d", total_ingested)
    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="netkeiba から races / race_results / horse_past_runs を一括取得するランナー",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/keiba.db",
        help="SQLite DB path",
    )
    parser.add_argument(
        "--year-from",
        type=int,
        required=True,
        help="開始年 (例: 2022)",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        required=True,
        help="終了年 (例: 2023)",
    )
    parser.add_argument(
        "--month-from",
        type=int,
        default=1,
        help="開始月 (1-12)",
    )
    parser.add_argument(
        "--month-to",
        type=int,
        default=12,
        help="終了月 (1-12)",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=2.5,
        help="各レース間の sleep 秒数（2〜3秒推奨）",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=None,
        help="全体での最大レース数（テスト用）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ingest(
        db_path=args.db,
        year_from=args.year_from,
        year_to=args.year_to,
        month_from=args.month_from,
        month_to=args.month_to,
        sleep_sec=args.sleep_sec,
        max_races=args.max_races,
    )
