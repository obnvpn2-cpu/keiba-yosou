#!/usr/bin/env python3
"""
fetch_odds_snapshots.py - Fetch and Store Pre-race Odds Snapshots

前日締め運用のためのオッズスナップショット取得スクリプト。
レース開催前のオッズを取得し、odds_snapshots テーブルに保存する。

【使用方法】
1. 単一レースのオッズ取得:
   python scripts/fetch_odds_snapshots.py --race-id 202406050811

2. 日付範囲のオッズ取得:
   python scripts/fetch_odds_snapshots.py --date 2024-12-28

3. 明日のレースのオッズ取得 (cron用):
   python scripts/fetch_odds_snapshots.py --tomorrow

4. HTMLデバッグ保存:
   python scripts/fetch_odds_snapshots.py --race-id 202406050811 --save-html

【decision_cutoff の考え方】
- デフォルト: 前日21:00 JST
- オッズは observed_at (取得時刻) と共に保存
- 評価時は observed_at が decision_cutoff 以前のスナップショットのみを使用

【安全装置】
- 取得した馬数が race_results の馬数と一致しない場合は保存しない（欠損防止）
- リトライ最大2回（指数バックオフ）後もダメならスキップ

Usage:
    python scripts/fetch_odds_snapshots.py --race-id 202406050811
    python scripts/fetch_odds_snapshots.py --date 2024-12-28 --db netkeiba.db
    python scripts/fetch_odds_snapshots.py --tomorrow
    python scripts/fetch_odds_snapshots.py --race-id 202406050811 --save-html
"""

import argparse
import logging
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.upsert import UpsertHelper
from src.db.schema_migration import run_migrations

try:
    import requests
    from bs4 import BeautifulSoup
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False
    requests = None
    BeautifulSoup = None


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

# User-Agent rotation list (same as scraper.py)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

# Retry configuration
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2.0  # seconds

# Sleep configuration (between requests)
DEFAULT_SLEEP_MIN = 2.0
DEFAULT_SLEEP_MAX = 3.5


# ============================================================
# Data Classes
# ============================================================


@dataclass
class OddsSnapshot:
    """Single horse odds snapshot."""
    race_id: str
    horse_no: int
    observed_at: str  # ISO format datetime
    win_odds: Optional[float]
    popularity: Optional[int]
    source: str = "netkeiba"


# ============================================================
# HTTP Client
# ============================================================


class OddsHttpClient:
    """Simple HTTP client with UA rotation, sleep, and retry."""

    ODDS_BASE_URL = "https://race.netkeiba.com/odds/index.html"

    def __init__(
        self,
        sleep_min: float = DEFAULT_SLEEP_MIN,
        sleep_max: float = DEFAULT_SLEEP_MAX,
    ):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: Optional[float] = None

    def _get_random_ua(self) -> str:
        return random.choice(USER_AGENTS)

    def _wait_if_needed(self) -> None:
        """Sleep between requests to avoid rate limiting."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            target_sleep = random.uniform(self.sleep_min, self.sleep_max)
            if elapsed < target_sleep:
                sleep_time = target_sleep - elapsed
                logger.debug(f"Sleeping {sleep_time:.2f}s between requests")
                time.sleep(sleep_time)

    def fetch_odds_page(self, race_id: str, timeout: int = 30) -> Tuple[Optional[str], int]:
        """
        Fetch odds page HTML with retry and backoff.

        Args:
            race_id: 12-digit race ID
            timeout: Request timeout in seconds

        Returns:
            Tuple of (html_content or None, status_code)
        """
        params = {
            "type": "b1",  # 単勝オッズ
            "race_id": race_id,
        }

        last_status = 0

        for attempt in range(MAX_RETRIES + 1):
            self._wait_if_needed()

            ua = self._get_random_ua()
            headers = {"User-Agent": ua}

            if attempt > 0:
                backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info(f"Retry {attempt}/{MAX_RETRIES} after {backoff:.1f}s (UA: {ua[:40]}...)")
                time.sleep(backoff)

            try:
                logger.debug(f"Fetching odds for {race_id} (attempt {attempt + 1})")
                response = self.session.get(
                    self.ODDS_BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                self._last_request_time = time.time()
                last_status = response.status_code

                if response.status_code == 200:
                    html = response.content.decode("utf-8", errors="replace")
                    return html, 200

                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"HTTP {response.status_code} for {race_id}, will retry")
                    continue

                # Other errors (400, 404, etc.) - don't retry
                logger.warning(f"HTTP {response.status_code} for {race_id}, not retrying")
                return None, response.status_code

            except requests.Timeout:
                logger.warning(f"Timeout fetching odds for {race_id}")
                last_status = 0
                continue
            except requests.RequestException as e:
                logger.warning(f"Request error for {race_id}: {e}")
                last_status = 0
                continue

        logger.error(f"Failed to fetch odds for {race_id} after {MAX_RETRIES + 1} attempts")
        return None, last_status

    def close(self):
        self.session.close()


# ============================================================
# Odds Page Parser
# ============================================================


def parse_odds_page(html: str, race_id: str, observed_at: str) -> List[OddsSnapshot]:
    """
    Parse odds page HTML and extract win odds for ALL horses.

    The netkeiba odds page (https://race.netkeiba.com/odds/index.html?type=b1&race_id=XXX)
    has the full odds table with all horses. We must parse the complete table,
    not just the "人気順" (popularity ranking) partial view.

    Args:
        html: HTML content of the odds page
        race_id: 12-digit race ID
        observed_at: ISO format timestamp of observation

    Returns:
        List of OddsSnapshot objects for all horses
    """
    soup = BeautifulSoup(html, "html.parser")
    snapshots: List[OddsSnapshot] = []

    # Strategy 1: Look for the main odds table with all horses
    # The full table typically has class "RaceOdds_HorseList" or similar
    # and contains rows with data-umaban attribute

    # Try multiple selectors for the main odds container
    odds_rows = []

    # Pattern 1: Direct row selection with data attributes
    odds_rows = soup.select("tr[data-umaban]")
    if odds_rows:
        logger.debug(f"Found {len(odds_rows)} rows with data-umaban attribute")
        for row in odds_rows:
            snapshot = _parse_odds_row_v2(row, race_id, observed_at)
            if snapshot:
                snapshots.append(snapshot)
        if snapshots:
            return snapshots

    # Pattern 2: Table with Odds class containing full horse list
    for table_selector in [
        "table.RaceOdds_HorseList_Table",
        "table.Odds_Table",
        "div.RaceOdds_HorseList table",
        "div#odds_tanpuku_block table",
        "div.Tanpuku table",
    ]:
        table = soup.select_one(table_selector)
        if table:
            rows = table.select("tr")
            logger.debug(f"Found table '{table_selector}' with {len(rows)} rows")
            for row in rows:
                snapshot = _parse_odds_row_v2(row, race_id, observed_at)
                if snapshot:
                    snapshots.append(snapshot)
            if snapshots:
                return snapshots

    # Pattern 3: Look for HorseList rows anywhere in page
    horse_list_rows = soup.select("tr.HorseList")
    if horse_list_rows:
        logger.debug(f"Found {len(horse_list_rows)} HorseList rows")
        for row in horse_list_rows:
            snapshot = _parse_odds_row_v2(row, race_id, observed_at)
            if snapshot:
                snapshots.append(snapshot)
        if snapshots:
            return snapshots

    # Pattern 4: Generic table row parsing
    all_tables = soup.select("table")
    for table in all_tables:
        rows = table.select("tr")
        temp_snapshots = []
        for row in rows:
            snapshot = _parse_odds_row_v2(row, race_id, observed_at)
            if snapshot:
                temp_snapshots.append(snapshot)
        # If this table has more horses than we found so far, use it
        if len(temp_snapshots) > len(snapshots):
            snapshots = temp_snapshots

    # Pattern 5: Try parsing from embedded JavaScript/JSON data
    if not snapshots:
        snapshots = _parse_odds_from_script(soup, race_id, observed_at)

    return snapshots


def _parse_odds_row_v2(row, race_id: str, observed_at: str) -> Optional[OddsSnapshot]:
    """
    Parse a single row from the odds table.
    Handles multiple HTML structures used by netkeiba.
    """
    try:
        cells = row.select("td")
        if len(cells) < 2:
            return None

        horse_no: Optional[int] = None
        win_odds: Optional[float] = None
        popularity: Optional[int] = None

        # ===== Extract horse number =====

        # Method 1: data-umaban or data-odds-umaban attribute on row
        umaban_attr = row.get("data-umaban") or row.get("data-odds-umaban")
        if umaban_attr and str(umaban_attr).isdigit():
            horse_no = int(umaban_attr)

        # Method 2: Umaban cell (usually first or second cell)
        if horse_no is None:
            for cell in cells[:3]:
                cell_classes = cell.get("class", [])
                # Look for Umaban class
                if any("maban" in c.lower() for c in cell_classes):
                    text = cell.get_text(strip=True)
                    if text.isdigit():
                        horse_no = int(text)
                        break
                # Check for span with Umaban class
                umaban_span = cell.select_one("span.Umaban, span.umaban")
                if umaban_span:
                    text = umaban_span.get_text(strip=True)
                    if text.isdigit():
                        horse_no = int(text)
                        break

        # Method 3: First numeric cell (common pattern)
        if horse_no is None:
            for cell in cells[:2]:
                text = cell.get_text(strip=True)
                if text.isdigit() and 1 <= int(text) <= 18:
                    horse_no = int(text)
                    break

        if horse_no is None:
            return None

        # ===== Extract odds and popularity =====

        for cell in cells:
            cell_classes = " ".join(cell.get("class", []))
            cell_text = cell.get_text(strip=True)

            # Look for odds value
            if win_odds is None:
                # Check for Odds class
                if "odds" in cell_classes.lower():
                    try:
                        val = float(cell_text.replace(",", ""))
                        if 1.0 <= val <= 9999.0:
                            win_odds = val
                            continue
                    except ValueError:
                        pass

                # Check for span with Odds class inside cell
                odds_span = cell.select_one("span.Odds, span.odds")
                if odds_span:
                    try:
                        val = float(odds_span.get_text(strip=True).replace(",", ""))
                        if 1.0 <= val <= 9999.0:
                            win_odds = val
                            continue
                    except ValueError:
                        pass

            # Look for popularity value
            if popularity is None:
                if "popular" in cell_classes.lower() or "ninki" in cell_classes.lower():
                    pop_match = re.search(r"(\d+)", cell_text)
                    if pop_match:
                        popularity = int(pop_match.group(1))
                        continue

                # Check for "人気" text
                if "人気" in cell_text:
                    pop_match = re.search(r"(\d+)", cell_text)
                    if pop_match:
                        popularity = int(pop_match.group(1))
                        continue

        # Try position-based extraction if class-based didn't work
        if win_odds is None:
            for cell in cells[1:]:
                cell_text = cell.get_text(strip=True)
                # Skip if it contains Japanese characters (likely horse name)
                if re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", cell_text):
                    continue
                try:
                    val = float(cell_text.replace(",", ""))
                    if 1.0 <= val <= 9999.0:
                        win_odds = val
                        break
                except ValueError:
                    continue

        # Must have at least odds or popularity to be valid
        if win_odds is None and popularity is None:
            return None

        return OddsSnapshot(
            race_id=race_id,
            horse_no=horse_no,
            observed_at=observed_at,
            win_odds=win_odds,
            popularity=popularity,
            source="netkeiba",
        )

    except Exception as e:
        logger.debug(f"Failed to parse odds row: {e}")
        return None


def _parse_odds_from_script(soup: BeautifulSoup, race_id: str, observed_at: str) -> List[OddsSnapshot]:
    """Try to parse odds from embedded JavaScript data."""
    import json

    snapshots = []

    for script in soup.select("script"):
        script_text = script.get_text()

        # Look for odds data patterns in JavaScript
        if "oddsList" in script_text or '"odds"' in script_text or "'odds'" in script_text:
            # Pattern 1: oddsList = [...];
            json_match = re.search(r"oddsList\s*=\s*(\[.*?\]);", script_text, re.DOTALL)
            if json_match:
                try:
                    odds_data = json.loads(json_match.group(1))
                    for item in odds_data:
                        if isinstance(item, dict):
                            horse_no = item.get("umaban") or item.get("horse_no") or item.get("no")
                            win_odds = item.get("odds") or item.get("win_odds") or item.get("tanso")
                            popularity = item.get("ninki") or item.get("popularity") or item.get("pop")

                            if horse_no:
                                snapshots.append(OddsSnapshot(
                                    race_id=race_id,
                                    horse_no=int(horse_no),
                                    observed_at=observed_at,
                                    win_odds=float(win_odds) if win_odds else None,
                                    popularity=int(popularity) if popularity else None,
                                    source="netkeiba",
                                ))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    logger.debug(f"Failed to parse JSON odds data: {e}")

            # Pattern 2: Inline object with odds data
            obj_match = re.search(r"\{[^{}]*\"odds\"[^{}]*\}", script_text)
            if obj_match and not snapshots:
                try:
                    obj = json.loads(obj_match.group(0))
                    # Handle various structures
                    if isinstance(obj.get("odds"), list):
                        for i, o in enumerate(obj["odds"], 1):
                            snapshots.append(OddsSnapshot(
                                race_id=race_id,
                                horse_no=i,
                                observed_at=observed_at,
                                win_odds=float(o) if o else None,
                                popularity=None,
                                source="netkeiba",
                            ))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass

    return snapshots


# ============================================================
# Odds Fetcher
# ============================================================


class OddsFetcher:
    """Fetches and stores odds snapshots from netkeiba with safety checks."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        sleep_min: float = DEFAULT_SLEEP_MIN,
        sleep_max: float = DEFAULT_SLEEP_MAX,
        save_html_dir: Optional[str] = None,
    ):
        """
        Args:
            conn: SQLite database connection
            sleep_min: Minimum sleep between requests (seconds)
            sleep_max: Maximum sleep between requests (seconds)
            save_html_dir: Directory to save HTML for debugging (None = don't save)
        """
        self.conn = conn
        self.client = OddsHttpClient(sleep_min=sleep_min, sleep_max=sleep_max)
        self.upsert_helper = UpsertHelper(conn)
        self.save_html_dir = save_html_dir

        # Ensure odds_snapshots table exists
        run_migrations(conn)

        # Create HTML save directory if specified
        if self.save_html_dir:
            Path(self.save_html_dir).mkdir(parents=True, exist_ok=True)

    def get_expected_horse_count(self, race_id: str) -> Optional[int]:
        """
        Get expected number of horses from race_results table.

        This is the authoritative source for horse count, more reliable than races.head_count.
        """
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM race_results WHERE race_id = ?",
            (race_id,)
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] > 0 else None

    def fetch_odds_for_race(self, race_id: str) -> Tuple[List[OddsSnapshot], Optional[str]]:
        """
        Fetch odds for a single race with retry.

        Args:
            race_id: 12-digit race ID

        Returns:
            Tuple of (list of snapshots, HTML content for debugging)
        """
        if not re.match(r"^\d{12}$", race_id):
            raise ValueError(f"Invalid race_id format: {race_id}")

        logger.info(f"Fetching odds for race {race_id}")

        observed_at = datetime.now().isoformat()

        # Retry loop for parsing (in case of transient HTML issues)
        for attempt in range(MAX_RETRIES + 1):
            html, status = self.client.fetch_odds_page(race_id)

            if html is None:
                logger.warning(f"Failed to fetch odds page for {race_id}: HTTP {status}")
                if attempt < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.info(f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    continue
                return [], None

            # Save HTML for debugging if requested
            if self.save_html_dir:
                html_path = Path(self.save_html_dir) / f"odds_html_{race_id}.html"
                html_path.write_text(html, encoding="utf-8")
                logger.info(f"Saved HTML to {html_path}")

            # Parse odds
            snapshots = parse_odds_page(html, race_id, observed_at)

            # Debug summary
            if snapshots:
                horse_nos = [s.horse_no for s in snapshots]
                logger.debug(
                    f"Parsed {len(snapshots)} entries for {race_id}: "
                    f"horse_no min={min(horse_nos)} max={max(horse_nos)}"
                )
            else:
                logger.debug(f"No odds parsed for {race_id}")

            return snapshots, html

        return [], None

    def fetch_and_store_odds(self, race_id: str) -> int:
        """
        Fetch odds for a race and store in database with safety checks.

        SAFETY: Only stores if parsed count matches expected count from race_results.
        This prevents storing incomplete data (e.g., only 8 horses instead of 16).

        Args:
            race_id: 12-digit race ID

        Returns:
            Number of snapshots stored (0 if failed or incomplete)
        """
        # Get expected count from race_results
        expected = self.get_expected_horse_count(race_id)
        if expected is None:
            logger.warning(f"[{race_id}] No race_results found, cannot validate - skipping")
            return 0

        # Fetch with retry
        for attempt in range(MAX_RETRIES + 1):
            snapshots, html = self.fetch_odds_for_race(race_id)

            if not snapshots:
                logger.warning(f"[{race_id}] No odds parsed (attempt {attempt + 1})")
                if attempt < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                return 0

            parsed_count = len(snapshots)

            # Safety check: parsed count must match expected
            if parsed_count != expected:
                logger.warning(
                    f"[{race_id}] Count mismatch: parsed={parsed_count}, expected={expected} - "
                    f"NOT STORING (attempt {attempt + 1}/{MAX_RETRIES + 1})"
                )
                if attempt < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.info(f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    continue
                # Final failure - don't store incomplete data
                return 0

            # Success - store the data
            rows = [
                {
                    "race_id": s.race_id,
                    "horse_no": s.horse_no,
                    "observed_at": s.observed_at,
                    "win_odds": s.win_odds,
                    "popularity": s.popularity,
                    "source": s.source,
                }
                for s in snapshots
            ]

            count = self.upsert_helper.upsert_odds_snapshots(rows)
            logger.info(f"[{race_id}] Stored {count} odds snapshots (expected={expected})")
            return count

        return 0

    def fetch_races_for_date(self, date_str: str) -> List[str]:
        """Get list of race IDs for a given date."""
        cursor = self.conn.execute(
            "SELECT race_id FROM races WHERE date = ? ORDER BY race_id",
            (date_str,)
        )
        return [row[0] for row in cursor.fetchall()]

    def fetch_all_for_date(self, date_str: str) -> Dict[str, int]:
        """
        Fetch odds for all races on a given date.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            Dict mapping race_id to stored count (0 for failures)
        """
        race_ids = self.fetch_races_for_date(date_str)

        if not race_ids:
            logger.warning(f"No races found for date {date_str}")
            return {}

        logger.info(f"Fetching odds for {len(race_ids)} races on {date_str}")

        results: Dict[str, int] = {}
        total_stored = 0
        success_count = 0
        fail_count = 0

        for race_id in race_ids:
            try:
                count = self.fetch_and_store_odds(race_id)
                results[race_id] = count
                total_stored += count
                if count > 0:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"[{race_id}] Unexpected error: {e}")
                results[race_id] = 0
                fail_count += 1

        logger.info(
            f"Summary for {date_str}: "
            f"success={success_count}, failed={fail_count}, total_stored={total_stored}"
        )
        return results

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# CLI
# ============================================================


def main() -> int:
    if not _DEPS_AVAILABLE:
        print("Required packages: pip install requests beautifulsoup4")
        return 1

    parser = argparse.ArgumentParser(
        description="Fetch and store pre-race odds snapshots with safety checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch odds for a single race
    python scripts/fetch_odds_snapshots.py --race-id 202406050811

    # Fetch odds for all races on a date
    python scripts/fetch_odds_snapshots.py --date 2024-12-28

    # Fetch odds for tomorrow's races
    python scripts/fetch_odds_snapshots.py --tomorrow

    # Debug: save HTML files
    python scripts/fetch_odds_snapshots.py --race-id 202406050811 --save-html

    # Dry run (show what would be fetched)
    python scripts/fetch_odds_snapshots.py --date 2024-12-28 --dry-run

Safety Features:
    - Only stores if parsed horse count matches race_results count
    - Retries up to 2 times with exponential backoff
    - Logs warnings for any mismatches
""",
    )

    parser.add_argument(
        "--db",
        type=str,
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)",
    )
    parser.add_argument(
        "--race-id",
        type=str,
        help="Fetch odds for a specific race ID",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Fetch odds for all races on this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--tomorrow",
        action="store_true",
        help="Fetch odds for tomorrow's races",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without storing",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save fetched HTML to artifacts/ directory for debugging",
    )
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=DEFAULT_SLEEP_MIN,
        help=f"Minimum sleep between requests (default: {DEFAULT_SLEEP_MIN})",
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=DEFAULT_SLEEP_MAX,
        help=f"Maximum sleep between requests (default: {DEFAULT_SLEEP_MAX})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate arguments
    if not any([args.race_id, args.date, args.tomorrow]):
        parser.error("One of --race-id, --date, or --tomorrow is required")

    # Resolve database path
    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return 1

    # HTML save directory
    save_html_dir = "artifacts" if args.save_html else None

    logger.info("=" * 60)
    logger.info("Odds Snapshot Fetcher (with safety checks)")
    logger.info("=" * 60)
    logger.info(f"Database: {db_path}")
    if save_html_dir:
        logger.info(f"HTML save directory: {save_html_dir}")

    conn = sqlite3.connect(db_path)

    try:
        with OddsFetcher(
            conn,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            save_html_dir=save_html_dir,
        ) as fetcher:
            if args.race_id:
                # Single race
                expected = fetcher.get_expected_horse_count(args.race_id)
                logger.info(f"Expected horses (from race_results): {expected}")

                if args.dry_run:
                    logger.info(f"[DRY-RUN] Would fetch odds for race {args.race_id}")
                    return 0

                count = fetcher.fetch_and_store_odds(args.race_id)
                if count > 0:
                    logger.info(f"SUCCESS: Stored {count} odds snapshots")
                else:
                    logger.warning("FAILED: No odds stored (check warnings above)")
                    return 1

            elif args.date or args.tomorrow:
                # Date-based
                if args.tomorrow:
                    target_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    target_date = args.date

                logger.info(f"Target date: {target_date}")

                if args.dry_run:
                    race_ids = fetcher.fetch_races_for_date(target_date)
                    logger.info(f"[DRY-RUN] Would fetch odds for {len(race_ids)} races:")
                    for rid in race_ids:
                        expected = fetcher.get_expected_horse_count(rid)
                        logger.info(f"  - {rid} (expected: {expected} horses)")
                    return 0

                results = fetcher.fetch_all_for_date(target_date)
                total = sum(results.values())
                success = sum(1 for c in results.values() if c > 0)
                failed = sum(1 for c in results.values() if c == 0)
                logger.info(f"Total: Stored {total} odds snapshots ({success} races OK, {failed} failed)")

        logger.info("=" * 60)
        logger.info("Done")
        logger.info("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
