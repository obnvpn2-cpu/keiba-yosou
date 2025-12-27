#!/usr/bin/env python3
"""
fetch_odds_snapshots.py - Fetch and Store Pre-race Odds Snapshots (API方式)

前日締め運用のためのオッズスナップショット取得スクリプト。
netkeiba APIを使用して単勝オッズを全馬分取得し、odds_snapshots テーブルに保存する。

【重要: API方式への移行】
従来のHTMLパースでは左右2列分割(1-8 / 9-16)の問題で8頭しか取得できなかった。
API方式により全馬のオッズを確実に取得する。

API Endpoint:
    https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1
    - type=1: 単勝オッズ

【使用方法】
1. 単一レースのオッズ取得:
   python scripts/fetch_odds_snapshots.py --race-id 202406050811

2. 日付範囲のオッズ取得:
   python scripts/fetch_odds_snapshots.py --date 2024-12-28

3. 明日のレースのオッズ取得 (cron用):
   python scripts/fetch_odds_snapshots.py --tomorrow

【decision_cutoff の考え方】
- デフォルト: 前日21:00 JST
- オッズは observed_at (取得時刻) と共に保存
- 評価時は observed_at が decision_cutoff 以前のスナップショットのみを使用

【popularity の扱い】
- APIがpopularityを返さない場合、win_odds昇順で人気順位を自前計算
- 同オッズの場合は同順位

【安全装置】
- 取得した馬数が race_results の馬数より大幅に少ない場合は警告
- races.head_count の半分以下の場合はWARNログを出力
- リトライ最大2回（指数バックオフ）後もダメならスキップ

Usage:
    python scripts/fetch_odds_snapshots.py --race-id 202406050811
    python scripts/fetch_odds_snapshots.py --date 2024-12-28 --db netkeiba.db
    python scripts/fetch_odds_snapshots.py --tomorrow
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
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.upsert import UpsertHelper

try:
    import requests
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False
    requests = None


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

# API endpoint for win odds
ODDS_API_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"

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
    source: str = "netkeiba_api_type1"


# ============================================================
# HTTP Client for API
# ============================================================


class OddsApiClient:
    """HTTP client for netkeiba odds API with UA rotation, sleep, and retry."""

    def __init__(
        self,
        sleep_min: float = DEFAULT_SLEEP_MIN,
        sleep_max: float = DEFAULT_SLEEP_MAX,
    ):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
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

    def fetch_odds_api(
        self,
        race_id: str,
        odds_type: int = 1,
        timeout: int = 30,
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Fetch odds data from netkeiba API.

        Args:
            race_id: 12-digit race ID
            odds_type: Odds type (1 = 単勝, 2 = 複勝, etc.)
            timeout: Request timeout in seconds

        Returns:
            Tuple of (JSON response dict or None, status_code)
        """
        params = {
            "race_id": race_id,
            "type": str(odds_type),
        }

        last_status = 0

        for attempt in range(MAX_RETRIES + 1):
            self._wait_if_needed()

            ua = self._get_random_ua()
            headers = {
                "User-Agent": ua,
                "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
            }

            if attempt > 0:
                backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info(f"Retry {attempt}/{MAX_RETRIES} after {backoff:.1f}s")
                time.sleep(backoff)

            try:
                logger.debug(f"Fetching API odds for {race_id} type={odds_type} (attempt {attempt + 1})")
                response = self.session.get(
                    ODDS_API_URL,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                self._last_request_time = time.time()
                last_status = response.status_code

                if response.status_code == 200:
                    try:
                        # The API returns JSONP-like format or pure JSON
                        content = response.text.strip()

                        # Handle JSONP wrapper if present (e.g., callback({...}))
                        if content.startswith("callback(") and content.endswith(")"):
                            content = content[9:-1]

                        import json
                        data = json.loads(content)
                        return data, 200
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error for {race_id}: {e}")
                        logger.debug(f"Response content (first 500 chars): {response.text[:500]}")
                        if attempt < MAX_RETRIES:
                            continue
                        return None, 200

                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"HTTP {response.status_code} for {race_id}, will retry")
                    continue

                # Other errors (400, 404, etc.) - don't retry
                logger.warning(f"HTTP {response.status_code} for {race_id}, not retrying")
                return None, response.status_code

            except requests.Timeout:
                logger.warning(f"Timeout fetching API odds for {race_id}")
                last_status = 0
                continue
            except requests.RequestException as e:
                logger.warning(f"Request error for {race_id}: {e}")
                last_status = 0
                continue

        logger.error(f"Failed to fetch API odds for {race_id} after {MAX_RETRIES + 1} attempts")
        return None, last_status

    def close(self):
        self.session.close()


# ============================================================
# API Response Parser
# ============================================================


def parse_api_response(
    data: Dict[str, Any],
    race_id: str,
    observed_at: str,
) -> List[OddsSnapshot]:
    """
    Parse netkeiba odds API response and extract win odds for all horses.

    The API response structure (type=1, 単勝) is typically:
    {
        "data": {
            "odds": {
                "1": {"odds": "1.5", "popular": "1", ...},
                "2": {"odds": "3.2", "popular": "2", ...},
                ...
            }
        }
    }

    Or alternative structure:
    {
        "data": {
            "1": {"odds": "1.5", ...},
            "2": {"odds": "3.2", ...},
            ...
        }
    }

    Args:
        data: Parsed JSON response from API
        race_id: 12-digit race ID
        observed_at: ISO format timestamp of observation

    Returns:
        List of OddsSnapshot objects for all horses
    """
    snapshots: List[OddsSnapshot] = []

    if not data:
        logger.warning(f"[{race_id}] Empty API response")
        return []

    # Extract odds data from various possible structures
    odds_dict: Optional[Dict[str, Any]] = None

    # Try structure 1: data.odds
    if isinstance(data.get("data"), dict):
        if "odds" in data["data"]:
            odds_dict = data["data"]["odds"]
        else:
            # Try structure 2: data itself contains horse numbers
            odds_dict = data["data"]

    # Try structure 3: direct odds at top level
    if odds_dict is None and isinstance(data.get("odds"), dict):
        odds_dict = data["odds"]

    # Try structure 4: data is a list
    if odds_dict is None and isinstance(data.get("data"), list):
        odds_dict = {str(i + 1): item for i, item in enumerate(data["data"]) if isinstance(item, dict)}

    if not odds_dict:
        logger.warning(f"[{race_id}] Could not find odds data in API response")
        logger.debug(f"[{race_id}] API response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        return []

    # Parse each horse's odds
    raw_entries: List[Tuple[int, Optional[float], Optional[int]]] = []

    for key, value in odds_dict.items():
        try:
            # Skip non-numeric keys
            if not str(key).isdigit():
                continue

            horse_no = int(key)

            # Extract odds
            win_odds: Optional[float] = None
            api_popularity: Optional[int] = None

            if isinstance(value, dict):
                # Get odds from various possible field names
                odds_val = value.get("odds") or value.get("tanso") or value.get("win_odds")
                if odds_val is not None:
                    try:
                        win_odds = float(str(odds_val).replace(",", ""))
                    except (ValueError, TypeError):
                        pass

                # Get popularity if available
                pop_val = value.get("popular") or value.get("ninki") or value.get("popularity")
                if pop_val is not None:
                    try:
                        api_popularity = int(pop_val)
                    except (ValueError, TypeError):
                        pass

            elif isinstance(value, (int, float, str)):
                # Direct odds value
                try:
                    win_odds = float(str(value).replace(",", ""))
                except (ValueError, TypeError):
                    pass

            # Only add if we have valid odds
            if win_odds is not None and 1.0 <= win_odds <= 9999.0:
                raw_entries.append((horse_no, win_odds, api_popularity))
            elif win_odds is not None:
                # Odds outside normal range
                logger.debug(f"[{race_id}] Horse {horse_no}: odds {win_odds} outside range, skipping")

        except Exception as e:
            logger.debug(f"[{race_id}] Error parsing horse {key}: {e}")
            continue

    if not raw_entries:
        logger.warning(f"[{race_id}] No valid odds entries parsed from API")
        return []

    # Calculate popularity from odds if not provided by API
    # Sort by odds ascending, assign rank (same odds = same rank)
    sorted_entries = sorted(raw_entries, key=lambda x: (x[1] or 9999.0))

    popularity_map: Dict[int, int] = {}
    current_rank = 1
    prev_odds = None

    for i, (horse_no, win_odds, api_pop) in enumerate(sorted_entries):
        if api_pop is not None:
            # Use API-provided popularity
            popularity_map[horse_no] = api_pop
        else:
            # Calculate from odds ranking
            if win_odds != prev_odds:
                current_rank = i + 1
            popularity_map[horse_no] = current_rank
            prev_odds = win_odds

    # Create snapshot objects
    for horse_no, win_odds, api_pop in raw_entries:
        snapshots.append(OddsSnapshot(
            race_id=race_id,
            horse_no=horse_no,
            observed_at=observed_at,
            win_odds=win_odds,
            popularity=popularity_map.get(horse_no),
            source="netkeiba_api_type1",
        ))

    return snapshots


# ============================================================
# Odds Fetcher
# ============================================================


class OddsFetcher:
    """Fetches and stores odds snapshots from netkeiba API with safety checks."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        sleep_min: float = DEFAULT_SLEEP_MIN,
        sleep_max: float = DEFAULT_SLEEP_MAX,
        save_response_dir: Optional[str] = None,
    ):
        """
        Args:
            conn: SQLite database connection
            sleep_min: Minimum sleep between requests (seconds)
            sleep_max: Maximum sleep between requests (seconds)
            save_response_dir: Directory to save API responses for debugging (None = don't save)
        """
        self.conn = conn
        self.client = OddsApiClient(sleep_min=sleep_min, sleep_max=sleep_max)
        self.upsert_helper = UpsertHelper(conn)
        self.save_response_dir = save_response_dir

        # Ensure odds_snapshots table exists (skip if already exists)
        self._ensure_odds_snapshots_table()

        # Create response save directory if specified
        if self.save_response_dir:
            Path(self.save_response_dir).mkdir(parents=True, exist_ok=True)

    def _ensure_odds_snapshots_table(self) -> None:
        """
        Ensure odds_snapshots table exists.

        Creates the table directly if it doesn't exist, without running full migrations.
        This is more robust for testing and avoids migration errors on partial DBs.
        """
        # Check if table exists
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='odds_snapshots'"
        )
        if cursor.fetchone()[0] > 0:
            return  # Table already exists

        # Create the table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                race_id TEXT NOT NULL,
                horse_no INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                win_odds REAL,
                popularity INTEGER,
                source TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (race_id, horse_no, observed_at)
            )
        """)
        self.conn.commit()
        logger.info("Created odds_snapshots table")

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

    def get_head_count_from_races(self, race_id: str) -> Optional[int]:
        """Get head_count from races table (for comparison)."""
        cursor = self.conn.execute(
            "SELECT head_count FROM races WHERE race_id = ?",
            (race_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def fetch_odds_for_race(self, race_id: str) -> List[OddsSnapshot]:
        """
        Fetch odds for a single race from API.

        Args:
            race_id: 12-digit race ID

        Returns:
            List of OddsSnapshot objects
        """
        if not re.match(r"^\d{12}$", race_id):
            raise ValueError(f"Invalid race_id format: {race_id}")

        logger.info(f"Fetching odds for race {race_id}")

        observed_at = datetime.now().isoformat()

        data, status = self.client.fetch_odds_api(race_id, odds_type=1)

        if data is None:
            logger.warning(f"[{race_id}] API returned no data (HTTP {status})")
            return []

        # Save response for debugging if requested
        if self.save_response_dir:
            import json
            response_path = Path(self.save_response_dir) / f"odds_api_{race_id}.json"
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved API response to {response_path}")

        # Parse response
        snapshots = parse_api_response(data, race_id, observed_at)

        if snapshots:
            horse_nos = [s.horse_no for s in snapshots]
            logger.debug(
                f"[{race_id}] Parsed {len(snapshots)} entries: "
                f"horse_no min={min(horse_nos)} max={max(horse_nos)}"
            )
        else:
            logger.debug(f"[{race_id}] No odds parsed from API")

        return snapshots

    def fetch_and_store_odds(self, race_id: str) -> int:
        """
        Fetch odds for a race and store in database with safety checks.

        SAFETY: Warns if parsed count is much less than expected (< half of head_count).
        Still stores data even if count differs (API may have different timing than race_results).

        Args:
            race_id: 12-digit race ID

        Returns:
            Number of snapshots stored (0 if failed or empty)
        """
        # Get expected count from race_results
        expected = self.get_expected_horse_count(race_id)
        head_count = self.get_head_count_from_races(race_id)

        if expected is None and head_count is None:
            logger.warning(f"[{race_id}] No race_results or races.head_count found - proceeding anyway")

        # Fetch with retry
        for attempt in range(MAX_RETRIES + 1):
            snapshots = self.fetch_odds_for_race(race_id)

            if not snapshots:
                logger.warning(f"[{race_id}] No odds parsed (attempt {attempt + 1})")
                if attempt < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                return 0

            parsed_count = len(snapshots)

            # Safety check: warn if count is much less than expected
            reference_count = expected or head_count
            if reference_count is not None:
                if parsed_count < reference_count // 2:
                    logger.warning(
                        f"[{race_id}] SEVERE: parsed={parsed_count} < half of expected={reference_count} - "
                        f"possible data issue (attempt {attempt + 1})"
                    )
                    if attempt < MAX_RETRIES:
                        backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                        time.sleep(backoff)
                        continue
                elif parsed_count != reference_count:
                    logger.info(
                        f"[{race_id}] Count differs: parsed={parsed_count}, expected={reference_count} - "
                        f"proceeding (minor difference)"
                    )

            # Store the data
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
            logger.info(f"[{race_id}] Stored {count} odds snapshots (expected={reference_count})")
            return count

        return 0

    def fetch_races_for_date(self, date_str: str) -> List[str]:
        """Get list of race IDs for a given date from DB."""
        cursor = self.conn.execute(
            "SELECT race_id FROM races WHERE date = ? ORDER BY race_id",
            (date_str,)
        )
        race_ids = [row[0] for row in cursor.fetchall()]

        if not race_ids:
            logger.warning(
                f"No races found in DB for date {date_str}. "
                f"Note: This script requires races to be already ingested into the database. "
                f"Use 'python -m src.ingestion.ingest_runner --date {date_str}' first if needed."
            )

        return race_ids

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
        print("Required packages: pip install requests")
        return 1

    parser = argparse.ArgumentParser(
        description="Fetch and store pre-race odds snapshots via API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch odds for a single race
    python scripts/fetch_odds_snapshots.py --race-id 202406050811

    # Fetch odds for all races on a date
    python scripts/fetch_odds_snapshots.py --date 2024-12-28

    # Fetch odds for tomorrow's races
    python scripts/fetch_odds_snapshots.py --tomorrow

    # Debug: save API responses
    python scripts/fetch_odds_snapshots.py --race-id 202406050811 --save-response

    # Dry run (show what would be fetched)
    python scripts/fetch_odds_snapshots.py --date 2024-12-28 --dry-run

API Method:
    This script uses the netkeiba API endpoint:
    https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1

    type=1 returns 単勝 (win) odds in JSON format.
    This avoids the HTML parsing issues that caused the 8-horse limit bug.

Safety Features:
    - Warns if parsed count is less than half of expected
    - Retries up to 2 times with exponential backoff
    - Logs detailed info for debugging

Verification SQL:
    -- Check snapshot counts vs expected
    SELECT r.race_id, r.head_count,
           COUNT(os.horse_no) AS snap_cnt,
           (r.head_count - COUNT(os.horse_no)) AS diff
    FROM races r
    LEFT JOIN odds_snapshots os ON os.race_id=r.race_id
    WHERE r.date='2024-12-28'
    GROUP BY r.race_id, r.head_count
    ORDER BY diff DESC, r.race_id;
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
        "--save-response",
        action="store_true",
        help="Save API responses to artifacts/ directory for debugging",
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

    # Response save directory
    save_response_dir = "artifacts" if args.save_response else None

    logger.info("=" * 60)
    logger.info("Odds Snapshot Fetcher (API method)")
    logger.info("=" * 60)
    logger.info(f"Database: {db_path}")
    logger.info(f"API endpoint: {ODDS_API_URL}")
    if save_response_dir:
        logger.info(f"Response save directory: {save_response_dir}")

    conn = sqlite3.connect(db_path)

    try:
        with OddsFetcher(
            conn,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            save_response_dir=save_response_dir,
        ) as fetcher:
            if args.race_id:
                # Single race
                expected = fetcher.get_expected_horse_count(args.race_id)
                head_count = fetcher.get_head_count_from_races(args.race_id)
                logger.info(f"Expected horses (race_results): {expected}, head_count: {head_count}")

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
                        head_count = fetcher.get_head_count_from_races(rid)
                        logger.info(f"  - {rid} (race_results: {expected}, head_count: {head_count})")
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
