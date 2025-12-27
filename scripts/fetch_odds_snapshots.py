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

【decision_cutoff の考え方】
- デフォルト: 前日21:00 JST
- オッズは observed_at (取得時刻) と共に保存
- 評価時は observed_at が decision_cutoff 以前のスナップショットのみを使用

Usage:
    python scripts/fetch_odds_snapshots.py --race-id 202406050811
    python scripts/fetch_odds_snapshots.py --date 2024-12-28 --db netkeiba.db
    python scripts/fetch_odds_snapshots.py --tomorrow
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.scraper import NetkeibaClient, BASE_URL
from src.db.upsert import UpsertHelper
from src.db.schema_migration import run_migrations, table_exists

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup4 is required. Install with: pip install beautifulsoup4")
    sys.exit(1)


logger = logging.getLogger(__name__)


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
# Odds Page Parser
# ============================================================


def parse_odds_page(html: str, race_id: str, observed_at: str) -> List[OddsSnapshot]:
    """
    Parse odds page HTML and extract win odds for each horse.

    The odds page URL is typically:
    https://race.netkeiba.com/odds/index.html?type=b1&race_id=XXXXXXXXXXXX

    Args:
        html: HTML content of the odds page
        race_id: 12-digit race ID
        observed_at: ISO format timestamp of observation

    Returns:
        List of OddsSnapshot objects
    """
    soup = BeautifulSoup(html, "html.parser")
    snapshots = []

    # Try multiple table selectors for odds
    # Pattern 1: Tanso odds table (単勝オッズ)
    odds_table = soup.select_one("table.Odds_Table, table.RaceOdds_HorseList_Table")

    if not odds_table:
        # Pattern 2: Generic odds list
        odds_table = soup.select_one("div.Odds_HorseList_Wrapper table")

    if not odds_table:
        # Pattern 3: Direct row parsing from page
        rows = soup.select("tr[data-odds-umaban], tr.HorseList")
        if rows:
            for row in rows:
                snapshot = _parse_odds_row_alternate(row, race_id, observed_at)
                if snapshot:
                    snapshots.append(snapshot)
            return snapshots

    if odds_table:
        rows = odds_table.select("tr")
        for row in rows:
            snapshot = _parse_odds_row(row, race_id, observed_at)
            if snapshot:
                snapshots.append(snapshot)

    # If no table found, try parsing from JSON data in script
    if not snapshots:
        snapshots = _parse_odds_from_script(soup, race_id, observed_at)

    return snapshots


def _parse_odds_row(row, race_id: str, observed_at: str) -> Optional[OddsSnapshot]:
    """Parse a single row from the odds table."""
    try:
        cells = row.select("td")
        if len(cells) < 2:
            return None

        # Horse number - try multiple patterns
        horse_no = None

        # Pattern 1: data-umaban attribute
        umaban_attr = row.get("data-umaban") or row.get("data-odds-umaban")
        if umaban_attr:
            horse_no = int(umaban_attr)

        # Pattern 2: First cell contains horse number
        if horse_no is None:
            first_cell = cells[0].get_text(strip=True)
            if first_cell.isdigit():
                horse_no = int(first_cell)

        # Pattern 3: Look for class with number
        if horse_no is None:
            num_cell = row.select_one("td.Num, td.Umaban, span.Umaban")
            if num_cell:
                num_text = num_cell.get_text(strip=True)
                if num_text.isdigit():
                    horse_no = int(num_text)

        if horse_no is None:
            return None

        # Win odds - look for odds value
        win_odds = None
        popularity = None

        for cell in cells:
            cell_class = " ".join(cell.get("class", []))
            cell_text = cell.get_text(strip=True)

            # Look for odds cell
            if "Odds" in cell_class or "odds" in cell_class.lower():
                try:
                    win_odds = float(cell_text.replace(",", ""))
                except ValueError:
                    pass

            # Look for popularity cell
            if "Popular" in cell_class or "Ninki" in cell_class or "人気" in cell_text:
                pop_match = re.search(r"\d+", cell_text)
                if pop_match:
                    popularity = int(pop_match.group())

        # If no odds found in class-based cells, try position-based
        if win_odds is None:
            for i, cell in enumerate(cells[1:], 1):
                cell_text = cell.get_text(strip=True)
                # Skip if it's a horse name (contains kanji)
                if re.search(r"[\u4e00-\u9fff]", cell_text):
                    continue
                try:
                    val = float(cell_text.replace(",", ""))
                    if 1.0 <= val <= 1000.0:  # Reasonable odds range
                        win_odds = val
                        break
                except ValueError:
                    continue

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


def _parse_odds_row_alternate(row, race_id: str, observed_at: str) -> Optional[OddsSnapshot]:
    """Parse odds row with alternate structure."""
    try:
        # Get horse number
        horse_no = None
        umaban_elem = row.select_one("span.Umaban, td.Umaban")
        if umaban_elem:
            horse_no = int(umaban_elem.get_text(strip=True))
        else:
            data_umaban = row.get("data-umaban") or row.get("data-odds-umaban")
            if data_umaban:
                horse_no = int(data_umaban)

        if not horse_no:
            return None

        # Get odds
        odds_elem = row.select_one("span.Odds, td.Odds")
        win_odds = None
        if odds_elem:
            odds_text = odds_elem.get_text(strip=True)
            try:
                win_odds = float(odds_text.replace(",", ""))
            except ValueError:
                pass

        # Get popularity
        popularity = None
        pop_elem = row.select_one("span.Popular, td.Popular, span.Ninki, td.Ninki")
        if pop_elem:
            pop_text = pop_elem.get_text(strip=True)
            pop_match = re.search(r"\d+", pop_text)
            if pop_match:
                popularity = int(pop_match.group())

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
        logger.debug(f"Failed to parse alternate odds row: {e}")
        return None


def _parse_odds_from_script(soup: BeautifulSoup, race_id: str, observed_at: str) -> List[OddsSnapshot]:
    """Try to parse odds from embedded JavaScript data."""
    import json

    snapshots = []

    for script in soup.select("script"):
        script_text = script.get_text()

        # Look for odds data patterns
        if "oddsList" in script_text or "Odds" in script_text:
            # Try to extract JSON data
            json_match = re.search(r"oddsList\s*=\s*(\[.*?\]);", script_text, re.DOTALL)
            if json_match:
                try:
                    odds_data = json.loads(json_match.group(1))
                    for item in odds_data:
                        if isinstance(item, dict):
                            horse_no = item.get("umaban") or item.get("horse_no")
                            win_odds = item.get("odds") or item.get("win_odds")
                            popularity = item.get("ninki") or item.get("popularity")

                            if horse_no:
                                snapshots.append(OddsSnapshot(
                                    race_id=race_id,
                                    horse_no=int(horse_no),
                                    observed_at=observed_at,
                                    win_odds=float(win_odds) if win_odds else None,
                                    popularity=int(popularity) if popularity else None,
                                    source="netkeiba",
                                ))
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug(f"Failed to parse JSON odds: {e}")

    return snapshots


# ============================================================
# Odds Fetcher
# ============================================================


class OddsFetcher:
    """Fetches and stores odds snapshots from netkeiba."""

    # Odds page URL template
    ODDS_URL_TEMPLATE = "https://race.netkeiba.com/odds/index.html"

    def __init__(self, conn: sqlite3.Connection, client: Optional[NetkeibaClient] = None):
        """
        Args:
            conn: SQLite database connection
            client: NetkeibaClient instance (creates new one if None)
        """
        self.conn = conn
        self.client = client or NetkeibaClient()
        self.upsert_helper = UpsertHelper(conn)

        # Ensure odds_snapshots table exists
        run_migrations(conn)

    def fetch_odds_for_race(self, race_id: str) -> List[OddsSnapshot]:
        """
        Fetch odds for a single race.

        Args:
            race_id: 12-digit race ID

        Returns:
            List of OddsSnapshot objects
        """
        if not re.match(r"^\d{12}$", race_id):
            raise ValueError(f"Invalid race_id format: {race_id}")

        logger.info(f"Fetching odds for race {race_id}")

        # Build URL for tanso (単勝) odds
        params = {
            "type": "b1",  # 単勝
            "race_id": race_id,
        }

        observed_at = datetime.now().isoformat()

        try:
            response = self.client.get(self.ODDS_URL_TEMPLATE, params=params)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch odds page: HTTP {response.status_code}")
                return []

            # Decode response (odds page is usually UTF-8)
            html = response.content.decode("utf-8", errors="replace")

            snapshots = parse_odds_page(html, race_id, observed_at)
            logger.info(f"Parsed {len(snapshots)} odds entries for race {race_id}")

            return snapshots

        except Exception as e:
            logger.error(f"Failed to fetch odds for race {race_id}: {e}")
            return []

    def fetch_and_store_odds(self, race_id: str) -> int:
        """
        Fetch odds for a race and store in database.

        Args:
            race_id: 12-digit race ID

        Returns:
            Number of snapshots stored
        """
        snapshots = self.fetch_odds_for_race(race_id)

        if not snapshots:
            logger.warning(f"No odds found for race {race_id}")
            return 0

        # Convert to dict list for UPSERT
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
        logger.info(f"Stored {count} odds snapshots for race {race_id}")
        return count

    def fetch_races_for_date(self, date_str: str) -> List[str]:
        """
        Get list of race IDs for a given date.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            List of race IDs
        """
        # Query races table for the date
        cursor = self.conn.execute(
            "SELECT race_id FROM races WHERE date = ? ORDER BY race_id",
            (date_str,)
        )
        return [row[0] for row in cursor.fetchall()]

    def fetch_all_for_date(self, date_str: str) -> int:
        """
        Fetch odds for all races on a given date.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            Total number of snapshots stored
        """
        race_ids = self.fetch_races_for_date(date_str)

        if not race_ids:
            logger.warning(f"No races found for date {date_str}")
            return 0

        logger.info(f"Fetching odds for {len(race_ids)} races on {date_str}")

        total = 0
        for race_id in race_ids:
            try:
                count = self.fetch_and_store_odds(race_id)
                total += count
            except Exception as e:
                logger.error(f"Failed to fetch odds for {race_id}: {e}")

        logger.info(f"Total: Stored {total} odds snapshots for {date_str}")
        return total

    def close(self):
        """Close the client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# CLI
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and store pre-race odds snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch odds for a single race
    python scripts/fetch_odds_snapshots.py --race-id 202406050811

    # Fetch odds for all races on a date
    python scripts/fetch_odds_snapshots.py --date 2024-12-28

    # Fetch odds for tomorrow's races
    python scripts/fetch_odds_snapshots.py --tomorrow

    # Dry run (show what would be fetched)
    python scripts/fetch_odds_snapshots.py --date 2024-12-28 --dry-run
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
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
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

    logger.info("=" * 60)
    logger.info("Odds Snapshot Fetcher")
    logger.info("=" * 60)
    logger.info(f"Database: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        with OddsFetcher(conn) as fetcher:
            if args.race_id:
                # Single race
                if args.dry_run:
                    logger.info(f"[DRY-RUN] Would fetch odds for race {args.race_id}")
                    return 0

                count = fetcher.fetch_and_store_odds(args.race_id)
                logger.info(f"Stored {count} odds snapshots")

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
                        logger.info(f"  - {rid}")
                    return 0

                count = fetcher.fetch_all_for_date(target_date)
                logger.info(f"Total: Stored {count} odds snapshots")

        logger.info("=" * 60)
        logger.info("Done")
        logger.info("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
