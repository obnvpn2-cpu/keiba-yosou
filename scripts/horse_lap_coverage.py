"""horse_laps のカバレッジを確認するユーティリティ。

使用例:

    python scripts/horse_lap_coverage.py --db netkeiba.db
    python scripts/horse_lap_coverage.py --db netkeiba.db --race-ids 202406050811 202408070108
"""

import argparse
import logging
import sqlite3
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_summary(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
    """race_results と horse_laps の突合サマリを返す。"""

    sql = """
        SELECT COUNT(*) AS laps, COUNT(DISTINCT race_id) AS races,
               SUM(CASE WHEN lap_cnt=0 THEN 1 ELSE 0 END) AS races_missing,
               SUM(CASE WHEN lap_horses < result_horses THEN 1 ELSE 0 END) AS races_partial
        FROM (
          SELECT rr.race_id,
                 COUNT(DISTINCT hl.horse_id) AS lap_horses,
                 COUNT(DISTINCT rr.horse_id) AS result_horses,
                 COUNT(hl.horse_id) AS lap_cnt
          FROM race_results rr
          LEFT JOIN horse_laps hl ON rr.race_id = hl.race_id
          GROUP BY rr.race_id
        )
    """
    return conn.execute(sql).fetchone()


def fetch_race_detail(conn: sqlite3.Connection, race_id: str) -> dict:
    """特定レースの head_count / lap_horses / 欠損馬一覧を返す。"""

    head_count = conn.execute(
        "SELECT COUNT(DISTINCT horse_id) FROM race_results WHERE race_id = ?",
        (race_id,),
    ).fetchone()[0]
    lap_horses = conn.execute(
        "SELECT COUNT(DISTINCT horse_id) FROM horse_laps WHERE race_id = ?",
        (race_id,),
    ).fetchone()[0]
    missing = conn.execute(
        """
        SELECT horse_id
        FROM race_results
        WHERE race_id = ?
        EXCEPT
        SELECT horse_id FROM horse_laps WHERE race_id = ?
        ORDER BY horse_id
        """,
        (race_id, race_id),
    ).fetchall()
    return {
        "race_id": race_id,
        "head_count": head_count,
        "lap_horses": lap_horses,
        "missing_horses": [row[0] for row in missing],
    }


def log_detail(conn: sqlite3.Connection, race_ids: Iterable[str]) -> None:
    for rid in race_ids:
        detail = fetch_race_detail(conn, rid)
        logger.info(
            "race_id=%s head_count=%d lap_horses=%d missing=%d",
            detail["race_id"],
            detail["head_count"],
            detail["lap_horses"],
            len(detail["missing_horses"]),
        )
        if detail["missing_horses"]:
            logger.info("  missing horse_ids: %s", ",".join(detail["missing_horses"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="horse_laps coverage checker")
    parser.add_argument("--db", default="netkeiba.db", help="SQLite DB path")
    parser.add_argument("--race-ids", nargs="+", default=None, help="race_id list to detail")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    summary = fetch_summary(conn)
    logger.info(
        "horse_laps summary laps=%d races=%d missing=%d partial=%d",
        summary[0],
        summary[1],
        summary[2],
        summary[3],
    )

    if args.race_ids:
        log_detail(conn, args.race_ids)


if __name__ == "__main__":
    main()
