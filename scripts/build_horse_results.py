#!/usr/bin/env python3
"""
build_horse_results.py

馬の過去成績を時系列で保持する horse_results テーブルを構築する。

Usage:
    python scripts/build_horse_results.py --db netkeiba.db
"""

import argparse
import logging
import os
import sqlite3
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


HORSE_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS horse_results (
    horse_id          TEXT NOT NULL,
    race_id           TEXT NOT NULL,
    race_date         TEXT,
    place             TEXT,
    course_type       TEXT,
    distance          INTEGER,
    track_condition   TEXT,
    field_size        INTEGER,
    race_class        TEXT,
    grade             TEXT,
    finish_order      INTEGER,
    finish_status     TEXT,
    frame_no          INTEGER,
    horse_no          INTEGER,
    time_sec          REAL,
    last_3f           REAL,
    win_odds          REAL,
    popularity        INTEGER,
    body_weight       INTEGER,
    body_weight_diff  INTEGER,
    prize_money       REAL,
    start_index       INTEGER,
    PRIMARY KEY (horse_id, race_id)
);
"""


def load_horse_race_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    races と race_results を JOIN して馬の全出走履歴を取得する。
    """
    query = """
    SELECT
        rr.horse_id,
        rr.race_id,
        r.date AS race_date,
        r.place,
        r.course_type,
        r.distance,
        r.track_condition,
        r.head_count AS field_size,
        r.race_class,
        r.grade,
        rr.finish_order,
        rr.finish_status,
        rr.frame_no,
        rr.horse_no,
        rr.time_sec,
        rr.last_3f,
        rr.win_odds,
        rr.popularity,
        rr.body_weight,
        rr.body_weight_diff,
        rr.prize_money
    FROM race_results AS rr
    INNER JOIN races AS r
        ON rr.race_id = r.race_id
    WHERE r.date IS NOT NULL
    ORDER BY rr.horse_id, r.date
    """
    df = pd.read_sql_query(query, conn)
    logger.info(f"Loaded {len(df):,} horse race records from races + race_results")
    return df


def assign_start_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとに race_date でソートし、start_index (1, 2, 3, ...) を付与する。
    """
    df = df.copy()
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
    df["start_index"] = df.groupby("horse_id").cumcount() + 1
    logger.info("Assigned start_index to each horse's race history")
    return df


def create_horse_results_table(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """
    horse_results テーブルを DROP → CREATE → INSERT する。
    """
    cursor = conn.cursor()

    # DROP existing table
    cursor.execute("DROP TABLE IF EXISTS horse_results")
    logger.info("Dropped existing horse_results table (if any)")

    # CREATE table
    cursor.execute(HORSE_RESULTS_DDL)
    logger.info("Created horse_results table")

    # INSERT data
    df.to_sql("horse_results", conn, if_exists="append", index=False)
    logger.info(f"Inserted {len(df):,} rows into horse_results")

    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build horse_results table from races and race_results"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"Using database: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Load race history
        df = load_horse_race_history(conn)

        if df.empty:
            logger.warning("No race records found. Exiting without creating horse_results table.")
            return

        # Assign start_index
        df = assign_start_index(df)

        # Create table and insert
        create_horse_results_table(conn, df)

        logger.info(f"✅ horse_results table built successfully: {len(df):,} rows")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
