#!/usr/bin/env python3
"""
upsert.py - Idempotent UPSERT Functions for keiba-yosou

Provides standardized UPSERT operations for all tables with:
- Batch processing via executemany
- Transaction management
- Optional "don't overwrite NULL" mode
- Proper conflict handling

Table Configurations:
- races: PK=race_id, update all cols on conflict
- race_results: PK=(race_id, horse_id), update all cols
- horse_results: PK=(horse_id, race_id), update all cols
- horse_laps: PK=(race_id, horse_id, section_m), update time_sec, position
- lap_times: PK=(race_id, lap_index), update distance_m, time_sec
- payouts: PK=(race_id, bet_type, combination), update payout, popularity
- corners: PK=race_id, update all corners
- short_comments: PK=(race_id, horse_id), update all
- feature_table: PK=(race_id, horse_id), update all
- feature_table_v2: UNIQUE=(race_id, horse_id), update all
- feature_table_v3: UNIQUE=(race_id, horse_id), update all

Usage:
    from src.db.upsert import UpsertHelper, upsert_dataframe

    helper = UpsertHelper(conn)
    helper.upsert_races([race1, race2, ...])
    helper.upsert_race_results([result1, result2, ...])

    # Or for DataFrames (feature tables)
    upsert_dataframe(conn, df, "feature_table_v3", ["race_id", "horse_id"])
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple, Sequence
from dataclasses import dataclass, fields, asdict
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Table Configuration
# ============================================================

@dataclass
class TableConfig:
    """Configuration for a table's UPSERT behavior."""
    table_name: str
    primary_key: List[str]  # Columns for ON CONFLICT
    update_columns: Optional[List[str]] = None  # None = all non-PK columns
    exclude_from_update: Optional[List[str]] = None  # Columns to never update
    has_updated_at: bool = False  # Auto-update updated_at column
    has_created_at: bool = False  # Don't overwrite created_at on update


# Pre-defined configurations for all tables
TABLE_CONFIGS = {
    "races": TableConfig(
        table_name="races",
        primary_key=["race_id"],
        exclude_from_update=["created_at"],
        has_updated_at=True,
        has_created_at=True,
    ),
    "race_results": TableConfig(
        table_name="race_results",
        primary_key=["race_id", "horse_id"],
    ),
    "horse_results": TableConfig(
        table_name="horse_results",
        primary_key=["horse_id", "race_id"],
    ),
    "horse_laps": TableConfig(
        table_name="horse_laps",
        primary_key=["race_id", "horse_id", "section_m"],
    ),
    "lap_times": TableConfig(
        table_name="lap_times",
        primary_key=["race_id", "lap_index"],
    ),
    "payouts": TableConfig(
        table_name="payouts",
        primary_key=["race_id", "bet_type", "combination"],
    ),
    "corners": TableConfig(
        table_name="corners",
        primary_key=["race_id"],
    ),
    "short_comments": TableConfig(
        table_name="short_comments",
        primary_key=["race_id", "horse_id"],
    ),
    "feature_table": TableConfig(
        table_name="feature_table",
        primary_key=["race_id", "horse_id"],
        exclude_from_update=["created_at"],
        has_updated_at=True,
        has_created_at=True,
    ),
    "feature_table_v2": TableConfig(
        table_name="feature_table_v2",
        primary_key=["race_id", "horse_id"],
        exclude_from_update=["created_at"],
        has_updated_at=True,
        has_created_at=True,
    ),
    "feature_table_v3": TableConfig(
        table_name="feature_table_v3",
        primary_key=["race_id", "horse_id"],
        exclude_from_update=["created_at"],
        has_updated_at=True,
        has_created_at=True,
    ),
    "odds_snapshots": TableConfig(
        table_name="odds_snapshots",
        primary_key=["race_id", "horse_no", "observed_at"],
        exclude_from_update=["created_at"],
        has_updated_at=True,
        has_created_at=True,
    ),
}


# ============================================================
# Helper Functions
# ============================================================


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get list of column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def safe_value(v: Any) -> Any:
    """Convert a value to a SQLite-compatible type."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return int(v)
    if pd.isna(v):
        return None
    return v


def build_upsert_sql(
    table_name: str,
    columns: List[str],
    conflict_columns: List[str],
    update_columns: Optional[List[str]] = None,
    skip_null_updates: bool = False,
) -> str:
    """
    Build an UPSERT SQL statement.

    Args:
        table_name: Table name
        columns: All columns to insert
        conflict_columns: Columns for ON CONFLICT
        update_columns: Columns to update on conflict (None = all non-conflict)
        skip_null_updates: If True, use COALESCE to preserve existing non-NULL values

    Returns:
        SQL string with placeholders
    """
    if update_columns is None:
        update_columns = [c for c in columns if c not in conflict_columns]

    col_list = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    conflict_list = ", ".join(conflict_columns)

    if not update_columns:
        # No columns to update - just ignore conflicts
        return f"""
            INSERT INTO {table_name} ({col_list})
            VALUES ({placeholders})
            ON CONFLICT({conflict_list}) DO NOTHING
        """

    if skip_null_updates:
        # Use COALESCE to preserve existing non-NULL values
        update_set = ", ".join([
            f"{c} = COALESCE(excluded.{c}, {table_name}.{c})"
            for c in update_columns
        ])
    else:
        update_set = ", ".join([
            f"{c} = excluded.{c}"
            for c in update_columns
        ])

    return f"""
        INSERT INTO {table_name} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT({conflict_list}) DO UPDATE SET
            {update_set}
    """


# ============================================================
# UpsertHelper Class
# ============================================================


class UpsertHelper:
    """
    Helper class for performing UPSERT operations on keiba-yosou tables.

    Features:
    - Batch processing with executemany
    - Transaction management
    - Configurable per-table behavior
    - Optional "don't overwrite NULL" mode
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        batch_size: int = 1000,
        skip_null_updates: bool = False,
    ):
        """
        Args:
            conn: SQLite connection
            batch_size: Number of rows to process in each batch
            skip_null_updates: If True, don't overwrite existing non-NULL values with NULL
        """
        self.conn = conn
        self.batch_size = batch_size
        self.skip_null_updates = skip_null_updates

    def upsert_rows(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
        config: Optional[TableConfig] = None,
    ) -> int:
        """
        UPSERT multiple rows into a table.

        Args:
            table_name: Target table
            rows: List of dicts with column:value pairs
            config: Optional TableConfig (uses default if not provided)

        Returns:
            Number of rows processed
        """
        if not rows:
            return 0

        config = config or TABLE_CONFIGS.get(table_name)
        if config is None:
            raise ValueError(f"No configuration found for table: {table_name}")

        # Get actual table columns
        table_columns = get_table_columns(self.conn, table_name)
        if not table_columns:
            raise ValueError(f"Table {table_name} not found or has no columns")

        # Determine columns from first row, filtered to existing columns
        row_columns = [c for c in rows[0].keys() if c in table_columns]

        # Determine update columns
        update_columns = [
            c for c in row_columns
            if c not in config.primary_key
            and (config.exclude_from_update is None or c not in config.exclude_from_update)
        ]

        # Add updated_at if table has it
        if config.has_updated_at and "updated_at" in table_columns and "updated_at" not in row_columns:
            row_columns.append("updated_at")
            update_columns.append("updated_at")

        sql = build_upsert_sql(
            table_name,
            row_columns,
            config.primary_key,
            update_columns,
            self.skip_null_updates,
        )

        # Process in batches
        total = 0
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            values_batch = []

            for row in batch:
                values = []
                for col in row_columns:
                    if col == "updated_at" and col not in row:
                        values.append(datetime.now().isoformat())
                    else:
                        values.append(safe_value(row.get(col)))
                values_batch.append(tuple(values))

            try:
                self.conn.executemany(sql, values_batch)
                total += len(batch)
            except sqlite3.Error as e:
                logger.error(f"UPSERT failed for {table_name}: {e}")
                logger.error(f"SQL: {sql}")
                logger.error(f"Sample row: {values_batch[0] if values_batch else 'N/A'}")
                raise

        self.conn.commit()
        logger.debug(f"Upserted {total:,} rows into {table_name}")
        return total

    def upsert_dataclasses(
        self,
        table_name: str,
        items: Sequence[Any],
        config: Optional[TableConfig] = None,
    ) -> int:
        """
        UPSERT dataclass instances into a table.

        Args:
            table_name: Target table
            items: List of dataclass instances
            config: Optional TableConfig

        Returns:
            Number of rows processed
        """
        if not items:
            return 0

        rows = [asdict(item) for item in items]
        return self.upsert_rows(table_name, rows, config)

    # Convenience methods for specific tables

    def upsert_races(self, races: List[Dict[str, Any]]) -> int:
        """UPSERT race data."""
        return self.upsert_rows("races", races)

    def upsert_race_results(self, results: List[Dict[str, Any]]) -> int:
        """UPSERT race result data."""
        return self.upsert_rows("race_results", results)

    def upsert_horse_results(self, results: List[Dict[str, Any]]) -> int:
        """UPSERT horse result data."""
        return self.upsert_rows("horse_results", results)

    def upsert_horse_laps(self, laps: List[Dict[str, Any]]) -> int:
        """UPSERT horse lap data."""
        return self.upsert_rows("horse_laps", laps)

    def upsert_lap_times(self, laps: List[Dict[str, Any]]) -> int:
        """UPSERT lap time data."""
        return self.upsert_rows("lap_times", laps)

    def upsert_payouts(self, payouts: List[Dict[str, Any]]) -> int:
        """UPSERT payout data."""
        return self.upsert_rows("payouts", payouts)

    def upsert_corners(self, corners: List[Dict[str, Any]]) -> int:
        """UPSERT corner data."""
        return self.upsert_rows("corners", corners)

    def upsert_short_comments(self, comments: List[Dict[str, Any]]) -> int:
        """UPSERT short comment data."""
        return self.upsert_rows("short_comments", comments)

    def upsert_odds_snapshots(self, snapshots: List[Dict[str, Any]]) -> int:
        """UPSERT odds snapshot data."""
        return self.upsert_rows("odds_snapshots", snapshots)


# ============================================================
# DataFrame UPSERT Functions
# ============================================================


def upsert_dataframe(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_name: str,
    key_columns: List[str],
    batch_size: int = 1000,
    skip_null_updates: bool = False,
) -> int:
    """
    UPSERT a DataFrame into a table.

    This is the preferred method for feature tables created with pandas.

    Args:
        conn: SQLite connection
        df: DataFrame to upsert
        table_name: Target table
        key_columns: Columns that form the unique key
        batch_size: Batch size for executemany
        skip_null_updates: Don't overwrite existing non-NULL values with NULL

    Returns:
        Number of rows processed
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for {table_name}, nothing to upsert")
        return 0

    # Get actual table columns
    table_columns = get_table_columns(conn, table_name)
    if not table_columns:
        raise ValueError(f"Table {table_name} not found or has no columns")

    # Filter DataFrame columns to only those in the table
    df_columns = [c for c in df.columns if c in table_columns]
    if not df_columns:
        raise ValueError(f"No matching columns between DataFrame and table {table_name}")

    # Ensure key columns are in the DataFrame
    for key in key_columns:
        if key not in df_columns:
            raise ValueError(f"Key column {key} not found in DataFrame")

    # Determine update columns
    update_columns = [c for c in df_columns if c not in key_columns]

    # Add updated_at if present in table
    if "updated_at" in table_columns and "updated_at" not in df_columns:
        df_columns.append("updated_at")
        update_columns.append("updated_at")

    sql = build_upsert_sql(
        table_name,
        df_columns,
        key_columns,
        update_columns,
        skip_null_updates,
    )

    # Convert DataFrame to list of tuples
    total = 0
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        values_batch = []

        for _, row in batch_df.iterrows():
            values = []
            for col in df_columns:
                if col == "updated_at" and col not in df.columns:
                    values.append(datetime.now().isoformat())
                else:
                    values.append(safe_value(row.get(col)))
            values_batch.append(tuple(values))

        try:
            conn.executemany(sql, values_batch)
            total += len(values_batch)
        except sqlite3.Error as e:
            logger.error(f"UPSERT failed for {table_name}: {e}")
            logger.error(f"SQL: {sql}")
            raise

    conn.commit()
    logger.info(f"Upserted {total:,} rows into {table_name}")
    return total


def upsert_feature_table_v2(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    batch_size: int = 1000,
) -> int:
    """UPSERT DataFrame into feature_table_v2."""
    return upsert_dataframe(
        conn, df, "feature_table_v2",
        key_columns=["race_id", "horse_id"],
        batch_size=batch_size,
    )


def upsert_feature_table_v3(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    batch_size: int = 1000,
) -> int:
    """UPSERT DataFrame into feature_table_v3."""
    return upsert_dataframe(
        conn, df, "feature_table_v3",
        key_columns=["race_id", "horse_id"],
        batch_size=batch_size,
    )


def upsert_odds_snapshots_df(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    batch_size: int = 1000,
) -> int:
    """UPSERT DataFrame into odds_snapshots."""
    return upsert_dataframe(
        conn, df, "odds_snapshots",
        key_columns=["race_id", "horse_no", "observed_at"],
        batch_size=batch_size,
    )


# ============================================================
# Create Table with UPSERT Support
# ============================================================


def create_table_from_df(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_name: str,
    key_columns: List[str],
    if_exists: str = "append",  # "fail", "replace", "append"
) -> None:
    """
    Create a table from a DataFrame with proper UNIQUE constraint.

    Unlike pandas to_sql, this creates the UNIQUE INDEX for idempotency.

    Args:
        conn: SQLite connection
        df: DataFrame to create table from
        table_name: Target table name
        key_columns: Columns for UNIQUE constraint
        if_exists: What to do if table exists
    """
    from .schema_migration import index_exists

    # Check if table exists
    cursor = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    table_exists = cursor.fetchone()[0] > 0

    if table_exists:
        if if_exists == "fail":
            raise ValueError(f"Table {table_name} already exists")
        elif if_exists == "replace":
            conn.execute(f"DROP TABLE {table_name}")
            conn.commit()
            table_exists = False

    if not table_exists:
        # Create table using pandas (gets the types right)
        df.head(0).to_sql(table_name, conn, index=False, if_exists="fail")
        conn.commit()

    # Ensure UNIQUE index exists
    index_name = f"idx_{table_name}_pk"
    if not index_exists(conn, index_name):
        key_col_str = ", ".join(key_columns)
        conn.execute(f"CREATE UNIQUE INDEX {index_name} ON {table_name} ({key_col_str})")
        conn.commit()
        logger.info(f"Created UNIQUE INDEX {index_name} on {table_name}")

    # Now UPSERT the data
    if not df.empty:
        upsert_dataframe(conn, df, table_name, key_columns)


# ============================================================
# CLI Testing
# ============================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test UPSERT functionality")
    parser.add_argument("--db", default="netkeiba.db", help="Database path")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    # Simple test
    conn = sqlite3.connect(args.db)
    try:
        # Test building SQL
        sql = build_upsert_sql(
            "feature_table_v3",
            ["race_id", "horse_id", "target_win", "target_in3"],
            ["race_id", "horse_id"],
            ["target_win", "target_in3"],
        )
        print("Generated SQL:")
        print(sql)
    finally:
        conn.close()
