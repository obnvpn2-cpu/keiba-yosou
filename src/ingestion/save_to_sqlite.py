import logging
from pathlib import Path

import pandas as pd

from .sqlite_store import RaceResultSQLiteStore

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main():
    configure_logging()

    data_dir = Path("./data")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    csv_files = sorted(data_dir.glob("race_*.csv"))
    if not csv_files:
        logger.warning(f"No race_*.csv files found under {data_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files")

    with RaceResultSQLiteStore("./data/keiba.db") as store:
        total = 0
        for path in csv_files:
            try:
                logger.info(f"Loading {path}")
                df = pd.read_csv(path)
                inserted = store.insert_race_results(df)
                logger.info(f"{path.name}: inserted {inserted} rows")
                total += inserted
            except Exception as e:
                logger.exception(f"Failed to import {path}: {e}")

    logger.info(f"Done. Total inserted rows: {total}")


if __name__ == "__main__":
    main()
