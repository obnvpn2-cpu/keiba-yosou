"""
netkeiba ingestion パイプライン

db.netkeiba.com からレース結果をスクレイピングし、
SQLite に保存する ingestion パイプライン。
"""

from .models import (
    Race,
    RaceResult,
    Payout,
    Corner,
    LapTime,
    HorseLap,
    HorseShortComment,
    ParsedRaceData,
    JRA_PLACE_CODES,
    PLACE_CODE_MAP,
)
from .scraper import NetkeibaClient, create_session, get_client
from .race_list import fetch_race_ids, fetch_race_ids_by_year, fetch_race_ids_by_month
from .parser import parse_race_page
from .db import Database, init_database, get_database
from .ingest_runner import run_ingestion

__all__ = [
    # Models
    "Race",
    "RaceResult",
    "Payout",
    "Corner",
    "LapTime",
    "HorseLap",
    "HorseShortComment",
    "ParsedRaceData",
    "JRA_PLACE_CODES",
    "PLACE_CODE_MAP",
    # Scraper
    "NetkeibaClient",
    "create_session",
    "get_client",
    # Race list
    "fetch_race_ids",
    "fetch_race_ids_by_year",
    "fetch_race_ids_by_month",
    # Parser
    "parse_race_page",
    # Database
    "Database",
    "init_database",
    "get_database",
    # Runner
    "run_ingestion",
]

__version__ = "1.0.0"
