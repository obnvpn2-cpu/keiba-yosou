"""
netkeiba ingestion パイプライン - データベース操作

SQLite への接続、テーブル作成、データの Upsert を行う。
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from .models import (
    Race,
    RaceResult,
    Payout,
    Corner,
    LapTime,
    HorseLap,
    HorseShortComment,
    ParsedRaceData,
)

logger = logging.getLogger(__name__)

# デフォルトのDBパス
DEFAULT_DB_PATH = Path("netkeiba.db")


# テーブル作成SQL
CREATE_TABLES_SQL = """
-- レース基本情報
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,
    date TEXT,
    place TEXT,
    kai INTEGER,
    nichime INTEGER,
    race_no INTEGER,
    name TEXT,
    grade TEXT,
    race_class TEXT,
    course_type TEXT,
    distance INTEGER,
    course_turn TEXT,
    course_inout TEXT,
    weather TEXT,
    track_condition TEXT,
    start_time TEXT,
    baba_index INTEGER,
    baba_comment TEXT,
    analysis_comment TEXT,
    head_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 出走馬成績
CREATE TABLE IF NOT EXISTS race_results (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    finish_order INTEGER,
    finish_status TEXT,
    frame_no INTEGER,
    horse_no INTEGER,
    horse_name TEXT,
    sex TEXT,
    age INTEGER,
    weight REAL,
    jockey_id TEXT,
    jockey_name TEXT,
    time_str TEXT,
    time_sec REAL,
    margin TEXT,
    passing_order TEXT,
    last_3f REAL,
    win_odds REAL,
    popularity INTEGER,
    body_weight INTEGER,
    body_weight_diff INTEGER,
    time_index INTEGER,
    trainer_id TEXT,
    trainer_name TEXT,
    trainer_region TEXT,
    owner_id TEXT,
    owner_name TEXT,
    prize_money REAL,
    remark_text TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- 払い戻し
CREATE TABLE IF NOT EXISTS payouts (
    race_id TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    payout INTEGER NOT NULL,
    popularity INTEGER,
    PRIMARY KEY (race_id, bet_type, combination)
);

-- コーナー通過順位
CREATE TABLE IF NOT EXISTS corners (
    race_id TEXT PRIMARY KEY,
    corner_1 TEXT,
    corner_2 TEXT,
    corner_3 TEXT,
    corner_4 TEXT
);

-- レース全体ラップタイム
CREATE TABLE IF NOT EXISTS lap_times (
    race_id TEXT NOT NULL,
    lap_index INTEGER NOT NULL,
    distance_m INTEGER NOT NULL,
    time_sec REAL NOT NULL,
    PRIMARY KEY (race_id, lap_index)
);

-- 個別馬ラップタイム（マスター会員限定）
CREATE TABLE IF NOT EXISTS horse_laps (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    section_m INTEGER NOT NULL,
    time_sec REAL,
    position INTEGER,
    PRIMARY KEY (race_id, horse_id, section_m)
);

-- 注目馬の短評（マスター会員限定）
CREATE TABLE IF NOT EXISTS short_comments (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    horse_name TEXT,
    finish_order INTEGER,
    comment TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_place ON races(place);
CREATE INDEX IF NOT EXISTS idx_race_results_horse_id ON race_results(horse_id);
CREATE INDEX IF NOT EXISTS idx_race_results_jockey_id ON race_results(jockey_id);
CREATE INDEX IF NOT EXISTS idx_race_results_trainer_id ON race_results(trainer_id);
CREATE INDEX IF NOT EXISTS idx_horse_laps_horse_id ON horse_laps(horse_id);
"""


class Database:
    """SQLite データベース操作クラス。"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
    
    @property
    def conn(self) -> sqlite3.Connection:
        """データベース接続を取得する（遅延初期化）。"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            # 外部キー制約を有効化
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn
    
    def close(self) -> None:
        """データベース接続を閉じる。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "Database":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def create_tables(self) -> None:
        """全テーブルを作成する。"""
        logger.info(f"Creating tables in {self.db_path}")
        self.conn.executescript(CREATE_TABLES_SQL)
        self.conn.commit()
    
    def save_race_data(self, data: ParsedRaceData) -> None:
        """
        パースしたレースデータを保存する。
        
        Args:
            data: ParsedRaceData
        """
        try:
            self._save_race(data.race)
            self._save_results(data.results)
            self._save_payouts(data.payouts)
            
            if data.corner:
                self._save_corner(data.corner)
            
            self._save_lap_times(data.lap_times)
            self._save_horse_laps(data.horse_laps)
            self._save_short_comments(data.short_comments, data.results)
            
            self.conn.commit()
            logger.debug(f"Saved race data for {data.race.race_id}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to save race data for {data.race.race_id}: {e}")
            raise
    
    def _save_race(self, race: Race) -> None:
        """レース基本情報を保存する。"""
        sql = """
        INSERT OR REPLACE INTO races (
            race_id, date, place, kai, nichime, race_no, name, grade, race_class,
            course_type, distance, course_turn, course_inout, weather, track_condition,
            start_time, baba_index, baba_comment, analysis_comment, head_count,
            updated_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            CURRENT_TIMESTAMP
        )
        """
        self.conn.execute(sql, (
            race.race_id,
            race.date.isoformat() if race.date else None,
            race.place,
            race.kai,
            race.nichime,
            race.race_no,
            race.name,
            race.grade,
            race.race_class,
            race.course_type,
            race.distance,
            race.course_turn,
            race.course_inout,
            race.weather,
            race.track_condition,
            race.start_time,
            race.baba_index,
            race.baba_comment,
            race.analysis_comment,
            race.head_count,
        ))
    
    def _save_results(self, results: list[RaceResult]) -> None:
        """出走馬成績を保存する。"""
        if not results:
            return
        
        sql = """
        INSERT OR REPLACE INTO race_results (
            race_id, horse_id, finish_order, finish_status, frame_no, horse_no,
            horse_name, sex, age, weight, jockey_id, jockey_name,
            time_str, time_sec, margin, passing_order, last_3f,
            win_odds, popularity, body_weight, body_weight_diff, time_index,
            trainer_id, trainer_name, trainer_region, owner_id, owner_name,
            prize_money, remark_text
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?
        )
        """
        for r in results:
            self.conn.execute(sql, (
                r.race_id, r.horse_id, r.finish_order, r.finish_status, r.frame_no, r.horse_no,
                r.horse_name, r.sex, r.age, r.weight, r.jockey_id, r.jockey_name,
                r.time_str, r.time_sec, r.margin, r.passing_order, r.last_3f,
                r.win_odds, r.popularity, r.body_weight, r.body_weight_diff, r.time_index,
                r.trainer_id, r.trainer_name, r.trainer_region, r.owner_id, r.owner_name,
                r.prize_money, r.remark_text,
            ))
    
    def _save_payouts(self, payouts: list[Payout]) -> None:
        """払い戻し情報を保存する。"""
        if not payouts:
            return
        
        sql = """
        INSERT OR REPLACE INTO payouts (
            race_id, bet_type, combination, payout, popularity
        ) VALUES (?, ?, ?, ?, ?)
        """
        for p in payouts:
            self.conn.execute(sql, (
                p.race_id, p.bet_type, p.combination, p.payout, p.popularity,
            ))
    
    def _save_corner(self, corner: Corner) -> None:
        """コーナー通過順位を保存する。"""
        sql = """
        INSERT OR REPLACE INTO corners (
            race_id, corner_1, corner_2, corner_3, corner_4
        ) VALUES (?, ?, ?, ?, ?)
        """
        self.conn.execute(sql, (
            corner.race_id, corner.corner_1, corner.corner_2,
            corner.corner_3, corner.corner_4,
        ))
    
    def _save_lap_times(self, lap_times: list[LapTime]) -> None:
        """レース全体ラップタイムを保存する。"""
        if not lap_times:
            return
        
        sql = """
        INSERT OR REPLACE INTO lap_times (
            race_id, lap_index, distance_m, time_sec
        ) VALUES (?, ?, ?, ?)
        """
        for lt in lap_times:
            self.conn.execute(sql, (
                lt.race_id, lt.lap_index, lt.distance_m, lt.time_sec,
            ))
    
    def _save_horse_laps(self, horse_laps: list[HorseLap]) -> None:
        """個別馬ラップタイムを保存する。

        同じレースIDについては一度全部削除してから入れ直す。
        → 以前のバグで入った変な section_m（3m, 4m など）が残らないようにする。
        """
        if not horse_laps:
            return

        # 対象レースIDをまとめて削除
        race_ids = {hl.race_id for hl in horse_laps}
        for rid in race_ids:
            self.conn.execute(
                "DELETE FROM horse_laps WHERE race_id = ?",
                (rid,),
            )

        sql = """
        INSERT OR REPLACE INTO horse_laps (
            race_id, horse_id, section_m, time_sec, position
        ) VALUES (?, ?, ?, ?, ?)
        """
        for hl in horse_laps:
            self.conn.execute(sql, (
                hl.race_id,
                hl.horse_id,
                hl.section_m,
                hl.time_sec,
                hl.position,
            ))

    def _save_short_comments(
            self,
            comments: list[HorseShortComment],
            results: list[RaceResult],
        ) -> None:
            """短評を保存する（horse_id を結果から補完）。"""
            if not comments:
                return
            
            # 馬名から horse_id へのマッピングを作成
            name_to_horse_id = {r.horse_name: r.horse_id for r in results if r.horse_name}
            
            sql = """
            INSERT OR REPLACE INTO short_comments (
                race_id, horse_id, horse_name, finish_order, comment
            ) VALUES (?, ?, ?, ?, ?)
            """
            for c in comments:
                # horse_id を補完
                horse_id = c.horse_id or name_to_horse_id.get(c.horse_name, "")
                if not horse_id:
                    logger.warning(f"Could not find horse_id for {c.horse_name}")
                    continue
                
                self.conn.execute(sql, (
                    c.race_id, horse_id, c.horse_name, c.finish_order, c.comment,
                ))


    def race_exists(self, race_id: str) -> bool:
        """
        レースが既に存在するかチェックする。
        """
        cursor = self.conn.execute(
            "SELECT 1 FROM races WHERE race_id = ?",
            (race_id,)
        )
        return cursor.fetchone() is not None

    def get_existing_race_ids(self, race_ids: list[str]) -> set[str]:
        """
        指定したレースIDのうち、既にDBに存在するものをセットで返す。
        """
        if not race_ids:
            return set()

        placeholders = ",".join("?" for _ in race_ids)
        sql = f"SELECT race_id FROM races WHERE race_id IN ({placeholders})"
        cursor = self.conn.execute(sql, race_ids)
        return {row[0] for row in cursor.fetchall()}

    def get_race_ids_missing_horse_laps(self, race_ids: list[str]) -> set[str]:
        """horse_laps が1件も入っていないレースIDの集合を返す。"""

        if not race_ids:
            return set()

        placeholders = ",".join("?" for _ in race_ids)
        sql = f"""
            SELECT r.race_id
            FROM races r
            LEFT JOIN (
                SELECT race_id, COUNT(*) AS lap_cnt
                FROM horse_laps
                WHERE race_id IN ({placeholders})
                GROUP BY race_id
            ) hl ON r.race_id = hl.race_id
            WHERE r.race_id IN ({placeholders})
              AND COALESCE(hl.lap_cnt, 0) = 0
        """
        cursor = self.conn.execute(sql, list(race_ids) + list(race_ids))
        return {row[0] for row in cursor.fetchall()}

    def get_race_count(self) -> int:
        """レース数を取得する。"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM races")
        return cursor.fetchone()[0]

    def get_result_count(self) -> int:
        """成績レコード数を取得する。"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM race_results")
        return cursor.fetchone()[0]






@contextmanager
def get_database(db_path: Optional[Path] = None):
    """
    データベース接続のコンテキストマネージャ。
    
    Args:
        db_path: データベースファイルのパス
    
    Yields:
        Database インスタンス
    """
    db = Database(db_path)
    try:
        yield db
    finally:
        db.close()


def init_database(db_path: Optional[Path] = None) -> Database:
    """
    データベースを初期化する。
    
    Args:
        db_path: データベースファイルのパス
    
    Returns:
        Database インスタンス
    """
    db = Database(db_path)
    db.create_tables()
    return db
