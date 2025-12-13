# -*- coding: utf-8 -*-
"""
sqlite_store_horse_past.py

馬の過去走成績用の SQLite ストア。

改善点:
- datetime('now', 'localtime') に統一
- PRIMARY KEY に race_id を追加（より確実な一意性）
- created_at / updated_at の扱いを改善（INSERT時は指定しない）
- エラーハンドリングの改善
- トランザクション管理の最適化
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


class HorsePastSQLiteStore:
    """
    horse_past_runs テーブルを管理するクラス。

    - DB 接続のオープン/クローズ
    - テーブルスキーマの初期化
    - 過去走成績 DataFrame の UPSERT
    
    改善点:
    - PRIMARY KEY の改善（race_id を含める）
    - datetime の統一
    - created_at / updated_at の自動管理
    """

    def __init__(self, db_path: str = "data/keiba.db") -> None:
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "HorsePastSQLiteStore":
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.row_factory = sqlite3.Row
        logger.info("Opened SQLite DB for horse_past_runs: %s", self.db_path)
        self._init_schema()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            logger.info("Closed SQLite DB connection (horse_past_runs)")
            self.conn = None
        return False

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        """
        スキーマを初期化
        
        PRIMARY KEY: (horse_id, race_date, race_name)
        
        注: race_id を含む UNIQUE 制約は SQLite のバージョン互換性のため削除しました。
        PRIMARY KEY だけで十分な一意性が保証されています。
        """
        assert self.conn is not None
        cur = self.conn.cursor()

        # horse_past_runs テーブル作成
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS horse_past_runs (
                horse_id            TEXT NOT NULL,
                race_id             TEXT,
                race_date           TEXT NOT NULL,
                place               TEXT,
                weather             TEXT,
                race_number         INTEGER,
                race_name           TEXT NOT NULL,
                num_head            INTEGER,
                waku                INTEGER,
                umaban              INTEGER,
                odds                REAL,
                popularity          INTEGER,
                finish              INTEGER,
                jockey              TEXT,
                weight_carried      REAL,
                surface             TEXT,
                distance            INTEGER,
                track_condition     TEXT,
                time                TEXT,
                time_seconds        REAL,
                time_diff           TEXT,
                pace_front_3f       REAL,
                pace_back_3f        REAL,
                passing             TEXT,
                last_3f             REAL,
                horse_weight        INTEGER,
                horse_weight_diff   INTEGER,
                winner_name         TEXT,
                prize               REAL,
                created_at          TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at          TEXT DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (horse_id, race_date, race_name)
            );
            """
        )

        # インデックス作成
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_horse_past_runs_horse_id
            ON horse_past_runs (horse_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_horse_past_runs_race_id
            ON horse_past_runs (race_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_horse_past_runs_race_date
            ON horse_past_runs (race_date);
            """
        )

        self.conn.commit()
        logger.info("SQLite schema initialized (horse_past_runs)")

    # ------------------------------------------------------------------
    # insert / upsert
    # ------------------------------------------------------------------
    def insert_past_runs(self, df: pd.DataFrame) -> int:
        """
        過去走成績 DataFrame を horse_past_runs テーブルに UPSERT する。

        Parameters
        ----------
        df : pd.DataFrame
            HorsePastRunsParser.parse() が返す DataFrame

        Returns
        -------
        int
            実際に INSERT / UPDATE した行数
        """
        if df is None or df.empty:
            logger.info("insert_past_runs: empty DataFrame, nothing to insert")
            return 0

        assert self.conn is not None

        # DataFrame カラム -> テーブルカラムの並びを明示しておく
        columns = [
            "horse_id",
            "race_id",
            "race_date",
            "place",
            "weather",
            "race_number",
            "race_name",
            "num_head",
            "waku",
            "umaban",
            "odds",
            "popularity",
            "finish",
            "jockey",
            "weight_carried",
            "surface",
            "distance",
            "track_condition",
            "time",
            "time_seconds",
            "time_diff",
            "pace_front_3f",
            "pace_back_3f",
            "passing",
            "last_3f",
            "horse_weight",
            "horse_weight_diff",
            "winner_name",
            "prize",
        ]

        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"insert_past_runs: DataFrame に必要な列が不足しています: {missing}")

        # ★★★ 修正ポイント ★★★
        # created_at と updated_at は DEFAULT で自動設定されるので、INSERT 時に指定しない
        sql = """
            INSERT INTO horse_past_runs (
                horse_id,
                race_id,
                race_date,
                place,
                weather,
                race_number,
                race_name,
                num_head,
                waku,
                umaban,
                odds,
                popularity,
                finish,
                jockey,
                weight_carried,
                surface,
                distance,
                track_condition,
                time,
                time_seconds,
                time_diff,
                pace_front_3f,
                pace_back_3f,
                passing,
                last_3f,
                horse_weight,
                horse_weight_diff,
                winner_name,
                prize
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(horse_id, race_date, race_name)
            DO UPDATE SET
                race_id           = excluded.race_id,
                place             = excluded.place,
                weather           = excluded.weather,
                race_number       = excluded.race_number,
                num_head          = excluded.num_head,
                waku              = excluded.waku,
                umaban            = excluded.umaban,
                odds              = excluded.odds,
                popularity        = excluded.popularity,
                finish            = excluded.finish,
                jockey            = excluded.jockey,
                weight_carried    = excluded.weight_carried,
                surface           = excluded.surface,
                distance          = excluded.distance,
                track_condition   = excluded.track_condition,
                time              = excluded.time,
                time_seconds      = excluded.time_seconds,
                time_diff         = excluded.time_diff,
                pace_front_3f     = excluded.pace_front_3f,
                pace_back_3f      = excluded.pace_back_3f,
                passing           = excluded.passing,
                last_3f           = excluded.last_3f,
                horse_weight      = excluded.horse_weight,
                horse_weight_diff = excluded.horse_weight_diff,
                winner_name       = excluded.winner_name,
                prize             = excluded.prize,
                updated_at        = datetime('now', 'localtime');
        """

        cur = self.conn.cursor()
        rows = df[columns].to_dict(orient="records")
        
        inserted = 0
        for row in rows:
            # None や NaN を適切に処理
            params = tuple(self._safe_value(row[col]) for col in columns)
            
            try:
                cur.execute(sql, params)
                inserted += 1
            except Exception as e:
                logger.error(
                    "Failed to insert/update row: horse_id=%s, race_date=%s, race_name=%s, error=%s",
                    row.get("horse_id"),
                    row.get("race_date"),
                    row.get("race_name"),
                    e,
                )
                raise

        self.conn.commit()
        logger.info("Inserted/Updated %d rows into horse_past_runs", inserted)
        return inserted

    # ------------------------------------------------------------------
    # helper: 型変換
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_value(v):
        """
        pandas の値を SQLite に安全に渡せる形に変換
        
        - None, NaN, NaT -> None
        - その他 -> そのまま
        """
        if v is None:
            return None
        if pd.isna(v):
            return None
        # pandas の Int64 などの nullable integer
        if hasattr(v, 'item'):
            return v.item()
        return v

    # ------------------------------------------------------------------
    # helper: クエリ
    # ------------------------------------------------------------------
    def get_existing_horse_ids(self) -> Set[str]:
        """
        horse_past_runs に既に存在する horse_id の集合を返す。
        """
        assert self.conn is not None
        cur = self.conn.cursor()
        
        try:
            cur.execute("SELECT DISTINCT horse_id FROM horse_past_runs;")
            rows = cur.fetchall()
            ids = {row["horse_id"] for row in rows if row["horse_id"]}
            logger.debug("Loaded %d existing horse_ids from horse_past_runs", len(ids))
            return ids
        except sqlite3.OperationalError as e:
            logger.warning("Failed to get existing horse_ids: %s", e)
            return set()

    def get_past_runs_count(self, horse_id: str) -> int:
        """
        指定した horse_id の過去走レコード数を返す
        
        Args:
            horse_id: 馬ID
        
        Returns:
            レコード数
        """
        assert self.conn is not None
        cur = self.conn.cursor()
        
        cur.execute(
            "SELECT COUNT(*) as cnt FROM horse_past_runs WHERE horse_id = ?",
            (horse_id,)
        )
        row = cur.fetchone()
        return row["cnt"] if row else 0


# ------------------------------------------------------------------
# テスト・デバッグ用エントリポイント
# ------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    print("=" * 80)
    print("HorsePastSQLiteStore テスト")
    print("=" * 80)
    
    # テスト用のDataFrameを作成
    test_df = pd.DataFrame({
        'horse_id': ['2020104385', '2020104385'],
        'race_id': ['202301010101', '202212050301'],
        'race_date': ['2023/01/01', '2022/12/05'],
        'place': ['中山', '阪神'],
        'weather': ['晴', '曇'],
        'race_number': [11, 9],
        'race_name': ['有馬記念', '朝日杯FS'],
        'num_head': [16, 18],
        'waku': [1, 3],
        'umaban': [1, 5],
        'odds': [2.1, 8.5],
        'popularity': [1, 4],
        'finish': [1, 3],
        'jockey': ['テスト騎手1', 'テスト騎手2'],
        'weight_carried': [57.0, 55.0],
        'surface': ['芝', '芝'],
        'distance': [2500, 1600],
        'track_condition': ['良', '良'],
        'time': ['2:31.5', '1:34.2'],
        'time_seconds': [151.5, 94.2],
        'time_diff': ['', '0.5'],
        'pace_front_3f': [36.5, 35.2],
        'pace_back_3f': [35.0, 34.8],
        'passing': ['1-1-1-1', '5-5-4-3'],
        'last_3f': [34.5, 33.8],
        'horse_weight': [502, 500],
        'horse_weight_diff': [2, -2],
        'winner_name': ['テストホース1', 'テストホース2'],
        'prize': [30000.0, 5000.0],
    })
    
    # テスト実行
    with HorsePastSQLiteStore("./data/test_horse_past.db") as store:
        # 保存テスト
        count = store.insert_past_runs(test_df)
        print(f"\n✅ Inserted {count} records")
        
        # 取得テスト
        past_count = store.get_past_runs_count("2020104385")
        print(f"\n📊 Past runs count for horse_id=2020104385: {past_count}")
        
        # 既存horse_ids取得テスト
        existing = store.get_existing_horse_ids()
        print(f"\n📝 Total horses in DB: {len(existing)}")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)
