# -*- coding: utf-8 -*-
"""
SQLite ストア - レース結果の保存

改善点:
- INSERT 文の列数と VALUES の値数を完全に一致
- created_at / updated_at は DEFAULT で自動管理（INSERT 時に指定不要）
- INSERT OR REPLACE ではなく INSERT ... ON CONFLICT DO UPDATE を使用
- 型変換の安全性を強化（pd.isna 対応）
- レース情報（course_type, distance, track_condition）も保存
- 詳細なログ出力
- 検索機能の追加（get_race_info, get_race_results, get_all_race_ids）
- トランザクション管理の改善
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RaceResultSQLiteStore:
    """
    SQLite にレース結果を保存するための薄いラッパ
    
    Features:
        - races テーブル: レース情報（race_id, race_name, course_type, distance, track_condition）
        - race_results テーブル: 出走馬ごとの結果
        - created_at / updated_at の自動管理
        - INSERT ... ON CONFLICT DO UPDATE による安全な UPSERT
        - 型変換の安全性（int/float/str/None）
        - トランザクション管理
    
    Example:
        >>> with RaceResultSQLiteStore() as store:
        ...     store.insert_race_results(df)
    """

    def __init__(self, db_path: str = "./data/keiba.db") -> None:
        """
        Args:
            db_path: SQLite データベースファイルのパス
        """
        self.db_path = Path(db_path)
        # data フォルダがなければ作る
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """データベースに接続してスキーマを初期化"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            # 外部キー有効化
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Opened SQLite DB: {self.db_path}")
            self._init_schema()

    def close(self) -> None:
        """データベース接続をクローズ"""
        if self.conn is not None:
            self.conn.close()
            logger.info("Closed SQLite DB connection")
            self.conn = None

    # Context manager 対応
    def __enter__(self):
        """Context manager: with 文で使用可能"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager: 自動クローズ"""
        self.close()
        return False

    # ------------------------------------------------------------------
    # スキーマ定義
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        """
        データベーススキーマを初期化
        
        テーブル:
            - races: レース情報
            - race_results: 出走馬ごとの結果
        
        既存のテーブルがある場合は、マイグレーションを実行して新しいカラムを追加
        """
        assert self.conn is not None
        cur = self.conn.cursor()

        # レース単位のテーブル
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS races (
                race_id TEXT PRIMARY KEY,
                race_name TEXT,
                course_type TEXT,
                distance INTEGER,
                track_condition TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )

        # 出走馬ごとの結果テーブル
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS race_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                horse_id TEXT,
                horse_name TEXT,
                jockey_id TEXT,
                jockey_name TEXT,
                trainer_name TEXT,
                position INTEGER,
                frame_no INTEGER,
                horse_no INTEGER,
                sex_age TEXT,
                carried_weight REAL,
                time_str TEXT,
                time_seconds REAL,
                margin TEXT,
                win_odds REAL,
                popularity INTEGER,
                body_weight TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                UNIQUE(race_id, horse_id, horse_no),
                FOREIGN KEY(race_id) REFERENCES races(race_id) ON DELETE CASCADE
            )
            """
        )

        # マイグレーション: 既存のテーブルに新しいカラムを追加
        self._migrate_schema(cur)

        self.conn.commit()
        logger.info("SQLite schema initialized (races, race_results)")

    def _migrate_schema(self, cur) -> None:
        """
        既存のテーブルに新しいカラムを追加（マイグレーション）
        
        Args:
            cur: SQLite カーソル
        """
        # races テーブルのカラム一覧を取得
        cur.execute("PRAGMA table_info(races)")
        races_columns = {row[1] for row in cur.fetchall()}
        
        # course_type カラムがない場合は追加
        if "course_type" not in races_columns:
            logger.info("Migrating: Adding course_type column to races table")
            cur.execute("ALTER TABLE races ADD COLUMN course_type TEXT")
        
        # distance カラムがない場合は追加
        if "distance" not in races_columns:
            logger.info("Migrating: Adding distance column to races table")
            cur.execute("ALTER TABLE races ADD COLUMN distance INTEGER")
        
        # track_condition カラムがない場合は追加
        if "track_condition" not in races_columns:
            logger.info("Migrating: Adding track_condition column to races table")
            cur.execute("ALTER TABLE races ADD COLUMN track_condition TEXT")

    # ------------------------------------------------------------------
    # 型変換ヘルパ
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_int(v) -> Optional[int]:
        """
        安全に int に変換
        
        Args:
            v: 変換する値
        
        Returns:
            int または None
        """
        try:
            if v is None:
                return None
            if pd.isna(v):
                return None
            if isinstance(v, int):
                return v
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            # "1.0" みたいなのも来るので float 経由
            return int(float(s))
        except Exception:
            return None

    @staticmethod
    def _safe_float(v) -> Optional[float]:
        """
        安全に float に変換
        
        Args:
            v: 変換する値
        
        Returns:
            float または None
        """
        try:
            if v is None:
                return None
            if pd.isna(v):
                return None
            if isinstance(v, (float, int)):
                return float(v)
            s = str(v).strip().replace(",", "")
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _safe_str(v) -> Optional[str]:
        """
        安全に str に変換（None や空文字列は None を返す）
        
        Args:
            v: 変換する値
        
        Returns:
            str または None
        """
        try:
            if v is None:
                return None
            if pd.isna(v):
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return s
        except Exception:
            return None

    # ------------------------------------------------------------------
    # DataFrame → DB への保存
    # ------------------------------------------------------------------
    def insert_race_results(self, df: pd.DataFrame) -> int:
        """
        parser_race_result.py が返す DataFrame を受け取り SQLite に保存する。

        期待カラム:
        ['着順', '枠番', '馬番', '馬名', '性齢', '斤量', '騎手', 'タイム', '着差',
         '単勝', '人気', '馬体重', '調教師', 'horse_id', 'jockey_id',
         'race_id', 'race_name', '着順_数値', '単勝_数値', 'タイム秒',
         'course_type', 'distance', 'track_condition']  # レース情報も含む可能性
        
        Args:
            df: レース結果の DataFrame
        
        Returns:
            挿入/更新した行数
        
        Example:
            >>> df = pipeline.scrape_race_result("202301010101")
            >>> store.insert_race_results(df)
            16
        """

        if df.empty:
            logger.warning("insert_race_results: empty DataFrame, skipping")
            return 0

        assert self.conn is not None
        cur = self.conn.cursor()

        # race 単位の情報（race_id, race_name）は全行同じ前提
        race_id = self._safe_str(df["race_id"].iloc[0])
        race_name = self._safe_str(df.get("race_name", pd.Series([None])).iloc[0])
        course_type = self._safe_str(df.get("course_type", pd.Series([None])).iloc[0])
        distance = self._safe_int(df.get("distance", pd.Series([None])).iloc[0])
        track_condition = self._safe_str(df.get("track_condition", pd.Series([None])).iloc[0])

        if not race_id:
            logger.error("race_id is missing in DataFrame")
            return 0

        logger.info(f"Inserting race: {race_id} - {race_name}")

        # races テーブル UPSERT
        cur.execute(
            """
            INSERT INTO races (race_id, race_name, course_type, distance, track_condition)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(race_id) DO UPDATE SET
              race_name = excluded.race_name,
              course_type = excluded.course_type,
              distance = excluded.distance,
              track_condition = excluded.track_condition,
              updated_at = datetime('now', 'localtime')
            """,
            (race_id, race_name, course_type, distance, track_condition),
        )

        inserted = 0
        for idx, row in df.iterrows():
            # DataFrameから値を取得
            horse_id = self._safe_str(row.get("horse_id"))
            jockey_id = self._safe_str(row.get("jockey_id"))
            horse_name = self._safe_str(row.get("馬名"))
            jockey_name = self._safe_str(row.get("騎手"))
            trainer_name = self._safe_str(row.get("調教師"))

            # 着順は数値版を優先、なければ元の値
            position = self._safe_int(row.get("着順_数値", row.get("着順")))
            frame_no = self._safe_int(row.get("枠番"))
            horse_no = self._safe_int(row.get("馬番"))
            sex_age = self._safe_str(row.get("性齢"))
            carried_weight = self._safe_float(row.get("斤量"))
            
            time_str = self._safe_str(row.get("タイム"))
            time_seconds = self._safe_float(row.get("タイム秒"))
            margin = self._safe_str(row.get("着差"))
            
            # オッズは数値版を優先、なければ元の値
            win_odds = self._safe_float(row.get("単勝_数値", row.get("単勝")))
            popularity = self._safe_int(row.get("人気"))
            body_weight = self._safe_str(row.get("馬体重"))

            # ★★★ 修正ポイント ★★★
            # INSERT 文の列リストと VALUES の値数を完全に一致させる
            # created_at と updated_at は DEFAULT で自動設定されるので、INSERT 時に指定しない
            cur.execute(
                """
                INSERT INTO race_results (
                    race_id,
                    horse_id,
                    horse_name,
                    jockey_id,
                    jockey_name,
                    trainer_name,
                    position,
                    frame_no,
                    horse_no,
                    sex_age,
                    carried_weight,
                    time_str,
                    time_seconds,
                    margin,
                    win_odds,
                    popularity,
                    body_weight
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(race_id, horse_id, horse_no) DO UPDATE SET
                    horse_name = excluded.horse_name,
                    jockey_id = excluded.jockey_id,
                    jockey_name = excluded.jockey_name,
                    trainer_name = excluded.trainer_name,
                    position = excluded.position,
                    frame_no = excluded.frame_no,
                    sex_age = excluded.sex_age,
                    carried_weight = excluded.carried_weight,
                    time_str = excluded.time_str,
                    time_seconds = excluded.time_seconds,
                    margin = excluded.margin,
                    win_odds = excluded.win_odds,
                    popularity = excluded.popularity,
                    body_weight = excluded.body_weight,
                    updated_at = datetime('now', 'localtime')
                """,
                (
                    race_id,
                    horse_id,
                    horse_name,
                    jockey_id,
                    jockey_name,
                    trainer_name,
                    position,
                    frame_no,
                    horse_no,
                    sex_age,
                    carried_weight,
                    time_str,
                    time_seconds,
                    margin,
                    win_odds,
                    popularity,
                    body_weight,
                ),
            )
            inserted += 1

        self.conn.commit()
        logger.info(
            f"Inserted/updated {inserted} rows into race_results for race_id={race_id}"
        )
        return inserted

    # ------------------------------------------------------------------
    # 検索機能（オプション）
    # ------------------------------------------------------------------
    def get_race_info(self, race_id: str) -> Optional[Dict[str, Any]]:
        """
        レース情報を取得
        
        Args:
            race_id: レースID
        
        Returns:
            レース情報の辞書、または None
        
        Example:
            >>> race_info = store.get_race_info("202301010101")
            >>> print(race_info['race_name'])
            有馬記念
        """
        assert self.conn is not None
        cur = self.conn.cursor()
        
        cur.execute(
            """
            SELECT race_id, race_name, course_type, distance, track_condition, created_at, updated_at
            FROM races
            WHERE race_id = ?
            """,
            (race_id,)
        )
        
        row = cur.fetchone()
        if row is None:
            return None
        
        return {
            "race_id": row[0],
            "race_name": row[1],
            "course_type": row[2],
            "distance": row[3],
            "track_condition": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    def get_race_results(self, race_id: str) -> pd.DataFrame:
        """
        レース結果を取得
        
        Args:
            race_id: レースID
        
        Returns:
            レース結果の DataFrame
        
        Example:
            >>> df = store.get_race_results("202301010101")
            >>> print(df[['position', 'horse_name', 'jockey_name']])
        """
        assert self.conn is not None
        
        query = """
            SELECT 
                id,
                race_id,
                horse_id,
                horse_name,
                jockey_id,
                jockey_name,
                trainer_name,
                position,
                frame_no,
                horse_no,
                sex_age,
                carried_weight,
                time_str,
                time_seconds,
                margin,
                win_odds,
                popularity,
                body_weight,
                created_at,
                updated_at
            FROM race_results
            WHERE race_id = ?
            ORDER BY position
        """
        
        df = pd.read_sql_query(query, self.conn, params=(race_id,))
        return df

    def get_all_race_ids(self) -> list:
        """
        全てのレースIDを取得
        
        Returns:
            レースIDのリスト
        
        Example:
            >>> race_ids = store.get_all_race_ids()
            >>> print(f"Total races: {len(race_ids)}")
        """
        assert self.conn is not None
        cur = self.conn.cursor()
        
        cur.execute("SELECT race_id FROM races ORDER BY race_id DESC")
        
        return [row[0] for row in cur.fetchall()]


# ------------------------------------------------------------------
# 使用例
# ------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from pathlib import Path
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # テスト用の DataFrame を作成
    test_df = pd.DataFrame({
        '着順': ['1', '2', '3'],
        '枠番': ['1', '2', '3'],
        '馬番': ['1', '2', '3'],
        '馬名': ['テストホース1', 'テストホース2', 'テストホース3'],
        '性齢': ['牡3', '牝4', '牡5'],
        '斤量': [56.0, 54.0, 57.0],
        '騎手': ['テスト騎手1', 'テスト騎手2', 'テスト騎手3'],
        'タイム': ['1:23.4', '1:23.5', '1:23.6'],
        '着差': ['', 'クビ', 'アタマ'],
        '単勝': [2.1, 3.5, 10.2],
        '人気': [1, 2, 5],
        '馬体重': ['502(+2)', '484(-4)', '478(0)'],
        '調教師': ['調教師1', '調教師2', '調教師3'],
        'horse_id': ['2020104385', '2019105509', '2018106234'],
        'jockey_id': ['01168', '05399', '00123'],
        'race_id': ['202301010101', '202301010101', '202301010101'],
        'race_name': ['テストレース', 'テストレース', 'テストレース'],
        '着順_数値': [1, 2, 3],
        '単勝_数値': [2.1, 3.5, 10.2],
        'タイム秒': [83.4, 83.5, 83.6],
        'course_type': ['芝', '芝', '芝'],
        'distance': [2500, 2500, 2500],
        'track_condition': ['良', '良', '良'],
    })
    
    # テスト実行
    print("=" * 80)
    print("SQLite Store テスト")
    print("=" * 80)
    
    with RaceResultSQLiteStore("./data/test_keiba.db") as store:
        # 保存テスト
        count = store.insert_race_results(test_df)
        print(f"\n✅ Inserted {count} rows")
        
        # 取得テスト
        race_info = store.get_race_info("202301010101")
        print(f"\n📊 Race info:")
        for key, value in race_info.items():
            print(f"  {key}: {value}")
        
        # レース結果取得テスト
        df_result = store.get_race_results("202301010101")
        print(f"\n🏇 Race results ({len(df_result)} rows):")
        print(df_result[['position', 'horse_name', 'jockey_name', 'time_seconds']])
        
        # 全レースID取得テスト
        all_race_ids = store.get_all_race_ids()
        print(f"\n📝 Total races in DB: {len(all_race_ids)}")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)
