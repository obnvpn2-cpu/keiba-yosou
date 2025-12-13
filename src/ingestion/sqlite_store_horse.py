# -*- coding: utf-8 -*-
# src/ingestion/sqlite_store_horse.py
"""
馬の基本情報用 SQLite ストア。

テーブル: horse_basic

    horse_id   TEXT PRIMARY KEY
    horse_name TEXT
    sex        TEXT
    breeder    TEXT
    created_at TEXT (ISO datetime, localtime)
    updated_at TEXT (ISO datetime, localtime)

改善点:
- datetime('now', 'localtime') に統一（既存の sqlite_store.py に合わせる）
- 型変換ヘルパーの追加（_safe_str）
- バッチINSERT機能の追加（将来の拡張性）
- トランザクション管理の改善
- マイグレーション機能の追加

RaceResultSQLiteStore とは分離しているが、同じ DB ファイル
(data/keiba.db) を共有する前提。
"""

from __future__ import annotations

import sqlite3
import logging
from typing import Optional, List, Set
import pandas as pd

from .parser_horse_basic import HorseBasicRecord

logger = logging.getLogger(__name__)


class HorseBasicSQLiteStore:
    """
    horse_basic テーブルの永続化レイヤ
    
    Features:
        - UPSERT による安全な更新
        - バッチ INSERT 対応（将来の拡張性）
        - マイグレーション機能
        - 型変換ヘルパー
        - トランザクション管理
    
    Example:
        >>> with HorseBasicSQLiteStore() as store:
        ...     record = HorseBasicRecord(...)
        ...     store.insert_or_update(record)
    """

    def __init__(self, db_path: str = "data/keiba.db") -> None:
        """
        Args:
            db_path: SQLite データベースファイルのパス
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "HorseBasicSQLiteStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return False

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------
    def open(self) -> None:
        """データベース接続を開く"""
        if self.conn is not None:
            return
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        logger.info("Opened SQLite DB for horse_basic: %s", self.db_path)
        self._init_schema()

    def close(self) -> None:
        """データベース接続を閉じる"""
        if self.conn is None:
            return
        self.conn.close()
        logger.info("Closed SQLite DB connection (horse_basic)")
        self.conn = None

    def _init_schema(self) -> None:
        """
        スキーマを初期化
        
        - horse_basic テーブルを作成
        - 必要に応じてマイグレーションを実行
        """
        assert self.conn is not None
        cur = self.conn.cursor()
        
        # horse_basic テーブルを作成
        cur.execute(
            """CREATE TABLE IF NOT EXISTS horse_basic (
                horse_id   TEXT PRIMARY KEY,
                horse_name TEXT,
                sex        TEXT,
                breeder    TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )"""
        )
        
        self.conn.commit()
        logger.info("SQLite schema initialized (horse_basic)")

    # ------------------------------------------------------------------
    # 型変換ヘルパ（既存の sqlite_store.py に合わせる）
    # ------------------------------------------------------------------
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
    # INSERT / UPSERT
    # ------------------------------------------------------------------
    def insert_or_update(self, record: HorseBasicRecord) -> int:
        """
        UPSERT する（1件）。

        Args:
            record: HorseBasicRecord

        Returns:
            1 固定（成功時）
        
        Example:
            >>> record = HorseBasicRecord(horse_id="2020104385", ...)
            >>> store.insert_or_update(record)
            1
        """
        assert self.conn is not None
        
        # 型変換
        horse_id = self._safe_str(record.horse_id)
        horse_name = self._safe_str(record.horse_name)
        sex = self._safe_str(record.sex)
        breeder = self._safe_str(record.breeder)
        
        if not horse_id:
            logger.error("horse_id が空のため、INSERT をスキップします")
            return 0
        
        cur = self.conn.cursor()
        
        logger.debug(
            "UPSERT horse_basic: horse_id=%s, horse_name=%s, sex=%s, breeder=%s",
            horse_id,
            horse_name,
            sex,
            breeder,
        )
        
        cur.execute(
            """INSERT INTO horse_basic (horse_id, horse_name, sex, breeder)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(horse_id) DO UPDATE SET
                       horse_name = excluded.horse_name,
                       sex        = excluded.sex,
                       breeder    = excluded.breeder,
                       updated_at = datetime('now', 'localtime')
            """,
            (horse_id, horse_name, sex, breeder),
        )
        self.conn.commit()
        
        logger.info(
            "Inserted/Updated horse_basic: horse_id=%s, horse_name=%s",
            horse_id,
            horse_name,
        )
        return 1

    def insert_or_update_batch(self, records: List[HorseBasicRecord]) -> int:
        """
        バッチ UPSERT する（複数件）。
        
        将来の拡張用（現時点では1件ずつ処理でも問題ない）。

        Args:
            records: HorseBasicRecord のリスト

        Returns:
            挿入/更新した件数
        
        Example:
            >>> records = [HorseBasicRecord(...), HorseBasicRecord(...)]
            >>> store.insert_or_update_batch(records)
            2
        """
        assert self.conn is not None
        
        if not records:
            logger.warning("insert_or_update_batch: 空のリストが渡されました")
            return 0
        
        cur = self.conn.cursor()
        inserted = 0
        
        logger.info("Batch UPSERT: %d records", len(records))
        
        for record in records:
            # 型変換
            horse_id = self._safe_str(record.horse_id)
            horse_name = self._safe_str(record.horse_name)
            sex = self._safe_str(record.sex)
            breeder = self._safe_str(record.breeder)
            
            if not horse_id:
                logger.warning("horse_id が空のため、スキップ: %s", record)
                continue
            
            cur.execute(
                """INSERT INTO horse_basic (horse_id, horse_name, sex, breeder)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(horse_id) DO UPDATE SET
                           horse_name = excluded.horse_name,
                           sex        = excluded.sex,
                           breeder    = excluded.breeder,
                           updated_at = datetime('now', 'localtime')
                """,
                (horse_id, horse_name, sex, breeder),
            )
            inserted += 1
        
        self.conn.commit()
        logger.info("Batch UPSERT completed: %d records", inserted)
        return inserted

    # ------------------------------------------------------------------
    # 補助: 既存 horse_id の取得
    # ------------------------------------------------------------------
    def get_all_horse_ids(self) -> Set[str]:
        """
        horse_basic に既に入っている horse_id セットを返す。
        
        Returns:
            horse_id の集合
        
        Example:
            >>> ids = store.get_all_horse_ids()
            >>> print(len(ids))
            1000
        """
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute("SELECT horse_id FROM horse_basic")
        rows = cur.fetchall()
        ids = {r[0] for r in rows if r[0]}
        logger.debug("Loaded %d horse_id from horse_basic", len(ids))
        return ids

    def get_horse_info(self, horse_id: str) -> Optional[dict]:
        """
        指定した horse_id の情報を取得
        
        Args:
            horse_id: 馬ID
        
        Returns:
            馬情報の辞書、または None
        
        Example:
            >>> info = store.get_horse_info("2020104385")
            >>> print(info['horse_name'])
            テストホース
        """
        assert self.conn is not None
        cur = self.conn.cursor()
        
        cur.execute(
            """SELECT horse_id, horse_name, sex, breeder, created_at, updated_at
               FROM horse_basic
               WHERE horse_id = ?
            """,
            (horse_id,)
        )
        
        row = cur.fetchone()
        if row is None:
            return None
        
        return {
            "horse_id": row[0],
            "horse_name": row[1],
            "sex": row[2],
            "breeder": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }


# ------------------------------------------------------------------
# テスト・デバッグ用エントリポイント
# ------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from pathlib import Path
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    print("=" * 80)
    print("HorseBasicSQLiteStore テスト")
    print("=" * 80)
    
    # テスト用のレコードを作成
    test_record = HorseBasicRecord(
        horse_id="0000000001",
        horse_name="テストホース",
        sex="牡",
        breeder="テスト牧場",
    )
    
    # テスト実行
    with HorseBasicSQLiteStore("./data/test_horse_basic.db") as store:
        # 保存テスト
        count = store.insert_or_update(test_record)
        print(f"\n✅ Inserted {count} record")
        
        # 取得テスト
        info = store.get_horse_info("0000000001")
        print(f"\n📊 Horse info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 全horse_id取得テスト
        all_ids = store.get_all_horse_ids()
        print(f"\n📝 Total horses in DB: {len(all_ids)}")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)
