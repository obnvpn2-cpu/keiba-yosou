# -*- coding: utf-8 -*-
"""
sqlite_store_feature.py

feature_table 用の SQLite ストア

テーブル構造:
- race_id, horse_id をPRIMARY KEY
- ターゲット変数（target_win, target_in3, target_value）
- レース側特徴量（course, surface, distance, etc.）
- 馬のグローバル能力指標
- 距離カテゴリ別能力指標
- 直近フォーム
- その他の能力指標
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import sqlite3

logger = logging.getLogger(__name__)


# ==============================================================================
# テーブル作成
# ==============================================================================

def create_table_feature(conn: sqlite3.Connection) -> None:
    """
    feature_table を作成
    
    Args:
        conn: SQLite接続
    """
    cur = conn.cursor()
    
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_table (
            race_id TEXT NOT NULL,
            horse_id TEXT NOT NULL,
            
            -- ターゲット変数
            target_win INTEGER,
            target_in3 INTEGER,
            target_value INTEGER,
            
            -- レース側特徴量
            course TEXT,
            surface TEXT,
            surface_id INTEGER,            -- カテゴリID（0:芝, 1:ダ, 2:障害）
            distance INTEGER,
            distance_cat INTEGER,
            track_condition TEXT,
            track_condition_id INTEGER,    -- カテゴリID（0:良, 1:稍, 2:重, 3:不）
            field_size INTEGER,
            race_class TEXT,
            race_year INTEGER,
            race_month INTEGER,
            
            -- 枠番・馬番
            waku INTEGER,
            umaban INTEGER,
            
            -- 馬体重
            horse_weight INTEGER,
            horse_weight_diff INTEGER,
            
            -- 新馬フラグ
            is_first_run INTEGER,          -- 過去走0なら1
            
            -- 馬のグローバル能力指標
            n_starts_total INTEGER,
            win_rate_total REAL,
            in3_rate_total REAL,
            avg_finish_total REAL,
            std_finish_total REAL,
            
            -- 距離カテゴリ別能力指標
            n_starts_dist_cat INTEGER,
            win_rate_dist_cat REAL,
            in3_rate_dist_cat REAL,
            avg_finish_dist_cat REAL,
            avg_last3f_dist_cat REAL,
            
            -- 直近フォーム
            days_since_last_run INTEGER,
            recent_avg_finish_3 REAL,
            recent_best_finish_3 INTEGER,
            recent_avg_last3f_3 REAL,
            
            -- 馬場状態別能力指標
            n_starts_track_condition INTEGER,
            win_rate_track_condition REAL,
            
            -- コース別能力指標
            n_starts_course INTEGER,
            win_rate_course REAL,

            -- 馬体重統計
            avg_horse_weight REAL,
            
            -- メタ情報
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT DEFAULT (datetime('now', 'localtime')),
            
            PRIMARY KEY (race_id, horse_id)
        );
        """
    )
    
    # インデックス作成
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_feature_table_race_id
        ON feature_table (race_id);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_feature_table_horse_id
        ON feature_table (horse_id);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_feature_table_target_win
        ON feature_table (target_win);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_feature_table_target_value
        ON feature_table (target_value);
        """
    )
    
    conn.commit()
    logger.info("feature_table created (or already exists)")


# ==============================================================================
# データ挿入
# ==============================================================================

def insert_feature_rows(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
) -> int:
    """
    feature_table にデータを挿入（INSERT OR REPLACE）
    
    Args:
        conn: SQLite接続
        df: 特徴量DataFrame
    
    Returns:
        挿入した行数
    """
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame, nothing to insert")
        return 0
    
    # カラムリスト
    columns = [
        "race_id",
        "horse_id",
        "target_win",
        "target_in3",
        "target_value",
        "course",
        "surface",
        "surface_id",  # NEW
        "distance",
        "distance_cat",
        "track_condition",
        "track_condition_id",  # NEW
        "field_size",
        "race_class",
        "race_year",
        "race_month",
        "waku",
        "umaban",
        "horse_weight",
        "horse_weight_diff",
        "is_first_run",  # NEW
        "n_starts_total",
        "win_rate_total",
        "in3_rate_total",
        "avg_finish_total",
        "std_finish_total",
        "n_starts_dist_cat",
        "win_rate_dist_cat",
        "in3_rate_dist_cat",
        "avg_finish_dist_cat",
        "avg_last3f_dist_cat",
        "days_since_last_run",
        "recent_avg_finish_3",
        "recent_best_finish_3",
        "recent_avg_last3f_3",
        "n_starts_track_condition",
        "win_rate_track_condition",
        "n_starts_course",
        "win_rate_course",
        "avg_horse_weight",
    ]
    
    # 不足しているカラムをチェック
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        raise ValueError(f"Missing columns: {missing}")
    
    # INSERT OR REPLACE SQL
    sql = f"""
        INSERT OR REPLACE INTO feature_table (
            {', '.join(columns)}
        )
        VALUES (
            {', '.join(['?'] * len(columns))}
        )
    """
    
    cur = conn.cursor()
    rows = df[columns].to_dict(orient="records")
    
    inserted = 0
    for row in rows:
        # None や NaN を適切に処理
        params = tuple(_safe_value(row[col]) for col in columns)
        
        try:
            cur.execute(sql, params)
            inserted += 1
        except Exception as e:
            logger.error(
                "Failed to insert row: race_id=%s, horse_id=%s, error=%s",
                row.get("race_id"),
                row.get("horse_id"),
                e,
            )
            raise
    
    conn.commit()
    logger.debug("Inserted %d rows into feature_table", inserted)
    
    return inserted


# ==============================================================================
# ヘルパー関数
# ==============================================================================

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


# ==============================================================================
# クエリ関数
# ==============================================================================

def get_feature_count(conn: sqlite3.Connection) -> int:
    """
    feature_table の総行数を取得
    
    Args:
        conn: SQLite接続
    
    Returns:
        行数
    """
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM feature_table")
    row = cur.fetchone()
    return row[0] if row else 0


def get_race_count(conn: sqlite3.Connection) -> int:
    """
    feature_table に含まれるレース数を取得
    
    Args:
        conn: SQLite接続
    
    Returns:
        レース数
    """
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT race_id) as cnt FROM feature_table")
    row = cur.fetchone()
    return row[0] if row else 0


def get_features_for_race(
    conn: sqlite3.Connection,
    race_id: str,
) -> pd.DataFrame:
    """
    指定したレースの特徴量を取得
    
    Args:
        conn: SQLite接続
        race_id: レースID
    
    Returns:
        特徴量DataFrame
    """
    df = pd.read_sql_query(
        """
        SELECT * FROM feature_table
        WHERE race_id = ?
        """,
        conn,
        params=(race_id,)
    )
    return df


def load_all_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    feature_table 全体をロード
    
    Args:
        conn: SQLite接続
    
    Returns:
        特徴量DataFrame
    """
    logger.info("Loading all features from feature_table...")
    df = pd.read_sql_query("SELECT * FROM feature_table", conn)
    logger.info("Loaded %d rows", len(df))
    return df


# ==============================================================================
# テスト・デバッグ用
# ==============================================================================

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 80)
    print("SQLite Store Feature Test")
    print("=" * 80)
    
    # テスト用のDataFrameを作成
    test_df = pd.DataFrame({
        'race_id': ['202301010101', '202301010101'],
        'horse_id': ['2020104385', '2020104386'],
        'target_win': [1, 0],
        'target_in3': [1, 1],
        'target_value': [0, 1],
        'course': ['中山', '中山'],
        'surface': ['芝', '芝'],
        'distance': [2500, 2500],
        'distance_cat': [2500, 2500],
        'track_condition': ['良', '良'],
        'field_size': [16, 16],
        'race_class': [None, None],
        'race_year': [2023, 2023],
        'race_month': [1, 1],
        'waku': [1, 2],
        'umaban': [1, 3],
        'horse_weight': [502, 480],
        'horse_weight_diff': [2, -3],
        'n_starts_total': [10, 15],
        'win_rate_total': [0.3, 0.2],
        'in3_rate_total': [0.6, 0.5],
        'avg_finish_total': [4.5, 5.2],
        'std_finish_total': [2.1, 2.5],
        'n_starts_dist_cat': [5, 8],
        'win_rate_dist_cat': [0.4, 0.25],
        'in3_rate_dist_cat': [0.6, 0.5],
        'avg_finish_dist_cat': [4.2, 5.0],
        'avg_last3f_dist_cat': [34.5, 35.2],
        'days_since_last_run': [30, 45],
        'recent_avg_finish_3': [3.7, 4.3],
        'recent_best_finish_3': [1, 2],
        'recent_avg_last3f_3': [34.2, 35.0],
        'n_starts_track_condition': [6, 9],
        'win_rate_track_condition': [0.33, 0.22],
        'n_starts_course': [4, 6],
        'win_rate_course': [0.25, 0.17],
        'avg_horse_weight': [500.0, 482.0],
    })
    
    # テスト実行
    conn = sqlite3.connect("./data/test_feature.db")
    
    try:
        # テーブル作成
        create_table_feature(conn)
        print("✅ Table created")
        
        # データ挿入
        count = insert_feature_rows(conn, test_df)
        print(f"✅ Inserted {count} rows")
        
        # 行数確認
        total = get_feature_count(conn)
        print(f"📊 Total rows: {total}")
        
        # レース数確認
        races = get_race_count(conn)
        print(f"📊 Total races: {races}")
        
        # データ取得
        df = get_features_for_race(conn, "202301010101")
        print(f"\n📝 Features for race_id=202301010101:")
        print(df[['horse_id', 'target_win', 'n_starts_total', 'win_rate_total']].to_string(index=False))
        
    finally:
        conn.close()
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
