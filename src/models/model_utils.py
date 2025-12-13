# src/models/model_utils.py (v2.0 - Reviewed & Improved)
"""
モデル学習・予測用のユーティリティ関数

v2.0の改善:
- 時系列分割の改善（年月日で連続的に切る）
- データリーク対策（train=validを許容しない）
- 小規模データでの適切なエラーハンドリング
- ログの改善
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)

# 特徴量から除外するカラム
FEATURE_EXCLUDE_COLS = [
    "race_id",
    "horse_id",
    # ターゲット
    "target_win",
    "target_in3",
    "target_value",
    # 説明用文字列カラム
    "course",
    "surface",
    "track_condition",
    "race_class",
    # メタ
    "created_at",
    "updated_at",
]


def load_feature_table(db_path: str) -> pd.DataFrame:
    """SQLite から feature_table 全件ロード"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM feature_table", conn)
    finally:
        conn.close()
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    特徴量として使うカラムの一覧を返す
    
    - 除外カラムを外す
    - object 型（文字列）は除外
    """
    cols: List[str] = []
    for c in df.columns:
        if c in FEATURE_EXCLUDE_COLS:
            continue
        if c.startswith("target_"):
            continue
        if df[c].dtype == "object":
            continue
        cols.append(c)
    return cols


def split_train_valid(
    df: pd.DataFrame,
    target_col: str,
    valid_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    学習用/検証用データを作成（時系列性を考慮）
    
    v2.0の改善:
    - 時系列分割を年月日で連続的に切る（valid_ratioを尊重）
    - train=validのデータリークを完全に排除
    - 小規模データでの適切なエラーハンドリング
    
    優先順位:
      1. race_year, race_month が複数ある → 時系列で最後のN%を検証用
      2. race_id が複数ある → race_id 単位で GroupShuffleSplit
      3. データが少ない → エラー（train=validは許容しない）
    """
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col} not found in DataFrame")

    feature_cols = get_feature_columns(df)
    
    # ケース1: 時系列分割（年月で連続的に切る）
    if "race_year" in df.columns and "race_month" in df.columns:
        # 年月でソート
        df = df.sort_values(["race_year", "race_month", "race_id"]).reset_index(drop=True)
        
        # 時系列で最後のN%を検証用
        n_valid = max(1, int(len(df) * valid_ratio))
        n_train = len(df) - n_valid
        
        if n_train < 1 or n_valid < 1:
            raise ValueError(
                f"Dataset too small for split: total={len(df)}, "
                f"train={n_train}, valid={n_valid}"
            )
        
        df_train = df.iloc[:n_train].reset_index(drop=True)
        df_valid = df.iloc[n_train:].reset_index(drop=True)
        
        logger.info(
            "Time-series split: train=%d rows (%.1f%%), valid=%d rows (%.1f%%)",
            len(df_train), 100 * len(df_train) / len(df),
            len(df_valid), 100 * len(df_valid) / len(df)
        )
        
        # 時系列分割の情報をログ
        if len(df_train) > 0 and len(df_valid) > 0:
            train_years = df_train["race_year"].unique()
            valid_years = df_valid["race_year"].unique()
            logger.info(
                "  Train years: %s, Valid years: %s",
                sorted(train_years),
                sorted(valid_years)
            )
    
    # ケース2: レースが複数あるなら race_id 単位で group split
    elif df["race_id"].nunique() >= 2 and len(df) >= 20:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=valid_ratio,
            random_state=random_state,
        )
        groups = df["race_id"]
        idx_train, idx_valid = next(gss.split(df, df[target_col], groups))
        df_train = df.iloc[idx_train].reset_index(drop=True)
        df_valid = df.iloc[idx_valid].reset_index(drop=True)
        
        logger.info(
            "Group split: train=%d rows (%d races), valid=%d rows (%d races)",
            len(df_train), df_train["race_id"].nunique(),
            len(df_valid), df_valid["race_id"].nunique()
        )
    
    # ケース3: データが少ない場合
    else:
        # 最低でも5行必要
        if len(df) < 5:
            raise ValueError(
                f"Dataset too small ({len(df)} rows). "
                f"Need at least 5 rows for train/valid split."
            )
        
        # 10行未満は警告
        if len(df) < 10:
            logger.warning(
                "Dataset is very small (%d rows). "
                "Validation metrics may be unreliable. "
                "Consider gathering more data.",
                len(df)
            )
        
        # 最低でも2行は検証用に確保
        test_size = max(0.2, 2 / len(df))
        
        df_train, df_valid = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        
        logger.info(
            "Random split: train=%d rows (%.1f%%), valid=%d rows (%.1f%%)",
            len(df_train), 100 * len(df_train) / len(df),
            len(df_valid), 100 * len(df_valid) / len(df)
        )
    
    # 共通処理
    X_train = df_train[feature_cols]
    X_valid = df_valid[feature_cols]
    y_train = df_train[target_col]
    y_valid = df_valid[target_col]
    
    # 検証セットのクラス分布を確認
    if len(y_valid) > 0:
        n_positive = (y_valid == 1).sum()
        n_negative = (y_valid == 0).sum()
        logger.info(
            "  Valid target distribution: positive=%d (%.1f%%), negative=%d (%.1f%%)",
            n_positive, 100 * n_positive / len(y_valid) if len(y_valid) > 0 else 0,
            n_negative, 100 * n_negative / len(y_valid) if len(y_valid) > 0 else 0
        )
        
        # 片側クラスしかない場合は警告
        if n_positive == 0 or n_negative == 0:
            logger.warning(
                "Validation set has only one class! "
                "AUC calculation will fail. "
                "Consider using more data or different split strategy."
            )
    
    return X_train, X_valid, y_train, y_valid, feature_cols


def save_feature_columns(feature_cols: List[str], output_dir: str, target_name: str) -> None:
    """特徴量リストをJSON に保存"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"feature_columns_{target_name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    logger.info("Saved feature columns to %s", path)


def load_feature_columns(output_dir: str, target_name: str) -> List[str]:
    """特徴量リストをJSONから読み込み"""
    path = Path(output_dir) / f"feature_columns_{target_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        cols = json.load(f)
    logger.debug("Loaded %d feature columns from %s", len(cols), path)
    return cols


def load_features_for_races(
    db_path: str,
    race_ids: List[str],
) -> pd.DataFrame:
    """指定 race_id 群の feature_table を取得"""
    if not race_ids:
        raise ValueError("race_ids must not be empty")

    conn = sqlite3.connect(db_path)
    try:
        placeholder = ",".join("?" for _ in race_ids)
        sql = f"""
            SELECT *
            FROM feature_table
            WHERE race_id IN ({placeholder})
            ORDER BY race_id, horse_id
        """
        df = pd.read_sql_query(sql, conn, params=race_ids)
    finally:
        conn.close()
    
    logger.info("Loaded %d rows for %d races", len(df), len(race_ids))
    return df
