#!/usr/bin/env python3
"""
build_feature_table_v2.py

情報リークゼロを保証した履歴特徴量を計算し、feature_table_v2 を構築する。

重要な原則：
- すべての hr_* 特徴量は「そのレースより前のレース」のみから計算
- groupby + shift(1) を徹底使用
- QA コードで検証

Usage:
    python scripts/build_feature_table_v2.py --db netkeiba.db
"""

import argparse
import logging
import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_feature_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    logger.info(f"Loading base feature table: {table_name}")
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    logger.info(f"feature_table rows: {len(df):,}")
    return df


def load_horse_results(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Loading horse_results")
    df = pd.read_sql_query("SELECT * FROM horse_results", conn)
    logger.info(f"horse_results rows: {len(df):,}")
    return df


def compute_horse_history_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    horse_results から「そのレース時点での過去の」履歴系特徴量を計算。

    重要: すべての特徴量は shift(1) を使って「今走を除外」する。
    """

    # 必要な列チェック
    cols_needed = [
        "horse_id", "race_id", "race_date", "finish_order",
        "last_3f", "win_odds", "popularity", "prize_money",
    ]
    missing = [c for c in cols_needed if c not in hr.columns]
    if missing:
        raise RuntimeError(f"horse_results に必要なカラムが不足: {missing}")

    df = hr[cols_needed].copy()

    # 日付を datetime に変換
    df["race_date_dt"] = pd.to_datetime(df["race_date"])

    # ソート（horse_id, race_date, race_id）
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"]).reset_index(drop=True)

    # 着順フラグ
    df["is_win"] = (df["finish_order"] == 1).astype(int)
    df["is_in3"] = df["finish_order"].between(1, 3).astype(int)

    # グループごと
    g = df.groupby("horse_id", group_keys=False)

    # ========================================
    # 1. キャリア通算統計（直前まで）
    # ========================================

    # 出走数（今走を含まない）
    df["hr_career_starts"] = g.cumcount()

    # 勝利数・複勝数（今走を含まない）
    # cumsum() は今走を含むので、shift(1) で1つ前にずらしてから cumsum 相当を得る
    df["hr_career_wins"] = g["is_win"].apply(lambda x: x.shift(1).fillna(0).cumsum())
    df["hr_career_in3"] = g["is_in3"].apply(lambda x: x.shift(1).fillna(0).cumsum())

    # 勝率・複勝率
    starts_safe = df["hr_career_starts"].replace(0, np.nan)
    df["hr_career_win_rate"] = df["hr_career_wins"] / starts_safe
    df["hr_career_in3_rate"] = df["hr_career_in3"] / starts_safe

    # 平均着順（過去のみ）
    def expanding_mean_prev(s: pd.Series) -> pd.Series:
        """直前までの expanding mean"""
        shifted = s.shift(1)
        return shifted.expanding().mean()

    df["hr_career_avg_finish"] = g["finish_order"].apply(expanding_mean_prev)

    # 標準偏差（過去のみ）
    def expanding_std_prev(s: pd.Series) -> pd.Series:
        """直前までの expanding std"""
        shifted = s.shift(1)
        return shifted.expanding().std()

    df["hr_career_std_finish"] = g["finish_order"].apply(expanding_std_prev)

    # 上がり・人気・賞金の平均（過去のみ）
    df["hr_career_avg_last3f"] = g["last_3f"].apply(expanding_mean_prev)
    df["hr_career_avg_popularity"] = g["popularity"].apply(expanding_mean_prev)

    # 賞金累計（過去のみ）
    df["hr_career_total_prize"] = g["prize_money"].apply(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    df["hr_career_avg_prize"] = g["prize_money"].apply(expanding_mean_prev)

    # ========================================
    # 2. 直近3走統計（直前まで）
    # ========================================

    def rolling_mean_prev(s: pd.Series, window: int = 3) -> pd.Series:
        """直前までの rolling mean"""
        return s.shift(1).rolling(window=window, min_periods=1).mean()

    def rolling_min_prev(s: pd.Series, window: int = 3) -> pd.Series:
        """直前までの rolling min"""
        return s.shift(1).rolling(window=window, min_periods=1).min()

    def rolling_sum_prev(s: pd.Series, window: int = 3) -> pd.Series:
        """直前までの rolling sum"""
        return s.shift(1).rolling(window=window, min_periods=1).sum()

    # 直近3走の統計（すべて shift(1) で過去のみ）
    df["hr_recent3_starts"] = df["hr_career_starts"].clip(upper=3)
    df["hr_recent3_avg_finish"] = g["finish_order"].apply(rolling_mean_prev)
    df["hr_recent3_best_finish"] = g["finish_order"].apply(rolling_min_prev)
    df["hr_recent3_avg_last3f"] = g["last_3f"].apply(rolling_mean_prev)
    df["hr_recent3_avg_win_odds"] = g["win_odds"].apply(rolling_mean_prev)
    df["hr_recent3_avg_popularity"] = g["popularity"].apply(rolling_mean_prev)

    # 大敗カウント（finish_order >= 10）
    def rolling_big_loss_prev(s: pd.Series) -> pd.Series:
        big_loss_flag = (s >= 10).astype(int)
        return big_loss_flag.shift(1).rolling(window=3, min_periods=1).sum()

    df["hr_recent3_big_loss_count"] = g["finish_order"].apply(rolling_big_loss_prev)

    # ========================================
    # 2b. 直近5走統計（直前まで）
    # ========================================

    def rolling_mean_prev_5(s: pd.Series) -> pd.Series:
        """直前までの rolling mean (window=5)"""
        return s.shift(1).rolling(window=5, min_periods=1).mean()

    def rolling_min_prev_5(s: pd.Series) -> pd.Series:
        """直前までの rolling min (window=5)"""
        return s.shift(1).rolling(window=5, min_periods=1).min()

    def rolling_sum_prev_5(s: pd.Series) -> pd.Series:
        """直前までの rolling sum (window=5)"""
        return s.shift(1).rolling(window=5, min_periods=1).sum()

    def rolling_count_prev_5(s: pd.Series) -> pd.Series:
        """直前までの rolling count (window=5)"""
        return s.shift(1).rolling(window=5, min_periods=1).count()

    # 直近5走の出走数（最大5）
    df["hr_recent5_starts"] = df["hr_career_starts"].clip(upper=5)

    # 直近5走の平均着順
    df["hr_recent5_avg_finish"] = g["finish_order"].apply(rolling_mean_prev_5)

    # 直近5走の最良着順
    df["hr_recent5_best_finish"] = g["finish_order"].apply(rolling_min_prev_5)

    # 直近5走の上がり3F平均
    df["hr_recent5_avg_last3f"] = g["last_3f"].apply(rolling_mean_prev_5)

    # 直近5走の人気平均
    df["hr_recent5_avg_popularity"] = g["popularity"].apply(rolling_mean_prev_5)

    # 直近5走の大敗カウント（finish_order >= 10）
    def rolling_big_loss_prev_5(s: pd.Series) -> pd.Series:
        big_loss_flag = (s >= 10).astype(int)
        return big_loss_flag.shift(1).rolling(window=5, min_periods=1).sum()

    df["hr_recent5_big_loss_count"] = g["finish_order"].apply(rolling_big_loss_prev_5)

    # ========================================
    # 3. ローテーション
    # ========================================

    # 前走からの日数（今走 - 前走）
    df["hr_days_since_prev"] = g["race_date_dt"].diff().dt.days

    # ========================================
    # 4. トレンド系特徴量（直近3走）
    # ========================================

    def compute_recent3_finish_trend(s: pd.Series) -> pd.Series:
        """
        直近3走の着順トレンド
        trend = f3 - (f1 + f2) / k
        負の値 = 上向き（最近の方が良い）、正の値 = 下り坂
        """
        shifted = s.shift(1)  # 今走を除外
        result = pd.Series(index=s.index, dtype=float)

        for idx in range(len(s)):
            # 過去3走を取得（最大3走）
            past_3 = shifted.iloc[max(0, idx-2):idx+1]

            if len(past_3) == 0:
                result.iloc[idx] = np.nan
                continue

            if len(past_3) == 1:
                # 履歴1走のみ：トレンド計算不可
                result.iloc[idx] = 0.0
                continue

            # 最新のレース（f3）とそれ以前の平均
            f_recent = past_3.iloc[-1]
            f_prev_avg = past_3.iloc[:-1].mean()

            result.iloc[idx] = f_recent - f_prev_avg

        return result

    df["hr_recent3_finish_trend"] = g["finish_order"].apply(compute_recent3_finish_trend)

    def compute_recent3_last_vs_best_diff(s: pd.Series) -> pd.Series:
        """
        直近3走の「最新着順 - ベスト着順」
        0 = 最新レースがベスト、正の値 = ベストより悪化
        """
        shifted = s.shift(1)  # 今走を除外
        result = pd.Series(index=s.index, dtype=float)

        for idx in range(len(s)):
            # 過去3走を取得（最大3走）
            past_3 = shifted.iloc[max(0, idx-2):idx+1]

            if len(past_3) == 0:
                result.iloc[idx] = np.nan
                continue

            f_recent = past_3.iloc[-1]
            f_best = past_3.min()

            result.iloc[idx] = f_recent - f_best

        return result

    df["hr_recent3_last_vs_best_diff"] = g["finish_order"].apply(compute_recent3_last_vs_best_diff)

    # ========================================
    # 出力カラムを選択
    # ========================================

    out_cols = [
        "race_id",
        "horse_id",
        # career
        "hr_career_starts",
        "hr_career_wins",
        "hr_career_in3",
        "hr_career_win_rate",
        "hr_career_in3_rate",
        "hr_career_avg_finish",
        "hr_career_std_finish",
        "hr_career_avg_last3f",
        "hr_career_avg_popularity",
        "hr_career_total_prize",
        "hr_career_avg_prize",
        # recent3
        "hr_recent3_starts",
        "hr_recent3_avg_finish",
        "hr_recent3_best_finish",
        "hr_recent3_avg_last3f",
        "hr_recent3_avg_win_odds",
        "hr_recent3_avg_popularity",
        "hr_recent3_big_loss_count",
        # recent5
        "hr_recent5_starts",
        "hr_recent5_avg_finish",
        "hr_recent5_best_finish",
        "hr_recent5_avg_last3f",
        "hr_recent5_avg_popularity",
        "hr_recent5_big_loss_count",
        # trend
        "hr_recent3_finish_trend",
        "hr_recent3_last_vs_best_diff",
        # rotation
        "hr_days_since_prev",
    ]

    return df[out_cols].copy()


def run_qa_checks(df: pd.DataFrame, hr: pd.DataFrame) -> None:
    """
    QA チェック: 情報リークがないことを検証
    """
    logger.info("=" * 60)
    logger.info("Running QA checks for data leakage...")
    logger.info("=" * 60)

    # サンプル抽出（ランダムに100件）
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    # horse_results を race_date でソート
    hr_sorted = hr.sort_values(["horse_id", "race_date", "race_id"])

    errors = []

    for idx, row in sample_df.iterrows():
        horse_id = row["horse_id"]
        race_id = row["race_id"]

        # この馬の全履歴を取得
        horse_hist = hr_sorted[hr_sorted["horse_id"] == horse_id].reset_index(drop=True)

        # 今走のインデックス
        race_idx = horse_hist[horse_hist["race_id"] == race_id].index
        if len(race_idx) == 0:
            continue
        race_idx = race_idx[0]

        # 過去のレース（今走を含まない）
        past_races = horse_hist.iloc[:race_idx]

        # 検証1: hr_career_starts == 過去のレース数
        expected_starts = len(past_races)
        actual_starts = row["hr_career_starts"]
        if actual_starts != expected_starts:
            errors.append(
                f"[LEAK] horse_id={horse_id}, race_id={race_id}: "
                f"hr_career_starts={actual_starts}, expected={expected_starts}"
            )

        # 検証2: hr_career_wins == 過去の勝利数
        if len(past_races) > 0:
            expected_wins = (past_races["finish_order"] == 1).sum()
            actual_wins = row["hr_career_wins"]
            if actual_wins != expected_wins:
                errors.append(
                    f"[LEAK] horse_id={horse_id}, race_id={race_id}: "
                    f"hr_career_wins={actual_wins}, expected={expected_wins}"
                )

        # 検証3: hr_career_in3 == 過去の複勝数
        if len(past_races) > 0:
            expected_in3 = past_races["finish_order"].between(1, 3).sum()
            actual_in3 = row["hr_career_in3"]
            if actual_in3 != expected_in3:
                errors.append(
                    f"[LEAK] horse_id={horse_id}, race_id={race_id}: "
                    f"hr_career_in3={actual_in3}, expected={expected_in3}"
                )

    # 結果レポート
    if errors:
        logger.error(f"QA FAILED: {len(errors)} errors found!")
        for err in errors[:10]:  # 最初の10件だけ表示
            logger.error(err)
        if len(errors) > 10:
            logger.error(f"... and {len(errors) - 10} more errors")
        raise RuntimeError("Data leakage detected in hr_* features!")
    else:
        logger.info(f"✅ QA PASSED: No leakage detected in {sample_size} samples")

    logger.info("=" * 60)


def build_feature_table_v2(
    conn: sqlite3.Connection,
    base_table: str = "feature_table",
    target_table: str = "feature_table_v2",
    run_qa: bool = True,
) -> None:
    """
    feature_table_v2 を構築
    """
    base = load_feature_table(conn, base_table)
    hr = load_horse_results(conn)

    hist = compute_horse_history_features(hr)

    logger.info("Merging base feature_table with horse_history_features...")
    merged = base.merge(hist, on=["race_id", "horse_id"], how="left")
    logger.info(f"merged rows: {len(merged):,}")

    # QA チェック実行
    if run_qa:
        run_qa_checks(merged, hr)

    # SQLite に保存 (UNIQUE制約付き)
    logger.info(f"Saving {target_table} with UPSERT support...")

    # Import UPSERT utilities
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from src.db.upsert import create_table_from_df
        # Use new idempotent table creation
        create_table_from_df(
            conn, merged, target_table,
            key_columns=["race_id", "horse_id"],
            if_exists="replace"
        )
        logger.info(f"✅ Wrote table {target_table}: {len(merged):,} rows")
        logger.info(f"Created UNIQUE INDEX on ({target_table}.race_id, {target_table}.horse_id)")
    except ImportError:
        # Fallback to old method if src.db not available
        logger.warning("src.db.upsert not available, using pandas to_sql (no UNIQUE constraint)")
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {target_table}")
        conn.commit()
        merged.to_sql(target_table, conn, index=False)
        logger.info(f"✅ Wrote table {target_table}: {len(merged):,} rows")

    # hr_* カラム数を表示
    hr_cols = [c for c in merged.columns if c.startswith("hr_")]
    logger.info(f"hr_* columns ({len(hr_cols)}): {hr_cols}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature_table_v2 with leak-free history features"
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB (e.g., netkeiba.db)",
    )
    parser.add_argument(
        "--base-table",
        type=str,
        default="feature_table",
        help="Source feature table name",
    )
    parser.add_argument(
        "--target-table",
        type=str,
        default="feature_table_v2",
        help="Target feature table name",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA checks (not recommended)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        build_feature_table_v2(
            conn,
            base_table=args.base_table,
            target_table=args.target_table,
            run_qa=not args.skip_qa,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
