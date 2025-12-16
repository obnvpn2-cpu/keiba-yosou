import argparse
import logging
import sqlite3
from typing import List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_feature_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    logger.info(f"Loading base feature table: {table_name}")
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    logger.info(f"[INFO] feature_table rows: {len(df):,}")
    return df


def load_horse_results(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Loading horse_results")
    df = pd.read_sql_query("SELECT * FROM horse_results", conn)
    logger.info(f"[INFO] horse_results rows: {len(df):,}")
    return df


def compute_horse_history_features(hr: pd.DataFrame) -> pd.DataFrame:
    """
    horse_results から「そのレース時点」での履歴系特徴量を計算して返す。
    返り値は (race_id, horse_id) ごとに 1 行。
    """

    # 必要な列だけに絞っておく（適宜調整）
    cols_needed = [
        "horse_id",
        "race_id",
        "race_date",
        "finish_order",
        "last_3f",
        "win_odds",
        "popularity",
        "prize_money",
    ]
    missing = [c for c in cols_needed if c not in hr.columns]
    if missing:
        raise RuntimeError(f"horse_results に必要なカラムが不足しています: {missing}")

    df = hr[cols_needed].copy()

    # 日付を datetime に
    df["race_date_dt"] = pd.to_datetime(df["race_date"])

    # ソート（horse_id, race_date, race_id）
    df = df.sort_values(["horse_id", "race_date_dt", "race_id"])

    # 着順フラグ
    df["is_win"] = (df["finish_order"] == 1).astype(int)
    df["is_in3"] = df["finish_order"].between(1, 3).astype(int)

    # グループごと（馬ごと）
    g = df.groupby("horse_id", group_keys=False)

    # =========================
    # 1. キャリア通算（直前まで）
    # =========================
    # その馬にとって何戦目か（0,1,2,...）: これが「過去レース数」のベースになる
    df["hr_career_starts"] = g.cumcount()  # 直前までの出走数（今走を含まない）

    # 通算勝利数・in3数（直前まで）: cumsum を1つ前にずらす
    df["hr_career_wins"] = g["is_win"].cumsum() - df["is_win"]
    df["hr_career_in3"] = g["is_in3"].cumsum() - df["is_in3"]

    # レースが1戦もなければ NaN にしておく
    starts = df["hr_career_starts"].replace(0, np.nan)
    df["hr_career_win_rate"] = df["hr_career_wins"] / starts
    df["hr_career_in3_rate"] = df["hr_career_in3"] / starts

    # 通算平均着順・標準偏差（直前まで）
    def _expanding_mean_prev(s: pd.Series) -> pd.Series:
        # 直前までの expading mean: cumsum.shift / count.shift
        csum = s.cumsum()
        cnt = pd.Series(np.arange(1, len(s) + 1), index=s.index)
        # 直前まで → shift(1)
        return (csum.shift(1) / cnt.shift(1)).astype(float)

    def _expanding_std_prev(s: pd.Series) -> pd.Series:
        # 単純に「これまでの値」だけで std を計算（直前まで）
        out = []
        vals: List[float] = []
        for v in s:
            out.append(np.std(vals, ddof=0) if len(vals) > 0 else np.nan)
            vals.append(v)
        return pd.Series(out, index=s.index, dtype=float)

    df["hr_career_avg_finish"] = g["finish_order"].apply(_expanding_mean_prev)
    df["hr_career_std_finish"] = g["finish_order"].apply(_expanding_std_prev)

    # 上がり・人気・賞金の通算平均・累計（直前まで）
    def _expanding_mean_prev_generic(s: pd.Series) -> pd.Series:
        csum = s.fillna(0).cumsum()
        cnt = (~s.isna()).cumsum()
        return (csum.shift(1) / cnt.shift(1)).replace([np.inf], np.nan)

    df["hr_career_avg_last3f"] = g["last_3f"].apply(_expanding_mean_prev_generic)
    df["hr_career_avg_popularity"] = g["popularity"].apply(_expanding_mean_prev_generic)
    df["hr_career_total_prize"] = g["prize_money"].cumsum() - df["prize_money"]
    df["hr_career_avg_prize"] = _expanding_mean_prev_generic(g["prize_money"].apply(lambda x: x))

    # =========================
    # 2. 直近3走ウィンドウ（直前まで）
    # =========================

    def _rolling_last3_prev(s: pd.Series) -> pd.Series:
        # 直前までのデータに対して window=3 で rolling
        return s.shift(1).rolling(window=3, min_periods=1).mean()

    df["hr_recent3_starts"] = g.cumcount().clip(upper=3)  # 最大3
    df["hr_recent3_avg_finish"] = g["finish_order"].apply(_rolling_last3_prev)

    def _rolling_best3_prev(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=3, min_periods=1).min()

    df["hr_recent3_best_finish"] = g["finish_order"].apply(_rolling_best3_prev)
    df["hr_recent3_avg_last3f"] = g["last_3f"].apply(_rolling_last3_prev)
    df["hr_recent3_avg_win_odds"] = g["win_odds"].apply(_rolling_last3_prev)
    df["hr_recent3_avg_popularity"] = g["popularity"].apply(_rolling_last3_prev)

    # 大敗カウント（直近3走）：finish_order >= 10 を「大敗」と仮定
    def _rolling_big_loss_prev(s: pd.Series) -> pd.Series:
        flag = (s >= 10).astype(int)
        return flag.shift(1).rolling(window=3, min_periods=1).sum()

    df["hr_recent3_big_loss_count"] = g["finish_order"].apply(_rolling_big_loss_prev)

    # =========================
    # 3. ローテーション系
    # =========================

    df["hr_days_since_prev"] = g["race_date_dt"].diff().dt.days

    # 前走の条件（今はクラス・グレード・距離・馬場だけ拾う想定）
    # これは feature_table 側と JOIN するときに別途拾う or 後回しにしてもよいので、
    # ひとまずローテ日数だけでも OK。

    # =========================
    # 出力カラムを整理
    # =========================
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
        # rotation
        "hr_days_since_prev",
    ]

    return df[out_cols].copy()


def build_feature_table_v2(
    conn: sqlite3.Connection,
    base_table: str = "feature_table",
    target_table: str = "feature_table_v2",
) -> None:
    base = load_feature_table(conn, base_table)
    hr = load_horse_results(conn)

    hist = compute_horse_history_features(hr)

    logger.info("Merging base feature_table with horse_history_features...")
    merged = base.merge(hist, on=["race_id", "horse_id"], how="left")
    logger.info(f"[INFO] merged rows: {len(merged):,}")

    # SQLite に保存
    logger.info(f"Dropping and creating {target_table} in SQLite...")
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {target_table}")
    conn.commit()

    merged.to_sql(target_table, conn, index=False)
    logger.info(f"[INFO] wrote table {target_table}: {len(merged):,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature_table_v2 using horse_results-based history features.")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite DB (e.g., netkeiba.db)")
    parser.add_argument("--base-table", type=str, default="feature_table", help="Source feature table name")
    parser.add_argument("--target-table", type=str, default="feature_table_v2", help="Target feature table name")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        build_feature_table_v2(conn, base_table=args.base_table, target_table=args.target_table)
    finally:
        conn.close()


if __name__ == "__main__":
    main()