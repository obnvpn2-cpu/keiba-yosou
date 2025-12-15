import argparse
import os
import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EXCLUDED_COLUMNS = {
    "race_id",
    "horse_id",
    "race_class",
    "course",
    "surface",
    "track_condition",
    "created_at",
    "updated_at",
    "horse_no",
    "fukusho_payout",
    "target_win",
    "target_in3",
    "target_value",
}


def load_dataset(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        WITH f AS (
            SELECT
                race_id,
                horse_id,
                target_win,
                target_in3,
                target_value,
                course,
                surface,
                surface_id,
                distance,
                distance_cat,
                track_condition,
                track_condition_id,
                field_size,
                race_class,
                race_year,
                race_month,
                waku,
                umaban,
                horse_weight,
                horse_weight_diff,
                is_first_run,
                n_starts_total,
                win_rate_total,
                in3_rate_total,
                avg_finish_total,
                std_finish_total,
                n_starts_dist_cat,
                win_rate_dist_cat,
                in3_rate_dist_cat,
                avg_finish_dist_cat,
                avg_last3f_dist_cat,
                days_since_last_run,
                recent_avg_finish_3,
                recent_best_finish_3,
                recent_avg_last3f_3,
                n_starts_track_condition,
                win_rate_track_condition,
                n_starts_course,
                win_rate_course,
                avg_horse_weight,
                created_at,
                updated_at
            FROM feature_table
            -- ここでは年で絞らない（2021〜2024 を全部持ってくる）
        ),
        rr AS (
            SELECT
                race_id,
                horse_id,
                horse_no
            FROM race_results
        ),
        fuku AS (
            SELECT
                race_id,
                combination AS horse_no_str,
                payout AS fukusho_payout
            FROM payouts
            WHERE bet_type = '複勝'
        )
        SELECT
            f.*, rr.horse_no,
            fuku.fukusho_payout
        FROM f
        JOIN rr
          ON f.race_id = rr.race_id
         AND f.horse_id = rr.horse_id
        LEFT JOIN fuku
          ON f.race_id = fuku.race_id
         AND CAST(rr.horse_no AS TEXT) = fuku.horse_no_str
        WHERE f.race_year BETWEEN 2021 AND 2024
    """
    df = pd.read_sql_query(query, conn)
    print(f"[INFO] loaded dataset rows: {len(df):,}")
    missing_fuku = df["fukusho_payout"].isna().sum()
    print(f"[INFO] fukusho_payout missing rows: {missing_fuku:,}")
    return df


def split_by_race(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = df["race_year"] <= 2023
    test_mask = df["race_year"] == 2024

    train_race_ids = set(df.loc[train_mask, "race_id"].unique())
    test_race_ids = set(df.loc[test_mask, "race_id"].unique())

    df_train = df[df["race_id"].isin(train_race_ids)].sample(frac=1, random_state=42)
    df_test = df[df["race_id"].isin(test_race_ids)]

    n_races = len(set(df["race_id"].unique()))
    print(
        f"[INFO] races: total={n_races}, train={len(train_race_ids)}, test={len(test_race_ids)}"
    )
    print(f"[INFO] rows: train={len(df_train):,}, test={len(df_test):,}")
    return df_train, df_test


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    feature_cols: List[str] = []
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    print(f"[INFO] feature cols ({len(feature_cols)}): {feature_cols}")
    X = df[feature_cols].fillna(0)
    y = df["target_in3"].astype(int).values
    return X, y


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return scaler, model


def evaluate(df_test: pd.DataFrame, scaler: StandardScaler, model: LogisticRegression, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = float((y_pred == y_test).mean()) if len(y_test) else float("nan")
    print(f"[INFO] test accuracy (per horse, target_in3): {accuracy:.3f}")

    proba = model.predict_proba(X_test_scaled)[:, 1]
    df_test = df_test.copy()
    df_test["pred_in3_proba"] = proba

    bets = 0
    hits = 0
    total_payout = 0.0

    for race_id, race_df in df_test.groupby("race_id"):
        if race_df.empty:
            continue
        bets += 1
        best_idx = race_df["pred_in3_proba"].idxmax()
        chosen = race_df.loc[best_idx]
        payout = chosen.get("fukusho_payout")
        if pd.isna(payout):
            payout = 0.0
        total_payout += float(payout)
        if int(chosen["target_in3"]) == 1:
            hits += 1

    if bets > 0:
        hit_rate = hits / bets
        investment = bets * 100
        roi = total_payout / investment
    else:
        hit_rate = float("nan")
        investment = 0
        roi = float("nan")

    print("===== テスト結果（複勝 本命1頭買い）=====")
    print(f"レース数(bets): {bets}")
    print(f"的中数(hits):   {hits}")
    print(f"的中率:         {hit_rate:.3f}")
    print(f"総投資額:       {investment:,} 円")
    print(f"総払戻:         {int(total_payout):,} 円")
    print(f"回収率:         {roi * 100:.1f} %")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate logistic regression on feature_table")
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = load_dataset(conn)
        df_train, df_test = split_by_race(df)

        if df_train.empty or df_test.empty:
            raise RuntimeError("train or test dataset is empty; check data availability")

        X_train, y_train = build_feature_matrix(df_train)
        X_test, y_test = build_feature_matrix(df_test)

        scaler, model = train_model(X_train, y_train)
        evaluate(df_test, scaler, model, X_test, y_test)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
