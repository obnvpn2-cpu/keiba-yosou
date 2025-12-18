import argparse
import logging
import os
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
    # Check which feature table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'feature_table%'")
    available_tables = [row[0] for row in cursor.fetchall()]

    if 'feature_table_v2' in available_tables:
        feature_table_name = 'feature_table_v2'
        print(f"[INFO] Using feature_table_v2")

        # Check columns in feature_table_v2
        cursor.execute(f"PRAGMA table_info({feature_table_name})")
        table_columns = [row[1] for row in cursor.fetchall()]
        hr_cols_in_table = [c for c in table_columns if c.startswith("hr_")]
        print(f"[INFO] hr_* columns in {feature_table_name} table: {len(hr_cols_in_table)}")
        if hr_cols_in_table:
            print(f"[INFO] hr_* columns: {hr_cols_in_table}")
        else:
            print(f"[WARN] No hr_* columns found in {feature_table_name} table!")
            print(f"[WARN] This might indicate wrong database file or table needs rebuild")

    elif 'feature_table' in available_tables:
        feature_table_name = 'feature_table'
        print(f"[WARN] feature_table_v2 not found, falling back to feature_table")
        print(f"[WARN] This means hr_* features will NOT be available")
    else:
        raise RuntimeError(
            "Neither feature_table_v2 nor feature_table found in database.\n"
            "Please run:\n"
            "  python scripts/build_feature_table_v2.py --db <your_db_path>\n"
            "or check your database path."
        )

    # Build column list: base columns + hr_* columns
    base_cols = [
        "race_id", "horse_id", "target_win", "target_in3", "target_value",
        "course", "surface", "surface_id", "distance", "distance_cat",
        "track_condition", "track_condition_id", "field_size", "race_class",
        "race_year", "race_month", "waku", "umaban", "horse_weight",
        "horse_weight_diff", "is_first_run", "n_starts_total",
        "win_rate_total", "in3_rate_total", "avg_finish_total",
        "std_finish_total", "n_starts_dist_cat", "win_rate_dist_cat",
        "in3_rate_dist_cat", "avg_finish_dist_cat", "avg_last3f_dist_cat",
        "days_since_last_run", "recent_avg_finish_3", "recent_best_finish_3",
        "recent_avg_last3f_3", "n_starts_track_condition",
        "win_rate_track_condition", "n_starts_course", "win_rate_course",
        "avg_horse_weight"
    ]

    # Add hr_* columns if available
    select_cols = base_cols + hr_cols_in_table
    select_cols_sql = ",\n                ".join(select_cols)

    query = f"""
        WITH f AS (
            SELECT
                {select_cols_sql}
            FROM {feature_table_name}
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
            f.*,
            rr.horse_no,
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

    # Debug: Log query to verify SELECT *
    print(f"[DEBUG] SQL query:")
    print(query)
    print()

    df = pd.read_sql_query(query, conn)
    print(f"[INFO] loaded dataset rows: {len(df):,}")
    print(f"[DEBUG] Total columns in df: {len(df.columns)}")
    print(f"[DEBUG] All df columns: {df.columns.tolist()}")

    # Check for hr_* columns
    hr_cols_in_df = [c for c in df.columns if c.startswith("hr_")]
    print(f"[INFO] hr_* columns in dataset: {len(hr_cols_in_df)}")
    if hr_cols_in_df:
        print(f"[INFO] hr_* column names: {hr_cols_in_df}")
    else:
        print(f"[WARN] No hr_* columns found in dataset despite being in table!")

    missing_fuku = df["fukusho_payout"].isna().sum()
    print(f"[INFO] fukusho_payout missing rows: {missing_fuku:,}")
    return df


def split_by_race(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 年ベースで分割：2021–2023 を学習、2024 をテスト
    df_train = df[(df["race_year"] >= 2021) & (df["race_year"] <= 2023)].copy()
    df_test = df[df["race_year"] == 2024].copy()

    train_race_ids = df_train["race_id"].nunique()
    test_race_ids = df_test["race_id"].nunique()
    total_races = df["race_id"].nunique()

    print(f"[INFO] races: total={total_races}, train={train_race_ids}, test={test_race_ids}")
    print(f"[INFO] rows: train={len(df_train):,}, test={len(df_test):,}")
    return df_train, df_test



def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    feature_cols: List[str] = []
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    # Count hr_* features
    hr_feature_cols = [c for c in feature_cols if c.startswith("hr_")]

    print(f"[INFO] feature cols ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] hr_* feature cols ({len(hr_feature_cols)}): {hr_feature_cols}")

    X = df[feature_cols].fillna(0)
    y = df["target_in3"].astype(int).values
    return X, y


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return scaler, model


# ========================================
# Evaluation Functions
# ========================================


def evaluate_global_metrics(
    df: pd.DataFrame,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
) -> Dict[str, float]:
    """
    グローバル指標の計算：accuracy, ROC-AUC, log loss, Brier score
    """
    y_true = df[y_true_col].astype(int).values
    y_pred_prob = df[y_pred_col].values
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred_binary)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = float("nan")

    try:
        logloss = log_loss(y_true, y_pred_prob)
    except ValueError:
        logloss = float("nan")

    brier = brier_score_loss(y_true, y_pred_prob)

    logger.info("=" * 60)
    logger.info("Global Metrics (全体指標)")
    logger.info("=" * 60)
    logger.info(f"Accuracy:     {accuracy:.4f}")
    logger.info(f"ROC-AUC:      {roc_auc:.4f}")
    logger.info(f"Log Loss:     {logloss:.4f}")
    logger.info(f"Brier Score:  {brier:.4f}")
    logger.info("")

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "brier_score": brier,
    }


def evaluate_ranking(
    df: pd.DataFrame,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
) -> Dict[str, float]:
    """
    レース単位のランキング評価
    各レースで in3 の馬のうち、最も予測確率が高い馬の順位を評価
    """
    results = []

    for race_id, race_df in df.groupby("race_id"):
        if race_df.empty:
            continue

        # 予測確率でソート
        race_sorted = race_df.sort_values(y_pred_col, ascending=False).reset_index(drop=True)
        race_sorted["rank"] = range(1, len(race_sorted) + 1)

        # in3 の馬を取得
        in3_horses = race_sorted[race_sorted[y_true_col] == 1]

        if len(in3_horses) == 0:
            # in3 の馬が1頭もいないレース（稀だが安全のため）
            continue

        # in3 馬の中で最も予測確率が高い馬の順位
        best_in3_rank = in3_horses["rank"].min()
        results.append(best_in3_rank)

    if not results:
        logger.warning("No races with in3 horses found!")
        return {}

    results = np.array(results)
    hit_top1 = (results == 1).mean()
    hit_top2 = (results <= 2).mean()
    hit_top3 = (results <= 3).mean()
    avg_rank = results.mean()

    logger.info("=" * 60)
    logger.info("Ranking Evaluation (ランキング評価)")
    logger.info("=" * 60)
    logger.info(f"テストレース数:        {len(results):,}")
    logger.info(f"Top1 カバー率:         {hit_top1:.3f}")
    logger.info(f"Top2 カバー率:         {hit_top2:.3f}")
    logger.info(f"Top3 カバー率:         {hit_top3:.3f}")
    logger.info(f"平均 rank (best_in3):  {avg_rank:.2f}")
    logger.info("")

    return {
        "n_races": len(results),
        "hit_top1_rate": hit_top1,
        "hit_top2_rate": hit_top2,
        "hit_top3_rate": hit_top3,
        "avg_rank_best_in3": avg_rank,
    }


def evaluate_strategy_top1_all(
    df: pd.DataFrame,
    bet_amount: int = 100,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
    payout_col: str = "fukusho_payout",
) -> Dict[str, float]:
    """
    戦略A：全レースで予測確率最大の馬に複勝ベット
    """
    n_races = 0
    n_hits = 0
    total_return = 0.0

    for race_id, race_df in df.groupby("race_id"):
        if race_df.empty:
            continue

        n_races += 1
        best_idx = race_df[y_pred_col].idxmax()
        chosen = race_df.loc[best_idx]

        payout = chosen.get(payout_col)
        if pd.notna(payout):
            total_return += float(payout)
            if int(chosen[y_true_col]) == 1:
                n_hits += 1

    total_bet = n_races * bet_amount
    hit_rate = n_hits / n_races if n_races > 0 else 0.0
    roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Strategy: Top1 All Races (全レース top1 買い)")
    logger.info("=" * 60)
    logger.info(f"ベットレース数:  {n_races:,}")
    logger.info(f"的中数:          {n_hits:,}")
    logger.info(f"的中率:          {hit_rate:.3f}")
    logger.info(f"総投資額:        {total_bet:,} 円")
    logger.info(f"総払戻:          {int(total_return):,} 円")
    logger.info(f"回収率:          {roi:.1f} %")
    logger.info("")

    return {
        "n_races": n_races,
        "n_hits": n_hits,
        "hit_rate": hit_rate,
        "total_bet": total_bet,
        "total_return": total_return,
        "roi": roi,
    }


def evaluate_strategy_top1_thresholds(
    df: pd.DataFrame,
    thresholds: List[float],
    bet_amount: int = 100,
    y_pred_col: str = "pred_in3_prob",
    payout_col: str = "fukusho_payout",
    y_true_col: str = "target_in3",
) -> List[Dict[str, float]]:
    """
    閾値付き戦略：予測確率が閾値以上の場合のみベット
    """
    results = []

    for thr in thresholds:
        n_bet_races = 0
        n_hits = 0
        total_return = 0.0
        pred_probs = []

        for race_id, race_df in df.groupby("race_id"):
            if race_df.empty:
                continue

            best_idx = race_df[y_pred_col].idxmax()
            chosen = race_df.loc[best_idx]

            # 閾値チェック
            if chosen[y_pred_col] < thr:
                continue

            n_bet_races += 1
            pred_probs.append(chosen[y_pred_col])

            payout = chosen.get(payout_col)
            if pd.notna(payout):
                total_return += float(payout)
                if int(chosen[y_true_col]) == 1:
                    n_hits += 1

        total_bet = n_bet_races * bet_amount
        hit_rate = n_hits / n_bet_races if n_bet_races > 0 else 0.0
        roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0
        avg_pred = np.mean(pred_probs) if pred_probs else 0.0

        results.append({
            "threshold": thr,
            "n_bet_races": n_bet_races,
            "n_hits": n_hits,
            "hit_rate": hit_rate,
            "total_bet": total_bet,
            "total_return": total_return,
            "roi": roi,
            "avg_pred_prob": avg_pred,
        })

    logger.info("=" * 60)
    logger.info("Strategy: Top1 with Thresholds (閾値付き戦略)")
    logger.info("=" * 60)
    logger.info(f"{'Threshold':>10} {'Bets':>6} {'Hits':>6} {'Hit%':>7} {'ROI%':>8} {'AvgProb':>8}")
    logger.info("-" * 60)

    for r in results:
        logger.info(
            f"{r['threshold']:>10.2f} {r['n_bet_races']:>6} {r['n_hits']:>6} "
            f"{r['hit_rate']:>7.3f} {r['roi']:>8.1f} {r['avg_pred_prob']:>8.3f}"
        )
    logger.info("")

    return results


def evaluate_calibration(
    df: pd.DataFrame,
    y_true_col: str = "target_in3",
    y_pred_col: str = "pred_in3_prob",
    n_bins: int = 10,
) -> Dict:
    """
    キャリブレーション評価：予測確率と実際の的中率の一致度
    """
    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values

    # ビン番号を計算
    bins = np.floor(y_pred * n_bins).clip(0, n_bins - 1).astype(int)

    bin_stats = []
    total_n = len(y_pred)
    ece_sum = 0.0

    for bin_idx in range(n_bins):
        mask = (bins == bin_idx)
        n = mask.sum()

        if n == 0:
            continue

        avg_pred = y_pred[mask].mean()
        emp_rate = y_true[mask].mean()
        diff = avg_pred - emp_rate

        bin_stats.append({
            "bin": bin_idx,
            "n": n,
            "avg_pred": avg_pred,
            "emp_rate": emp_rate,
            "diff": diff,
        })

        ece_sum += (n / total_n) * abs(diff)

    logger.info("=" * 60)
    logger.info("Calibration Evaluation (キャリブレーション評価)")
    logger.info("=" * 60)
    logger.info(f"{'Bin':>4} {'N':>8} {'AvgPred':>10} {'EmpRate':>10} {'Diff':>10}")
    logger.info("-" * 60)

    for stat in bin_stats:
        logger.info(
            f"{stat['bin']:>4} {stat['n']:>8} {stat['avg_pred']:>10.4f} "
            f"{stat['emp_rate']:>10.4f} {stat['diff']:>10.4f}"
        )

    logger.info("-" * 60)
    logger.info(f"ECE (Expected Calibration Error): {ece_sum:.4f}")
    logger.info("")

    return {
        "bins": bin_stats,
        "ece": ece_sum,
    }


def evaluate_debug(df: pd.DataFrame, payout_col: str = "fukusho_payout") -> Dict:
    """
    デバッグ用チェック：DB 側の欠損状況を確認
    """
    n_total = len(df)
    n_missing_payout = df[payout_col].isna().sum()

    # レースごとの複勝払戻件数をチェック
    payout_counts = []
    for race_id, race_df in df.groupby("race_id"):
        n_payout = race_df[payout_col].notna().sum()
        payout_counts.append(n_payout)

    payout_counts = np.array(payout_counts)
    avg_payout_per_race = payout_counts.mean()
    min_payout_per_race = payout_counts.min()
    max_payout_per_race = payout_counts.max()

    logger.info("=" * 60)
    logger.info("Debug Information (デバッグ情報)")
    logger.info("=" * 60)
    logger.info(f"Total rows:                {n_total:,}")
    logger.info(f"Missing fukusho_payout:    {n_missing_payout:,} ({n_missing_payout/n_total*100:.1f}%)")
    logger.info(f"Races:                     {len(payout_counts):,}")
    logger.info(f"Avg payout per race:       {avg_payout_per_race:.1f}")
    logger.info(f"Min payout per race:       {min_payout_per_race}")
    logger.info(f"Max payout per race:       {max_payout_per_race}")

    # 複勝払戻が3件未満のレースを警告
    races_with_few_payouts = (payout_counts < 3).sum()
    if races_with_few_payouts > 0:
        logger.warning(f"Races with < 3 payouts:    {races_with_few_payouts:,}")

    logger.info("")

    return {
        "n_total": n_total,
        "n_missing_payout": n_missing_payout,
        "n_races": len(payout_counts),
        "avg_payout_per_race": avg_payout_per_race,
        "races_with_few_payouts": races_with_few_payouts,
    }


def run_sanity_check_payout(df: pd.DataFrame) -> None:
    """
    複勝払戻 JOIN の健全性チェック (SANITY CHECK)

    目的：fukusho_payout の欠損が「負け馬の自然な NULL」なのか、
         「JOIN 不整合による異常」なのかを判定する。
    """
    logger.info("=" * 80)
    logger.info("SANITY CHECK: fukusho_payout JOIN integrity")
    logger.info("=" * 80)

    # 必要なカラムの存在確認
    required_cols = ["fukusho_payout", "target_in3", "race_id"]
    optional_cols = ["horse_id", "umaban", "field_size"]

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        logger.error(f"FATAL: Required columns missing: {missing_required}")
        logger.error(f"Available columns (first 50): {df.columns.tolist()[:50]}")
        raise RuntimeError(f"Cannot run sanity check: missing columns {missing_required}")

    # 基本統計
    n_total = len(df)
    overall_missing = df["fukusho_payout"].isna().mean()

    logger.info(f"Total rows:                    {n_total:,}")
    logger.info(f"Overall fukusho_payout missing: {overall_missing:.1%} ({df['fukusho_payout'].isna().sum():,} rows)")
    logger.info("")

    # target_in3 別の欠損率
    df_in3 = df[df["target_in3"] == 1]
    df_not_in3 = df[df["target_in3"] == 0]

    in3_missing = df_in3["fukusho_payout"].isna().mean() if len(df_in3) > 0 else float("nan")
    notin3_missing = df_not_in3["fukusho_payout"].isna().mean() if len(df_not_in3) > 0 else float("nan")

    logger.info("Breakdown by target_in3:")
    logger.info(f"  in3 (target_in3==1):     {len(df_in3):,} rows, missing: {in3_missing:.1%}")
    logger.info(f"  not-in3 (target_in3==0): {len(df_not_in3):,} rows, missing: {notin3_missing:.1%}")
    logger.info("")

    # ⚠️ in3 なのに fukusho_payout が NULL の行（潜在的な JOIN 問題）
    in3_but_null = df_in3[df_in3["fukusho_payout"].isna()]
    n_in3_but_null = len(in3_but_null)

    if n_in3_but_null > 0:
        logger.warning(f"⚠️  Found {n_in3_but_null:,} rows where target_in3==1 BUT fukusho_payout is NULL!")
        logger.warning("    This may indicate JOIN issues or data corruption.")
        logger.warning("    Showing up to 30 samples:")
        logger.warning("")

        # サンプル表示
        sample_cols = ["race_id", "horse_id"] if "horse_id" in df.columns else ["race_id"]
        if "umaban" in df.columns:
            sample_cols.append("umaban")

        sample = in3_but_null[sample_cols].head(30)
        for idx, row in sample.iterrows():
            logger.warning(f"    {dict(row)}")

        if n_in3_but_null > 30:
            logger.warning(f"    ... and {n_in3_but_null - 30:,} more rows")
        logger.warning("")
    else:
        logger.info("✅ All in3 horses have fukusho_payout (no JOIN issues detected)")
        logger.info("")

    # レース単位での払戻件数チェック
    logger.info("Per-race payout count analysis:")

    payout_counts = []
    field_sizes = []

    for race_id, race_df in df.groupby("race_id"):
        n_payout = race_df["fukusho_payout"].notna().sum()
        payout_counts.append(n_payout)

        if "field_size" in df.columns:
            field_sizes.append(race_df["field_size"].iloc[0] if len(race_df) > 0 else 0)
        else:
            field_sizes.append(len(race_df))  # 出走頭数の推定

    payout_counts = np.array(payout_counts)
    field_sizes = np.array(field_sizes)

    # describe 統計
    logger.info(f"  Payout count per race:")
    logger.info(f"    Mean:   {payout_counts.mean():.2f}")
    logger.info(f"    Std:    {payout_counts.std():.2f}")
    logger.info(f"    Min:    {payout_counts.min()}")
    logger.info(f"    25%:    {np.percentile(payout_counts, 25):.0f}")
    logger.info(f"    50%:    {np.percentile(payout_counts, 50):.0f}")
    logger.info(f"    75%:    {np.percentile(payout_counts, 75):.0f}")
    logger.info(f"    Max:    {payout_counts.max()}")
    logger.info("")

    # < 3 件のレース（理論的には top3 なので 3 件あるべき）
    races_with_few = np.where(payout_counts < 3)[0]

    if len(races_with_few) > 0:
        logger.warning(f"⚠️  Found {len(races_with_few):,} races with < 3 payouts (expected 3 for top-3 finish)")
        logger.warning("    Showing up to 10 samples:")

        for i in races_with_few[:10]:
            race_id = df.groupby("race_id").groups.keys().__iter__()
            for _ in range(i + 1):
                race_id = next(iter(df.groupby("race_id").groups.keys()))

            race_ids_list = list(df.groupby("race_id").groups.keys())
            race_id_sample = race_ids_list[i] if i < len(race_ids_list) else "N/A"

            logger.warning(f"      race_id={race_id_sample}, payout_count={payout_counts[i]}, field_size={field_sizes[i]}")

        if len(races_with_few) > 10:
            logger.warning(f"      ... and {len(races_with_few) - 10:,} more races")
        logger.warning("")
    else:
        logger.info("✅ All races have >= 3 payouts (expected for top-3 finish)")
        logger.info("")

    # 期待欠損率の計算（理論値との比較）
    avg_field_size = field_sizes.mean()
    expected_missing = 1.0 - (3.0 / avg_field_size) if avg_field_size > 0 else float("nan")

    logger.info("Expected vs Actual missing rate:")
    logger.info(f"  Avg field size:        {avg_field_size:.1f} horses")
    logger.info(f"  Expected missing rate: {expected_missing:.1%} (1 - 3/{avg_field_size:.1f})")
    logger.info(f"  Actual missing rate:   {overall_missing:.1%}")
    logger.info(f"  Difference:            {(overall_missing - expected_missing):.1%}")

    if abs(overall_missing - expected_missing) < 0.05:
        logger.info("  ✅ Actual missing rate matches theory (within 5%)")
    else:
        logger.warning(f"  ⚠️  Actual missing rate differs from theory by {abs(overall_missing - expected_missing):.1%}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("")


def _evaluate_strategy(df: pd.DataFrame) -> None:
    """複勝本命1頭買い戦略を評価するヘルパー関数"""
    bets = 0
    hits = 0
    total_payout = 0.0

    for race_id, race_df in df.groupby("race_id"):
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

    print(f"ベットレース数: {bets}")
    print(f"的中数:         {hits}")
    print(f"的中率:         {hit_rate:.3f}")
    print(f"総投資額:       {investment:,} 円")
    print(f"総払戻:         {int(total_payout):,} 円")
    print(f"回収率:         {roi * 100:.1f} %")


def evaluate(df_test: pd.DataFrame, scaler: StandardScaler, model: LogisticRegression, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    """
    総合評価関数：各種評価メトリクスを実行
    """
    X_test_scaled = scaler.transform(X_test)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    df_test = df_test.copy()
    df_test["pred_in3_prob"] = proba

    # 1. グローバル指標
    evaluate_global_metrics(df_test)

    # 2. ランキング評価
    evaluate_ranking(df_test)

    # 3. 戦略A：全レースで top1 買い
    evaluate_strategy_top1_all(df_test)

    # 4. 閾値付き戦略
    thresholds = [0.25, 0.30, 0.35, 0.40]
    evaluate_strategy_top1_thresholds(df_test, thresholds=thresholds)

    # 5. キャリブレーション評価
    evaluate_calibration(df_test, n_bins=10)

    # 6. デバッグ情報
    evaluate_debug(df_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate logistic regression on feature_table")
    parser.add_argument(
        "--db",
        default="netkeiba.db",
        help="Path to SQLite database (default: netkeiba.db)",
    )
    parser.add_argument(
        "--sanity-payout",
        action="store_true",
        help="Run sanity check for fukusho_payout JOIN integrity",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run sanity check only and exit (implies --sanity-payout)",
    )
    args = parser.parse_args()

    # --sanity-only implies --sanity-payout
    if args.sanity_only:
        args.sanity_payout = True

    db_path = os.path.abspath(args.db)
    print(f"[INFO] Database path: {db_path}")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = load_dataset(conn)

        # Run sanity check if requested
        if args.sanity_payout:
            run_sanity_check_payout(df)

            # Exit if --sanity-only
            if args.sanity_only:
                logger.info("Sanity check completed. Exiting (--sanity-only mode).")
                return

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
