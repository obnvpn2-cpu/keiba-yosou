# -*- coding: utf-8 -*-
"""
diagnostics.py - Feature Diagnostics for FeaturePack v1

【機能】
1. LightGBM 標準重要度 (gain / split)
2. Permutation Importance (複数メトリクス対応)
3. Feature Group Importance (プレフィックスによるグループ化)
4. Segment Performance (コース・距離・馬場等のセグメント別分析)

【出力形式】
- CSV: 詳細な数値データ
- テキストレポート: 人間が読みやすい形式
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from sklearn.metrics import roc_auc_score, log_loss

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureImportanceResult:
    """
    LightGBM 標準重要度の結果
    """
    feature_name: str
    gain: float
    split: int
    gain_rank: int
    split_rank: int


@dataclass
class PermutationImportanceResult:
    """
    Permutation Importance の結果 (1特徴量)
    """
    feature_name: str
    # 各メトリクスの重要度 (baseline - shuffled)
    # 正の値 = その特徴量が有用
    delta_auc: float
    delta_logloss: float  # 負の値 = 有用 (loglossは小さいほど良い)
    delta_top1: float
    delta_top3: float
    delta_top5: float
    delta_mrr: float
    # ランク
    rank_auc: int = 0
    rank_logloss: int = 0
    rank_top1: int = 0
    rank_mrr: int = 0


@dataclass
class FeatureGroupImportanceResult:
    """
    Feature Group の重要度結果
    """
    group_name: str
    n_features: int
    # 集計重要度
    total_gain: float
    mean_gain: float
    max_gain: float
    total_split: int
    # 上位特徴量
    top_features: List[str]
    # Permutation (オプション)
    perm_delta_auc: Optional[float] = None
    perm_delta_mrr: Optional[float] = None


@dataclass
class SegmentPerformanceResult:
    """
    セグメント別パフォーマンス結果
    """
    segment_name: str  # e.g., "surface_id=0 (芝)"
    segment_key: str   # e.g., "surface_id"
    segment_value: Any
    # 基本統計
    n_races: int
    n_entries: int
    # メトリクス
    auc: float
    logloss: float
    top1_hit_rate: float
    top3_hit_rate: float
    top5_hit_rate: float
    mrr: float
    # 平均フィールドサイズ
    avg_field_size: float


@dataclass
class DiagnosticsReport:
    """
    診断レポート全体
    """
    # 基本情報
    dataset_name: str
    n_samples: int
    n_races: int
    n_features: int

    # LightGBM 重要度
    feature_importance: List[FeatureImportanceResult] = field(default_factory=list)

    # Permutation Importance
    permutation_importance: List[PermutationImportanceResult] = field(default_factory=list)

    # Feature Group Importance
    group_importance: List[FeatureGroupImportanceResult] = field(default_factory=list)

    # Segment Performance
    segment_performance: List[SegmentPerformanceResult] = field(default_factory=list)

    # メタ情報 (fail-soft 用)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    segment_skipped: bool = False
    segment_skip_reason: Optional[str] = None


# =============================================================================
# Feature Groups Definition
# =============================================================================

# プレフィックスによるグループ定義
FEATURE_GROUP_PREFIXES = {
    "base_race": ["place_id", "surface_id", "distance", "track_condition_id",
                  "course_turn_id", "course_inout_id", "race_year", "race_month",
                  "race_day_of_week", "race_no", "grade_id", "race_class_id",
                  "field_size", "waku", "umaban", "sex_id", "age", "weight"],
    "horse_form": ["h_"],
    "pace_position": ["pace_", "pos_"],
    "class_prize": ["class_", "prize_"],
    "jockey_trainer": ["j_", "t_"],
    "pedigree": ["ped_hash_", "anc_hash_"],
    "market": ["market_"],
}


def get_feature_group(feature_name: str) -> str:
    """
    特徴量名からグループ名を取得
    """
    # 1. prefix-based matching
    for group_name, prefixes in FEATURE_GROUP_PREFIXES.items():
        for prefix in prefixes:
            if feature_name.startswith(prefix):
                return group_name

    # 2. base_race の完全一致チェック
    base_race_exact = [
        "place_id", "surface_id", "distance", "distance_cat",
        "track_condition_id", "course_turn_id", "course_inout_id",
        "race_year", "race_month", "race_day_of_week", "race_no",
        "grade_id", "race_class_id", "field_size",
        "waku", "umaban", "umaban_norm", "sex_id", "age", "weight",
    ]
    if feature_name in base_race_exact:
        return "base_race"

    return "other"


# =============================================================================
# Segment Definitions
# =============================================================================

SEGMENT_DEFINITIONS = {
    "surface_id": {
        0: "芝",
        1: "ダート",
        2: "障害",
    },
    "distance_cat": {
        1000: "~1200m",
        1200: "1200-1400m",
        1400: "1400-1600m",
        1600: "1600-1800m",
        1800: "1800-2000m",
        2000: "2000-2200m",
        2200: "2200-2400m",
        2400: "2400m+",
        3000: "3000m+",
    },
    "track_condition_id": {
        0: "良",
        1: "稍重",
        2: "重",
        3: "不良",
    },
    "grade_id": {
        0: "G1",
        1: "G2",
        2: "G3",
        3: "OP",
        4: "Listed",
        5: "条件戦",
    },
    "field_size_cat": {
        "small": "~10頭",
        "medium": "11-14頭",
        "large": "15頭+",
    },
}


# =============================================================================
# LightGBM Feature Importance
# =============================================================================

def compute_lgbm_importance(
    model: Any,
    feature_cols: List[str],
) -> List[FeatureImportanceResult]:
    """
    LightGBM モデルから gain/split 重要度を計算

    Args:
        model: 学習済み LightGBM モデル
        feature_cols: 特徴量カラムのリスト

    Returns:
        FeatureImportanceResult のリスト (gain 降順)
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed")

    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")

    results = []
    for i, feat_name in enumerate(feature_cols):
        results.append(FeatureImportanceResult(
            feature_name=feat_name,
            gain=float(gain[i]),
            split=int(split[i]),
            gain_rank=0,  # 後で設定
            split_rank=0,
        ))

    # ランク計算 (gain 降順)
    results_sorted_gain = sorted(results, key=lambda x: x.gain, reverse=True)
    for rank, r in enumerate(results_sorted_gain, 1):
        r.gain_rank = rank

    # ランク計算 (split 降順)
    results_sorted_split = sorted(results, key=lambda x: x.split, reverse=True)
    for rank, r in enumerate(results_sorted_split, 1):
        r.split_rank = rank

    # gain 降順で返す
    return results_sorted_gain


# =============================================================================
# Permutation Importance
# =============================================================================

def _compute_metrics_for_df(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    race_ids: pd.Series,
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    データに対する各種メトリクスを計算

    Returns:
        {
            "auc": float,
            "logloss": float,
            "top1": float,
            "top3": float,
            "top5": float,
            "mrr": float,
        }
    """
    y_pred_proba = model.predict(X[feature_cols].fillna(-999), num_iteration=model.best_iteration)

    # AUC, LogLoss
    try:
        auc = roc_auc_score(y, y_pred_proba)
    except:
        auc = 0.5

    try:
        logloss_val = log_loss(y, y_pred_proba)
    except:
        logloss_val = 1.0

    # Ranking metrics (per-race)
    result_df = pd.DataFrame({
        "race_id": race_ids.values,
        "target": y.values,
        "pred_proba": y_pred_proba,
    })

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks = []
    n_races = 0

    for race_id, race_df in result_df.groupby("race_id"):
        if race_df["target"].sum() == 0:
            continue  # 勝ち馬なし

        n_races += 1
        race_df = race_df.sort_values("pred_proba", ascending=False).reset_index(drop=True)
        race_df["pred_rank"] = range(1, len(race_df) + 1)

        winner_row = race_df[race_df["target"] == 1].iloc[0]
        winner_rank = winner_row["pred_rank"]

        if winner_rank == 1:
            top1_hits += 1
        if winner_rank <= 3:
            top3_hits += 1
        if winner_rank <= 5:
            top5_hits += 1
        reciprocal_ranks.append(1.0 / winner_rank)

    if n_races == 0:
        return {
            "auc": auc,
            "logloss": logloss_val,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "mrr": 0.0,
        }

    return {
        "auc": auc,
        "logloss": logloss_val,
        "top1": top1_hits / n_races,
        "top3": top3_hits / n_races,
        "top5": top5_hits / n_races,
        "mrr": np.mean(reciprocal_ranks),
    }


def compute_permutation_importance(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_win",
    n_repeats: int = 3,
    random_state: int = 42,
    top_n: Optional[int] = None,
) -> List[PermutationImportanceResult]:
    """
    Permutation Importance を計算

    各特徴量をシャッフルして、メトリクスの変化を測定。
    変化が大きい = その特徴量が重要。

    Args:
        model: 学習済みモデル
        df: 評価データ (race_id, target_col, feature_cols が必要)
        feature_cols: 特徴量カラム
        target_col: ターゲットカラム
        n_repeats: シャッフル繰り返し回数
        random_state: 乱数シード
        top_n: 上位N個の特徴量のみ計算 (None = 全特徴量)

    Returns:
        PermutationImportanceResult のリスト
    """
    np.random.seed(random_state)

    X = df[feature_cols].copy()
    y = df[target_col]
    race_ids = df["race_id"]

    # ベースラインメトリクス
    logger.info("Computing baseline metrics...")
    baseline = _compute_metrics_for_df(model, X, y, race_ids, feature_cols)
    logger.info("  Baseline AUC: %.4f, LogLoss: %.4f, Top1: %.2f%%, MRR: %.4f",
                baseline["auc"], baseline["logloss"], baseline["top1"] * 100, baseline["mrr"])

    # 計算対象の特徴量を絞り込み
    if top_n is not None:
        # まず gain importance で上位を取得
        gain = model.feature_importance(importance_type="gain")
        top_indices = np.argsort(gain)[::-1][:top_n]
        target_features = [feature_cols[i] for i in top_indices]
    else:
        target_features = feature_cols

    results = []
    n_total = len(target_features)

    for idx, feat_name in enumerate(target_features):
        if (idx + 1) % 20 == 0:
            logger.info("  Progress: %d/%d features", idx + 1, n_total)

        deltas = {k: [] for k in ["auc", "logloss", "top1", "top3", "top5", "mrr"]}

        for repeat in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[feat_name] = np.random.permutation(X_shuffled[feat_name].values)

            shuffled_metrics = _compute_metrics_for_df(
                model, X_shuffled, y, race_ids, feature_cols
            )

            for key in deltas:
                if key == "logloss":
                    # logloss は小さいほど良いので、増加が重要度
                    deltas[key].append(shuffled_metrics[key] - baseline[key])
                else:
                    # 他は大きいほど良いので、減少が重要度
                    deltas[key].append(baseline[key] - shuffled_metrics[key])

        results.append(PermutationImportanceResult(
            feature_name=feat_name,
            delta_auc=float(np.mean(deltas["auc"])),
            delta_logloss=float(np.mean(deltas["logloss"])),
            delta_top1=float(np.mean(deltas["top1"])),
            delta_top3=float(np.mean(deltas["top3"])),
            delta_top5=float(np.mean(deltas["top5"])),
            delta_mrr=float(np.mean(deltas["mrr"])),
        ))

    # ランク計算
    for metric, attr in [
        ("auc", "rank_auc"),
        ("logloss", "rank_logloss"),
        ("top1", "rank_top1"),
        ("mrr", "rank_mrr"),
    ]:
        sorted_results = sorted(
            results,
            key=lambda x: getattr(x, f"delta_{metric}"),
            reverse=True
        )
        for rank, r in enumerate(sorted_results, 1):
            setattr(r, attr, rank)

    # AUC 降順でソート
    results.sort(key=lambda x: x.delta_auc, reverse=True)

    logger.info("Permutation importance computed for %d features", len(results))

    return results


# =============================================================================
# Feature Group Importance
# =============================================================================

def compute_group_importance(
    lgbm_importance: List[FeatureImportanceResult],
    perm_importance: Optional[List[PermutationImportanceResult]] = None,
) -> List[FeatureGroupImportanceResult]:
    """
    特徴量グループごとの重要度を集計

    Args:
        lgbm_importance: LightGBM 重要度リスト
        perm_importance: Permutation 重要度リスト (オプション)

    Returns:
        FeatureGroupImportanceResult のリスト
    """
    # グループごとに集計
    groups: Dict[str, List[FeatureImportanceResult]] = {}
    for feat in lgbm_importance:
        group_name = get_feature_group(feat.feature_name)
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(feat)

    # Permutation 重要度をマッピング
    perm_map = {}
    if perm_importance:
        for p in perm_importance:
            perm_map[p.feature_name] = p

    results = []
    for group_name, features in groups.items():
        gains = [f.gain for f in features]
        splits = [f.split for f in features]

        # 上位3特徴量 (gain順)
        top_features = sorted(features, key=lambda x: x.gain, reverse=True)[:3]
        top_feature_names = [f.feature_name for f in top_features]

        # Permutation 重要度の集計 (グループ内の全特徴量の平均)
        perm_auc = None
        perm_mrr = None
        if perm_importance:
            group_perm = [perm_map[f.feature_name] for f in features if f.feature_name in perm_map]
            if group_perm:
                perm_auc = np.mean([p.delta_auc for p in group_perm])
                perm_mrr = np.mean([p.delta_mrr for p in group_perm])

        results.append(FeatureGroupImportanceResult(
            group_name=group_name,
            n_features=len(features),
            total_gain=sum(gains),
            mean_gain=np.mean(gains),
            max_gain=max(gains),
            total_split=sum(splits),
            top_features=top_feature_names,
            perm_delta_auc=perm_auc,
            perm_delta_mrr=perm_mrr,
        ))

    # total_gain 降順でソート
    results.sort(key=lambda x: x.total_gain, reverse=True)

    return results


# =============================================================================
# Segment Performance
# =============================================================================

def compute_segment_performance(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_win",
    segment_keys: Optional[List[str]] = None,
) -> Tuple[List[SegmentPerformanceResult], List[str]]:
    """
    セグメント別のモデルパフォーマンスを計算

    Args:
        model: 学習済みモデル
        df: 評価データ
        feature_cols: 特徴量カラム
        target_col: ターゲットカラム
        segment_keys: セグメント化するカラム (None = デフォルト)

    Returns:
        Tuple of (SegmentPerformanceResult のリスト, 警告メッセージのリスト)
    """
    warnings = []

    if segment_keys is None:
        segment_keys = ["surface_id", "distance_cat", "track_condition_id", "grade_id"]

    # 欠損列をチェックして警告
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        msg = f"Missing {len(missing_cols)} feature columns in df (e.g., {missing_cols[:5]})"
        logger.warning(msg)
        warnings.append(msg)

    if len(available_cols) == 0:
        msg = "No feature columns available in df, skipping segment performance"
        logger.warning(msg)
        warnings.append(msg)
        return [], warnings

    # 予測確率を計算（利用可能な列のみ使用）
    X = df[available_cols].fillna(-999)
    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)

    work_df = df[["race_id", target_col] + [k for k in segment_keys if k in df.columns]].copy()
    work_df["pred_proba"] = y_pred_proba

    # field_size_cat を追加
    if "field_size" in df.columns:
        work_df["field_size"] = df["field_size"]
        work_df["field_size_cat"] = pd.cut(
            df["field_size"],
            bins=[0, 10, 14, 100],
            labels=["small", "medium", "large"]
        )
        if "field_size_cat" not in segment_keys:
            segment_keys = list(segment_keys) + ["field_size_cat"]

    results = []

    for seg_key in segment_keys:
        if seg_key not in work_df.columns:
            logger.warning("Segment key '%s' not found in data", seg_key)
            continue

        # セグメント定義を取得
        seg_def = SEGMENT_DEFINITIONS.get(seg_key, {})

        for seg_value in work_df[seg_key].dropna().unique():
            seg_df = work_df[work_df[seg_key] == seg_value]

            if len(seg_df) == 0:
                continue

            # セグメント名
            seg_label = seg_def.get(seg_value, str(seg_value))
            seg_name = f"{seg_key}={seg_value} ({seg_label})"

            # メトリクス計算
            y = seg_df[target_col]
            y_pred = seg_df["pred_proba"]

            try:
                auc = roc_auc_score(y, y_pred)
            except:
                auc = float("nan")

            try:
                logloss_val = log_loss(y, y_pred)
            except:
                logloss_val = float("nan")

            # Ranking metrics
            top1_hits = 0
            top3_hits = 0
            top5_hits = 0
            reciprocal_ranks = []
            n_races = 0
            field_sizes = []

            for race_id, race_df in seg_df.groupby("race_id"):
                if race_df[target_col].sum() == 0:
                    continue

                n_races += 1
                field_sizes.append(len(race_df))

                race_df = race_df.sort_values("pred_proba", ascending=False).reset_index(drop=True)
                race_df["pred_rank"] = range(1, len(race_df) + 1)

                winner_row = race_df[race_df[target_col] == 1].iloc[0]
                winner_rank = winner_row["pred_rank"]

                if winner_rank == 1:
                    top1_hits += 1
                if winner_rank <= 3:
                    top3_hits += 1
                if winner_rank <= 5:
                    top5_hits += 1
                reciprocal_ranks.append(1.0 / winner_rank)

            if n_races == 0:
                continue

            results.append(SegmentPerformanceResult(
                segment_name=seg_name,
                segment_key=seg_key,
                segment_value=seg_value,
                n_races=n_races,
                n_entries=len(seg_df),
                auc=auc,
                logloss=logloss_val,
                top1_hit_rate=top1_hits / n_races,
                top3_hit_rate=top3_hits / n_races,
                top5_hit_rate=top5_hits / n_races,
                mrr=np.mean(reciprocal_ranks),
                avg_field_size=np.mean(field_sizes),
            ))

    # segment_key, segment_value でソート
    results.sort(key=lambda x: (x.segment_key, str(x.segment_value)))

    return results, warnings


# =============================================================================
# Full Diagnostics
# =============================================================================

def run_diagnostics(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_win",
    dataset_name: str = "test",
    compute_perm: bool = True,
    perm_top_n: Optional[int] = 50,
    perm_n_repeats: int = 3,
    segment_keys: Optional[List[str]] = None,
) -> DiagnosticsReport:
    """
    フル診断を実行 (fail-soft: 部分エラーでも続行)

    Args:
        model: 学習済みモデル
        df: 評価データ
        feature_cols: 特徴量カラム
        target_col: ターゲットカラム
        dataset_name: データセット名
        compute_perm: Permutation Importance を計算するか
        perm_top_n: Permutation で評価する上位N特徴量 (None = 全て)
        perm_n_repeats: Permutation の繰り返し回数
        segment_keys: セグメントキー

    Returns:
        DiagnosticsReport
    """
    errors = []
    warnings = []
    segment_skipped = False
    segment_skip_reason = None

    logger.info("=" * 70)
    logger.info("Running Feature Diagnostics for %s", dataset_name)
    logger.info("=" * 70)
    logger.info("  Samples: %d", len(df))
    logger.info("  Races: %d", df["race_id"].nunique())
    logger.info("  Features: %d", len(feature_cols))

    # 利用可能な特徴量を確認
    available_feature_cols = [c for c in feature_cols if c in df.columns]
    missing_feature_cols = [c for c in feature_cols if c not in df.columns]
    if missing_feature_cols:
        msg = f"Missing {len(missing_feature_cols)} feature columns (e.g., {missing_feature_cols[:5]})"
        logger.warning(msg)
        warnings.append(msg)
        logger.info("  Available features: %d / %d", len(available_feature_cols), len(feature_cols))

    # 1. LightGBM Importance
    lgbm_imp = []
    logger.info("")
    logger.info("1. Computing LightGBM Importance...")
    try:
        lgbm_imp = compute_lgbm_importance(model, feature_cols)
        logger.info("   Top 10 (gain):")
        for i, feat in enumerate(lgbm_imp[:10], 1):
            logger.info("     %2d. %s: %.2f", i, feat.feature_name, feat.gain)
    except Exception as e:
        msg = f"LightGBM importance failed: {e}"
        logger.error(msg)
        errors.append(msg)

    # 2. Permutation Importance (オプション)
    perm_imp = []
    if compute_perm and available_feature_cols:
        logger.info("")
        logger.info("2. Computing Permutation Importance (top %s, %d repeats)...",
                    perm_top_n or "all", perm_n_repeats)
        try:
            perm_imp = compute_permutation_importance(
                model, df, available_feature_cols, target_col,
                n_repeats=perm_n_repeats,
                top_n=perm_top_n,
            )
            logger.info("   Top 10 (delta_auc):")
            for i, feat in enumerate(perm_imp[:10], 1):
                logger.info("     %2d. %s: AUC=%.4f, MRR=%.4f",
                            i, feat.feature_name, feat.delta_auc, feat.delta_mrr)
        except Exception as e:
            msg = f"Permutation importance failed: {e}"
            logger.error(msg)
            errors.append(msg)
    elif not available_feature_cols:
        msg = "Skipping permutation importance: no available feature columns"
        logger.warning(msg)
        warnings.append(msg)

    # 3. Feature Group Importance
    group_imp = []
    if lgbm_imp:
        logger.info("")
        logger.info("3. Computing Feature Group Importance...")
        try:
            group_imp = compute_group_importance(lgbm_imp, perm_imp if perm_imp else None)
            logger.info("   Group Summary:")
            for g in group_imp:
                perm_str = ""
                if g.perm_delta_auc is not None:
                    perm_str = f", perm_auc={g.perm_delta_auc:.4f}"
                logger.info("     %s: %d features, gain=%.2f%s",
                            g.group_name, g.n_features, g.total_gain, perm_str)
        except Exception as e:
            msg = f"Group importance failed: {e}"
            logger.error(msg)
            errors.append(msg)

    # 4. Segment Performance (fail-soft)
    seg_perf = []
    seg_warnings = []
    logger.info("")
    logger.info("4. Computing Segment Performance...")
    try:
        seg_perf, seg_warnings = compute_segment_performance(
            model, df, available_feature_cols, target_col, segment_keys
        )
        warnings.extend(seg_warnings)
        if seg_perf:
            logger.info("   Segment Summary:")
            for seg in seg_perf:
                logger.info("     %s: %d races, Top1=%.1f%%, MRR=%.3f",
                            seg.segment_name, seg.n_races,
                            seg.top1_hit_rate * 100, seg.mrr)
        else:
            segment_skipped = True
            segment_skip_reason = "No segment results computed"
            if seg_warnings:
                segment_skip_reason = "; ".join(seg_warnings)
            logger.warning("   Segment performance skipped: %s", segment_skip_reason)
    except Exception as e:
        segment_skipped = True
        segment_skip_reason = str(e)
        msg = f"Segment performance failed: {e}"
        logger.error(msg)
        errors.append(msg)

    logger.info("")
    if errors:
        logger.warning("Diagnostics completed with %d error(s)", len(errors))
    else:
        logger.info("Diagnostics complete.")

    return DiagnosticsReport(
        dataset_name=dataset_name,
        n_samples=len(df),
        n_races=df["race_id"].nunique(),
        n_features=len(feature_cols),
        feature_importance=lgbm_imp,
        permutation_importance=perm_imp,
        group_importance=group_imp,
        segment_performance=seg_perf,
        errors=errors,
        warnings=warnings,
        segment_skipped=segment_skipped,
        segment_skip_reason=segment_skip_reason,
    )


# =============================================================================
# Save Diagnostics
# =============================================================================

def save_diagnostics(
    report: DiagnosticsReport,
    output_dir: str,
    target_col: str = "target_win",
) -> None:
    """
    診断結果を保存

    Args:
        report: DiagnosticsReport
        output_dir: 出力ディレクトリ
        target_col: ターゲットカラム名 (ファイル名に使用)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = report.dataset_name
    suffix = f"{target_col}_{dataset}_v4"

    # 1. LightGBM Importance CSV
    if report.feature_importance:
        df_lgbm = pd.DataFrame([asdict(r) for r in report.feature_importance])
        lgbm_path = out_path / f"feature_importance_{suffix}.csv"
        df_lgbm.to_csv(lgbm_path, index=False)
        logger.info("Saved: %s", lgbm_path)

    # 2. Permutation Importance CSV
    if report.permutation_importance:
        df_perm = pd.DataFrame([asdict(r) for r in report.permutation_importance])
        perm_path = out_path / f"permutation_importance_{suffix}.csv"
        df_perm.to_csv(perm_path, index=False)
        logger.info("Saved: %s", perm_path)

    # 3. Group Importance CSV
    if report.group_importance:
        rows = []
        for g in report.group_importance:
            rows.append({
                "group_name": g.group_name,
                "n_features": g.n_features,
                "total_gain": g.total_gain,
                "mean_gain": g.mean_gain,
                "max_gain": g.max_gain,
                "total_split": g.total_split,
                "top_features": ", ".join(g.top_features),
                "perm_delta_auc": g.perm_delta_auc,
                "perm_delta_mrr": g.perm_delta_mrr,
            })
        df_group = pd.DataFrame(rows)
        group_path = out_path / f"group_importance_{suffix}.csv"
        df_group.to_csv(group_path, index=False)
        logger.info("Saved: %s", group_path)

    # 4. Segment Performance CSV
    if report.segment_performance:
        df_seg = pd.DataFrame([asdict(r) for r in report.segment_performance])
        seg_path = out_path / f"segment_performance_{suffix}.csv"
        df_seg.to_csv(seg_path, index=False)
        logger.info("Saved: %s", seg_path)

    # 5. Text Report
    report_path = out_path / f"diagnostics_report_{suffix}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Feature Diagnostics Report: {dataset}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Samples: {report.n_samples:,}\n")
        f.write(f"Races: {report.n_races:,}\n")
        f.write(f"Features: {report.n_features}\n\n")

        # Top 20 LightGBM Importance
        f.write("-" * 50 + "\n")
        f.write("Top 20 Features (LightGBM Gain)\n")
        f.write("-" * 50 + "\n")
        for i, feat in enumerate(report.feature_importance[:20], 1):
            f.write(f"{i:2d}. {feat.feature_name}: {feat.gain:.2f}\n")
        f.write("\n")

        # Top 20 Permutation Importance
        if report.permutation_importance:
            f.write("-" * 50 + "\n")
            f.write("Top 20 Features (Permutation - AUC)\n")
            f.write("-" * 50 + "\n")
            for i, feat in enumerate(report.permutation_importance[:20], 1):
                f.write(f"{i:2d}. {feat.feature_name}: delta_auc={feat.delta_auc:.4f}, delta_mrr={feat.delta_mrr:.4f}\n")
            f.write("\n")

        # Group Importance
        f.write("-" * 50 + "\n")
        f.write("Feature Group Importance\n")
        f.write("-" * 50 + "\n")
        for g in report.group_importance:
            f.write(f"{g.group_name}:\n")
            f.write(f"  Features: {g.n_features}\n")
            f.write(f"  Total Gain: {g.total_gain:.2f}\n")
            f.write(f"  Mean Gain: {g.mean_gain:.2f}\n")
            f.write(f"  Top: {', '.join(g.top_features[:3])}\n")
            if g.perm_delta_auc is not None:
                f.write(f"  Perm AUC: {g.perm_delta_auc:.4f}\n")
            f.write("\n")

        # Segment Performance
        f.write("-" * 50 + "\n")
        f.write("Segment Performance\n")
        f.write("-" * 50 + "\n")
        current_key = None
        for seg in report.segment_performance:
            if seg.segment_key != current_key:
                f.write(f"\n{seg.segment_key}:\n")
                current_key = seg.segment_key
            f.write(f"  {seg.segment_name}:\n")
            f.write(f"    Races: {seg.n_races:,}, Entries: {seg.n_entries:,}\n")
            f.write(f"    AUC: {seg.auc:.4f}, LogLoss: {seg.logloss:.4f}\n")
            f.write(f"    Top1: {seg.top1_hit_rate*100:.1f}%, Top3: {seg.top3_hit_rate*100:.1f}%, MRR: {seg.mrr:.4f}\n")

    logger.info("Saved: %s", report_path)

    # 6. JSON Summary
    summary = {
        "dataset_name": report.dataset_name,
        "n_samples": report.n_samples,
        "n_races": report.n_races,
        "n_features": report.n_features,
        "top10_features_gain": [
            {"name": f.feature_name, "gain": f.gain}
            for f in report.feature_importance[:10]
        ],
        "group_importance": [
            {"name": g.group_name, "total_gain": g.total_gain, "n_features": g.n_features}
            for g in report.group_importance
        ],
        # メタ情報
        "segment_skipped": report.segment_skipped,
        "segment_skip_reason": report.segment_skip_reason,
        "warnings": report.warnings if report.warnings else [],
        "errors": report.errors if report.errors else [],
    }
    if report.permutation_importance:
        summary["top10_features_perm_auc"] = [
            {"name": f.feature_name, "delta_auc": f.delta_auc, "delta_mrr": f.delta_mrr}
            for f in report.permutation_importance[:10]
        ]

    summary_path = out_path / f"diagnostics_summary_{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", summary_path)
