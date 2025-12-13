# src/scenario/horse_features.py

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# 脚質ラベル
RUN_STYLE_NIGE = "逃げ"
RUN_STYLE_SENKO = "先行"
RUN_STYLE_SASHI = "差し"
RUN_STYLE_OIKOMI = "追込"
RUN_STYLE_UNKNOWN = "不明"

RUN_STYLE_ORDER = [
    RUN_STYLE_NIGE,
    RUN_STYLE_SENKO,
    RUN_STYLE_SASHI,
    RUN_STYLE_OIKOMI,
]


@dataclass(frozen=True)
class RunStyleConfig:
    """
    脚質推定用のしきい値設定。
    ratio = 通過順位 / 頭数 で判断する。
    """
    nige_max_ratio: float = 0.20   # 0.0〜0.2 → 逃げ
    senko_max_ratio: float = 0.40 # 0.2〜0.4 → 先行
    sashi_max_ratio: float = 0.70 # 0.4〜0.7 → 差し
    # それ以上 → 追込


@dataclass(frozen=True)
class JockeyAggressivenessConfig:
    """
    騎手の「先行・逃げ気質」スコアを計算するための設定。
    """
    # 脚質 → 数値スコアへの変換
    score_nige: float = 1.0
    score_senko: float = 0.7
    score_sashi: float = 0.4
    score_oikomi: float = 0.1

    # 何鞍以上あればその騎手のスコアを信頼するか
    min_rides: int = 20

    # 騎手ごとのサンプルが少ないときに寄せる「全体平均」への重み
    shrinkage_strength: float = 0.5  # 0.0: 全く寄せない, 1.0: 全部平均に寄せる


# ==============================
# 1. 単レースの脚質推定
# ==============================

def _parse_passing(passing: str) -> List[int]:
    """
    'passing' 列（例: '1-1-1-1', '5-4-3-2'）を int のリストに変換する。
    パースに失敗したら空リスト。
    """
    if not isinstance(passing, str):
        return []

    raw = passing.replace(" ", "")
    if not raw:
        return []

    parts = raw.split("-")
    positions: List[int] = []
    for p in parts:
        try:
            positions.append(int(p))
        except ValueError:
            # '1(2)' みたいな変な値が来たらスキップ
            continue
    return positions


def classify_run_style_for_row(
    passing: Optional[str],
    field_size: Optional[int],
    config: RunStyleConfig,
) -> str:
    """
    単一レース（1走分）の passing / field_size から脚質を推定する。

    - ratio = 平均通過順位 / 頭数
    - ratio が小さいほど前に行っている＝「逃げ・先行」寄り
    """
    if field_size is None or field_size <= 0:
        return RUN_STYLE_UNKNOWN

    positions = _parse_passing(passing)
    if not positions:
        return RUN_STYLE_UNKNOWN

    avg_pos = sum(positions) / len(positions)
    ratio = avg_pos / float(field_size)

    if ratio <= config.nige_max_ratio:
        return RUN_STYLE_NIGE
    elif ratio <= config.senko_max_ratio:
        return RUN_STYLE_SENKO
    elif ratio <= config.sashi_max_ratio:
        return RUN_STYLE_SASHI
    else:
        return RUN_STYLE_OIKOMI


def infer_horse_run_style(
    horse_past_df: pd.DataFrame,
    config: RunStyleConfig | None = None,
    recent_n: int = 10,
) -> str:
    """
    特定の馬の過去走 DataFrame から「その馬の代表的な脚質」を推定する。

    - 直近 recent_n 走までを見る（古すぎる走りは無視）
    - 各レースで classify_run_style_for_row を呼ぶ
    - 最頻出の脚質を、その馬の脚質とみなす
    """
    if config is None:
        config = RunStyleConfig()

    if horse_past_df.empty:
        return RUN_STYLE_UNKNOWN

    # 日付があればそれでソートしたいが、ないケースも考えて race_id で擬似ソート
    df = horse_past_df.copy()
    if "race_date" in df.columns:
        df = df.sort_values("race_date")
    elif "race_id" in df.columns:
        df = df.sort_values("race_id")

    df = df.tail(recent_n)

    style_counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        style = classify_run_style_for_row(
            passing=row.get("passing"),
            field_size=row.get("field_size"),
            config=config,
        )
        if style == RUN_STYLE_UNKNOWN:
            continue
        style_counts[style] = style_counts.get(style, 0) + 1

    if not style_counts:
        return RUN_STYLE_UNKNOWN

    # 最頻出を採用（同数なら「逃げ > 先行 > 差し > 追込」の優先順）
    best_style = RUN_STYLE_UNKNOWN
    best_count = -1

    for style in RUN_STYLE_ORDER:
        cnt = style_counts.get(style, 0)
        if cnt > best_count:
            best_count = cnt
            best_style = style

    return best_style


# ==============================
# 2. 騎手の「先行・逃げ気質」スコア
# ==============================

def build_jockey_aggressiveness_map(
    horse_past_runs: pd.DataFrame,
    run_style_config: RunStyleConfig | None = None,
    config: JockeyAggressivenessConfig | None = None,
) -> Dict[str, float]:
    """
    horse_past_runs 全体から「騎手ごとの先行・逃げ気質スコア」を計算する。

    戻り値: {jockey_name: aggressiveness_score(0.0〜1.0)}
    """
    if run_style_config is None:
        run_style_config = RunStyleConfig()
    if config is None:
        config = JockeyAggressivenessConfig()

    if horse_past_runs.empty:
        return {}

    if "jockey" not in horse_past_runs.columns:
        logger.warning("horse_past_runs に 'jockey' 列がないため、騎手スコアを計算できません。")
        return {}

    df = horse_past_runs[
        horse_past_runs["jockey"].notna()
        & horse_past_runs["passing"].notna()
        & horse_past_runs["field_size"].notna()
    ].copy()

    if df.empty:
        return {}

    # 各レースごとに脚質を判定
    def _row_style(row) -> str:
        return classify_run_style_for_row(
            passing=row.get("passing"),
            field_size=row.get("field_size"),
            config=run_style_config,
        )

    df["run_style"] = df.apply(_row_style, axis=1)

    # 脚質 → スコア
    style_score_map = {
        RUN_STYLE_NIGE: config.score_nige,
        RUN_STYLE_SENKO: config.score_senko,
        RUN_STYLE_SASHI: config.score_sashi,
        RUN_STYLE_OIKOMI: config.score_oikomi,
    }

    df = df[df["run_style"] != RUN_STYLE_UNKNOWN].copy()
    if df.empty:
        return {}

    df["style_score"] = df["run_style"].map(style_score_map)

    grouped = df.groupby("jockey")["style_score"]
    mean_scores = grouped.mean()
    counts = grouped.count()

    if mean_scores.empty:
        return {}

    global_mean = float(mean_scores.mean())

    result: Dict[str, float] = {}
    for jockey_name, mean_score in mean_scores.items():
        n = int(counts.loc[jockey_name])
        # サンプルが少ない騎手は全体平均に寄せる（単純なシュリンク）
        if n < config.min_rides:
            w = config.shrinkage_strength
            score = (1.0 - w) * float(mean_score) + w * global_mean
        else:
            score = float(mean_score)
        # 0.0〜1.0 にクリップ
        score = max(0.0, min(1.0, score))
        result[jockey_name] = score

    logger.info(
        "Computed jockey aggressiveness for %d jockeys (global_mean=%.3f)",
        len(result),
        global_mean,
    )
    return result


# ==============================
# 3. 特定レースの horse_features を構築
# ==============================

def build_horse_features_for_race(
    conn: sqlite3.Connection,
    race_id: str,
    run_style_config: RunStyleConfig | None = None,
    jockey_config: JockeyAggressivenessConfig | None = None,
    recent_n_runs_for_style: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    1レース分の horse_features を構築する。

    返り値:
        {
            horse_id: {
                "frame_no": int | None,
                "run_style": str,  # "逃げ" / "先行" / "差し" / "追込" / "不明"
                "jockey_name": str | None,
                "jockey_aggressiveness": float | None,  # 0.0〜1.0
            },
            ...
        }
    """
    if run_style_config is None:
        run_style_config = RunStyleConfig()
    if jockey_config is None:
        jockey_config = JockeyAggressivenessConfig()

    # --- 1. race_results からそのレースの馬一覧を取得 ---
    race_results = pd.read_sql_query(
        """
        SELECT
            race_id,
            horse_id,
            horse_name,
            frame_no,
            jockey_name
        FROM race_results
        WHERE race_id = ?
        """,
        conn,
        params=[race_id],
    )

    if race_results.empty:
        logger.warning("race_results に race_id=%s のデータが見つかりません。", race_id)
        return {}

    horse_ids = race_results["horse_id"].dropna().unique().tolist()
    logger.info("Race %s: %d horses in race_results", race_id, len(horse_ids))

    # --- 2. horse_past_runs から対象馬の過去走を取得 ---
    if not horse_ids:
        return {}

    placeholders = ",".join(["?"] * len(horse_ids))
    horse_past_runs = pd.read_sql_query(
        f"""
        SELECT
            horse_id,
            race_id,
            race_date,
            field_size,
            passing,
            jockey  -- 名前
        FROM horse_past_runs
        WHERE horse_id IN ({placeholders})
        """,
        conn,
        params=horse_ids,
    )

    # --- 3. 全過去走から騎手の aggressiveness スコアを計算 ---
    #   ※ 本当は「全 DB の過去走」でやるのがベストだが、
    #     まずは対象馬ぶんだけで叩き台として動かす。
    jockey_aggr_map = build_jockey_aggressiveness_map(
        horse_past_runs=horse_past_runs,
        run_style_config=run_style_config,
        config=jockey_config,
    )

    # --- 4. 各馬の過去走から脚質を推定 ---
    horse_features: Dict[str, Dict[str, Any]] = {}

    for _, row in race_results.iterrows():
        hid = row.get("horse_id")
        if not hid:
            continue

        frame_no = row.get("frame_no")
        try:
            frame_no_int = int(frame_no) if frame_no is not None else None
        except (TypeError, ValueError):
            frame_no_int = None

        jockey_name = row.get("jockey_name")

        past_df = horse_past_runs[horse_past_runs["horse_id"] == hid]
        run_style = infer_horse_run_style(
            horse_past_df=past_df,
            config=run_style_config,
            recent_n=recent_n_runs_for_style,
        )

        # 騎手 aggressiveness は horse_past_runs 上の jockey 列をキーにする
        jockey_score = None
        if isinstance(jockey_name, str):
            jockey_score = jockey_aggr_map.get(jockey_name)

        horse_features[hid] = {
            "frame_no": frame_no_int,
            "run_style": run_style,
            "jockey_name": jockey_name,
            "jockey_aggressiveness": jockey_score,
        }

    logger.info(
        "Built horse_features for race %s: %d horses",
        race_id,
        len(horse_features),
    )
    return horse_features
