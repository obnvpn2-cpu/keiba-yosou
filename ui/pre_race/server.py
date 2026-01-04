#!/usr/bin/env python3
"""
Pre-race Scenario UI Server (v2.2)

シナリオUIを提供するローカルサーバー。
race_<race_id>.json を読み込み、シナリオ補正を適用し、結果を保存する。

v2.2 更新:
- 補正理由の構造化（カテゴリ別: pace, lane_bias, style_bias, track_condition, race_flow）
- 全体コメントをテンプレート化（概況/バイアス/展開/注目）
- 履歴ミニ要約（上位入れ替わり要約）
- 確率変化の明示（before/after + diff）

v2.1 更新:
- 日付フォルダ選択 → レース選択の2段階UX
- summary_<date>.json から人間向けレース一覧表示
- 保存先を artifacts/pre_race/outputs/<date>/<race_id>/<timestamp>.json に変更
- 履歴ロード機能追加
- エラーリカバリ強化

v2.0 更新:
- トラックバイアスを2軸に分離（進路バイアス + 脚質バイアス）
- メモ解析API追加（提案チップ生成）
- レース一覧にメタ情報付与
- 全体コメント生成
- 馬タグ生成

起動方法:
    python ui/pre_race/server.py

Usage:
    1. サーバー起動後、http://localhost:8080 にアクセス
    2. 日付フォルダを選択
    3. レースを選択（場所/R/レース名/距離/発走時刻で表示）
    4. シナリオを入力し「適用」をクリック
    5. 補正後のランキングと変化量を確認
    6. 「保存」で結果をJSONに出力（履歴から再読み込み可能）
"""

import json
import logging
import os
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
import socketserver

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_pre_race.memo_parser import MemoParser, parse_memo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Scenario Adjustment Logic (v2.0 - Two-axis Bias)
# =============================================================================

# Pace adjustment coefficients (脚質への影響)
PACE_COEF = {
    "S": {  # Slow pace - 前が有利
        "逃げ": {"win": 1.20, "in3": 1.12},
        "先行": {"win": 1.12, "in3": 1.08},
        "差し": {"win": 0.92, "in3": 0.95},
        "追込": {"win": 0.82, "in3": 0.88},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "M": {  # Medium pace - フラット
        "逃げ": {"win": 1.00, "in3": 1.00},
        "先行": {"win": 1.00, "in3": 1.00},
        "差し": {"win": 1.00, "in3": 1.00},
        "追込": {"win": 1.00, "in3": 1.00},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "H": {  # High pace - 差しが有利
        "逃げ": {"win": 0.80, "in3": 0.88},
        "先行": {"win": 0.90, "in3": 0.94},
        "差し": {"win": 1.12, "in3": 1.08},
        "追込": {"win": 1.22, "in3": 1.14},
        "不明": {"win": 1.00, "in3": 1.00},
    },
}

# Track condition adjustment
TRACK_COEF = {
    "良": 1.00,
    "稍重": 0.99,
    "重": 0.97,
    "不良": 0.94,
}

# 進路バイアス係数 (lane_bias)
# 「どのラインが伸びるか」に基づく補正
# 重要: 枠へ直結しない。推定される通り道（lane）への作用
LANE_BIAS_COEF = {
    "inner": {  # 内が伸びる馬場
        # 推定される通り道に対する補正
        "inner_lane": {"win": 1.06, "in3": 1.04},   # 内を通る馬
        "middle_lane": {"win": 1.00, "in3": 1.00}, # 中を通る馬
        "outer_lane": {"win": 0.96, "in3": 0.97},  # 外を回す馬
    },
    "middle": {  # 真ん中が良い
        "inner_lane": {"win": 1.00, "in3": 1.00},
        "middle_lane": {"win": 1.03, "in3": 1.02},
        "outer_lane": {"win": 1.00, "in3": 1.00},
    },
    "outer": {  # 外が伸びる馬場
        "inner_lane": {"win": 0.96, "in3": 0.97},
        "middle_lane": {"win": 1.00, "in3": 1.00},
        "outer_lane": {"win": 1.06, "in3": 1.04},
    },
    "flat": {  # バイアスなし
        "inner_lane": {"win": 1.00, "in3": 1.00},
        "middle_lane": {"win": 1.00, "in3": 1.00},
        "outer_lane": {"win": 1.00, "in3": 1.00},
    },
}

# 脚質バイアス係数 (style_bias)
# 「前が残るか/差しが届くか」に基づく補正
STYLE_BIAS_COEF = {
    "front": {  # 前が残りやすい
        "逃げ": {"win": 1.10, "in3": 1.06},
        "先行": {"win": 1.06, "in3": 1.04},
        "差し": {"win": 0.95, "in3": 0.97},
        "追込": {"win": 0.90, "in3": 0.94},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "closer": {  # 差しが届きやすい
        "逃げ": {"win": 0.90, "in3": 0.94},
        "先行": {"win": 0.95, "in3": 0.97},
        "差し": {"win": 1.06, "in3": 1.04},
        "追込": {"win": 1.10, "in3": 1.06},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "flat": {  # バイアスなし
        "逃げ": {"win": 1.00, "in3": 1.00},
        "先行": {"win": 1.00, "in3": 1.00},
        "差し": {"win": 1.00, "in3": 1.00},
        "追込": {"win": 1.00, "in3": 1.00},
        "不明": {"win": 1.00, "in3": 1.00},
    },
}


def estimate_run_style(
    umaban: int,
    front_runner_ids: List[str],
    horse_id: str,
    entry: Optional[Dict[str, Any]] = None
) -> str:
    """
    脚質を推定

    優先順位:
    1. JSONに脚質情報があればそれを使用
    2. front_runner_ids に含まれていれば逃げ
    3. 馬番から弱い推定（不確実）
    """
    # 1. JSONに脚質情報があれば使用
    if entry:
        run_style = entry.get("run_style") or entry.get("style")
        if run_style and run_style in ["逃げ", "先行", "差し", "追込"]:
            return run_style

    # 2. 指定された逃げ馬
    if horse_id in front_runner_ids:
        return "逃げ"

    # 3. 馬番から弱い推定（あくまで補助）
    # 内枠は先行寄り、外枠は差し寄りの傾向はあるが確実ではない
    if umaban <= 4:
        return "先行"
    elif umaban <= 10:
        return "差し"
    else:
        return "追込"


def estimate_lane(
    umaban: int,
    field_size: int,
    run_style: str,
    entry: Optional[Dict[str, Any]] = None
) -> str:
    """
    推定される通り道（lane）を推定

    重要: 枠番 ≠ 通り道
    - 内枠の差し馬は外へ持ち出すことがある
    - 外枠の逃げ馬は内に切れ込む
    - これらを考慮した「推定」を行う

    Returns:
        "inner_lane", "middle_lane", "outer_lane"
    """
    # JSONにlane情報があれば使用
    if entry:
        lane = entry.get("estimated_lane")
        if lane and lane in ["inner_lane", "middle_lane", "outer_lane"]:
            return lane

    # 枠と脚質から推定
    if field_size <= 8:
        inner_max = 2
        outer_min = 7
    else:
        inner_max = max(2, int(field_size * 0.25))
        outer_min = field_size - max(2, int(field_size * 0.25)) + 1

    # 基本は枠から推定
    if umaban <= inner_max:
        base_lane = "inner"
    elif umaban >= outer_min:
        base_lane = "outer"
    else:
        base_lane = "middle"

    # 脚質による調整（弱い推定）
    # 差し・追込は外へ持ち出す傾向がある（ただし確実ではない）
    if run_style in ["差し", "追込"] and base_lane == "inner":
        # 内枠の差し馬は中~外へ持ち出す可能性
        return "middle_lane"
    elif run_style in ["逃げ", "先行"] and base_lane == "outer":
        # 外枠の逃げ馬は内へ切れ込む可能性
        return "middle_lane"

    return f"{base_lane}_lane"


def generate_horse_tags(
    entry: Dict[str, Any],
    run_style: str,
    lane: str,
    is_front_runner: bool
) -> List[str]:
    """馬ごとのタグを生成"""
    tags = []

    # 脚質タグ
    if is_front_runner:
        tags.append("逃げ想定")
    elif run_style:
        tags.append(f"{run_style}想定")

    # 推定進路タグ（弱い推定であることを示す）
    lane_labels = {
        "inner_lane": "内目通過想定",
        "middle_lane": "中通過想定",
        "outer_lane": "外目通過想定",
    }
    if lane in lane_labels:
        tags.append(lane_labels[lane])

    # 特徴量ベースのタグ（JSONにあれば）
    if entry.get("h_days_since_last") is not None:
        days = entry.get("h_days_since_last", 0)
        if days > 90:
            tags.append("休み明け")
        elif days > 60:
            tags.append("間隔あり")

    if entry.get("h_recent3_avg_finish") is not None:
        avg_finish = entry.get("h_recent3_avg_finish", 10)
        if avg_finish <= 2.0:
            tags.append("近走好調")
        elif avg_finish >= 8.0:
            tags.append("近走不振")

    if entry.get("h_recent3_avg_last3f") is not None:
        avg_last3f = entry.get("h_recent3_avg_last3f", 35)
        if avg_last3f <= 34.0:
            tags.append("末脚○")
        elif avg_last3f >= 37.0:
            tags.append("末脚△")

    return tags


def generate_race_comment(
    race_data: Dict[str, Any],
    scenario: Dict[str, Any],
    adjusted_entries: List[Dict[str, Any]]
) -> str:
    """
    全体コメントを生成（後方互換 - 旧形式のテキスト）

    入力情報のみから説明可能な範囲で生成（捏造禁止）
    """
    parts = []

    race_name = race_data.get("name") or race_data.get("race_name") or ""
    distance = race_data.get("distance", 0)
    course = race_data.get("course", "")

    # レース基本情報
    if race_name:
        parts.append(f"【{race_name}】")
    if distance and course:
        parts.append(f"{course}{distance}m。")

    # ペース予想
    pace = scenario.get("pace", "M")
    pace_labels = {"S": "スローペース", "M": "平均ペース", "H": "ハイペース"}
    parts.append(f"予想ペースは{pace_labels.get(pace, '不明')}。")

    # 脚質バイアス
    style_bias = scenario.get("style_bias", "flat")
    if style_bias == "front":
        parts.append("前残り傾向で逃げ・先行馬に有利な展開が見込まれる。")
    elif style_bias == "closer":
        parts.append("差し・追込が届きやすい展開が見込まれる。")

    # 進路バイアス
    lane_bias = scenario.get("lane_bias", "flat")
    if lane_bias == "inner":
        parts.append("内めのラインが伸びる馬場状態。")
    elif lane_bias == "outer":
        parts.append("外めのラインが伸びる馬場状態。")
    elif lane_bias == "middle":
        parts.append("馬場中央が良い状態。")

    # 馬場状態
    track = scenario.get("track_condition", "良")
    if track in ["重", "不良"]:
        parts.append(f"馬場は{track}で、パワーが求められる。")
    elif track == "稍重":
        parts.append("やや時計がかかる馬場。")

    # 逃げ馬想定
    front_runners = [e for e in adjusted_entries if e.get("run_style") == "逃げ"]
    if len(front_runners) == 1:
        parts.append(f"逃げは{front_runners[0].get('name', '?')}が想定される単騎逃げ見込み。")
    elif len(front_runners) >= 2:
        names = "、".join([e.get("name", "?") for e in front_runners[:3]])
        parts.append(f"逃げ候補は{names}など複数おり、競り合いの可能性あり。")

    # メモからの補足
    notes = scenario.get("notes", "")
    if notes:
        parts.append(f"補足：{notes[:100]}")

    return " ".join(parts)


def generate_race_comment_structured(
    race_data: Dict[str, Any],
    scenario: Dict[str, Any],
    adjusted_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    v2.2/A4: 構造化された全体コメントを生成

    セクション:
    1) レース概況（頭数/距離/馬場/想定ペース）
    2) バイアス見立て（lane_bias/style_bias の意味と今回の入力）
    3) 展開の焦点（逃げ先行の数/隊列のイメージ：メモ由来 race_flow 反映分のみ）
    4) 注目ポイント（上位入れ替わり要因：理由カテゴリから要約）

    A4: 捏造禁止ルール:
    - 全ての情報は入力データ（race_data, scenario, entries）からのみ生成
    - DB/特徴量に存在しない情報（血統等）は含めない
    - 推測・憶測は含めない（「〜かもしれない」等は書かない）
    """
    race_name = race_data.get("name") or race_data.get("race_name") or ""
    distance = race_data.get("distance", 0)
    course = race_data.get("course", "")
    place = race_data.get("place", "")
    head_count = len(adjusted_entries)

    pace = scenario.get("pace", "M")
    track_condition = scenario.get("track_condition", "良")
    lane_bias = scenario.get("lane_bias", "flat")
    style_bias = scenario.get("style_bias", "flat")

    # 1) レース概況
    overview_parts = []
    if race_name:
        overview_parts.append(f"【{race_name}】")
    if place:
        overview_parts.append(f"{place}")
    if course and distance:
        overview_parts.append(f"{course}{distance}m")
    overview_parts.append(f"{head_count}頭立て")

    pace_labels = {"S": "スローペース", "M": "平均ペース", "H": "ハイペース"}
    overview_parts.append(f"予想ペース：{pace_labels.get(pace, '不明')}")

    track_labels = {"良": "良馬場", "稍重": "稍重馬場", "重": "重馬場", "不良": "不良馬場"}
    overview_parts.append(f"馬場：{track_labels.get(track_condition, track_condition)}")

    overview = "／".join(overview_parts)

    # 2) バイアス見立て
    bias_parts = []

    lane_bias_labels = {
        "inner": "内伸び（内ラチ沿いが有利）",
        "middle": "中央良好（馬場中央が走りやすい）",
        "outer": "外伸び（外めのラインが有利）",
        "flat": "フラット（進路バイアスなし）",
    }
    bias_parts.append(f"進路：{lane_bias_labels.get(lane_bias, lane_bias)}")

    style_bias_labels = {
        "front": "前残り傾向（逃げ・先行有利）",
        "closer": "差し届き傾向（差し・追込有利）",
        "flat": "フラット（脚質バイアスなし）",
    }
    bias_parts.append(f"脚質：{style_bias_labels.get(style_bias, style_bias)}")

    bias = "／".join(bias_parts)

    # 3) 展開の焦点
    front_runners = [e for e in adjusted_entries if e.get("run_style") == "逃げ"]
    senkou = [e for e in adjusted_entries if e.get("run_style") == "先行"]

    focus_parts = []
    if len(front_runners) == 0:
        focus_parts.append("逃げ馬不在")
    elif len(front_runners) == 1:
        focus_parts.append(f"逃げ候補は{front_runners[0].get('name', '?')}の単騎想定")
    elif len(front_runners) >= 2:
        names = "・".join([e.get("name", "?") for e in front_runners[:3]])
        focus_parts.append(f"逃げ候補複数（{names}）→ペース競り合いの可能性")

    n_front = len(front_runners) + len(senkou)
    if n_front >= 5:
        focus_parts.append("先行馬多め→ペース流れやすい")
    elif n_front <= 2:
        focus_parts.append("先行馬少→スロー傾向")

    focus = "。".join(focus_parts) if focus_parts else "特記なし"

    # 4) 注目ポイント（上位入れ替わり要因）
    highlight_parts = []

    # 大きな順位変化のある馬を抽出
    big_changes = sorted(
        [e for e in adjusted_entries if abs(e.get("rank_change_win", 0)) >= 2],
        key=lambda x: abs(x.get("rank_change_win", 0)),
        reverse=True
    )[:3]

    for entry in big_changes:
        change = entry.get("rank_change_win", 0)
        arrow = "↑" if change > 0 else "↓"
        reasons_struct = entry.get("reasons_structured", {})

        # 理由カテゴリから主要因を特定
        main_reason = None
        for cat in ["pace", "style_bias", "lane_bias"]:
            if cat in reasons_struct:
                main_reason = reasons_struct[cat].get("text", "")
                break

        if main_reason:
            highlight_parts.append(
                f"{entry.get('name', '?')} {arrow}{abs(change)}位（{main_reason}）"
            )

    highlight = "／".join(highlight_parts) if highlight_parts else "大きな順位変動なし"

    # 補足（メモ）
    notes = scenario.get("notes", "")
    supplement = notes[:150] if notes else ""

    return {
        "overview": overview,
        "bias": bias,
        "focus": focus,
        "highlight": highlight,
        "supplement": supplement,
    }


def generate_rank_mini_summary(adjusted_entries: List[Dict[str, Any]]) -> str:
    """
    v2.2/A3: 履歴用のミニ要約を生成（直感的な形式）

    例: "◎ Horse1 (3→1 ↑2), ○ Horse2 (1→2), ▲ Horse3 (5→3 ↑2)"
    """
    # 補正後ランキングでソート
    sorted_entries = sorted(
        adjusted_entries,
        key=lambda x: x.get("adj_rank_win", 99)
    )

    rank_symbols = ["◎", "○", "▲", "△", "☆"]
    summary_parts = []

    for i, entry in enumerate(sorted_entries[:3], 0):
        base_rank = entry.get("base_rank_win", 99)
        adj_rank = entry.get("adj_rank_win", 99)
        name = entry.get("name", "?")
        symbol = rank_symbols[i] if i < len(rank_symbols) else ""

        if base_rank != adj_rank:
            change = base_rank - adj_rank  # 正=順位上昇
            arrow = f"↑{change}" if change > 0 else f"↓{abs(change)}"
            summary_parts.append(f"{symbol}{name}({base_rank}→{adj_rank} {arrow})")
        else:
            summary_parts.append(f"{symbol}{name}({adj_rank})")

    return " ".join(summary_parts)


def apply_scenario_adjustment(
    race_data: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """
    シナリオ補正を適用（v2.0）

    Args:
        race_data: race_<race_id>.json の内容
        scenario: シナリオ入力
            - pace: "S" / "M" / "H"
            - track_condition: "良" / "稍重" / "重" / "不良"
            - lane_bias: "inner" / "middle" / "outer" / "flat"
            - style_bias: "front" / "closer" / "flat"
            - front_runner_ids: ["h001", ...]
            - notes: "自由記述"

    Returns:
        補正結果
    """
    entries = race_data.get("entries", [])
    field_size = len(entries)

    pace = scenario.get("pace", "M")
    track_condition = scenario.get("track_condition", "良")
    lane_bias = scenario.get("lane_bias", "flat")
    style_bias = scenario.get("style_bias", "flat")
    front_runner_ids = scenario.get("front_runner_ids", [])

    # 補正結果を格納
    adjusted_entries = []

    for entry in entries:
        horse_id = entry.get("horse_id", "")
        umaban = entry.get("umaban", 0) or 0
        name = entry.get("name", "")

        base_p_win = entry.get("p_win", 0.0) or 0.0
        base_p_in3 = entry.get("p_in3", 0.0) or 0.0
        base_rank_win = entry.get("rank_win", 99)
        base_rank_in3 = entry.get("rank_in3", 99)

        is_front_runner = horse_id in front_runner_ids

        # 脚質推定
        run_style = estimate_run_style(umaban, front_runner_ids, horse_id, entry)

        # 進路推定（枠番 + 脚質から）
        lane = estimate_lane(umaban, field_size, run_style, entry)

        # 補正係数を計算
        pace_coef = PACE_COEF.get(pace, PACE_COEF["M"]).get(run_style, {"win": 1.0, "in3": 1.0})
        track_coef = TRACK_COEF.get(track_condition, 1.0)
        lane_coef = LANE_BIAS_COEF.get(lane_bias, LANE_BIAS_COEF["flat"]).get(lane, {"win": 1.0, "in3": 1.0})
        style_coef = STYLE_BIAS_COEF.get(style_bias, STYLE_BIAS_COEF["flat"]).get(run_style, {"win": 1.0, "in3": 1.0})

        # 補正後確率
        adj_p_win = base_p_win * pace_coef["win"] * track_coef * lane_coef["win"] * style_coef["win"]
        adj_p_in3 = base_p_in3 * pace_coef["in3"] * track_coef * lane_coef["in3"] * style_coef["in3"]

        # 補正理由を生成（カテゴリ別の構造化）
        reasons = []  # 後方互換のためのフラットリスト
        reasons_structured = {}  # v2.2: カテゴリ別の構造化

        # ペース補正
        if pace != "M":
            if pace_coef["win"] > 1.0:
                reason_text = f"ペース{pace}で{run_style}有利 (+{(pace_coef['win']-1)*100:.0f}%)"
                reasons.append(reason_text)
                reasons_structured["pace"] = {
                    "category": "pace",
                    "input": pace,
                    "effect": "positive",
                    "text": reason_text,
                    "coef": round(pace_coef["win"], 3),
                }
            elif pace_coef["win"] < 1.0:
                reason_text = f"ペース{pace}で{run_style}不利 ({(pace_coef['win']-1)*100:.0f}%)"
                reasons.append(reason_text)
                reasons_structured["pace"] = {
                    "category": "pace",
                    "input": pace,
                    "effect": "negative",
                    "text": reason_text,
                    "coef": round(pace_coef["win"], 3),
                }

        # 脚質バイアス補正
        if style_bias != "flat":
            style_label = {"front": "前残り", "closer": "差し向き"}.get(style_bias, style_bias)
            if style_coef["win"] > 1.0:
                reason_text = f"{style_label}バイアスで有利 (+{(style_coef['win']-1)*100:.0f}%)"
                reasons.append(reason_text)
                reasons_structured["style_bias"] = {
                    "category": "style_bias",
                    "input": style_bias,
                    "effect": "positive",
                    "text": reason_text,
                    "coef": round(style_coef["win"], 3),
                }
            elif style_coef["win"] < 1.0:
                reason_text = f"{style_label}バイアスで不利 ({(style_coef['win']-1)*100:.0f}%)"
                reasons.append(reason_text)
                reasons_structured["style_bias"] = {
                    "category": "style_bias",
                    "input": style_bias,
                    "effect": "negative",
                    "text": reason_text,
                    "coef": round(style_coef["win"], 3),
                }

        # 進路バイアス補正
        if lane_bias != "flat":
            lane_label = {"inner": "内伸び", "middle": "中央", "outer": "外伸び"}.get(lane_bias, lane_bias)
            est_lane_label = {"inner_lane": "内目", "middle_lane": "中", "outer_lane": "外目"}.get(lane, lane)
            if lane_coef["win"] > 1.0:
                reason_text = f"{lane_label}馬場で{est_lane_label}通過有利 (+{(lane_coef['win']-1)*100:.0f}%)"
                reasons.append(reason_text)
                reasons_structured["lane_bias"] = {
                    "category": "lane_bias",
                    "input": lane_bias,
                    "effect": "positive",
                    "text": reason_text,
                    "coef": round(lane_coef["win"], 3),
                }
            elif lane_coef["win"] < 1.0:
                reason_text = f"{lane_label}馬場で{est_lane_label}通過不利 ({(lane_coef['win']-1)*100:.0f}%)"
                reasons.append(reason_text)
                reasons_structured["lane_bias"] = {
                    "category": "lane_bias",
                    "input": lane_bias,
                    "effect": "negative",
                    "text": reason_text,
                    "coef": round(lane_coef["win"], 3),
                }

        # 逃げ想定馬
        if is_front_runner:
            reasons.append("逃げ想定馬")
            reasons_structured["race_flow"] = {
                "category": "race_flow",
                "input": "front_runner",
                "effect": "info",
                "text": "逃げ想定馬",
                "coef": None,
            }

        # 馬場状態
        if track_coef < 1.0:
            reason_text = f"馬場{track_condition}で減衰 ({(track_coef-1)*100:.0f}%)"
            reasons.append(reason_text)
            reasons_structured["track_condition"] = {
                "category": "track_condition",
                "input": track_condition,
                "effect": "negative",
                "text": reason_text,
                "coef": round(track_coef, 3),
            }

        # タグ生成
        tags = generate_horse_tags(entry, run_style, lane, is_front_runner)

        adjusted_entries.append({
            "horse_id": horse_id,
            "umaban": umaban,
            "name": name,
            "base_p_win": round(base_p_win, 4),
            "base_p_in3": round(base_p_in3, 4),
            "base_rank_win": base_rank_win,
            "base_rank_in3": base_rank_in3,
            "adj_p_win": round(adj_p_win, 6),
            "adj_p_in3": round(adj_p_in3, 6),
            "run_style": run_style,
            "lane": lane,
            "reasons": reasons,
            "reasons_structured": reasons_structured,  # v2.2: カテゴリ別構造化
            "tags": tags,
            "jockey": entry.get("jockey"),
            "trainer": entry.get("trainer"),
        })

    # 正規化
    total_win = sum(e["adj_p_win"] for e in adjusted_entries)
    total_in3 = sum(e["adj_p_in3"] for e in adjusted_entries)

    if total_win > 0:
        for e in adjusted_entries:
            e["adj_p_win"] = round(e["adj_p_win"] / total_win, 4)

    if total_in3 > 3.0:
        scale = 3.0 / total_in3
        for e in adjusted_entries:
            e["adj_p_in3"] = round(e["adj_p_in3"] * scale, 4)

    # 補正後ランキングを計算
    sorted_by_win = sorted(adjusted_entries, key=lambda x: x["adj_p_win"], reverse=True)
    for rank, entry in enumerate(sorted_by_win, 1):
        entry["adj_rank_win"] = rank
        entry["rank_change_win"] = entry["base_rank_win"] - rank  # 正=上昇、負=下降

    sorted_by_in3 = sorted(adjusted_entries, key=lambda x: x["adj_p_in3"], reverse=True)
    for rank, entry in enumerate(sorted_by_in3, 1):
        entry["adj_rank_in3"] = rank
        entry["rank_change_in3"] = entry["base_rank_in3"] - rank

    # umaban順に並べ直す
    adjusted_entries.sort(key=lambda x: x["umaban"])

    # 全体コメント生成
    race_comment = generate_race_comment(race_data, scenario, adjusted_entries)
    race_comment_structured = generate_race_comment_structured(race_data, scenario, adjusted_entries)
    rank_mini_summary = generate_rank_mini_summary(adjusted_entries)

    return {
        "race_id": race_data.get("race_id"),
        "race_name": race_data.get("name") or race_data.get("race_name"),
        "date": race_data.get("date"),
        "place": race_data.get("place"),
        "race_no": race_data.get("race_no"),
        "distance": race_data.get("distance"),
        "course": race_data.get("course"),
        "grade": race_data.get("grade"),
        "scenario": scenario,
        "entries": adjusted_entries,
        "race_comment": race_comment,
        "race_comment_structured": race_comment_structured,  # v2.2: 構造化コメント
        "rank_mini_summary": rank_mini_summary,  # v2.2: 履歴用ミニ要約
        "adjusted_at": datetime.now().isoformat(),
    }


# =============================================================================
# HTTP Server
# =============================================================================

class ScenarioUIHandler(SimpleHTTPRequestHandler):
    """シナリオUIのHTTPハンドラー"""

    def __init__(self, *args, **kwargs):
        # UI directory as the base
        self.ui_dir = Path(__file__).parent
        super().__init__(*args, directory=str(self.ui_dir), **kwargs)

    def do_GET(self):
        """GET リクエスト処理"""
        parsed = urlparse(self.path)

        if parsed.path == "/api/list-dates":
            self._handle_list_dates(parsed)
        elif parsed.path == "/api/list-races":
            self._handle_list_races(parsed)
        elif parsed.path == "/api/load-race":
            self._handle_load_race(parsed)
        elif parsed.path == "/api/list-history":
            self._handle_list_history(parsed)
        elif parsed.path == "/api/load-history":
            self._handle_load_history(parsed)
        else:
            # 静的ファイル配信
            super().do_GET()

    def do_POST(self):
        """POST リクエスト処理"""
        parsed = urlparse(self.path)

        if parsed.path == "/api/apply-scenario":
            self._handle_apply_scenario()
        elif parsed.path == "/api/save-result":
            self._handle_save_result()
        elif parsed.path == "/api/parse-memo":
            self._handle_parse_memo()
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data: Any, status: int = 200):
        """JSON レスポンスを送信"""
        response = json.dumps(data, ensure_ascii=False, indent=2)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(response.encode("utf-8")))
        self.end_headers()
        self.wfile.write(response.encode("utf-8"))

    def _handle_list_dates(self, parsed):
        """日付フォルダ一覧を返す (A2: エラー時のガイダンス付き)"""
        artifacts_dir = PROJECT_ROOT / "artifacts" / "pre_race"
        dates = []

        if not artifacts_dir.exists():
            self._send_json({
                "dates": [],
                "error": "artifacts/pre_race/ ディレクトリが見つかりません",
                "guidance": "python scripts/generate_pre_race_materials.py --date YYYY-MM-DD を実行してデータを生成してください。",
            })
            return

        for date_dir in sorted(artifacts_dir.iterdir(), reverse=True):
            if date_dir.is_dir() and date_dir.name != "outputs":
                # Check for summary file or race files
                summary_file = date_dir / f"summary_{date_dir.name}.json"
                race_files = list(date_dir.glob("race_*.json"))

                if summary_file.exists() or race_files:
                    n_races = 0
                    # Try to get race count from summary
                    if summary_file.exists():
                        try:
                            with open(summary_file, "r", encoding="utf-8") as f:
                                summary = json.load(f)
                            n_races = summary.get("n_races", len(summary.get("races", [])))
                        except Exception:
                            n_races = len(race_files)
                    else:
                        n_races = len(race_files)

                    dates.append({
                        "date": date_dir.name,
                        "n_races": n_races,
                        "has_summary": summary_file.exists(),
                    })

        if not dates:
            self._send_json({
                "dates": [],
                "warning": "日付フォルダが見つかりません",
                "guidance": "python scripts/generate_pre_race_materials.py --date YYYY-MM-DD を実行してデータを生成してください。",
            })
            return

        self._send_json({"dates": dates[:30]})  # Latest 30 dates

    def _handle_list_races(self, parsed):
        """指定日のレース一覧を返す（summary JSONから人間向け表示）"""
        query = parse_qs(parsed.query)
        target_date = query.get("date", [""])[0]

        artifacts_dir = PROJECT_ROOT / "artifacts" / "pre_race"
        races = []

        if not target_date:
            # 日付指定なし: 全日付のレース一覧（後方互換）
            if artifacts_dir.exists():
                for date_dir in sorted(artifacts_dir.iterdir(), reverse=True):
                    if date_dir.is_dir() and date_dir.name != "outputs":
                        races.extend(self._load_races_for_date(date_dir))
            self._send_json({"races": races[:50]})
            return

        # 日付指定あり: summary JSONを優先的に使用
        date_dir = artifacts_dir / target_date
        if not date_dir.exists():
            self._send_json({"races": [], "error": f"Date folder not found: {target_date}"})
            return

        summary_file = date_dir / f"summary_{target_date}.json"

        if summary_file.exists():
            # summary JSONから人間向け情報を取得
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                for race_info in summary.get("races", []):
                    race_id = race_info.get("race_id", "")
                    race_file = date_dir / f"race_{race_id}.json"

                    place = race_info.get("place", "")
                    race_no = race_info.get("race_no")
                    race_name = race_info.get("name", "")
                    grade = race_info.get("grade", "")
                    distance = race_info.get("distance")
                    n_entries = race_info.get("n_entries", 0)

                    # Get start_time from race file if available
                    start_time = ""
                    if race_file.exists():
                        try:
                            with open(race_file, "r", encoding="utf-8") as rf:
                                race_data = json.load(rf)
                            start_time = race_data.get("start_time", "")
                            if not n_entries:
                                n_entries = len(race_data.get("entries", []))
                        except Exception:
                            pass

                    # Human-readable label: 中山5R 有馬記念(G1) 芝2500m 15:25 12頭
                    label_parts = []
                    if place:
                        label_parts.append(place)
                    if race_no:
                        label_parts.append(f"{race_no}R")
                    if race_name:
                        label_parts.append(race_name)
                    if grade:
                        label_parts.append(f"({grade})")
                    if distance:
                        label_parts.append(f"{distance}m")
                    if start_time:
                        label_parts.append(start_time)
                    if n_entries:
                        label_parts.append(f"{n_entries}頭")

                    races.append({
                        "path": str(race_file.relative_to(PROJECT_ROOT)) if race_file.exists() else "",
                        "date": target_date,
                        "race_id": race_id,
                        "label": " ".join(label_parts),
                        "meta": {
                            "place": place,
                            "race_no": race_no,
                            "race_name": race_name,
                            "grade": grade,
                            "distance": distance,
                            "start_time": start_time,
                            "n_entries": n_entries,
                        }
                    })

            except Exception as e:
                logger.warning(f"Failed to read summary {summary_file}: {e}")
                # Fall back to reading individual race files
                races = self._load_races_for_date(date_dir)
        else:
            # No summary file, read individual race files
            races = self._load_races_for_date(date_dir)

        self._send_json({"races": races})

    def _load_races_for_date(self, date_dir: Path) -> List[Dict[str, Any]]:
        """日付ディレクトリからレース一覧を読み込む（フォールバック用）"""
        races = []
        for race_file in sorted(date_dir.glob("race_*.json")):
            try:
                with open(race_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                race_id = data.get("race_id") or race_file.stem.replace("race_", "")
                race_name = data.get("name") or data.get("race_name") or ""
                place = data.get("place") or ""
                race_no = data.get("race_no")
                distance = data.get("distance")
                course = data.get("course") or ""
                grade = data.get("grade") or ""
                start_time = data.get("start_time") or ""
                head_count = data.get("head_count") or len(data.get("entries", []))

                # Human-readable label
                label_parts = []
                if place:
                    label_parts.append(place)
                if race_no:
                    label_parts.append(f"{race_no}R")
                if race_name:
                    label_parts.append(race_name)
                if grade:
                    label_parts.append(f"({grade})")
                if course and distance:
                    label_parts.append(f"{course}{distance}m")
                elif distance:
                    label_parts.append(f"{distance}m")
                if start_time:
                    label_parts.append(start_time)
                if head_count:
                    label_parts.append(f"{head_count}頭")

                races.append({
                    "path": str(race_file.relative_to(PROJECT_ROOT)),
                    "date": date_dir.name,
                    "race_id": race_id,
                    "label": " ".join(label_parts),
                    "meta": {
                        "place": place,
                        "race_no": race_no,
                        "race_name": race_name,
                        "grade": grade,
                        "distance": distance,
                        "course": course,
                        "start_time": start_time,
                        "n_entries": head_count,
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to read {race_file}: {e}")
                races.append({
                    "path": str(race_file.relative_to(PROJECT_ROOT)),
                    "date": date_dir.name,
                    "race_id": race_file.stem.replace("race_", ""),
                    "label": f"{date_dir.name} - {race_file.stem}",
                    "meta": {},
                })
        return races

    def _handle_load_race(self, parsed):
        """レースデータを読み込む (A2: エラー時のガイダンス付き)"""
        query = parse_qs(parsed.query)
        race_path = query.get("path", [""])[0]

        if not race_path:
            self._send_json({
                "error": "path parameter required",
                "guidance": "レースを選択してください。",
            }, 400)
            return

        full_path = PROJECT_ROOT / race_path
        if not full_path.exists():
            self._send_json({
                "error": f"レースファイルが見つかりません: {race_path}",
                "guidance": "python scripts/generate_pre_race_materials.py --date YYYY-MM-DD を実行してデータを生成してください。",
            }, 404)
            return

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # A2: 必須フィールドの検証
            if "entries" not in data or len(data.get("entries", [])) == 0:
                self._send_json({
                    "error": "レースデータにentriesがありません",
                    "guidance": "レースJSONファイルが正しく生成されているか確認してください。",
                }, 400)
                return

            self._send_json(data)
        except json.JSONDecodeError as e:
            self._send_json({
                "error": f"JSONパースエラー: {e}",
                "guidance": "レースJSONファイルが破損している可能性があります。再生成してください。",
            }, 500)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_apply_scenario(self):
        """シナリオ補正を適用"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body.decode("utf-8"))
            race_data = request.get("race_data", {})
            scenario = request.get("scenario", {})

            result = apply_scenario_adjustment(race_data, scenario)
            self._send_json(result)
        except Exception as e:
            logger.exception("Error applying scenario")
            self._send_json({"error": str(e)}, 500)

    def _handle_parse_memo(self):
        """メモをパースして提案チップを返す"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body.decode("utf-8"))
            memo_text = request.get("memo", "")

            result = parse_memo(memo_text)
            self._send_json(result)
        except Exception as e:
            logger.exception("Error parsing memo")
            self._send_json({"error": str(e)}, 500)

    def _handle_save_result(self):
        """結果を保存 (artifacts/pre_race/outputs/<date>/<race_id>/<timestamp>.json)"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body.decode("utf-8"))
            result = request.get("result", {})

            # Extract date and race_id
            race_date = result.get("date", datetime.now().strftime("%Y-%m-%d"))
            race_id = result.get("race_id", "unknown")

            # New path structure: artifacts/pre_race/outputs/<date>/<race_id>/<timestamp>.json
            output_dir = PROJECT_ROOT / "artifacts" / "pre_race" / "outputs" / race_date / race_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Filename: <timestamp>.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.json"

            output_path = output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            self._send_json({
                "status": "saved",
                "path": str(output_path.relative_to(PROJECT_ROOT)),
                "date": race_date,
                "race_id": race_id,
                "timestamp": timestamp,
            })
        except Exception as e:
            logger.exception("Error saving result")
            self._send_json({"error": str(e)}, 500)

    def _handle_list_history(self, parsed):
        """履歴一覧を返す（日付/レースIDでフィルタ可能）"""
        query = parse_qs(parsed.query)
        target_date = query.get("date", [""])[0]
        target_race_id = query.get("race_id", [""])[0]

        outputs_dir = PROJECT_ROOT / "artifacts" / "pre_race" / "outputs"
        history = []

        if not outputs_dir.exists():
            self._send_json({"history": []})
            return

        try:
            # Iterate through date folders
            date_dirs = [outputs_dir / target_date] if target_date else sorted(outputs_dir.iterdir(), reverse=True)

            for date_dir in date_dirs:
                if not date_dir.is_dir():
                    continue

                # Iterate through race_id folders
                race_dirs = [date_dir / target_race_id] if target_race_id else sorted(date_dir.iterdir(), reverse=True)

                for race_dir in race_dirs:
                    if not race_dir.is_dir():
                        continue

                    # Iterate through saved result files
                    for result_file in sorted(race_dir.glob("*.json"), reverse=True):
                        try:
                            with open(result_file, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            # v2.2: メモ有無とミニ要約を追加
                            scenario = data.get("scenario", {})
                            has_notes = bool(scenario.get("notes", "").strip())
                            mini_summary = data.get("rank_mini_summary", "")

                            history.append({
                                "path": str(result_file.relative_to(PROJECT_ROOT)),
                                "date": date_dir.name,
                                "race_id": race_dir.name,
                                "timestamp": result_file.stem,
                                "race_name": data.get("race_name", ""),
                                "place": data.get("place", ""),
                                "race_no": data.get("race_no"),
                                "scenario": scenario,
                                "adjusted_at": data.get("adjusted_at", ""),
                                "has_notes": has_notes,  # v2.2
                                "rank_mini_summary": mini_summary,  # v2.2
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read history file {result_file}: {e}")
                            history.append({
                                "path": str(result_file.relative_to(PROJECT_ROOT)),
                                "date": date_dir.name,
                                "race_id": race_dir.name,
                                "timestamp": result_file.stem,
                                "error": str(e),
                            })

            self._send_json({"history": history[:100]})  # Latest 100
        except Exception as e:
            logger.exception("Error listing history")
            self._send_json({"history": [], "error": str(e)})

    def _handle_load_history(self, parsed):
        """履歴データを読み込む"""
        query = parse_qs(parsed.query)
        history_path = query.get("path", [""])[0]

        if not history_path:
            self._send_json({"error": "path parameter required"}, 400)
            return

        full_path = PROJECT_ROOT / history_path
        if not full_path.exists():
            self._send_json({"error": f"History file not found: {history_path}"}, 404)
            return

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._send_json(data)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def log_message(self, format, *args):
        """ログ出力をカスタマイズ"""
        logger.info("%s - %s", self.address_string(), format % args)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """スレッド対応HTTPサーバー"""
    allow_reuse_address = True


def main():
    """メイン関数"""
    port = int(os.environ.get("PORT", 8080))

    server = ThreadedHTTPServer(("", port), ScenarioUIHandler)

    logger.info("=" * 60)
    logger.info("Pre-race Scenario UI Server v2.2")
    logger.info("=" * 60)
    logger.info("  URL: http://localhost:%d", port)
    logger.info("  UI:  %s", Path(__file__).parent / "index.html")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
