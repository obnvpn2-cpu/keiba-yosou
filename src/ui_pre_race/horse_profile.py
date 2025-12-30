# -*- coding: utf-8 -*-
"""
Horse Profile Generator for Pre-race UI

馬ごとの概要/補足/コメントを特徴量データから生成する。
捏造禁止: DBに存在する特徴量のみを根拠に使用する。

Usage:
    from src.ui_pre_race.horse_profile import generate_horse_profile

    profile = generate_horse_profile(entry, feature_data)
    # profile = {
    #     "overview": "近走好調の差し馬。距離適性高い。",
    #     "supplement": "直近3走平均着順2.3着。...",
    #     "comments": ["近走安定", "末脚○", ...],
    #     "evidence": [{"key": "h_recent3_avg_finish", "value": 2.3, "reason": "..."}]
    # }
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Pre-race unsafe features (当日情報のため除外)
PRE_RACE_UNSAFE_FEATURES = {
    "h_body_weight",
    "h_body_weight_diff",
    "h_body_weight_dev",
    "market_win_odds",
    "market_popularity",
    "market_odds_rank",
}


@dataclass
class Evidence:
    """プロフィール根拠"""
    key: str              # 特徴量名
    value: Any            # 値
    reason: str           # 説明


@dataclass
class HorseProfile:
    """馬プロフィール"""
    overview: str                     # 概要 (1-2行)
    supplement: str                   # 補足 (箇条書き可)
    comments: List[str] = field(default_factory=list)   # コメント (短文列挙)
    evidence: List[Evidence] = field(default_factory=list)  # 根拠リスト


def generate_horse_profile(
    entry: Dict[str, Any],
    feature_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    馬プロフィールを生成する

    Args:
        entry: race_<race_id>.json の entry (umaban, name, p_win, etc.)
        feature_data: feature_table_v4 の行データ (optional)

    Returns:
        プロフィール辞書
        {
            "overview": str,
            "supplement": str,
            "comments": [str, ...],
            "evidence": [{"key": str, "value": Any, "reason": str}, ...]
        }

    Note:
        - 捏造禁止: DBに存在する特徴量のみを使用
        - pre-race unsafe features (馬体重など) は使用しない
        - 血統は現状特徴量に無い前提で文章に入れない
    """
    # Merge entry and feature_data
    data = {}
    if feature_data:
        data.update(feature_data)
    data.update(entry)  # entry が優先

    evidence_list = []
    comments = []
    overview_parts = []
    supplement_parts = []

    # ===========================================================================
    # 1. 近走成績から特徴抽出
    # ===========================================================================
    recent3_avg = _safe_float(data.get("h_recent3_avg_finish"))
    if recent3_avg is not None:
        if recent3_avg <= 2.0:
            comments.append("近走好調")
            overview_parts.append("近走好調")
            evidence_list.append(Evidence(
                key="h_recent3_avg_finish",
                value=round(recent3_avg, 1),
                reason="直近3走の平均着順が2.0以下で好調"
            ))
        elif recent3_avg <= 4.0:
            comments.append("近走安定")
            evidence_list.append(Evidence(
                key="h_recent3_avg_finish",
                value=round(recent3_avg, 1),
                reason="直近3走の平均着順が4.0以下で安定"
            ))
        elif recent3_avg >= 10.0:
            comments.append("近走不振")
            evidence_list.append(Evidence(
                key="h_recent3_avg_finish",
                value=round(recent3_avg, 1),
                reason="直近3走の平均着順が10.0以上で低調"
            ))
        supplement_parts.append(f"直近3走平均着順: {recent3_avg:.1f}着")

    # Recent best finish
    recent3_best = _safe_int(data.get("h_recent3_best_finish"))
    if recent3_best is not None and recent3_best == 1:
        comments.append("近走勝ち馬")
        evidence_list.append(Evidence(
            key="h_recent3_best_finish",
            value=recent3_best,
            reason="直近3走で1着あり"
        ))

    # ===========================================================================
    # 2. 末脚特徴 (上がり3F)
    # ===========================================================================
    recent3_last3f = _safe_float(data.get("h_recent3_avg_last3f"))
    avg_last3f = _safe_float(data.get("h_avg_last3f"))
    best_last3f = _safe_float(data.get("h_best_last3f"))

    if recent3_last3f is not None:
        if recent3_last3f <= 34.0:
            comments.append("末脚鋭い")
            overview_parts.append("末脚型")
            evidence_list.append(Evidence(
                key="h_recent3_avg_last3f",
                value=round(recent3_last3f, 1),
                reason="直近3走の平均上がり3Fが34.0秒以下で鋭い末脚"
            ))
        elif recent3_last3f >= 37.0:
            comments.append("上がり平凡")
            evidence_list.append(Evidence(
                key="h_recent3_avg_last3f",
                value=round(recent3_last3f, 1),
                reason="直近3走の平均上がり3Fが37.0秒以上でやや鈍い"
            ))
        supplement_parts.append(f"直近3走平均上がり3F: {recent3_last3f:.1f}秒")

    if best_last3f is not None and best_last3f <= 33.0:
        comments.append("瞬発力あり")
        evidence_list.append(Evidence(
            key="h_best_last3f",
            value=round(best_last3f, 1),
            reason="過去最速上がり3Fが33.0秒以下で高い瞬発力"
        ))

    # ===========================================================================
    # 3. 休養・間隔
    # ===========================================================================
    days_since_last = _safe_int(data.get("h_days_since_last"))
    if days_since_last is not None:
        if days_since_last >= 180:
            comments.append("長期休み明け")
            evidence_list.append(Evidence(
                key="h_days_since_last",
                value=days_since_last,
                reason="前走から180日以上経過で長期休養明け"
            ))
        elif days_since_last >= 90:
            comments.append("休み明け")
            evidence_list.append(Evidence(
                key="h_days_since_last",
                value=days_since_last,
                reason="前走から90日以上経過で休み明け"
            ))
        elif days_since_last <= 14:
            comments.append("連闘")
            evidence_list.append(Evidence(
                key="h_days_since_last",
                value=days_since_last,
                reason="前走から14日以内で連闘"
            ))
        supplement_parts.append(f"前走から: {days_since_last}日")

    # ===========================================================================
    # 4. 距離適性
    # ===========================================================================
    win_rate_dist = _safe_float(data.get("h_win_rate_dist"))
    in3_rate_dist = _safe_float(data.get("h_in3_rate_dist"))
    n_starts_dist = _safe_int(data.get("h_n_starts_dist"))

    if n_starts_dist is not None and n_starts_dist >= 3:
        if win_rate_dist is not None and win_rate_dist >= 0.25:
            comments.append("距離◎")
            overview_parts.append("距離適性高い")
            evidence_list.append(Evidence(
                key="h_win_rate_dist",
                value=f"{win_rate_dist*100:.0f}%",
                reason=f"同距離カテゴリで勝率{win_rate_dist*100:.0f}% (出走{n_starts_dist}回)"
            ))
        elif in3_rate_dist is not None and in3_rate_dist >= 0.5:
            comments.append("距離○")
            evidence_list.append(Evidence(
                key="h_in3_rate_dist",
                value=f"{in3_rate_dist*100:.0f}%",
                reason=f"同距離カテゴリで複勝率{in3_rate_dist*100:.0f}%"
            ))
        elif in3_rate_dist is not None and in3_rate_dist < 0.2 and n_starts_dist >= 5:
            comments.append("距離△")
            evidence_list.append(Evidence(
                key="h_in3_rate_dist",
                value=f"{in3_rate_dist*100:.0f}%",
                reason=f"同距離カテゴリで複勝率{in3_rate_dist*100:.0f}%と低い"
            ))

    # ===========================================================================
    # 5. コース適性
    # ===========================================================================
    win_rate_course = _safe_float(data.get("h_win_rate_course"))
    n_starts_course = _safe_int(data.get("h_n_starts_course"))

    if n_starts_course is not None and n_starts_course >= 2:
        if win_rate_course is not None and win_rate_course >= 0.3:
            comments.append("コース◎")
            evidence_list.append(Evidence(
                key="h_win_rate_course",
                value=f"{win_rate_course*100:.0f}%",
                reason=f"同コースで勝率{win_rate_course*100:.0f}% (出走{n_starts_course}回)"
            ))

    # ===========================================================================
    # 6. 馬場適性 (track condition)
    # ===========================================================================
    win_rate_track = _safe_float(data.get("h_win_rate_track"))
    n_starts_track = _safe_int(data.get("h_n_starts_track"))

    if n_starts_track is not None and n_starts_track >= 2:
        if win_rate_track is not None and win_rate_track >= 0.3:
            comments.append("馬場◎")
            evidence_list.append(Evidence(
                key="h_win_rate_track",
                value=f"{win_rate_track*100:.0f}%",
                reason=f"同馬場状態で勝率{win_rate_track*100:.0f}%"
            ))

    # ===========================================================================
    # 7. 脚質特徴 (running style)
    # ===========================================================================
    running_style_id = _safe_int(data.get("h_running_style_id"))
    style_label = _get_running_style_label(running_style_id)

    if style_label:
        comments.append(f"{style_label}脚質")
        overview_parts.append(f"{style_label}脚質")
        evidence_list.append(Evidence(
            key="h_running_style_id",
            value=running_style_id,
            reason=f"過去の位置取りから{style_label}に分類"
        ))

    # Position change tendency
    pos_change = _safe_float(data.get("h_pos_change_tendency"))
    if pos_change is not None:
        if pos_change <= -3.0:
            comments.append("捲り脚")
            evidence_list.append(Evidence(
                key="h_pos_change_tendency",
                value=round(pos_change, 1),
                reason="終盤に位置を大きく上げる傾向"
            ))
        elif pos_change >= 2.0:
            comments.append("終い甘い")
            evidence_list.append(Evidence(
                key="h_pos_change_tendency",
                value=round(pos_change, 1),
                reason="終盤に位置を下げる傾向"
            ))

    # ===========================================================================
    # 8. 騎手・調教師成績
    # ===========================================================================
    j_win_rate = _safe_float(data.get("j_win_rate"))
    j_in3_rate = _safe_float(data.get("j_in3_rate"))

    if j_win_rate is not None and j_win_rate >= 0.15:
        comments.append("J高勝率")
        evidence_list.append(Evidence(
            key="j_win_rate",
            value=f"{j_win_rate*100:.1f}%",
            reason=f"騎手の通算勝率が{j_win_rate*100:.1f}%と高い"
        ))
    elif j_in3_rate is not None and j_in3_rate >= 0.35:
        comments.append("J安定")
        evidence_list.append(Evidence(
            key="j_in3_rate",
            value=f"{j_in3_rate*100:.1f}%",
            reason=f"騎手の通算複勝率が{j_in3_rate*100:.1f}%"
        ))

    jh_combo_win_rate = _safe_float(data.get("jh_combo_win_rate"))
    jh_n_combos = _safe_int(data.get("jh_n_combos"))
    if jh_n_combos is not None and jh_n_combos >= 3:
        if jh_combo_win_rate is not None and jh_combo_win_rate >= 0.25:
            comments.append("コンビ◎")
            evidence_list.append(Evidence(
                key="jh_combo_win_rate",
                value=f"{jh_combo_win_rate*100:.0f}%",
                reason=f"この騎手とのコンビで勝率{jh_combo_win_rate*100:.0f}% ({jh_n_combos}回騎乗)"
            ))

    t_win_rate = _safe_float(data.get("t_win_rate"))
    if t_win_rate is not None and t_win_rate >= 0.12:
        comments.append("T高勝率")
        evidence_list.append(Evidence(
            key="t_win_rate",
            value=f"{t_win_rate*100:.1f}%",
            reason=f"調教師の通算勝率が{t_win_rate*100:.1f}%と高い"
        ))

    # ===========================================================================
    # 9. 重賞実績
    # ===========================================================================
    n_g1_runs = _safe_int(data.get("h_n_g1_runs"))
    n_g2_runs = _safe_int(data.get("h_n_g2_runs"))
    n_g3_runs = _safe_int(data.get("h_n_g3_runs"))
    graded_in3_rate = _safe_float(data.get("h_graded_in3_rate"))

    if n_g1_runs is not None and n_g1_runs >= 1:
        comments.append("G1経験")
        evidence_list.append(Evidence(
            key="h_n_g1_runs",
            value=n_g1_runs,
            reason=f"G1に{n_g1_runs}回出走経験あり"
        ))

    if graded_in3_rate is not None and graded_in3_rate >= 0.3:
        comments.append("重賞実績○")
        evidence_list.append(Evidence(
            key="h_graded_in3_rate",
            value=f"{graded_in3_rate*100:.0f}%",
            reason=f"重賞での複勝率{graded_in3_rate*100:.0f}%"
        ))

    # ===========================================================================
    # 10. 初出走フラグ
    # ===========================================================================
    is_first_run = _safe_int(data.get("h_is_first_run"))
    if is_first_run == 1:
        comments.append("初出走")
        overview_parts.append("初出走馬")
        evidence_list.append(Evidence(
            key="h_is_first_run",
            value=1,
            reason="過去出走データなし（新馬または転厩直後など）"
        ))

    # ===========================================================================
    # Build overview and supplement
    # ===========================================================================
    if not overview_parts:
        # デフォルト: 出走数ベースの表現
        n_starts = _safe_int(data.get("h_n_starts"))
        if n_starts is not None:
            if n_starts == 0:
                overview_parts.append("データなし")
            elif n_starts <= 3:
                overview_parts.append("キャリア浅い")
            else:
                overview_parts.append("経験馬")

    overview = "。".join(overview_parts) + "。" if overview_parts else ""
    supplement = " / ".join(supplement_parts) if supplement_parts else ""

    # Convert to dict format
    return {
        "overview": overview,
        "supplement": supplement,
        "comments": comments,
        "evidence": [
            {"key": e.key, "value": e.value, "reason": e.reason}
            for e in evidence_list
        ],
    }


def _safe_float(val: Any) -> Optional[float]:
    """安全にfloatに変換"""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """安全にintに変換"""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _get_running_style_label(style_id: Optional[int]) -> str:
    """脚質IDからラベルを取得"""
    if style_id is None:
        return ""
    labels = {
        0: "逃げ",
        1: "先行",
        2: "差し",
        3: "追込",
    }
    return labels.get(style_id, "")


# =============================================================================
# Utility functions for batch processing
# =============================================================================

def generate_profiles_for_race(
    entries: List[Dict[str, Any]],
    feature_data_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    レース全体のエントリーにプロフィールを付与

    Args:
        entries: エントリーリスト
        feature_data_map: horse_id -> feature_data のマップ

    Returns:
        プロフィール付きエントリーリスト
    """
    result = []
    for entry in entries:
        horse_id = entry.get("horse_id", "")
        feature_data = (feature_data_map or {}).get(horse_id, {})

        profile = generate_horse_profile(entry, feature_data)
        entry_with_profile = dict(entry)
        entry_with_profile["profile"] = profile

        result.append(entry_with_profile)

    return result


if __name__ == "__main__":
    # Simple test
    test_entry = {
        "horse_id": "test001",
        "umaban": 1,
        "name": "テストホース",
        "h_recent3_avg_finish": 2.3,
        "h_recent3_avg_last3f": 33.5,
        "h_days_since_last": 28,
        "h_win_rate_dist": 0.3,
        "h_n_starts_dist": 5,
        "h_running_style_id": 2,
        "j_win_rate": 0.12,
    }

    profile = generate_horse_profile(test_entry)

    print("=" * 60)
    print("Test Profile Generation")
    print("=" * 60)
    print(f"Overview: {profile['overview']}")
    print(f"Supplement: {profile['supplement']}")
    print(f"Comments: {profile['comments']}")
    print("\nEvidence:")
    for e in profile["evidence"]:
        print(f"  - {e['key']}: {e['value']} ({e['reason']})")
