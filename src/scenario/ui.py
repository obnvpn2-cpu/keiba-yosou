"""
UI向けのコンテキスト生成関数。

このモジュールは ScenarioScore から「UIでそのまま使えるJSON」を生成する。
フロントエンド（React/Next.js）への API レスポンスとしても、
LLMへの入力としても再利用できる共通インターフェイスを提供する。

Usage:
    from scenario import ScenarioScore
    from scenario.ui import build_scenario_ui_context

    # ScenarioAdjuster で得た ScenarioScore から UI 向け JSON を生成
    ui_context = build_scenario_ui_context(score, top_n=10)

    # FastAPI などでそのままレスポンスとして返せる
    return JSONResponse(content=ui_context)

    # LLMに渡す場合
    prompt = f"以下のシナリオについて解説してください:\\n{json.dumps(ui_context, ensure_ascii=False)}"
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from .score import ScenarioScore, ScenarioHorseScore


# ソートキーの型定義
SortByType = Literal["adj_win", "base_win", "adj_in3", "base_in3", "win_delta", "frame_no"]


def build_scenario_ui_context(
    score: ScenarioScore,
    *,
    sort_by: SortByType = "adj_win",
    top_n: Optional[int] = None,
    include_summary: bool = True,
) -> Dict[str, Any]:
    """
    単一の ScenarioScore から UI 向けの JSON を生成する。

    この関数は以下の用途で使うことを想定:
    - フロントエンド（React/Next.js）への API レスポンス
    - LLMに渡して「このシナリオの解説」を生成させる入力
    - CLI/ログでの確認

    Args:
        score: ScenarioAdjuster.adjust() が返した ScenarioScore
        sort_by: 馬リストのソートキー
            - "adj_win": シナリオ補正後の勝率（デフォルト）
            - "base_win": ベースモデルの勝率
            - "adj_in3": シナリオ補正後の3着内率
            - "base_in3": ベースモデルの3着内率
            - "win_delta": 勝率の変化幅
            - "frame_no": 枠番順
        top_n: 上位何頭まで返すか（None なら全頭）
        include_summary: サマリ情報を含めるか

    Returns:
        UI向けのJSON構造（dict）

    Example:
        >>> from scenario import ScenarioAdjuster, ScenarioSpec, RaceContext
        >>> from scenario.ui import build_scenario_ui_context
        >>>
        >>> adjuster = ScenarioAdjuster()
        >>> score = adjuster.adjust(spec, base_predictions, horse_features, horse_names)
        >>> ui_context = build_scenario_ui_context(score, top_n=10)
        >>> # FastAPI でそのまま返す or json.dumps() して LLM に渡す
    """
    spec = score.spec
    race_ctx = spec.race_context

    # 馬リストを取得してソート
    horses = _sort_horses(list(score.horses.values()), sort_by)

    # top_n でトリミング
    if top_n is not None:
        horses = horses[:top_n]

    # 各馬のデータを変換
    horses_data = [_build_horse_data(h) for h in horses]

    # 基本構造
    result: Dict[str, Any] = {
        "race": _build_race_data(race_ctx),
        "scenario": _build_scenario_data(spec),
        "horses": horses_data,
    }

    # サマリを追加
    if include_summary:
        result["summary"] = _build_summary_data(score, horses)

    return result


def _sort_horses(
    horses: List[ScenarioHorseScore],
    sort_by: SortByType,
) -> List[ScenarioHorseScore]:
    """馬リストをソートする"""
    sort_key_map = {
        "adj_win": lambda h: h.adj_win,
        "base_win": lambda h: h.base_win,
        "adj_in3": lambda h: h.adj_in3,
        "base_in3": lambda h: h.base_in3,
        "win_delta": lambda h: h.win_delta,
        "frame_no": lambda h: h.frame_no or 99,  # None は末尾へ
    }

    key_func = sort_key_map.get(sort_by, sort_key_map["adj_win"])

    # frame_no 以外は降順（高い順）
    reverse = sort_by != "frame_no"

    return sorted(horses, key=key_func, reverse=reverse)


def _build_race_data(race_ctx) -> Dict[str, Any]:
    """レース情報を構築"""
    return {
        "race_id": race_ctx.race_id,
        "race_name": race_ctx.race_name,
        "course": race_ctx.course,
        "surface": race_ctx.surface,
        "distance": race_ctx.distance,
        "race_date": race_ctx.race_date,
        "race_class": race_ctx.race_class,
    }


def _build_scenario_data(spec) -> Dict[str, Any]:
    """シナリオ情報を構築"""
    data = {
        "scenario_id": spec.scenario_id,
        "pace": spec.pace,
        "track_condition": spec.track_condition,
        "bias": spec.bias,
        "notes": spec.notes,
    }

    # 将来の拡張用: 馬場データがあれば追加
    if spec.cushion_value is not None:
        data["cushion_value"] = spec.cushion_value

    if not spec.moisture.turf.is_empty() or not spec.moisture.dirt.is_empty():
        data["moisture"] = spec.moisture.to_dict()

    # 脚質指定情報
    if spec.front_runner_ids:
        data["front_runner_count"] = len(spec.front_runner_ids)
    if spec.stalker_ids:
        data["stalker_count"] = len(spec.stalker_ids)
    if spec.closer_ids:
        data["closer_count"] = len(spec.closer_ids)
    if spec.weak_horse_ids:
        data["weak_horse_count"] = len(spec.weak_horse_ids)

    return data


def _build_horse_data(h: ScenarioHorseScore) -> Dict[str, Any]:
    """馬1頭分のデータを構築"""
    data = {
        "id": h.horse_id,
        "name": h.horse_name,
        "frame_no": h.frame_no,
        "gate_number": h.gate_number,
        "run_style": h.run_style_label,
        "win": {
            "base": _round_prob(h.base_win),
            "adj": _round_prob(h.adj_win),
            "delta": _round_prob(h.win_delta),
            "delta_pct": _round_pct(h.win_delta_pct) if h.base_win > 0 else None,
        },
        "in3": {
            "base": _round_prob(h.base_in3),
            "adj": _round_prob(h.adj_in3),
            "delta": _round_prob(h.in3_delta),
        },
        "reasons": h.get_reasons_japanese(),
    }

    # コメントがあれば追加
    if h.comment:
        data["comment"] = h.comment

    return data


def _build_summary_data(
    score: ScenarioScore,
    sorted_horses: List[ScenarioHorseScore],
) -> Dict[str, Any]:
    """サマリ情報を構築"""
    # 上位5頭のサマリ
    top_5 = sorted_horses[:5]
    top_by_adj_win = [
        {
            "id": h.horse_id,
            "name": h.horse_name,
            "adj_win": _round_prob(h.adj_win),
            "base_win": _round_prob(h.base_win),
            "run_style": h.run_style_label,
        }
        for h in top_5
    ]

    # プラス補正がかかった馬
    boosted = score.horses_with_positive_adjustment()[:5]
    boosted_horses = [
        {
            "id": h.horse_id,
            "name": h.horse_name,
            "win_delta": _round_prob(h.win_delta),
            "reasons": h.get_reasons_japanese(),
        }
        for h in boosted
    ]

    # マイナス補正がかかった馬
    dropped = score.horses_with_negative_adjustment()[:5]
    dropped_horses = [
        {
            "id": h.horse_id,
            "name": h.horse_name,
            "win_delta": _round_prob(h.win_delta),
            "reasons": h.get_reasons_japanese(),
        }
        for h in dropped
    ]

    # 堅実なヒモ候補
    placers = score.solid_placers()[:3]
    solid_placers = [
        {
            "id": h.horse_id,
            "name": h.horse_name,
            "adj_win": _round_prob(h.adj_win),
            "adj_in3": _round_prob(h.adj_in3),
        }
        for h in placers
    ]

    return {
        "top_by_adj_win": top_by_adj_win,
        "boosted_horses": boosted_horses,
        "dropped_horses": dropped_horses,
        "solid_placers": solid_placers,
        "total_horses": len(score.horses),
    }


def _round_prob(value: float, digits: int = 4) -> float:
    """確率値を丸める"""
    return round(value, digits)


def _round_pct(value: float, digits: int = 1) -> float:
    """パーセント値を丸める"""
    return round(value, digits)


# =============================================================================
# ペース日本語変換用のヘルパー
# =============================================================================

PACE_LABELS_JA = {
    "S": "スローペース",
    "M": "ミドルペース",
    "H": "ハイペース",
}

BIAS_LABELS_JA = {
    "内": "内有利",
    "外": "外有利",
    "フラット": "フラット",
}


def get_pace_label_ja(pace: str) -> str:
    """ペースの日本語ラベルを取得"""
    return PACE_LABELS_JA.get(pace, pace)


def get_bias_label_ja(bias: str) -> str:
    """バイアスの日本語ラベルを取得"""
    return BIAS_LABELS_JA.get(bias, bias)
