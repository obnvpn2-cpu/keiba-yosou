# scenario/compare.py

from __future__ import annotations

from typing import Dict, Any, List

from .score import ScenarioScore, ScenarioHorseScore


def build_scenario_comparison_context(
    base_score: ScenarioScore,
    scenario_a_score: ScenarioScore,
    scenario_b_score: ScenarioScore,
    *,
    top_n: int = 10,
    max_diff_horses: int = 10,
) -> Dict[str, Any]:
    """
    ベース予測と 2 つのシナリオ予測を比較し、LLM に渡しやすい JSON 形式のコンテキストを構築する。

    Args:
        base_score:
            「ベース予測」として扱う ScenarioScore。
            通常はシナリオ補正をかけていない（または中立的な）状態を想定する。
        scenario_a_score:
            比較対象となるシナリオ A の ScenarioScore。
        scenario_b_score:
            比較対象となるシナリオ B の ScenarioScore。
        top_n:
            各 ScenarioScore.to_llm_context() に渡すトップ表示頭数。
        max_diff_horses:
            差分リストに含める最大頭数。

    Returns:
        LLM にそのまま渡せる dict。
        構造はざっくり以下のイメージ::

            {
                "race": {...},
                "base": {...},        # base_score.to_llm_context()
                "scenario_a": {...},  # scenario_a_score.to_llm_context()
                "scenario_b": {...},  # scenario_b_score.to_llm_context()
                "comparison": {
                    "horses": [...],                  # 馬ごとの詳細差分
                    "top_movers_a_vs_base": [...],    # シナリオAで特に上がる馬
                    "top_movers_b_vs_base": [...],    # シナリオBで特に上がる馬
                    "top_diff_between_a_and_b": [...] # AとBで評価差が大きい馬
                }
            }
    """
    # レースIDが一致しているかチェック
    race_id = base_score.race_id
    if scenario_a_score.race_id != race_id or scenario_b_score.race_id != race_id:
        raise ValueError(
            f"Race ID mismatch: base={base_score.race_id}, "
            f"A={scenario_a_score.race_id}, B={scenario_b_score.race_id}"
        )

    # 馬IDの共通集合をとる（念のため不一致も考慮）
    common_ids = (
        set(base_score.horses.keys())
        & set(scenario_a_score.horses.keys())
        & set(scenario_b_score.horses.keys())
    )

    horse_diffs: List[Dict[str, Any]] = []

    for horse_id in sorted(common_ids):
        base_h: ScenarioHorseScore = base_score.horses[horse_id]
        a_h: ScenarioHorseScore = scenario_a_score.horses[horse_id]
        b_h: ScenarioHorseScore = scenario_b_score.horses[horse_id]

        name = base_h.horse_name or a_h.horse_name or b_h.horse_name or horse_id

        # ベースモデルの勝率・3着内率は base_score 側の base_* を基準とする
        base_win = float(base_h.base_win)
        base_in3 = float(base_h.base_in3)

        a_win = float(a_h.adj_win)
        a_in3 = float(a_h.adj_in3)

        b_win = float(b_h.adj_win)
        b_in3 = float(b_h.adj_in3)

        # シナリオごとの差分（勝率）
        delta_a_vs_base = a_win - base_win
        delta_b_vs_base = b_win - base_win
        delta_a_vs_b = a_win - b_win

        # パーセント表現（ベースが0のときは 0 とする）
        if base_win > 0:
            delta_a_vs_base_pct = (delta_a_vs_base / base_win) * 100.0
            delta_b_vs_base_pct = (delta_b_vs_base / base_win) * 100.0
        else:
            delta_a_vs_base_pct = 0.0
            delta_b_vs_base_pct = 0.0

        horse_diffs.append(
            {
                "id": horse_id,
                "name": name,
                "frame_no": base_h.frame_no,
                "gate_number": base_h.gate_number,
                "run_style": {
                    "base": base_h.run_style_label,
                    "scenario_a": a_h.run_style_label,
                    "scenario_b": b_h.run_style_label,
                },
                "win": {
                    "base": base_win,
                    "scenario_a": a_win,
                    "scenario_b": b_win,
                    "delta_a_vs_base": delta_a_vs_base,
                    "delta_b_vs_base": delta_b_vs_base,
                    "delta_a_vs_b": delta_a_vs_b,
                    "delta_a_vs_base_pct": delta_a_vs_base_pct,
                    "delta_b_vs_base_pct": delta_b_vs_base_pct,
                },
                "in3": {
                    "base": base_in3,
                    "scenario_a": a_in3,
                    "scenario_b": b_in3,
                    "delta_a_vs_base": a_in3 - base_in3,
                    "delta_b_vs_base": b_in3 - base_in3,
                    "delta_a_vs_b": a_in3 - b_in3,
                },
                "reasons": {
                    "scenario_a": a_h.get_reasons_japanese(),
                    "scenario_b": b_h.get_reasons_japanese(),
                },
            }
        )

    # 差分の大きい馬を絞り込むためのヘルパー
    def _top_by_key(
        key: str,
        positive_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        key には "win.delta_a_vs_base" のような文字列を渡す想定。
        """
        if key.startswith("win."):
            # "win.delta_a_vs_base" 形式を期待
            _, subkey = key.split(".", 1)

            def get_value(h: Dict[str, Any]) -> float:
                return float(h["win"].get(subkey, 0.0))

        else:
            def get_value(h: Dict[str, Any]) -> float:
                return float(h.get(key, 0.0))

        items = sorted(horse_diffs, key=get_value, reverse=True)
        if positive_only:
            items = [h for h in items if get_value(h) > 0]
        return items[:max_diff_horses]

    # シナリオAで特に評価が上がる馬（ベース比）
    top_movers_a_vs_base = _top_by_key("win.delta_a_vs_base", positive_only=True)

    # シナリオBで特に評価が上がる馬（ベース比）
    top_movers_b_vs_base = _top_by_key("win.delta_b_vs_base", positive_only=True)

    # AとBで評価差が大きい馬（絶対値でソート）
    def _abs_delta_a_vs_b(h: Dict[str, Any]) -> float:
        return abs(float(h["win"].get("delta_a_vs_b", 0.0)))

    top_diff_between_a_and_b = sorted(
        horse_diffs, key=_abs_delta_a_vs_b, reverse=True
    )[:max_diff_horses]

    return {
        "race": base_score.spec.race_context.to_dict(),
        "base": base_score.to_llm_context(top_n=top_n),
        "scenario_a": scenario_a_score.to_llm_context(top_n=top_n),
        "scenario_b": scenario_b_score.to_llm_context(top_n=top_n),
        "comparison": {
            "horses": horse_diffs,
            "top_movers_a_vs_base": top_movers_a_vs_base,
            "top_movers_b_vs_base": top_movers_b_vs_base,
            "top_diff_between_a_and_b": top_diff_between_a_and_b,
        },
    }
