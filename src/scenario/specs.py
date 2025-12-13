# src/scenario/specs.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, List


# 型エイリアス ---------------------------------------------------------

PaceType = Literal["S", "M", "H"]  # S=スロー, M=ミドル, H=ハイ
TrackConditionType = Literal["良", "稍重", "重", "不良"]
BiasType = Literal["内", "外", "フラット"]


# シナリオ定義 ---------------------------------------------------------

@dataclass
class ScenarioSpec:
    """
    人間が定義する『レースシナリオ』の入力情報。

    - モデルの生の予測 (base prob) に対して、
      「今日の馬場やペース、バイアス、含水率」をどう見るかを
      人間側が明示するためのコンテナ。
    """

    # --- コア想定 ---
    pace: PaceType                          # ペース想定: S/M/H
    track_condition: TrackConditionType     # 馬場: 良/稍重/重/不良
    bias: BiasType                          # バイアス: 内/外/フラット

    # --- 馬場・含水率関連 ---
    cushion_value: Optional[float] = None   # 当日朝のクッション値

    # JRA 公開の 4 区分に対応
    # 芝 / ダート × ゴール前 / 4コーナー
    moist_turf_goal: Optional[float] = None   # 芝・ゴール前 (%)
    moist_turf_4c: Optional[float] = None     # 芝・4コーナー (%)
    moist_dirt_goal: Optional[float] = None   # ダート・ゴール前 (%)
    moist_dirt_4c: Optional[float] = None     # ダート・4コーナー (%)

    # --- 馬ごとの主観情報 ---
    # 「シナリオ上」逃げ想定の馬 / 信頼度が低い馬 など
    front_runner_ids: List[str] = field(default_factory=list)  # 逃げそうな馬ID
    weak_horse_ids: List[str] = field(default_factory=list)    # 状態や信頼度の低い馬ID

    # --- 補足メモ ---
    notes: str = ""  # 自由記述

    # --- 補助メソッド（あって困らないユーティリティ） ---

    def to_dict(self) -> Dict[str, object]:
        """LLM やログに投げやすいように dict へ変換する簡易メソッド。"""
        return {
            "pace": self.pace,
            "track_condition": self.track_condition,
            "bias": self.bias,
            "cushion_value": self.cushion_value,
            "moist_turf_goal": self.moist_turf_goal,
            "moist_turf_4c": self.moist_turf_4c,
            "moist_dirt_goal": self.moist_dirt_goal,
            "moist_dirt_4c": self.moist_dirt_4c,
            "front_runner_ids": list(self.front_runner_ids),
            "weak_horse_ids": list(self.weak_horse_ids),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ScenarioSpec":
        """
        dict から ScenarioSpec を復元する簡易コンストラクタ。
        必須項目が足りない場合は KeyError を投げる想定。
        """
        return cls(
            pace=data["pace"],  # type: ignore[arg-type]
            track_condition=data["track_condition"],  # type: ignore[arg-type]
            bias=data["bias"],  # type: ignore[arg-type]
            cushion_value=data.get("cushion_value"),  # type: ignore[arg-type]
            moist_turf_goal=data.get("moist_turf_goal"),  # type: ignore[arg-type]
            moist_turf_4c=data.get("moist_turf_4c"),  # type: ignore[arg-type]
            moist_dirt_goal=data.get("moist_dirt_goal"),  # type: ignore[arg-type]
            moist_dirt_4c=data.get("moist_dirt_4c"),  # type: ignore[arg-type]
            front_runner_ids=list(data.get("front_runner_ids", [])),  # type: ignore[list-item]
            weak_horse_ids=list(data.get("weak_horse_ids", [])),      # type: ignore[list-item]
            notes=data.get("notes", "")  # type: ignore[arg-type]
        )


# シナリオ補正後スコア -------------------------------------------------

@dataclass
class ScenarioScore:
    """
    あるレース × あるシナリオに対する、
    『シナリオを踏まえた評価』をまとめるコンテナ。

    - シナリオのメタ情報（ペース・馬場・含水率など）
    - そのシナリオのもとでの、馬ごとの評価
      （ベース予測 + シナリオ補正後のスコア）

    実際の補正ロジックは別レイヤーで実装し、
    ここは「結果をどう保持するか」の入れ物だけにしている。
    """

    # --- シナリオ側メタ情報 ---
    pace: PaceType
    track_condition: TrackConditionType
    cushion_value: Optional[float]

    moist_turf_goal: Optional[float]
    moist_turf_4c: Optional[float]
    moist_dirt_goal: Optional[float]
    moist_dirt_4c: Optional[float]

    # --- 評価結果（馬ごと） ---
    # key: horse_id
    # value: {
    #   "base_win": 生の勝率 (BaseWinModel 出力),
    #   "base_in3": 生の3着内率,
    #   "adj_win": シナリオ補正後の勝率,
    #   "adj_in3": シナリオ補正後の3着内率,
    #   "comment": その馬への一言コメント など
    # }
    per_horse: Dict[str, Dict[str, float | str]] = field(default_factory=dict)

    # --- シナリオ全体のコメント ---
    # 例: 「前半から速くなりそうで、差し・追い込み有利」
    scenario_comment: str = ""

    def to_dict(self) -> Dict[str, object]:
        """JSON 保存や LLM 入力向けに dict 化する。"""
        return {
            "pace": self.pace,
            "track_condition": self.track_condition,
            "cushion_value": self.cushion_value,
            "moist_turf_goal": self.moist_turf_goal,
            "moist_turf_4c": self.moist_turf_4c,
            "moist_dirt_goal": self.moist_dirt_goal,
            "moist_dirt_4c": self.moist_dirt_4c,
            "per_horse": self.per_horse,
            "scenario_comment": self.scenario_comment,
        }

    @classmethod
    def from_spec(
        cls,
        spec: ScenarioSpec,
        per_horse: Dict[str, Dict[str, float | str]],
        scenario_comment: str = "",
    ) -> "ScenarioScore":
        """
        ScenarioSpec + 馬ごとの評価結果 から ScenarioScore を組み立てる補助メソッド。
        """
        return cls(
            pace=spec.pace,
            track_condition=spec.track_condition,
            cushion_value=spec.cushion_value,
            moist_turf_goal=spec.moist_turf_goal,
            moist_turf_4c=spec.moist_turf_4c,
            moist_dirt_goal=spec.moist_dirt_goal,
            moist_dirt_4c=spec.moist_dirt_4c,
            per_horse=per_horse,
            scenario_comment=scenario_comment,
        )
