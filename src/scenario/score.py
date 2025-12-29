"""
ScenarioScore / ScenarioHorseScore: シナリオ補正後の評価結果を格納するコンテナ。

このモジュールは scenario/ の出力側を担当し、spec.py に依存する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Iterator
from enum import Enum, auto

from .spec import ScenarioSpec


@dataclass
class BodyWeightContext:
    """
    馬体重に関するコンテキスト情報。

    LLM説明生成時に「体重変動に注意」「当日の馬体重確認」などの
    コメントを出すための材料として使用する。

    Attributes:
        # 過去ベース (pre-race safe: 前日までに確実に取得可能)
        avg_body_weight: 過去平均馬体重
        last_body_weight: 直近出走時の馬体重
        last_body_weight_diff: 直近出走時の増減
        recent3_avg_body_weight: 直近3走の平均体重
        recent3_std_body_weight: 直近3走の標準偏差
        recent3_trend: 直近3走の体重トレンド (正=増量傾向)
        body_weight_z: 体重z-score

        # 当日データ (race-day only: 前日運用では None)
        current_body_weight: 当日の馬体重 (None if not available)
        current_body_weight_diff: 当日の増減 (None if not available)
        current_body_weight_dev: 当日の偏差 (None if not available)
    """
    # 過去ベース (pre-race safe)
    avg_body_weight: Optional[float] = None
    last_body_weight: Optional[int] = None
    last_body_weight_diff: Optional[int] = None
    recent3_avg_body_weight: Optional[float] = None
    recent3_std_body_weight: Optional[float] = None
    recent3_trend: Optional[float] = None
    body_weight_z: Optional[float] = None

    # 当日データ (race-day only)
    current_body_weight: Optional[int] = None
    current_body_weight_diff: Optional[int] = None
    current_body_weight_dev: Optional[float] = None

    @property
    def has_current_data(self) -> bool:
        """当日体重データがあるか"""
        return self.current_body_weight is not None

    @property
    def has_historical_data(self) -> bool:
        """過去体重データがあるか"""
        return self.last_body_weight is not None

    @property
    def is_weight_volatile(self) -> bool:
        """体重変動が大きいか (std > 4kg を目安)"""
        if self.recent3_std_body_weight is None:
            return False
        return self.recent3_std_body_weight > 4.0

    @property
    def is_gaining_weight(self) -> bool:
        """増量傾向か"""
        if self.recent3_trend is None:
            return False
        return self.recent3_trend > 2.0

    @property
    def is_losing_weight(self) -> bool:
        """減量傾向か"""
        if self.recent3_trend is None:
            return False
        return self.recent3_trend < -2.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式にシリアライズ"""
        return {
            # 過去ベース
            "avg_body_weight": self.avg_body_weight,
            "last_body_weight": self.last_body_weight,
            "last_body_weight_diff": self.last_body_weight_diff,
            "recent3_avg_body_weight": self.recent3_avg_body_weight,
            "recent3_std_body_weight": self.recent3_std_body_weight,
            "recent3_trend": self.recent3_trend,
            "body_weight_z": self.body_weight_z,
            # 当日データ
            "current_body_weight": self.current_body_weight,
            "current_body_weight_diff": self.current_body_weight_diff,
            "current_body_weight_dev": self.current_body_weight_dev,
            # フラグ
            "has_current_data": self.has_current_data,
            "has_historical_data": self.has_historical_data,
            "is_weight_volatile": self.is_weight_volatile,
            "is_gaining_weight": self.is_gaining_weight,
            "is_losing_weight": self.is_losing_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BodyWeightContext":
        """辞書形式からデシリアライズ"""
        return cls(
            avg_body_weight=data.get("avg_body_weight"),
            last_body_weight=data.get("last_body_weight"),
            last_body_weight_diff=data.get("last_body_weight_diff"),
            recent3_avg_body_weight=data.get("recent3_avg_body_weight"),
            recent3_std_body_weight=data.get("recent3_std_body_weight"),
            recent3_trend=data.get("recent3_trend"),
            body_weight_z=data.get("body_weight_z"),
            current_body_weight=data.get("current_body_weight"),
            current_body_weight_diff=data.get("current_body_weight_diff"),
            current_body_weight_dev=data.get("current_body_weight_dev"),
        )

    def get_llm_notes(self) -> List[str]:
        """LLM向けの注意点リストを生成"""
        notes = []

        if self.is_weight_volatile:
            notes.append("体重変動大 (要確認)")

        if self.is_gaining_weight:
            notes.append("増量傾向")
        elif self.is_losing_weight:
            notes.append("減量傾向")

        if not self.has_current_data:
            notes.append("当日体重未確定")

        return notes


class AdjustmentReason(Enum):
    """
    補正理由のフラグ。

    LLM説明生成時に「なぜこの馬の評価が上がった/下がったか」を
    構造化して伝えるために使用する。
    """

    # ペース関連
    PACE_ADVANTAGE = auto()  # ペース想定で有利
    PACE_DISADVANTAGE = auto()  # ペース想定で不利

    # バイアス関連
    BIAS_ADVANTAGE = auto()  # コースバイアスで有利
    BIAS_DISADVANTAGE = auto()  # コースバイアスで不利

    # 馬場状態関連
    TRACK_SPECIALIST = auto()  # 馬場巧者（重馬場得意など）
    TRACK_WEAK = auto()  # 馬場苦手

    # 脚質関連（ベース脚質から判定）
    FRONT_RUNNER = auto()  # 逃げ馬（ベース脚質）
    STALKER = auto()  # 先行馬
    CLOSER = auto()  # 差し・追込馬

    # シナリオ指定の脚質（人間が明示的に指定）
    SCENARIO_FRONT_RUNNER = auto()  # シナリオで逃げ指定された馬
    SCENARIO_STALKER = auto()  # シナリオで先行指定された馬
    SCENARIO_CLOSER = auto()  # シナリオで差し指定された馬

    # 人間の主観的判断
    MARKED_WEAK = auto()  # 信頼度低として指定

    # 馬場含水率関連
    MOISTURE_BOOST = auto()  # 含水率条件で上昇
    MOISTURE_DROP = auto()  # 含水率条件で下降

    # 距離適性
    DISTANCE_BEST = auto()  # 距離ベスト
    DISTANCE_STRETCH = auto()  # 距離延長が不安
    DISTANCE_SHORTEN = auto()  # 距離短縮が不安

    def to_japanese(self) -> str:
        """日本語ラベルを返す（LLM説明生成用）"""
        labels = {
            AdjustmentReason.PACE_ADVANTAGE: "ペース有利",
            AdjustmentReason.PACE_DISADVANTAGE: "ペース不利",
            AdjustmentReason.BIAS_ADVANTAGE: "バイアス有利",
            AdjustmentReason.BIAS_DISADVANTAGE: "バイアス不利",
            AdjustmentReason.TRACK_SPECIALIST: "馬場巧者",
            AdjustmentReason.TRACK_WEAK: "馬場苦手",
            AdjustmentReason.FRONT_RUNNER: "逃げ",
            AdjustmentReason.STALKER: "先行",
            AdjustmentReason.CLOSER: "差し・追込",
            AdjustmentReason.SCENARIO_FRONT_RUNNER: "逃げ指定",
            AdjustmentReason.SCENARIO_STALKER: "先行指定",
            AdjustmentReason.SCENARIO_CLOSER: "差し指定",
            AdjustmentReason.MARKED_WEAK: "信頼度低",
            AdjustmentReason.MOISTURE_BOOST: "含水率プラス",
            AdjustmentReason.MOISTURE_DROP: "含水率マイナス",
            AdjustmentReason.DISTANCE_BEST: "距離ベスト",
            AdjustmentReason.DISTANCE_STRETCH: "距離延長不安",
            AdjustmentReason.DISTANCE_SHORTEN: "距離短縮不安",
        }
        return labels.get(self, self.name)


# 脚質ラベルの日本語変換マップ
RUN_STYLE_LABELS_JA: Dict[str, str] = {
    "逃げ": "逃げ",
    "先行": "先行",
    "差し": "差し",
    "追込": "追込",
    "その他": "その他",
}


@dataclass
class ScenarioHorseScore:
    """
    1頭分のシナリオ評価結果。

    ベースモデル（LightGBM）の予測値と、シナリオ補正後の値を
    両方保持することで、「どれだけ補正がかかったか」を可視化できる。

    Attributes:
        horse_id: 馬ID
        horse_name: 馬名（表示・説明用）
        frame_no: 枠番（1-8）
        gate_number: 馬番（1-18）
        run_style_label: 脚質ラベル（"逃げ"/"先行"/"差し"/"追込"/"その他"）
        base_win: ベースモデルの勝率
        base_in3: ベースモデルの3着内率
        adj_win: シナリオ補正後の勝率
        adj_in3: シナリオ補正後の3着内率
        adjustment_reasons: 補正がかかった理由のリスト
        comment: その馬への一言コメント（adjusterまたはLLMが生成）
    """

    horse_id: str
    horse_name: Optional[str] = None
    frame_no: Optional[int] = None  # 枠番 (1-8)
    gate_number: Optional[int] = None  # 馬番 (1-18)
    run_style_label: Optional[str] = None  # 脚質ラベル

    # ベース予測値
    base_win: float = 0.0
    base_in3: float = 0.0

    # 補正後予測値
    adj_win: float = 0.0
    adj_in3: float = 0.0

    # 補正理由
    adjustment_reasons: List[AdjustmentReason] = field(default_factory=list)

    # 体重コンテキスト (pre-race + race-day)
    body_weight_context: Optional[BodyWeightContext] = None

    # コメント
    comment: str = ""

    @property
    def post_position(self) -> Optional[int]:
        """枠番（frame_no のエイリアス、後方互換用）"""
        return self.frame_no

    @property
    def win_delta(self) -> float:
        """勝率の変化幅"""
        return self.adj_win - self.base_win

    @property
    def in3_delta(self) -> float:
        """3着内率の変化幅"""
        return self.adj_in3 - self.base_in3

    @property
    def win_delta_pct(self) -> float:
        """勝率の変化率（%）"""
        if self.base_win == 0:
            return 0.0
        return (self.win_delta / self.base_win) * 100

    @property
    def has_positive_adjustment(self) -> bool:
        """プラス補正がかかっているか"""
        return self.win_delta > 0.001 or self.in3_delta > 0.001

    @property
    def has_negative_adjustment(self) -> bool:
        """マイナス補正がかかっているか"""
        return self.win_delta < -0.001 or self.in3_delta < -0.001

    @property
    def is_solid_placer(self) -> bool:
        """堅実なヒモ候補か（3着内率が高いが勝率は低め）"""
        return self.adj_in3 > 0.3 and self.adj_win < 0.1

    def add_reason(self, reason: AdjustmentReason) -> None:
        """補正理由を追加（重複チェックあり）"""
        if reason not in self.adjustment_reasons:
            self.adjustment_reasons.append(reason)

    def get_reasons_japanese(self) -> List[str]:
        """補正理由の日本語リストを返す"""
        return [r.to_japanese() for r in self.adjustment_reasons]

    def get_run_style_japanese(self) -> str:
        """脚質ラベルの日本語を返す"""
        if self.run_style_label is None:
            return "不明"
        return RUN_STYLE_LABELS_JA.get(self.run_style_label, self.run_style_label)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式にシリアライズ"""
        result = {
            "horse_id": self.horse_id,
            "horse_name": self.horse_name,
            "frame_no": self.frame_no,
            "gate_number": self.gate_number,
            "run_style_label": self.run_style_label,
            "base_win": self.base_win,
            "base_in3": self.base_in3,
            "adj_win": self.adj_win,
            "adj_in3": self.adj_in3,
            "win_delta": self.win_delta,
            "in3_delta": self.in3_delta,
            "adjustment_reasons": [r.name for r in self.adjustment_reasons],
            "adjustment_reasons_ja": self.get_reasons_japanese(),
            "comment": self.comment,
        }
        # 体重コンテキスト
        if self.body_weight_context is not None:
            result["body_weight_context"] = self.body_weight_context.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioHorseScore:
        """辞書形式からデシリアライズ"""
        reasons = []
        for r in data.get("adjustment_reasons", []):
            if isinstance(r, str):
                try:
                    reasons.append(AdjustmentReason[r])
                except KeyError:
                    pass  # 不明な reason は無視
            elif isinstance(r, AdjustmentReason):
                reasons.append(r)

        # 後方互換: post_position があれば frame_no として使う
        frame_no = data.get("frame_no")
        if frame_no is None:
            frame_no = data.get("post_position")

        # 体重コンテキスト
        body_weight_ctx = None
        if "body_weight_context" in data and data["body_weight_context"]:
            body_weight_ctx = BodyWeightContext.from_dict(data["body_weight_context"])

        return cls(
            horse_id=data["horse_id"],
            horse_name=data.get("horse_name"),
            frame_no=frame_no,
            gate_number=data.get("gate_number"),
            run_style_label=data.get("run_style_label"),
            base_win=data.get("base_win", 0.0),
            base_in3=data.get("base_in3", 0.0),
            adj_win=data.get("adj_win", 0.0),
            adj_in3=data.get("adj_in3", 0.0),
            adjustment_reasons=reasons,
            body_weight_context=body_weight_ctx,
            comment=data.get("comment", ""),
        )


@dataclass
class ScenarioScore:
    """
    あるレース × あるシナリオに対する評価結果のコンテナ。

    ScenarioSpec への参照を持ち、重複したフィールドは持たない。
    馬ごとの評価結果を ScenarioHorseScore として格納する。

    Attributes:
        spec: 元のシナリオ定義（参照）
        horses: 馬ごとの評価結果（horse_id -> ScenarioHorseScore）
        scenario_comment: シナリオ全体のコメント
        recommended_horse_ids: このシナリオで推奨する馬ID（順位付き）
    """

    spec: ScenarioSpec
    horses: Dict[str, ScenarioHorseScore] = field(default_factory=dict)
    scenario_comment: str = ""
    recommended_horse_ids: List[str] = field(default_factory=list)

    @property
    def race_id(self) -> str:
        """レースIDへのショートカット"""
        return self.spec.race_id

    @property
    def scenario_id(self) -> str:
        """シナリオIDへのショートカット"""
        return self.spec.scenario_id

    @property
    def horse_count(self) -> int:
        """出走頭数"""
        return len(self.horses)

    def get_horse(self, horse_id: str) -> Optional[ScenarioHorseScore]:
        """馬IDから評価結果を取得"""
        return self.horses.get(horse_id)

    def add_horse(self, horse_score: ScenarioHorseScore) -> None:
        """馬の評価結果を追加"""
        self.horses[horse_score.horse_id] = horse_score

    def iter_horses(self) -> Iterator[ScenarioHorseScore]:
        """全馬の評価結果をイテレート"""
        yield from self.horses.values()

    def top_horses_by_adj_win(self, n: int = 5) -> List[ScenarioHorseScore]:
        """adj_win の上位n頭を返す"""
        sorted_horses = sorted(
            self.horses.values(), key=lambda h: h.adj_win, reverse=True
        )
        return sorted_horses[:n]

    def top_horses_by_adj_in3(self, n: int = 5) -> List[ScenarioHorseScore]:
        """adj_in3 の上位n頭を返す"""
        sorted_horses = sorted(
            self.horses.values(), key=lambda h: h.adj_in3, reverse=True
        )
        return sorted_horses[:n]

    def horses_with_positive_adjustment(self) -> List[ScenarioHorseScore]:
        """プラス補正がかかった馬のリストを返す（変化幅降順）"""
        positive = [h for h in self.horses.values() if h.has_positive_adjustment]
        return sorted(positive, key=lambda h: h.win_delta, reverse=True)

    def horses_with_negative_adjustment(self) -> List[ScenarioHorseScore]:
        """マイナス補正がかかった馬のリストを返す（変化幅昇順）"""
        negative = [h for h in self.horses.values() if h.has_negative_adjustment]
        return sorted(negative, key=lambda h: h.win_delta)

    def solid_placers(self) -> List[ScenarioHorseScore]:
        """堅実なヒモ候補を返す（adj_in3 降順）"""
        placers = [h for h in self.horses.values() if h.is_solid_placer]
        return sorted(placers, key=lambda h: h.adj_in3, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式にシリアライズ"""
        return {
            "scenario_id": self.scenario_id,
            "race_id": self.race_id,
            "spec": self.spec.to_dict(),
            "horses": {hid: h.to_dict() for hid, h in self.horses.items()},
            "scenario_comment": self.scenario_comment,
            "recommended_horse_ids": list(self.recommended_horse_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioScore:
        """辞書形式からデシリアライズ"""
        spec = ScenarioSpec.from_dict(data["spec"])
        horses = {
            hid: ScenarioHorseScore.from_dict(hdata)
            for hid, hdata in data.get("horses", {}).items()
        }
        return cls(
            spec=spec,
            horses=horses,
            scenario_comment=data.get("scenario_comment", ""),
            recommended_horse_ids=list(data.get("recommended_horse_ids", [])),
        )

    def to_llm_context(self, top_n: int = 10) -> Dict[str, Any]:
        """
        LLMに渡す用の軽量化されたコンテキスト。

        不要な詳細を削ぎ落とし、LLMが説明生成しやすい形式にする。
        in3（3着内率）も含めることで、「堅実なヒモ」「頭までは厳しい」
        などの表現をLLMがしやすくなる。
        """
        return {
            "scenario": {
                "id": self.scenario_id,
                "summary": self.spec.to_llm_summary(),
                "pace": self.spec.pace,
                "track_condition": self.spec.track_condition,
                "bias": self.spec.bias,
                "notes": self.spec.notes,
            },
            "race": self.spec.race_context.to_dict(),
            "top_horses": [
                {
                    "id": h.horse_id,
                    "name": h.horse_name,
                    "frame_no": h.frame_no,
                    "run_style": h.run_style_label,
                    "base_win": round(h.base_win, 4),
                    "adj_win": round(h.adj_win, 4),
                    "base_in3": round(h.base_in3, 4),
                    "adj_in3": round(h.adj_in3, 4),
                    "win_delta": round(h.win_delta, 4),
                    "win_delta_pct": round(h.win_delta_pct, 1),
                    "reasons": h.get_reasons_japanese(),
                    "weight_notes": h.body_weight_context.get_llm_notes() if h.body_weight_context else [],
                }
                for h in self.top_horses_by_adj_win(top_n)
            ],
            "boosted_horses": [
                {
                    "id": h.horse_id,
                    "name": h.horse_name,
                    "run_style": h.run_style_label,
                    "win_delta": round(h.win_delta, 4),
                    "in3_delta": round(h.in3_delta, 4),
                    "reasons": h.get_reasons_japanese(),
                }
                for h in self.horses_with_positive_adjustment()[:5]
            ],
            "dropped_horses": [
                {
                    "id": h.horse_id,
                    "name": h.horse_name,
                    "run_style": h.run_style_label,
                    "win_delta": round(h.win_delta, 4),
                    "in3_delta": round(h.in3_delta, 4),
                    "reasons": h.get_reasons_japanese(),
                }
                for h in self.horses_with_negative_adjustment()[:5]
            ],
            "solid_placers": [
                {
                    "id": h.horse_id,
                    "name": h.horse_name,
                    "adj_win": round(h.adj_win, 4),
                    "adj_in3": round(h.adj_in3, 4),
                    "reasons": h.get_reasons_japanese(),
                }
                for h in self.solid_placers()[:3]
            ],
        }

    def generate_summary_text(self) -> str:
        """
        シナリオ評価結果のサマリテキストを生成（デバッグ・ログ用）。
        """
        lines = [
            f"=== Scenario: {self.scenario_id} ===",
            f"Race: {self.race_id}",
            f"Conditions: {self.spec.to_llm_summary()}",
            "",
            "Top 5 by adjusted win rate:",
        ]

        for i, h in enumerate(self.top_horses_by_adj_win(5), 1):
            delta_str = f"+{h.win_delta:.3f}" if h.win_delta >= 0 else f"{h.win_delta:.3f}"
            name = h.horse_name or h.horse_id
            run_style = f"[{h.run_style_label}]" if h.run_style_label else ""
            reasons = ", ".join(h.get_reasons_japanese()) if h.adjustment_reasons else "-"
            lines.append(
                f"  {i}. {name} {run_style}: {h.base_win:.3f} -> {h.adj_win:.3f} ({delta_str}) [{reasons}]"
            )

        if self.scenario_comment:
            lines.extend(["", f"Comment: {self.scenario_comment}"])

        return "\n".join(lines)
