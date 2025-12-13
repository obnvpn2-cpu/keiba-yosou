"""
ScenarioSpec: 人間が定義する『レースシナリオ』の入力情報。

このモジュールは scenario/ の入力側を担当し、他のモジュールに依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from .types import (
    PaceType,
    TrackConditionType,
    BiasType,
    SurfaceType,
    DistanceCategoryType,
    JRACourseType,
)


@dataclass(frozen=True)
class MoistureReading:
    """
    含水率の1区画分（ゴール前 or 4コーナー）。
    
    JRAは芝・ダートそれぞれについて、ゴール前と4コーナーの2地点で
    含水率を計測・公開している。
    
    Attributes:
        goal: ゴール前の含水率（%）
        corner_4: 4コーナーの含水率（%）
    """
    goal: Optional[float] = None
    corner_4: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "goal": self.goal,
            "corner_4": self.corner_4,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MoistureReading:
        return cls(
            goal=data.get("goal"),
            corner_4=data.get("corner_4"),
        )
    
    def is_empty(self) -> bool:
        """含水率データが未設定かどうか"""
        return self.goal is None and self.corner_4 is None
    
    def average(self) -> Optional[float]:
        """2地点の平均値を返す（片方のみの場合はその値）"""
        values = [v for v in (self.goal, self.corner_4) if v is not None]
        if not values:
            return None
        return sum(values) / len(values)


@dataclass(frozen=True)
class TrackMoisture:
    """
    芝・ダートそれぞれの含水率をまとめた構造体。
    
    Attributes:
        turf: 芝コースの含水率
        dirt: ダートコースの含水率
    """
    turf: MoistureReading = field(default_factory=MoistureReading)
    dirt: MoistureReading = field(default_factory=MoistureReading)
    
    def to_dict(self) -> Dict[str, Dict[str, Optional[float]]]:
        return {
            "turf": self.turf.to_dict(),
            "dirt": self.dirt.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackMoisture:
        turf_data = data.get("turf", {})
        dirt_data = data.get("dirt", {})
        return cls(
            turf=MoistureReading.from_dict(turf_data) if turf_data else MoistureReading(),
            dirt=MoistureReading.from_dict(dirt_data) if dirt_data else MoistureReading(),
        )
    
    def get_for_surface(self, surface: SurfaceType) -> MoistureReading:
        """指定した馬場種別の含水率を返す"""
        return self.turf if surface == "turf" else self.dirt


@dataclass
class RaceContext:
    """
    レースの基本情報（LLM説明生成で必要になるメタ情報）。
    
    ScenarioSpec とは別に切り出すことで、
    シナリオ定義時に毎回入力しなくても済むようにする。
    DBから取得した情報をここにまとめる想定。
    
    Attributes:
        race_id: レースID（netkeiba形式）
        race_name: レース名（例: "日本ダービー"）
        course: 競馬場
        surface: 馬場種別
        distance: 距離（メートル）
        distance_cat: 距離カテゴリ
        race_date: レース日（YYYY-MM-DD形式）
        race_class: クラス（例: "G1", "3勝クラス"）
    """
    race_id: str
    race_name: Optional[str] = None
    course: Optional[JRACourseType] = None
    surface: Optional[SurfaceType] = None
    distance: Optional[int] = None
    distance_cat: Optional[DistanceCategoryType] = None
    race_date: Optional[str] = None
    race_class: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "race_id": self.race_id,
            "race_name": self.race_name,
            "course": self.course,
            "surface": self.surface,
            "distance": self.distance,
            "distance_cat": self.distance_cat,
            "race_date": self.race_date,
            "race_class": self.race_class,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RaceContext:
        return cls(
            race_id=data["race_id"],
            race_name=data.get("race_name"),
            course=data.get("course"),
            surface=data.get("surface"),
            distance=data.get("distance"),
            distance_cat=data.get("distance_cat"),
            race_date=data.get("race_date"),
            race_class=data.get("race_class"),
        )


@dataclass
class ScenarioSpec:
    """
    人間が定義する『レースシナリオ』の入力情報。
    
    シナリオとは、レース当日の馬場状態・展開・バイアスなどについての
    人間の予想・想定を構造化したもの。同一レースに対して複数のシナリオを
    定義し、比較することを想定している。
    
    Attributes:
        scenario_id: シナリオを一意に識別するID（複数シナリオ比較用）
        race_context: レースの基本情報
        pace: 想定ペース（S=スロー, M=ミドル, H=ハイ）
        track_condition: 馬場状態
        bias: コースバイアス
        cushion_value: 当日朝のクッション値
        moisture: 含水率（JRA 4区分）
        front_runner_ids: 逃げ想定馬のID
        stalker_ids: 先行想定馬のID
        closer_ids: 差し・追込想定馬のID
        weak_horse_ids: 状態不安・信頼度低い馬のID
        notes: 補足メモ
    
    Examples:
        >>> spec = ScenarioSpec(
        ...     scenario_id="slow_inner",
        ...     race_context=RaceContext(race_id="202305021211"),
        ...     pace="S",
        ...     track_condition="良",
        ...     bias="内",
        ...     front_runner_ids=["2019104567"],
        ...     notes="逃げ馬が1頭しかいないのでスロー濃厚",
        ... )
    """
    # 識別子
    scenario_id: str
    race_context: RaceContext
    
    # コア想定（必須）
    pace: PaceType
    track_condition: TrackConditionType
    bias: BiasType
    
    # 馬場データ（任意）
    cushion_value: Optional[float] = None
    moisture: TrackMoisture = field(default_factory=TrackMoisture)
    
    # 馬ごとの脚質・状態情報
    front_runner_ids: List[str] = field(default_factory=list)
    stalker_ids: List[str] = field(default_factory=list)
    closer_ids: List[str] = field(default_factory=list)
    weak_horse_ids: List[str] = field(default_factory=list)
    
    # 補足メモ
    notes: str = ""
    
    @property
    def race_id(self) -> str:
        """レースIDへのショートカット"""
        return self.race_context.race_id
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式にシリアライズ"""
        return {
            "scenario_id": self.scenario_id,
            "race_context": self.race_context.to_dict(),
            "pace": self.pace,
            "track_condition": self.track_condition,
            "bias": self.bias,
            "cushion_value": self.cushion_value,
            "moisture": self.moisture.to_dict(),
            "front_runner_ids": list(self.front_runner_ids),
            "stalker_ids": list(self.stalker_ids),
            "closer_ids": list(self.closer_ids),
            "weak_horse_ids": list(self.weak_horse_ids),
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioSpec:
        """辞書形式からデシリアライズ"""
        moisture_data = data.get("moisture", {})
        race_context_data = data.get("race_context", {})
        
        # 後方互換: race_context がなく race_id だけある場合
        if not race_context_data and "race_id" in data:
            race_context_data = {"race_id": data["race_id"]}
        
        return cls(
            scenario_id=data["scenario_id"],
            race_context=RaceContext.from_dict(race_context_data),
            pace=data["pace"],
            track_condition=data["track_condition"],
            bias=data["bias"],
            cushion_value=data.get("cushion_value"),
            moisture=TrackMoisture.from_dict(moisture_data) if moisture_data else TrackMoisture(),
            front_runner_ids=list(data.get("front_runner_ids", [])),
            stalker_ids=list(data.get("stalker_ids", [])),
            closer_ids=list(data.get("closer_ids", [])),
            weak_horse_ids=list(data.get("weak_horse_ids", [])),
            notes=data.get("notes", ""),
        )
    
    def get_running_style(self, horse_id: str) -> Optional[str]:
        """指定した馬の想定脚質を返す"""
        if horse_id in self.front_runner_ids:
            return "front_runner"
        elif horse_id in self.stalker_ids:
            return "stalker"
        elif horse_id in self.closer_ids:
            return "closer"
        return None
    
    def is_marked_weak(self, horse_id: str) -> bool:
        """指定した馬が信頼度低とマークされているか"""
        return horse_id in self.weak_horse_ids
    
    def to_llm_summary(self) -> str:
        """LLMに渡すための自然言語サマリを生成"""
        parts = []
        
        # ペース
        pace_map = {"S": "スローペース", "M": "ミドルペース", "H": "ハイペース"}
        parts.append(f"想定ペース: {pace_map.get(self.pace, self.pace)}")
        
        # 馬場
        parts.append(f"馬場状態: {self.track_condition}")
        
        # バイアス
        bias_map = {"内": "内有利", "外": "外有利", "フラット": "フラット"}
        parts.append(f"バイアス: {bias_map.get(self.bias, self.bias)}")
        
        # 逃げ馬情報
        if self.front_runner_ids:
            parts.append(f"逃げ想定: {len(self.front_runner_ids)}頭")
        
        # 補足
        if self.notes:
            parts.append(f"補足: {self.notes}")
        
        return " / ".join(parts)
