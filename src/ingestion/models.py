"""
netkeiba ingestion パイプライン - データモデル定義

レース、出走馬、払い戻し、ラップタイム等のデータ構造を定義する。
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class Race:
    """レース基本情報"""
    race_id: str  # 12桁の数値文字列
    date: Optional[date] = None
    place: Optional[str] = None  # 中山, 東京 など
    kai: Optional[int] = None  # 回次（5回中山 の 5）
    nichime: Optional[int] = None  # 日目（8日目 の 8）
    race_no: Optional[int] = None  # R番号（11R の 11）
    name: Optional[str] = None  # レース名（第68回有馬記念(GI)）
    grade: Optional[str] = None  # G1, G2, G3, OP, Listed など
    race_class: Optional[str] = None  # 3歳以上オープン (国際)(指)(定量)
    course_type: Optional[str] = None  # turf, dirt, steeple
    distance: Optional[int] = None  # メートル
    course_turn: Optional[str] = None  # 右, 左, 直線
    course_inout: Optional[str] = None  # 内, 外
    weather: Optional[str] = None  # 晴, 曇, 雨 など
    track_condition: Optional[str] = None  # 良, 稍重, 重, 不良
    start_time: Optional[str] = None  # "15:40" など
    baba_index: Optional[int] = None  # 馬場指数
    baba_comment: Optional[str] = None  # 馬場コメント
    analysis_comment: Optional[str] = None  # 分析コメント
    head_count: Optional[int] = None  # 出走頭数


@dataclass
class RaceResult:
    """出走馬の成績"""
    race_id: str
    horse_id: str
    finish_order: Optional[int] = None  # 着順（取消や除外は None）
    finish_status: Optional[str] = None  # 正常完走は None, それ以外は "取消", "除外", "中止", "失格"
    frame_no: Optional[int] = None  # 枠番
    horse_no: Optional[int] = None  # 馬番
    horse_name: Optional[str] = None
    sex: Optional[str] = None  # 牡, 牝, セ
    age: Optional[int] = None
    weight: Optional[float] = None  # 斤量
    jockey_id: Optional[str] = None
    jockey_name: Optional[str] = None
    time_str: Optional[str] = None  # "2:30.9"
    time_sec: Optional[float] = None  # 150.9
    margin: Optional[str] = None  # 着差（"1/2", "クビ", "アタマ" など）
    passing_order: Optional[str] = None  # 通過順（"13-13-8-3"）
    last_3f: Optional[float] = None  # 上り3F
    win_odds: Optional[float] = None  # 単勝オッズ
    popularity: Optional[int] = None  # 人気
    body_weight: Optional[int] = None  # 馬体重
    body_weight_diff: Optional[int] = None  # 馬体重増減
    time_index: Optional[int] = None  # タイム指数（プレミアム）
    trainer_id: Optional[str] = None
    trainer_name: Optional[str] = None
    trainer_region: Optional[str] = None  # 東, 西
    owner_id: Optional[str] = None
    owner_name: Optional[str] = None
    prize_money: Optional[float] = None  # 賞金（万円）
    remark_text: Optional[str] = None  # 備考（出遅れ など）


@dataclass
class Payout:
    """払い戻し情報"""
    race_id: str
    bet_type: str  # 単勝, 複勝, 枠連, 馬連, ワイド, 馬単, 三連複, 三連単
    combination: str  # "5", "5 - 16", "5 → 16 → 4" など
    payout: int  # 配当（円）
    popularity: Optional[int] = None  # 人気


@dataclass
class Corner:
    """コーナー通過順位"""
    race_id: str
    corner_1: Optional[str] = None
    corner_2: Optional[str] = None
    corner_3: Optional[str] = None
    corner_4: Optional[str] = None


@dataclass
class LapTime:
    """レース全体ラップタイム（200m単位）"""
    race_id: str
    lap_index: int  # 1始まり
    distance_m: int  # 累積距離（200, 400, ...）
    time_sec: float  # そのラップのタイム（秒）


@dataclass
class HorseLap:
    """個別馬のラップタイム（マスター会員限定）"""
    race_id: str
    horse_id: str
    section_m: int  # 区間距離（100, 300, 500, ...）
    time_sec: Optional[float] = None  # ラップタイム（秒）
    position: Optional[int] = None  # その区間での順位（あれば）


@dataclass
class HorseShortComment:
    """注目馬 レース後の短評（マスター会員限定）"""
    race_id: str
    horse_id: str
    horse_name: str
    finish_order: int
    comment: str


@dataclass
class ParsedRaceData:
    """パース結果をまとめたコンテナ"""
    race: Race
    results: list[RaceResult] = field(default_factory=list)
    payouts: list[Payout] = field(default_factory=list)
    corner: Optional[Corner] = None
    lap_times: list[LapTime] = field(default_factory=list)
    horse_laps: list[HorseLap] = field(default_factory=list)
    short_comments: list[HorseShortComment] = field(default_factory=list)


# 場コードのマッピング
PLACE_CODE_MAP = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

PLACE_NAME_TO_CODE = {v: k for k, v in PLACE_CODE_MAP.items()}

# JRA場コード一覧
JRA_PLACE_CODES = list(PLACE_CODE_MAP.keys())
