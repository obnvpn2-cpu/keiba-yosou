"""
シナリオレイヤで使用する共通の型定義。

Literal型で選択肢を固定することで、型チェッカーによる検証を有効にする。
将来の拡張（地方競馬・海外）を見据え、Union型での拡張ポイントをコメントで示す。
"""

from typing import Literal

# ペース想定
# S=スロー, M=ミドル, H=ハイ
PaceType = Literal["S", "M", "H"]

# 馬場状態（JRA標準）
# 地方競馬対応時: Union[JRATrackCondition, LocalTrackCondition] に拡張
TrackConditionType = Literal["良", "稍重", "重", "不良"]

# コースバイアス
BiasType = Literal["内", "外", "フラット"]

# 馬場種別
SurfaceType = Literal["turf", "dirt"]

# 距離カテゴリ（feature_builder.py と整合）
DistanceCategoryType = Literal["sprint", "mile", "intermediate", "long", "extended"]

# JRA競馬場コード
# 地方競馬対応時: Union[JRACourse, LocalCourse] に拡張
JRACourseType = Literal[
    "札幌", "函館", "福島", "新潟", "東京",
    "中山", "中京", "京都", "阪神", "小倉"
]
