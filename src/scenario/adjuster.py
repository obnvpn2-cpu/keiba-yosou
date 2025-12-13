"""
ScenarioAdjuster

人間が定義したレースシナリオ (ScenarioSpec) と
ベースモデルの予測値 (base_predictions)・簡易特徴量 (horse_features) を受け取り、
馬ごとの勝率 / 3着内率をシナリオに応じて補正するレイヤ。

このモジュールでは「何がどれくらい効いているか」を
できるだけテーブル化された係数として持たせている。

Note:
    このレイヤは「倍率を掛けるだけ」の補正レイヤ。
    確率レンジの最終調整（Platt/Isotonic等）は別途 Calibration レイヤで行う想定。
    normalize=True で正規化を行うが、別途 Calibrator を噛ませる場合は
    normalize=False にして二重キャリブレーションを避けること。

設計思想:
    - **ペース補正** は「どの脚質が得をするか」を担当
      - 差し・追込が有利になるのは「ハイペース(H)」が原因という設計
      - 追込 (deep_closer) がペース変化に一番強く反応
      - 差し (mid_chaser) は先行と追込の中間

    - **バイアス補正** は「どのレーン (inside/middle/outside) が速いか」だけを見る
      - 脚質には直接のプラス/マイナスを与えない
      - 「外バイアスだから差しが有利」ではなく、
        「外バイアス → outside レーンが相対的にマシになる」という設計
      - バイアス強度は頭数でスケール（8頭立てと18頭立てで外枠不利の度合いが違う）

    - **レーン推定** は枠番 + 脚質 + 頭数から決める
      - 内枠の差し/追込は「道中内→直線外に出す」ことが多いので
        "inside" ではなく "middle" 扱い
      - 真の inside バイアス恩恵は「内枠の逃げ・先行」だけ

    - コース固有の枠順バイアス（中山芝2500mは内有利など）はベースモデル側の
      特徴量で表現されている前提。このレイヤでは当日のトラックバイアスのみ扱う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Mapping, List, Tuple, Literal

import logging

from .spec import ScenarioSpec
from .score import ScenarioScore, ScenarioHorseScore, AdjustmentReason

logger = logging.getLogger(__name__)


# =============================================================================
# 型定義
# =============================================================================

# レーン種別
LaneType = Literal["inside", "middle", "outside"]

# 内部脚質カテゴリ（4分類）
RunStyleCategory = Literal["front_runner", "pace_keeper", "mid_chaser", "deep_closer", "other"]


# =============================================================================
# 定数
# =============================================================================

# 正規化済みの脚質ラベル
VALID_RUN_STYLES = ("逃げ", "先行", "差し", "追込")

# 脚質の別表記 → 正規化ラベルへのマッピング
RUN_STYLE_MAPPING: Dict[str, str] = {
    "NIGE": "逃げ",
    "SENKO": "先行",
    "SENKOU": "先行",
    "好位": "先行",
    "SASHI": "差し",
    "SASI": "差し",
    "OIKOMI": "追込",
    "追い込み": "追込",
    "自在": "先行",  # 自在は先行寄りとして扱う
}

# 日本語脚質ラベル → 内部カテゴリへのマッピング
RUN_STYLE_TO_CATEGORY: Dict[str, RunStyleCategory] = {
    "逃げ": "front_runner",
    "先行": "pace_keeper",
    "差し": "mid_chaser",
    "追込": "deep_closer",
    "その他": "other",
}

# ベース脚質ラベル → AdjustmentReason のマッピング
RUN_STYLE_TO_REASON: Dict[str, AdjustmentReason] = {
    "逃げ": AdjustmentReason.FRONT_RUNNER,
    "先行": AdjustmentReason.STALKER,
    "差し": AdjustmentReason.CLOSER,
    "追込": AdjustmentReason.CLOSER,
}


# =============================================================================
# 係数定義
# =============================================================================


@dataclass
class AdjustmentConfig:
    """
    シナリオ補正に使う各種係数テーブルをまとめた設定。

    - pace_category_* : ペース × 内部脚質カテゴリ の乗数
    - front_runner_*  : シナリオで「逃げ想定」に指定された馬への追加乗数
    - weak_horse_*    : 「信頼度低い」に指定された馬への減衰係数
    - bias_*          : トラックバイアス（内伸び・外伸び）のレーン別係数

    脚質カテゴリ:
        - front_runner: 逃げ（最前列、ハイペースで潰れやすい）
        - pace_keeper: 先行（2-4番手、スロー有利だがハイでも致命傷ではない）
        - mid_chaser: 差し（中団、ペース変動に中程度の影響）
        - deep_closer: 追込（後方一気、ペース変動に最も敏感）
        - other: 不明・その他
    """

    # ペース × 内部脚質カテゴリ 別の係数 (勝率用)
    # キー: ペース(S/M/H) → 内部カテゴリ
    pace_category_win: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ペース × 内部脚質カテゴリ 別の係数 (3着内率用)
    pace_category_in3: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # シナリオで front_runner_ids に含まれている馬への追加ブースト係数
    front_runner_win: Dict[str, float] = field(default_factory=dict)
    front_runner_in3: Dict[str, float] = field(default_factory=dict)

    # 「状態や信頼度の低い馬」への減衰係数
    weak_horse_win: float = 0.85
    weak_horse_in3: float = 0.90

    # 内伸びバイアス時の係数（inside レーンに有利、outside レーンに不利）
    # これらはフルスケール（18頭立て）での値。頭数で減衰される。
    bias_inner_win_adv: float = 1.05       # inside レーンの勝率ブースト
    bias_inner_in3_adv: float = 1.03       # inside レーンの3着内率ブースト
    bias_inner_win_disadv: float = 0.97    # outside レーンの勝率ペナルティ
    bias_inner_in3_disadv: float = 0.98    # outside レーンの3着内率ペナルティ

    # 外伸びバイアス時の係数（outside レーンに有利、inside レーンに不利）
    bias_outer_win_adv: float = 1.05       # outside レーンの勝率ブースト
    bias_outer_in3_adv: float = 1.03       # outside レーンの3着内率ブースト
    bias_outer_win_disadv: float = 0.97    # inside レーンの勝率ペナルティ
    bias_outer_in3_disadv: float = 0.98    # inside レーンの3着内率ペナルティ

    @classmethod
    def default(cls) -> "AdjustmentConfig":
        """
        デフォルト係数一式。

        ペース補正の設計思想:
        - スロー(S): 逃げ/先行が有利。追込が一番不利（展開が向かない）
        - ミドル(M): ほぼニュートラル（=1.0中心）
        - ハイ(H): 差し/追込が有利。逃げが一番潰れやすい

        追込 (deep_closer) は「ペース変化に最も敏感」という設計。
        差し (mid_chaser) は「先行と追込の中間」で穏やかな変動。

        バイアス補正:
        - バイアスはレーンだけを見る。脚質で直接有利/不利を与えない。
        - 「外バイアスだから差しが有利」ではなく、
          「外バイアス → outside レーンが相対的にマシになる」という設計。
        """
        pace_category_win = {
            "S": {
                "front_runner": 1.20,   # 逃げ: スローで最も有利
                "pace_keeper":  1.10,   # 先行: やや有利
                "mid_chaser":   0.95,   # 差し: やや不利
                "deep_closer":  0.85,   # 追込: 最も不利（展開が向かない）
                "other":        1.00,
            },
            "M": {
                "front_runner": 1.00,
                "pace_keeper":  1.00,
                "mid_chaser":   1.00,
                "deep_closer":  1.00,
                "other":        1.00,
            },
            "H": {
                "front_runner": 0.80,   # 逃げ: ハイで最も潰れやすい
                "pace_keeper":  0.90,   # 先行: やや不利
                "mid_chaser":   1.10,   # 差し: やや有利
                "deep_closer":  1.20,   # 追込: 最も有利（展開ハマり）
                "other":        1.00,
            },
        }

        pace_category_in3 = {
            "S": {
                "front_runner": 1.10,
                "pace_keeper":  1.05,
                "mid_chaser":   0.98,
                "deep_closer":  0.92,
                "other":        1.00,
            },
            "M": {
                "front_runner": 1.00,
                "pace_keeper":  1.00,
                "mid_chaser":   1.00,
                "deep_closer":  1.00,
                "other":        1.00,
            },
            "H": {
                "front_runner": 0.90,
                "pace_keeper":  0.95,
                "mid_chaser":   1.05,
                "deep_closer":  1.12,
                "other":        1.00,
            },
        }

        # front_runner_ids で明示的に「逃げ想定」にされている馬への追加補正
        # ペース補正との合計が強くなりすぎないよう、控えめに設定
        front_runner_win = {
            "S": 1.05,  # スロー想定での逃げ指定（ペース補正と合わせて約1.26倍）
            "M": 1.03,
            "H": 0.95,  # ハイペース想定ではリスクとして少し下げる
        }
        front_runner_in3 = {
            "S": 1.03,
            "M": 1.02,
            "H": 0.97,
        }

        return cls(
            pace_category_win=pace_category_win,
            pace_category_in3=pace_category_in3,
            front_runner_win=front_runner_win,
            front_runner_in3=front_runner_in3,
        )


# =============================================================================
# ユーティリティ
# =============================================================================


def _normalize_run_style(raw: Any) -> Optional[str]:
    """
    feature_table 等に入っている脚質表現をシナリオ側のキーに正規化する。

    想定する入力例:
        "逃げ", "先行", "差し", "追込"
        "NIGE", "SENKO", "SASHI", "OIKOMI"
        None, "", など

    戻り値:
        "逃げ" / "先行" / "差し" / "追込" / None
    """
    if raw is None:
        return None

    s = str(raw).strip()

    # 正規化済みの値ならそのまま返す
    if s in VALID_RUN_STYLES:
        return s

    # 英語表記・別表記を変換
    return RUN_STYLE_MAPPING.get(s.upper())


def _get_run_style_category(run_style_label: str) -> RunStyleCategory:
    """
    日本語脚質ラベル → 内部カテゴリに変換。

    Args:
        run_style_label: "逃げ"/"先行"/"差し"/"追込"/"その他"

    Returns:
        "front_runner"/"pace_keeper"/"mid_chaser"/"deep_closer"/"other"
    """
    return RUN_STYLE_TO_CATEGORY.get(run_style_label, "other")


def _infer_run_style_label(
    horse_id: str,
    features: Mapping[str, Any] | None,
    spec: ScenarioSpec,
) -> Tuple[str, bool]:
    """
    シナリオ補正で使う脚質ラベルを決める。

    優先順位:
        1. シナリオで front_runner_ids に含まれていれば強制的に「逃げ」
        2. features["run_style"] があればそれを正規化して使う
        3. spec.stalker_ids / spec.closer_ids を見て「先行」「差し」
        4. それでも決まらなければ "その他"

    Returns:
        (脚質ラベル, シナリオ指定かどうか)
    """
    # 1. シナリオ側で「逃げ想定」にされていれば最優先
    if horse_id in spec.front_runner_ids:
        return ("逃げ", True)

    # 2. 特徴量側で脚質があれば尊重
    if features is not None:
        raw_style = features.get("run_style")
        if raw_style is not None:
            style = _normalize_run_style(raw_style)
            if style is not None:
                return (style, False)

    # 3. シナリオ側の補助的な脚質指定
    if horse_id in spec.stalker_ids:
        return ("先行", True)
    if horse_id in spec.closer_ids:
        return ("差し", True)

    # 4. 何も情報がなければその他
    return ("その他", False)


def _field_size_factor(field_size: int) -> float:
    """
    頭数に基づくバイアス強度のスケール係数を計算。

    8頭立てと18頭立てでは外枠不利の度合いが全く違う。
    - 少頭数（8頭）：外枠でもそこまで絶望的ではない → 係数弱める
    - 多頭数（18頭）：外を回されるリスクが大きい → フルに効かせる

    Args:
        field_size: 出走頭数

    Returns:
        0.4 (8頭) 〜 1.0 (18頭) のスケール係数
    """
    x = max(8, min(field_size, 18))  # 8〜18にクリップ
    t = (x - 8) / (18 - 8)  # 0〜1 に正規化
    return 0.4 + 0.6 * t    # 0.4〜1.0


# =============================================================================
# 本体クラス
# =============================================================================


@dataclass
class ScenarioAdjuster:
    """
    ScenarioSpec と ベース予測値を受け取り、ScenarioScore を返す本体。

    base_predictions: {horse_id: {"win": float, "in3": float}}
    horse_features:   {horse_id: {feature_name: value}}
        - ここで使う想定のキー例:
            - "frame_no": 枠順 (1–8)
            - "run_style": 脚質
            - "jockey_name": 騎手名（現状はロジック未使用だが将来拡張用）

    Note:
        このクラスは「倍率を掛けるだけ」の補正レイヤ。
        確率レンジの最終調整は別途 Calibration レイヤで行う想定。

        トラックバイアスは「枠順」ではなく「直線で通るレーン」に対して補正。
        枠＋脚質＋頭数から推定したレーンにバイアス係数を適用することで、
        同じ外枠でも脚質によってバイアスの効き方が変わる設計になっている。

        重要な設計原則:
        - 「差し・追込が有利」はペース補正の責任（ハイペースで得をする）
        - 「レーンが有利」はバイアス補正の責任（脚質には直接作用しない）
    """

    config: AdjustmentConfig = field(default_factory=AdjustmentConfig.default)

    def adjust(
        self,
        spec: ScenarioSpec,
        base_predictions: Mapping[str, Mapping[str, float]],
        horse_features: Mapping[str, Mapping[str, Any]] | None = None,
        horse_names: Mapping[str, str] | None = None,
        normalize: bool = True,
    ) -> ScenarioScore:
        """
        シナリオ補正を適用し、ScenarioScore を返す。

        Args:
            spec: シナリオ定義
            base_predictions: 馬ごとのベース予測値
                {horse_id: {"win": float, "in3": float}}
            horse_features: 馬ごとの特徴量
                {horse_id: {"frame_no": int, "run_style": str, ...}}
            horse_names: 馬名のマッピング
                {horse_id: "馬名"}
            normalize: 勝率を正規化するかどうか（デフォルト: True）
                別途 Calibrator を使う場合は False にして二重補正を避ける

        Returns:
            ScenarioScore: 補正後の評価結果
        """
        horses: Dict[str, ScenarioHorseScore] = {}
        horse_features = horse_features or {}

        # 頭数を取得（バイアス強度スケーリング用）
        field_size = len(base_predictions)

        for horse_id, preds in base_predictions.items():
            base_win = float(preds.get("win", 0.0))
            base_in3 = float(preds.get("in3", 0.0))

            features = horse_features.get(horse_id, {})
            name = horse_names.get(horse_id) if horse_names is not None else None

            # ScenarioHorseScore を先に作成し、add_reason で理由を追加していく
            horse_score = ScenarioHorseScore(
                horse_id=horse_id,
                horse_name=name,
                frame_no=self._extract_frame_no(features),
                base_win=base_win,
                base_in3=base_in3,
                adj_win=base_win,
                adj_in3=base_in3,
            )

            # -----------------------------------------------------------------
            # ペース × 脚質
            # -----------------------------------------------------------------
            run_label, is_scenario_specified = _infer_run_style_label(
                horse_id, features, spec
            )
            horse_score.run_style_label = run_label
            pace = spec.pace  # "S" / "M" / "H"

            # 脚質ラベルを reason に追加
            if run_label in RUN_STYLE_TO_REASON:
                horse_score.add_reason(RUN_STYLE_TO_REASON[run_label])

            # シナリオ指定の脚質なら追加の reason
            if is_scenario_specified:
                if run_label == "逃げ":
                    horse_score.add_reason(AdjustmentReason.SCENARIO_FRONT_RUNNER)
                elif run_label == "先行":
                    horse_score.add_reason(AdjustmentReason.SCENARIO_STALKER)
                elif run_label in ("差し", "追込"):
                    horse_score.add_reason(AdjustmentReason.SCENARIO_CLOSER)

            # 内部カテゴリに変換してペース補正を適用
            category = _get_run_style_category(run_label)
            pace_win_table = self.config.pace_category_win.get(pace, {})
            pace_in3_table = self.config.pace_category_in3.get(pace, {})

            win_factor = pace_win_table.get(category, 1.0)
            in3_factor = pace_in3_table.get(category, 1.0)

            # 有利・不利を判定して reason に追加
            if win_factor > 1.0 or in3_factor > 1.0:
                horse_score.add_reason(AdjustmentReason.PACE_ADVANTAGE)
            elif win_factor < 1.0 or in3_factor < 1.0:
                horse_score.add_reason(AdjustmentReason.PACE_DISADVANTAGE)

            horse_score.adj_win *= win_factor
            horse_score.adj_in3 *= in3_factor

            # -----------------------------------------------------------------
            # 逃げ想定 (front_runner_ids) への追加補正
            # -----------------------------------------------------------------
            if horse_id in spec.front_runner_ids:
                fr_win_factor = self.config.front_runner_win.get(pace, 1.0)
                fr_in3_factor = self.config.front_runner_in3.get(pace, 1.0)

                horse_score.adj_win *= fr_win_factor
                horse_score.adj_in3 *= fr_in3_factor

            # -----------------------------------------------------------------
            # トラックバイアス（内伸び・外伸び）× レーン
            # -----------------------------------------------------------------
            if horse_score.frame_no is not None:
                # 枠番＋脚質＋頭数から直線で通りそうなレーンを推定
                lane = self._estimate_lane(
                    horse_score.frame_no,
                    category,
                    field_size,
                )

                # レーンに対してバイアス補正を適用（頭数でスケール）
                bias_win, bias_in3, bias_reason = self._apply_bias_adjustment(
                    spec.bias, lane, field_size
                )
                horse_score.adj_win *= bias_win
                horse_score.adj_in3 *= bias_in3
                if bias_reason is not None:
                    horse_score.add_reason(bias_reason)

            # -----------------------------------------------------------------
            # 「状態や信頼度の低い馬」への減衰
            # -----------------------------------------------------------------
            if horse_id in spec.weak_horse_ids:
                horse_score.adj_win *= self.config.weak_horse_win
                horse_score.adj_in3 *= self.config.weak_horse_in3
                horse_score.add_reason(AdjustmentReason.MARKED_WEAK)

            horses[horse_id] = horse_score

        # 正規化
        if normalize:
            self._normalize_probabilities(horses)

        return ScenarioScore(spec=spec, horses=horses)

    def _extract_frame_no(self, features: Mapping[str, Any]) -> Optional[int]:
        """
        特徴量から枠番を抽出する。

        Returns:
            枠番 (1-8) または None
        """
        if "frame_no" not in features:
            return None

        try:
            frame_no = int(features["frame_no"])
            if 1 <= frame_no <= 8:
                return frame_no
            return None
        except (TypeError, ValueError):
            return None

    def _estimate_lane(
        self,
        frame_no: int,
        category: RunStyleCategory,
        field_size: int,
    ) -> LaneType:
        """
        枠番・脚質・頭数から、直線で通りがちなレーンをざっくり推定する。

        設計思想:
            - バイアス補正は「レーン」だけを見る。脚質に直接ボーナスを与えない。
            - 差し/追込が有利かどうかは「ペース補正」の責任。
            - 「内バイアス有利」を受けられるのは「真の inside（内枠の逃げ/先行）」だけ。
            - 内枠の差し/追込は「道中内→直線外」が多いので middle 扱い。

        ロジック:
            1. 頭数からレーン境界を決める
               - 内側レーン: 上位約30%の枠（最低2枠）
               - 外側レーン: 下位約30%の枠（最低2枠）
               - 残り: middle

            2. 脚質カテゴリでレーン推定を調整
               - front_runner (逃げ): どの枠でも inside（スタートから内を取りに行く）
               - pace_keeper (先行): 内枠→inside, 中枠→多頭数ならmiddle/少頭数ならinside, 外枠→middle
               - mid_chaser (差し): 内枠→middle（道中内→直線外）, 中枠→middle, 外枠→outside
               - deep_closer (追込): 内枠→middle（道中内→直線外）, 中枠/外枠→outside
               - other: 素の枠ベースそのまま

        Args:
            frame_no: 枠番 (1-8)
            category: 内部脚質カテゴリ
            field_size: 出走頭数

        Returns:
            "inside" / "middle" / "outside"
        """
        # Step 1: 頭数に基づいてレーン境界を計算
        x = max(8, min(field_size, 18))  # 8〜18にクリップ
        inside_max = max(2, round(x * 0.3))      # 内側レーン: 枠1〜inside_max
        outside_min = x - max(2, round(x * 0.3)) + 1  # 外側レーン: outside_min〜x

        # Step 2: 枠番からベースラインレーンを決定
        if frame_no <= inside_max:
            baseline: LaneType = "inside"
        elif frame_no >= outside_min:
            baseline = "outside"
        else:
            baseline = "middle"

        # Step 3: 脚質カテゴリでレーン推定を調整
        # ポイント: 内枠の差し/追込は inside にしない（middle 止まり）

        if category == "front_runner":
            # 逃げ: どの枠でもスタートから内を取りに行く
            return "inside"

        elif category == "pace_keeper":
            # 先行: 内枠→inside, 中枠→頭数次第, 外枠→middle
            if baseline == "inside":
                return "inside"
            elif baseline == "outside":
                # 外枠の先行は完全 outside にはしない（ある程度内に潜れる）
                return "middle"
            else:
                # 中枠の先行: 多頭数(14頭以上)なら middle、少頭数なら inside 寄り
                return "middle" if field_size >= 14 else "inside"

        elif category == "mid_chaser":
            # 差し: 内枠→middle（道中イン→直線外）, 中枠→middle, 外枠→outside
            if baseline == "inside":
                # 内枠の差しは inside バイアスをフルには受けない
                return "middle"
            elif baseline == "outside":
                return "outside"
            else:
                return "middle"

        elif category == "deep_closer":
            # 追込: 内枠→middle（道中内→直線外）, 中枠/外枠→outside
            if baseline == "inside":
                # 内枠の追込も inside バイアスをフルには受けない
                return "middle"
            else:
                # 中枠・外枠の追込は外を回して進路確保
                return "outside"

        else:
            # other: ベースラインそのまま
            return baseline

    def _apply_bias_adjustment(
        self,
        bias: str,
        lane: LaneType,
        field_size: int,
    ) -> Tuple[float, float, Optional[AdjustmentReason]]:
        """
        トラックバイアスに基づく補正係数を計算する。

        「内伸び」のときは inside レーンを通る馬に有利、outside レーンに不利。
        「外伸び」のときはその逆。
        「フラット」または middle レーンの場合は補正なし。

        重要: バイアスは「レーン」だけを見る。
        脚質が得をするかどうかは「ペース補正」の責任であり、
        バイアス補正では脚質に直接のプラス/マイナスを与えない。

        頭数によるスケーリング:
        - 8頭立て: バイアス効果を 40% に抑制
        - 18頭立て: バイアス効果をフル (100%) 適用
        これにより「少頭数では外枠もそこまで絶望的ではない」を表現。

        Args:
            bias: バイアス ("内" / "外" / "フラット")
            lane: 推定レーン ("inside" / "middle" / "outside")
            field_size: 出走頭数

        Returns:
            (win係数, in3係数, 補正理由)
        """
        # 頭数に基づくスケール係数
        scale = _field_size_factor(field_size)

        if bias == "内":
            # 内伸びバイアス: inside レーンが有利
            if lane == "inside":
                # 係数を1.0基準でスケール: (original - 1.0) * scale + 1.0
                win_adj = (self.config.bias_inner_win_adv - 1.0) * scale + 1.0
                in3_adj = (self.config.bias_inner_in3_adv - 1.0) * scale + 1.0
                return (win_adj, in3_adj, AdjustmentReason.BIAS_ADVANTAGE)
            elif lane == "outside":
                win_adj = (self.config.bias_inner_win_disadv - 1.0) * scale + 1.0
                in3_adj = (self.config.bias_inner_in3_disadv - 1.0) * scale + 1.0
                return (win_adj, in3_adj, AdjustmentReason.BIAS_DISADVANTAGE)

        elif bias == "外":
            # 外伸びバイアス: outside レーンが有利
            if lane == "outside":
                win_adj = (self.config.bias_outer_win_adv - 1.0) * scale + 1.0
                in3_adj = (self.config.bias_outer_in3_adv - 1.0) * scale + 1.0
                return (win_adj, in3_adj, AdjustmentReason.BIAS_ADVANTAGE)
            elif lane == "inside":
                win_adj = (self.config.bias_outer_win_disadv - 1.0) * scale + 1.0
                in3_adj = (self.config.bias_outer_in3_disadv - 1.0) * scale + 1.0
                return (win_adj, in3_adj, AdjustmentReason.BIAS_DISADVANTAGE)

        # フラット or middle レーン → 補正なし
        return (1.0, 1.0, None)

    def _normalize_probabilities(self, horses: Dict[str, ScenarioHorseScore]) -> None:
        """
        確率の正規化。

        全馬の勝率合計が1になるように正規化する。
        3着内率は合計が約3になるように緩やかに正規化する。

        Note:
            別途 Calibrator（Platt/Isotonic等）を噛ませる場合は、
            adjust() で normalize=False を指定し、この処理をスキップすること。
            二重キャリブレーションを避けるため。
        """
        if not horses:
            return

        total_win = sum(h.adj_win for h in horses.values())
        total_in3 = sum(h.adj_in3 for h in horses.values())

        # 勝率の正規化（合計 = 1.0）
        if total_win > 0:
            for h in horses.values():
                h.adj_win = h.adj_win / total_win

        # 3着内率の正規化（合計が3を超えたら調整）
        if total_in3 > 3.0:
            scale = 3.0 / total_in3
            for h in horses.values():
                h.adj_in3 = h.adj_in3 * scale


# =============================================================================
# ファクトリ関数
# =============================================================================


def create_adjuster_with_custom_config(
    config: Optional[AdjustmentConfig] = None,
) -> ScenarioAdjuster:
    """
    カスタム係数で ScenarioAdjuster を生成するヘルパー。

    Args:
        config: 調整済みの AdjustmentConfig。
                None の場合は AdjustmentConfig.default() が使われる。

    Returns:
        ScenarioAdjuster インスタンス
    """
    if config is None:
        config = AdjustmentConfig.default()
    return ScenarioAdjuster(config=config)


# =============================================================================
# Sanity Check / デモ
# =============================================================================

if __name__ == "__main__":
    """
    簡単な動作確認用デモ。

    以下を確認:
    1. 8頭立て vs 18頭立てで外枠のバイアス補正が弱くなる
    2. 内枠の追込が「内バイアス有利」にならない（middle扱い止まり）
    3. スロー(S)で逃げ/先行が+、ハイ(H)で差し/追込が+になる
    """
    from .spec import RaceContext, ScenarioSpec

    print("=" * 60)
    print("Sanity Check: ScenarioAdjuster Refactoring")
    print("=" * 60)

    adjuster = ScenarioAdjuster()

    # --- Check 1: 頭数によるバイアススケーリング ---
    print("\n[Check 1] Field size scaling for bias")
    print("-" * 40)

    for fs in [8, 12, 18]:
        factor = _field_size_factor(fs)
        print(f"  {fs}頭立て: scale = {factor:.2f}")

    # 具体例: 外枠(frame_no=8) × 差し(mid_chaser) × 内バイアス
    print("\n  → 外枠・差しの馬が内バイアスで受けるペナルティ:")
    for fs in [8, 18]:
        lane = adjuster._estimate_lane(8, "mid_chaser", fs)
        win_coef, _, _ = adjuster._apply_bias_adjustment("内", lane, fs)
        print(f"     {fs}頭立て: lane={lane}, win_coef={win_coef:.4f}")

    # --- Check 2: 内枠の追込 → inside にならない ---
    print("\n[Check 2] Inner frame closer is NOT 'inside'")
    print("-" * 40)

    for category in ["front_runner", "pace_keeper", "mid_chaser", "deep_closer"]:
        lane = adjuster._estimate_lane(1, category, 18)  # 枠1, 18頭立て
        print(f"  枠1 × {category:12s} → lane = {lane}")

    # --- Check 3: ペース補正の方向性 ---
    print("\n[Check 3] Pace adjustments by category")
    print("-" * 40)

    for pace in ["S", "H"]:
        print(f"\n  Pace = {pace}:")
        table = adjuster.config.pace_category_win.get(pace, {})
        for cat in ["front_runner", "pace_keeper", "mid_chaser", "deep_closer"]:
            coef = table.get(cat, 1.0)
            direction = "有利" if coef > 1.0 else ("不利" if coef < 1.0 else "中立")
            print(f"    {cat:12s}: {coef:.2f} ({direction})")

    print("\n" + "=" * 60)
    print("Sanity check completed!")
    print("=" * 60)
