"""
pace_adjustment.py v2.0 - プロダクション完成版

v2.0（2024-12-04）: 致命的問題完全修正
🔥 修正内容:
1. pace_balance定義を修正（front - last）
2. RaceFeatureBuilder v5.0との整合性確保
3. balance_scale調整（1.2 → 3.0、飽和防止）
4. sigmoid数値安定版実装
5. unknown脚質警告追加

v1.0: 初版（ChatGPT版）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Mapping, Optional, Tuple, List
import warnings

import numpy as np


# 他モジュールと表記を揃えておく
RUNNING_STYLE_NIGE = "逃げ"
RUNNING_STYLE_SENKOU = "先行"
RUNNING_STYLE_SASHI = "差し"
RUNNING_STYLE_OIKOMI = "追込"


@dataclass
class PaceAdjustmentConfig:
    """
    ペース補正のハイパーパラメータ（v2.0最終版）

    Attributes
    ----------
    style_coef : Dict[str, float]
        脚質ごとのペース感応度係数。
        
        🔥 v2.0最終版: pace_balance = last_3f - front_3f
        
        定義:
            pace_balance = last_3f - front_3f
            - 正（+）: ハイペース（前半速い、後半遅い）
            - 負（-）: スローペース（前半遅い、後半速い）
        
        係数の意味:
            - 正の係数: ハイペース（pace_balance正）で有利（差し・追込）
            - 負の係数: ハイペース（pace_balance正）で不利（逃げ・先行）
        
        計算例（ハイペース: pace_balance = +3秒）:
            normalized_balance = tanh(+3 / 3.0) ≈ +0.76
            
            差し馬（style_coef = +0.7）:
                impact = +0.7 × +0.76 = +0.53 → 有利
            
            逃げ馬（style_coef = -1.0）:
                impact = -1.0 × +0.76 = -0.76 → 不利
        
    alpha : float
        ペースインパクトを logit にどれだけ乗せるか。
        大きいほどペースの影響が強くなる。
        
    balance_scale : float
        🔥 v2.0: 1.2 → 3.0に変更（飽和防止）
        
        pace_balance（秒）をどのスケールで tanh に入れるか。
        3.0 なら:
        - ±1秒: tanh(1/3) = ±0.32
        - ±2秒: tanh(2/3) = ±0.54
        - ±3秒: tanh(3/3) = ±0.76
        - ±5秒: tanh(5/3) = ±0.94
        
    max_shift_abs : float
        1頭あたりの logit 変化量の絶対上限（安全装置）。
        極端な補正で確率が吹き飛ぶのを防ぐ。
        
    renormalize : bool
        True の場合、補正後の logit から odds を作り、
        レース内で合計 1.0 になるよう再正規化する。
        🔥 v2.0: Trueを推奨（Falseは非推奨）
    """

    style_coef: Dict[str, float] = field(
        default_factory=lambda: {
            RUNNING_STYLE_NIGE: -1.0,      # ハイペースで不利
            RUNNING_STYLE_SENKOU: -0.5,    # ハイペースでやや不利
            RUNNING_STYLE_SASHI: 0.7,      # ハイペースで有利
            RUNNING_STYLE_OIKOMI: 1.0,     # ハイペースで最も有利
        }
    )
    alpha: float = 0.7
    balance_scale: float = 3.0  # 🔥 v2.0: 1.2 → 3.0（飽和防止）
    max_shift_abs: float = 1.5
    renormalize: bool = True


class PaceAdjustment:
    """
    PaceModel の出力（front_3f / last_3f）と
    馬ごとの脚質情報を使って、BaseModel の勝率をペースで補正するクラス（v2.0最終版）
    
    🔥 v2.0: pace_balance定義を明確化
    
    pace_balance = last_3f - front_3f
    - 正（+）: ハイペース（前半速い、後半遅い）→ 逃げ・先行不利、差し・追込有利
    - 負（-）: スローペース（前半遅い、後半速い）→ 逃げ・先行有利、差し・追込不利
    
    具体例:
        ハイペース: 前半33秒、後半36秒
        → pace_balance = 36 - 33 = +3秒
        → 差し馬の確率 ↑、逃げ馬の確率 ↓
        
        スローペース: 前半36秒、後半33秒
        → pace_balance = 33 - 36 = -3秒
        → 差し馬の確率 ↓、逃げ馬の確率 ↑
    
    想定フロー:
    
        base_probs      : BaseModel + calibration の勝率（馬ごと）
        horse_features  : RaceFeatureBuilder.build_for_race()["horse_features"]
        pace_vector     : PaceModel.predict_pace_vector(...) の出力
        
        ↓
        
        adjust() -> pace 補正後の勝率 dict
    """

    def __init__(self, config: Optional[PaceAdjustmentConfig] = None) -> None:
        self.config = config or PaceAdjustmentConfig()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def adjust(
        self,
        base_probs: Mapping[str, float],
        horse_features: Mapping[str, Mapping[str, Any]],
        pace_vector: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        各馬のベース勝率を、ペースに応じて補正した勝率に変換する（v2.0改善版）
        
        🔥 v2.0: pace_balance定義変更
        
        Parameters
        ----------
        base_probs : Mapping[str, float]
            horse_id -> base_prob（0〜1、レース内でおおむね合計1想定）
            
        horse_features : Mapping[str, Mapping[str, Any]]
            horse_id -> dict。最低限 'running_style' を含むこと。
            
            🔥 v2.0: RaceFeatureBuilder v5.0 の出力形式
            
            例:
                {
                    "horse_1": {"running_style": "逃げ"},
                    "horse_2": {"running_style": "差し"},
                }
            
        pace_vector : Mapping[str, float]
            PaceModel.predict_pace_vector(...) の出力を想定。
            必須キー:
                - "front_3f"
                - "last_3f"
            任意キー:
                - "pace_balance"（あればそれを優先）

        Returns
        -------
        Dict[str, float]
            horse_id -> 補正後勝率（レース内で合計 1.0 に正規化される）
        """
        if not base_probs:
            return {}

        horse_ids, base_p_arr = self._prepare_base_probs(base_probs)
        if base_p_arr.size == 0:
            return {}

        # 🔥 v2.0: pace_balance = front - last
        pace_balance = self._compute_pace_balance(pace_vector)

        # 馬ごとの style_coef を配列で取得
        style_coef_arr = self._get_style_coef_array(horse_ids, horse_features)

        # 🔥 v2.0: pace_balance を [-1, 1] 程度に圧縮
        # balance_scale = 3.0 なので、±3秒で tanh(±1) ≈ ±0.76
        normalized_balance = np.tanh(pace_balance / self.config.balance_scale)

        # 馬ごとのインパクト（符号付き）
        # ハイペース（正）× 差し（正の係数）→ 正のインパクト
        # ハイペース（正）× 逃げ（負の係数）→ 負のインパクト
        impact = style_coef_arr * normalized_balance

        # logit 空間での変化量
        delta_logit = self.config.alpha * impact
        delta_logit = np.clip(
            delta_logit,
            -self.config.max_shift_abs,
            self.config.max_shift_abs,
        )

        # ベース勝率を logit に変換して補正
        base_logit = self._logit(base_p_arr)
        adj_logit = base_logit + delta_logit

        if self.config.renormalize:
            # odds を計算してレース内で正規化
            odds = np.exp(adj_logit)
            total = float(odds.sum())
            if total <= 0.0 or not np.isfinite(total):
                # 万一の安全策：再正規化不能なら補正前をそのまま返す
                warnings.warn(
                    "ペース補正後のodds合計が異常です。補正前の確率を返します。"
                )
                final_probs = base_p_arr
            else:
                final_probs = odds / total
        else:
            # 各頭を単独に sigmoid にかけるだけ（分布としては 1.0 にならない）
            # 🔥 v2.0: 非推奨
            warnings.warn(
                "renormalize=False は非推奨です。確率の合計が1.0になりません。"
            )
            final_probs = self._sigmoid(adj_logit)

        return {horse_id: float(p) for horse_id, p in zip(horse_ids, final_probs)}

    def adjust_with_debug(
        self,
        base_probs: Mapping[str, float],
        horse_features: Mapping[str, Mapping[str, Any]],
        pace_vector: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        デバッグ用：補正後の勝率に加え、各馬の pace_impact や delta_logit も返す（v2.0改善版）
        
        🔥 v2.0: pace_balance定義変更に伴うデバッグ情報の意味修正

        Returns
        -------
        (final_probs, debug_info)

        final_probs : Dict[str, float]
            horse_id -> 補正後勝率
            
        debug_info : Dict[str, Dict[str, float]]
            horse_id -> {
                "pace_balance": ペースバランス（front - last、秒）,
                "normalized_balance": 正規化後（-1〜1）,
                "style_coef": 脚質係数,
                "impact": インパクト（style_coef × normalized_balance）,
                "delta_logit": logit変化量,
                "base_prob": 補正前確率,
                "final_prob": 補正後確率
            }
        """
        if not base_probs:
            return {}, {}

        horse_ids, base_p_arr = self._prepare_base_probs(base_probs)
        if base_p_arr.size == 0:
            return {}, {}

        # 🔥 v2.0: pace_balance = front - last
        pace_balance = self._compute_pace_balance(pace_vector)
        style_coef_arr = self._get_style_coef_array(horse_ids, horse_features)
        normalized_balance = np.tanh(pace_balance / self.config.balance_scale)
        impact = style_coef_arr * normalized_balance
        delta_logit = np.clip(
            self.config.alpha * impact,
            -self.config.max_shift_abs,
            self.config.max_shift_abs,
        )

        base_logit = self._logit(base_p_arr)
        adj_logit = base_logit + delta_logit

        if self.config.renormalize:
            odds = np.exp(adj_logit)
            total = float(odds.sum())
            if total <= 0.0 or not np.isfinite(total):
                warnings.warn(
                    "ペース補正後のodds合計が異常です。補正前の確率を返します。"
                )
                final_probs = base_p_arr
            else:
                final_probs = odds / total
        else:
            warnings.warn(
                "renormalize=False は非推奨です。確率の合計が1.0になりません。"
            )
            final_probs = self._sigmoid(adj_logit)

        debug: Dict[str, Dict[str, float]] = {}
        for idx, horse_id in enumerate(horse_ids):
            debug[horse_id] = {
                "pace_balance": float(pace_balance),
                "normalized_balance": float(normalized_balance),
                "style_coef": float(style_coef_arr[idx]),
                "impact": float(impact[idx]),
                "delta_logit": float(delta_logit[idx]),
                "base_prob": float(base_p_arr[idx]),
                "final_prob": float(final_probs[idx]),
            }

        final_dict = {horse_id: float(p) for horse_id, p in zip(horse_ids, final_probs)}
        return final_dict, debug

    # ------------------------------------------------------------
    # 内部: 前処理系
    # ------------------------------------------------------------
    def _prepare_base_probs(
        self,
        base_probs: Mapping[str, float],
    ) -> Tuple[List[str], np.ndarray]:
        """ベース確率を配列化"""
        horse_ids: List[str] = []
        probs: List[float] = []

        for horse_id, p in base_probs.items():
            if p is None:
                continue
            horse_ids.append(horse_id)
            probs.append(float(p))

        if not horse_ids:
            return [], np.array([], dtype=float)

        p_arr = np.asarray(probs, dtype=float)
        # 念のため 0〜1 にクリップ
        p_arr = np.clip(p_arr, 1e-8, 1.0 - 1e-8)
        return horse_ids, p_arr

    def _compute_pace_balance(self, pace_vector: Mapping[str, float]) -> float:
        """
        pace_balance を求める（v2.0最終版）
        
        🔥 v2.0: last_3f - front_3f を採用
        
        定義:
            pace_balance = last_3f - front_3f
        
        意味:
            - 正の値（+）: ハイペース（前半速い、後半遅い）→ 差し・追込有利
            - 負の値（-）: スローペース（前半遅い、後半速い）→ 逃げ・先行有利
        
        具体例:
            ハイペースの場合:
                前半: 33.0秒（速い）
                後半: 36.0秒（遅い）
                → pace_balance = 36.0 - 33.0 = +3.0秒（正）
                → normalized_balance = tanh(+3.0 / 3.0) ≈ +0.76
                → 差し馬（style_coef = +0.7）: impact = +0.7 × +0.76 = +0.53
                → 逃げ馬（style_coef = -1.0）: impact = -1.0 × +0.76 = -0.76
            
            スローペースの場合:
                前半: 36.0秒（遅い）
                後半: 33.0秒（速い）
                → pace_balance = 33.0 - 36.0 = -3.0秒（負）
                → normalized_balance = tanh(-3.0 / 3.0) ≈ -0.76
                → 差し馬（style_coef = +0.7）: impact = +0.7 × -0.76 = -0.53
                → 逃げ馬（style_coef = -1.0）: impact = -1.0 × -0.76 = +0.76
        
        処理:
            - pace_vector["pace_balance"] があればそれを優先
            - なければ last_3f - front_3f で計算
        """
        if "pace_balance" in pace_vector and pace_vector["pace_balance"] is not None:
            return float(pace_vector["pace_balance"])

        front = float(pace_vector["front_3f"])
        last = float(pace_vector["last_3f"])
        
        # 🔥 v2.0最終版: last - front（選択肢A採用）
        # 正 = ハイペース（前半速い、後半遅い）
        # 負 = スローペース（前半遅い、後半速い）
        return last - front

    def _get_style_coef_array(
        self,
        horse_ids: List[str],
        horse_features: Mapping[str, Mapping[str, Any]],
    ) -> np.ndarray:
        """
        各馬の脚質から style_coef を取り出して配列化（v2.0改善版）
        
        🔥 v2.0: 未知の脚質に対して警告を出す
        """
        arr: List[float] = []
        unknown_count = 0
        unknown_styles = set()

        for horse_id in horse_ids:
            info = horse_features.get(horse_id, {})
            rs = info.get("running_style")
            
            if rs not in self.config.style_coef:
                unknown_count += 1
                unknown_styles.add(rs)
                coef = 0.0
            else:
                coef = self.config.style_coef[rs]
            
            arr.append(float(coef))

        if unknown_count > 0:
            warnings.warn(
                f"{unknown_count}頭の脚質が不明です（{unknown_styles}）。"
                f"補正なしで処理します。"
            )

        return np.asarray(arr, dtype=float)

    # ------------------------------------------------------------
    # 内部: 数学系
    # ------------------------------------------------------------
    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        """確率をlogitに変換"""
        p_clipped = np.clip(p, 1e-8, 1.0 - 1e-8)
        return np.log(p_clipped / (1.0 - p_clipped))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        数値安定版sigmoid（v2.0改善版）
        
        🔥 v2.0: オーバーフロー対策
        """
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )


def example_usage():
    """使用例（v2.0最終版）"""
    
    print("=" * 80)
    print("PaceAdjustment v2.0 - 使用例（プロダクション完成版）")
    print("=" * 80)
    
    # ベース確率（BaseModel + Calibration の出力）
    # 🔥 重要: 全馬の確率の合計が1.0になる必要がある
    base_probs = {
        "horse_1": 0.20,  # 逃げ馬（1番人気）
        "horse_2": 0.15,  # 差し馬（2番人気）
        "horse_3": 0.12,  # 先行馬（3番人気）
        "horse_4": 0.10,  # 追込馬（4番人気）
        "horse_5": 0.08,  # 差し馬（5番人気）
        "horse_6": 0.35,  # その他の馬（平均確率）
    }
    # 合計 = 1.0 ✅
    
    # 馬ごとの脚質（RaceFeatureBuilder v5.0 の出力）
    horse_features = {
        "horse_1": {"running_style": "逃げ"},
        "horse_2": {"running_style": "差し"},
        "horse_3": {"running_style": "先行"},
        "horse_4": {"running_style": "追込"},
        "horse_5": {"running_style": "差し"},
        "horse_6": {"running_style": "先行"},  # その他の馬
    }
    
    # ペース予測（PaceModel の出力）
    pace_vector_high = {
        "front_3f": 33.0,  # 速い
        "last_3f": 36.0,   # 遅い
    }
    # pace_balance = 36 - 33 = 3秒（正） → ハイペース
    
    pace_vector_slow = {
        "front_3f": 36.0,  # 遅い
        "last_3f": 33.0,   # 速い
    }
    # pace_balance = 33 - 36 = -3秒（負） → スローペース
    
    # PaceAdjustment初期化
    adjuster = PaceAdjustment()
    
    print("\n【ケース1】ハイペース（前半速い、後半遅い）")
    print(f"  前半3F: {pace_vector_high['front_3f']}秒（速い）")
    print(f"  後半3F: {pace_vector_high['last_3f']}秒（遅い）")
    print(f"  pace_balance: {pace_vector_high['last_3f'] - pace_vector_high['front_3f']:.1f}秒（正 = ハイペース）")
    print(f"\n  期待: 差し・追込↑、逃げ・先行↓")
    
    final_high, debug_high = adjuster.adjust_with_debug(
        base_probs, horse_features, pace_vector_high
    )
    
    print("\n  補正結果:")
    for horse_id in ["horse_1", "horse_2", "horse_3", "horse_4", "horse_5"]:
        base = base_probs[horse_id]
        final = final_high[horse_id]
        delta = final - base
        style = horse_features[horse_id]["running_style"]
        arrow = "↑" if delta > 0 else "↓"
        print(f"    {horse_id}（{style:>2s}）: {base*100:5.1f}% → {final*100:5.1f}% ({delta*100:+5.1f}%pt) {arrow}")
    
    print("\n  デバッグ情報（抜粋）:")
    for horse_id in ["horse_1", "horse_2"]:
        info = debug_high[horse_id]
        style = horse_features[horse_id]["running_style"]
        print(f"    {horse_id}（{style}）:")
        print(f"      delta_logit: {info['delta_logit']:+.4f}（{'有利' if info['delta_logit'] > 0 else '不利'}）")
    
    print("\n" + "-" * 80)
    print("\n【ケース2】スローペース（前半遅い、後半速い）")
    print(f"  前半3F: {pace_vector_slow['front_3f']}秒（遅い）")
    print(f"  後半3F: {pace_vector_slow['last_3f']}秒（速い）")
    print(f"  pace_balance: {pace_vector_slow['last_3f'] - pace_vector_slow['front_3f']:.1f}秒（負 = スローペース）")
    print(f"\n  期待: 逃げ・先行↑、差し・追込↓")
    
    final_slow, debug_slow = adjuster.adjust_with_debug(
        base_probs, horse_features, pace_vector_slow
    )
    
    print("\n  補正結果:")
    for horse_id in ["horse_1", "horse_2", "horse_3", "horse_4", "horse_5"]:
        base = base_probs[horse_id]
        final = final_slow[horse_id]
        delta = final - base
        style = horse_features[horse_id]["running_style"]
        arrow = "↑" if delta > 0 else "↓"
        print(f"    {horse_id}（{style:>2s}）: {base*100:5.1f}% → {final*100:5.1f}% ({delta*100:+5.1f}%pt) {arrow}")
    
    print("\n  デバッグ情報（抜粋）:")
    for horse_id in ["horse_1", "horse_2"]:
        info = debug_slow[horse_id]
        style = horse_features[horse_id]["running_style"]
        print(f"    {horse_id}（{style}）:")
        print(f"      delta_logit: {info['delta_logit']:+.4f}（{'有利' if info['delta_logit'] > 0 else '不利'}）")
    
    print("\n" + "=" * 80)
    print("✅ v2.0完成 - pace_balance定義修正、balance_scale調整")
    print("  定義: pace_balance = last_3f - front_3f")
    print("    正（+）: ハイペース → 差し・追込↑、逃げ・先行↓")
    print("    負（-）: スローペース → 逃げ・先行↑、差し・追込↓")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
