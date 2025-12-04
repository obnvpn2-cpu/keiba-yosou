"""
確率統合モジュール

目的: ベース予測 + 馬場補正 + ペース補正をlog-odds空間で統合
正規化: レース内でSoftmax正規化（合計=1.0）
"""

import pandas as pd
import numpy as np
from scipy.special import logit, expit
from typing import Dict, List, Tuple, Optional


class ProbabilityIntegrator:
    """
    確率統合クラス
    
    統合方式:
    final_logit = logit(calibrated_base) + delta_baba + delta_pace
    final_prob = sigmoid(final_logit)
    
    レース内正規化:
    normalized_prob = softmax(final_logit)
    """
    
    def __init__(self):
        pass
    
    def integrate(
        self,
        calibrated_base: np.ndarray,
        delta_baba: np.ndarray,
        delta_pace: np.ndarray,
        race_ids: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        確率を統合
        
        Args:
            calibrated_base: 校正済みベース予測
            delta_baba: 馬場補正（log-odds）
            delta_pace: ペース補正（log-odds）
            race_ids: レースID（正規化用）
        
        Returns:
            (最終確率, 詳細情報)
        """
        
        # Step 1: log-odds計算
        calibrated_base_clipped = np.clip(calibrated_base, 0.001, 0.999)
        logit_base = logit(calibrated_base_clipped)
        
        # Step 2: log-odds統合
        logit_final = logit_base + delta_baba + delta_pace
        
        # Step 3: sigmoid変換
        prob_before_normalize = expit(logit_final)
        
        # Step 4: レース内でSoftmax正規化
        normalized_prob = self._softmax_by_race(logit_final, race_ids)
        
        # 詳細情報
        details = {
            'logit_base': logit_base,
            'delta_baba': delta_baba,
            'delta_pace': delta_pace,
            'logit_final': logit_final,
            'prob_before_normalize': prob_before_normalize,
            'normalized_prob': normalized_prob
        }
        
        return normalized_prob, details
    
    def _softmax_by_race(
        self,
        logits: np.ndarray,
        race_ids: np.ndarray
    ) -> np.ndarray:
        """
        レース内でSoftmax正規化
        
        Args:
            logits: log-odds
            race_ids: レースID
        
        Returns:
            正規化後の確率
        """
        
        normalized_probs = np.zeros_like(logits)
        
        for race_id in np.unique(race_ids):
            race_mask = race_ids == race_id
            race_logits = logits[race_mask]
            
            # Softmax（数値安定性のため最大値を引く）
            max_logit = np.max(race_logits)
            exp_logits = np.exp(race_logits - max_logit)
            race_probs = exp_logits / np.sum(exp_logits)
            
            normalized_probs[race_mask] = race_probs
        
        return normalized_probs
    
    def validate_probabilities(
        self,
        probs: np.ndarray,
        race_ids: np.ndarray
    ) -> Dict[str, bool]:
        """
        確率の妥当性を検証
        
        Returns:
            検証結果の辞書
        """
        
        results = {
            'all_positive': np.all(probs >= 0),
            'all_below_one': np.all(probs <= 1),
            'race_sums_to_one': True
        }
        
        # レース内の確率の合計が1.0かチェック
        for race_id in np.unique(race_ids):
            race_mask = race_ids == race_id
            race_sum = np.sum(probs[race_mask])
            
            if not np.isclose(race_sum, 1.0, atol=1e-5):
                results['race_sums_to_one'] = False
                print(f"警告: レース {race_id} の確率合計が1.0ではありません: {race_sum:.6f}")
        
        return results


class PredictionPipeline:
    """
    予測パイプライン全体を管理
    
    フロー:
    1. ベースモデル予測
    2. 確率キャリブレーション
    3. 馬場補正
    4. ペース補正
    5. 確率統合
    """
    
    def __init__(
        self,
        base_model,
        calibrator,
        baba_model,
        pace_model,
        integrator
    ):
        """
        Args:
            base_model: ベースモデル
            calibrator: キャリブレーター
            baba_model: 馬場補正モデル
            pace_model: ペース補正モデル
            integrator: 確率統合器
        """
        
        self.base_model = base_model
        self.calibrator = calibrator
        self.baba_model = baba_model
        self.pace_model = pace_model
        self.integrator = integrator
    
    def predict(
        self,
        X_base: pd.DataFrame,
        X_baba: pd.DataFrame,
        X_pace: pd.DataFrame,
        race_ids: np.ndarray,
        horse_baba_counts: Optional[np.ndarray] = None,
        horse_pace_counts: Optional[np.ndarray] = None
    ) -> Dict:
        """
        エンドツーエンド予測
        
        Args:
            X_base: ベースモデル用特徴量
            X_baba: 馬場補正用特徴量
            X_pace: ペース補正用特徴量
            race_ids: レースID
            horse_baba_counts: 馬場データ量
            horse_pace_counts: ペースデータ量
        
        Returns:
            予測結果の辞書
        """
        
        # Step 1: ベース予測
        base_pred = self.base_model.predict(X_base)
        
        # Step 2: キャリブレーション
        calibrated_pred = self.calibrator.transform(base_pred)
        
        # Step 3: 馬場補正
        delta_baba = self.baba_model.predict(
            X_baba,
            apply_shrinkage=True,
            horse_baba_race_counts=horse_baba_counts
        )
        
        after_baba = self.baba_model.apply_adjustment(
            calibrated_pred,
            delta_baba
        )
        
        # Step 4: ペース補正
        delta_pace = self.pace_model.predict(
            X_pace,
            apply_shrinkage=True,
            horse_pace_race_counts=horse_pace_counts
        )
        
        # Step 5: 確率統合
        final_prob, details = self.integrator.integrate(
            calibrated_pred,
            delta_baba,
            delta_pace,
            race_ids
        )
        
        # 検証
        validation = self.integrator.validate_probabilities(
            final_prob,
            race_ids
        )
        
        return {
            'base_pred': base_pred,
            'calibrated_pred': calibrated_pred,
            'after_baba': after_baba,
            'delta_baba': delta_baba,
            'delta_pace': delta_pace,
            'final_prob': final_prob,
            'details': details,
            'validation': validation
        }


def example_usage():
    """使用例"""
    
    # ダミーデータ
    np.random.seed(42)
    
    # 3レース、各18頭
    n_races = 3
    n_horses_per_race = 18
    n_total = n_races * n_horses_per_race
    
    # レースID
    race_ids = np.repeat([f'race_{i}' for i in range(n_races)], n_horses_per_race)
    
    # ダミー予測
    calibrated_base = np.random.beta(2, 15, n_total)
    delta_baba = np.random.normal(0, 0.3, n_total)
    delta_pace = np.random.normal(0, 0.3, n_total)
    
    # 統合
    integrator = ProbabilityIntegrator()
    final_prob, details = integrator.integrate(
        calibrated_base,
        delta_baba,
        delta_pace,
        race_ids
    )
    
    # 検証
    validation = integrator.validate_probabilities(final_prob, race_ids)
    
    print("=== 確率統合の例 ===")
    for i in range(3):
        race_id = f'race_{i}'
        race_mask = race_ids == race_id
        race_probs = final_prob[race_mask]
        
        print(f"\n{race_id}:")
        print(f"  確率合計: {np.sum(race_probs):.6f}")
        print(f"  最大確率: {np.max(race_probs):.4f}")
        print(f"  最小確率: {np.min(race_probs):.4f}")
        print(f"  上位3頭: {np.sort(race_probs)[-3:][::-1]}")
    
    print("\n=== 検証結果 ===")
    for key, value in validation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()
