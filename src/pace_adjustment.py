"""
ペース補正モデル（Pace Delta）

目的: ペースによる勝率の変化をlog-odds空間で補正
学習: 目的変数 = logit(smoothed_actual) - logit(calibrated_pred_win + delta_baba)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.special import logit, expit
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class PaceAdjustmentModel:
    """
    ペース補正モデル
    
    アプローチ:
    1. 馬場補正後の予測とのlog-odds差を学習
    2. ペース偏差（連続値）を特徴量として使用
    3. データ量に応じたshrinkage
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        clip_sigma: float = 3.0,
        min_data_for_full_weight: int = 8,
        params: Optional[Dict] = None
    ):
        """
        Args:
            alpha: Laplace smoothing係数
            clip_sigma: 外れ値クリップ（σの何倍まで）
            min_data_for_full_weight: 完全信頼に必要なデータ数
            params: LightGBMパラメータ
        """
        
        self.alpha = alpha
        self.clip_sigma = clip_sigma
        self.min_data_for_full_weight = min_data_for_full_weight
        
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 15,
                'max_depth': 5,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1
            }
        
        self.params = params
        self.model = None
        self.feature_names = None
        self.target_mean = None
        self.target_std = None
    
    def prepare_training_data(
        self,
        after_baba_pred: np.ndarray,
        actual_win: np.ndarray,
        features: pd.DataFrame,
        horse_pace_race_counts: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        学習データを準備
        
        Args:
            after_baba_pred: 馬場補正後の予測
            actual_win: 実際の勝利（1/0）
            features: ペース関連の特徴量
            horse_pace_race_counts: 各馬のペース条件別レース数
        
        Returns:
            (特徴量, ターゲット, サンプルウェイト)
        """
        
        # Smoothing
        smoothed_actual = (actual_win + self.alpha) / (1 + 2 * self.alpha)
        
        # log-odds差を計算
        # target = logit(smoothed_actual) - logit(after_baba_pred)
        
        after_baba_pred_clipped = np.clip(after_baba_pred, 0.001, 0.999)
        smoothed_actual_clipped = np.clip(smoothed_actual, 0.001, 0.999)
        
        logit_pred = logit(after_baba_pred_clipped)
        logit_actual = logit(smoothed_actual_clipped)
        
        target = logit_actual - logit_pred
        
        # 外れ値をクリップ
        self.target_mean = np.mean(target)
        self.target_std = np.std(target)
        
        target_clipped = np.clip(
            target,
            self.target_mean - self.clip_sigma * self.target_std,
            self.target_mean + self.clip_sigma * self.target_std
        )
        
        # サンプルウェイト
        if horse_pace_race_counts is not None:
            sample_weight = np.minimum(
                1.0,
                horse_pace_race_counts / self.min_data_for_full_weight
            )
        else:
            sample_weight = np.ones(len(target_clipped))
        
        return features, target_clipped, sample_weight
    
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None
    ):
        """
        モデルを訓練
        
        Args:
            X: 特徴量
            y: ターゲット（log-odds差）
            sample_weight: サンプルウェイト
            X_val: 検証特徴量
            y_val: 検証ターゲット
            sample_weight_val: 検証サンプルウェイト
        """
        
        self.feature_names = X.columns.tolist()
        
        # データセット作成
        train_data = lgb.Dataset(
            X,
            label=y,
            weight=sample_weight,
            free_raw_data=False
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                weight=sample_weight_val,
                reference=train_data,
                free_raw_data=False
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # 訓練
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"\nペース補正モデル - 最適イテレーション: {self.model.best_iteration}")
    
    def predict(
        self,
        X: pd.DataFrame,
        apply_shrinkage: bool = True,
        horse_pace_race_counts: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        log-odds補正を予測
        
        Args:
            X: 特徴量
            apply_shrinkage: データ量に応じた縮小を適用するか
            horse_pace_race_counts: 各馬のペース条件別レース数
        
        Returns:
            log-odds補正（delta_pace）
        """
        
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        delta = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Shrinkage適用
        if apply_shrinkage and horse_pace_race_counts is not None:
            confidence = np.minimum(
                1.0,
                horse_pace_race_counts / self.min_data_for_full_weight
            )
            delta = delta * confidence
        
        return delta
    
    def apply_adjustment(
        self,
        after_baba_pred: np.ndarray,
        delta_pace: np.ndarray
    ) -> np.ndarray:
        """
        補正を適用して最終確率を計算
        
        final_prob = sigmoid(logit(after_baba_pred) + delta_pace)
        
        Args:
            after_baba_pred: 馬場補正後の予測
            delta_pace: ペース補正（log-odds）
        
        Returns:
            補正後の確率
        """
        
        # log-odds計算
        after_baba_pred_clipped = np.clip(after_baba_pred, 0.001, 0.999)
        logit_pred = logit(after_baba_pred_clipped)
        
        # 補正
        logit_final = logit_pred + delta_pace
        
        # 確率に戻す
        final_prob = expit(logit_final)
        
        return final_prob
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: int = 10
    ) -> pd.DataFrame:
        """特徴量の重要度を取得"""
        
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return df


class PaceAdjustmentFeatureExtractor:
    """
    ペース補正用の特徴量を抽出
    """
    
    @staticmethod
    def extract_features(
        race_data: pd.DataFrame,
        horse_history: pd.DataFrame,
        predicted_pace_deviation: np.ndarray
    ) -> pd.DataFrame:
        """
        ペース補正用の特徴量を抽出
        
        Args:
            race_data: レース情報
            horse_history: 各馬の過去成績
            predicted_pace_deviation: 予測ペース偏差（秒）
        
        Returns:
            特徴量DataFrame
        """
        
        features = []
        
        for idx, (_, race) in enumerate(race_data.iterrows()):
            horse_id = race['horse_id']
            horse_past = horse_history[
                horse_history['horse_id'] == horse_id
            ]
            
            # 馬の脚質判定
            if len(horse_past) > 0:
                avg_corner1 = horse_past['corner1_position'].mean()
                avg_corner4 = horse_past['corner4_position'].mean()
                
                # 脚質分類
                if avg_corner1 <= 3:
                    running_style = 0  # 逃げ・先行
                elif avg_corner4 <= avg_corner1 + 3:
                    running_style = 1  # 差し
                else:
                    running_style = 2  # 追込
                
                # ペース別成績
                high_pace_races = horse_past[
                    horse_past['pace_deviation'] < -0.5
                ]
                slow_pace_races = horse_past[
                    horse_past['pace_deviation'] > 0.5
                ]
                
                high_pace_win_rate = (
                    high_pace_races['finish_position'] == 1
                ).mean() if len(high_pace_races) > 0 else 0.1
                
                slow_pace_win_rate = (
                    slow_pace_races['finish_position'] == 1
                ).mean() if len(slow_pace_races) > 0 else 0.1
                
                high_pace_count = len(high_pace_races)
                slow_pace_count = len(slow_pace_races)
                
                # 早期スピード能力
                best_early_speed = horse_past['front_3f'].min() if 'front_3f' in horse_past.columns else 35
            else:
                running_style = 1
                avg_corner1 = 9
                avg_corner4 = 9
                high_pace_win_rate = 0.1
                slow_pace_win_rate = 0.1
                high_pace_count = 0
                slow_pace_count = 0
                best_early_speed = 35
            
            features.append({
                # 馬の特性
                'running_style': running_style,
                'avg_corner1_position': avg_corner1,
                'avg_corner4_position': avg_corner4,
                'best_early_speed': best_early_speed,
                
                # ペース別成績
                'horse_high_pace_win_rate': high_pace_win_rate,
                'horse_slow_pace_win_rate': slow_pace_win_rate,
                'horse_high_pace_count': high_pace_count,
                'horse_slow_pace_count': slow_pace_count,
                
                # 予測ペース
                'predicted_pace_deviation': predicted_pace_deviation[idx],
                'predicted_pace_deviation_abs': abs(predicted_pace_deviation[idx]),
                
                # レース情報
                'distance': race['distance'],
                'n_escape': race.get('n_escape', 2)
            })
        
        return pd.DataFrame(features)


def example_usage():
    """使用例"""
    
    # ダミーデータ
    np.random.seed(42)
    n = 1000
    
    # 馬場補正後の予測
    after_baba_pred = np.random.beta(2, 8, n)
    
    # 実際の結果
    actual_win = np.random.binomial(1, 0.1, n)
    
    # 特徴量（ペース関連）
    predicted_pace_deviation = np.random.normal(0, 1, n)
    
    features = pd.DataFrame({
        'running_style': np.random.choice([0, 1, 2], n),
        'avg_corner1_position': np.random.uniform(1, 15, n),
        'avg_corner4_position': np.random.uniform(1, 15, n),
        'best_early_speed': np.random.uniform(32, 38, n),
        'horse_high_pace_win_rate': np.random.beta(2, 8, n),
        'horse_slow_pace_win_rate': np.random.beta(2, 8, n),
        'horse_high_pace_count': np.random.randint(0, 20, n),
        'horse_slow_pace_count': np.random.randint(0, 20, n),
        'predicted_pace_deviation': predicted_pace_deviation,
        'predicted_pace_deviation_abs': np.abs(predicted_pace_deviation),
        'distance': np.random.choice([1600, 1800, 2000], n),
        'n_escape': np.random.randint(0, 5, n)
    })
    
    # ペースレース数（shrinkage用）
    horse_pace_counts = features['horse_high_pace_count'] + \
                       features['horse_slow_pace_count']
    
    # モデル訓練
    model = PaceAdjustmentModel(alpha=0.5)
    
    X, y, weights = model.prepare_training_data(
        after_baba_pred[:800],
        actual_win[:800],
        features.iloc[:800],
        horse_pace_counts[:800].values
    )
    
    model.train(X, y, sample_weight=weights)
    
    # 予測
    delta_pace = model.predict(
        features.iloc[800:],
        apply_shrinkage=True,
        horse_pace_race_counts=horse_pace_counts[800:].values
    )
    
    # 補正適用
    final_prob = model.apply_adjustment(
        after_baba_pred[800:],
        delta_pace
    )
    
    print("\n=== ペース補正の例 ===")
    print("馬場補正後:", after_baba_pred[800:805])
    print("delta_pace:", delta_pace[:5])
    print("最終確率:", final_prob[:5])
    
    # 特徴量重要度
    print("\n=== 特徴量重要度 ===")
    print(model.get_feature_importance())


if __name__ == "__main__":
    example_usage()
