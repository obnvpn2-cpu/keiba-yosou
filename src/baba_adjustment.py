"""
馬場補正モデル（Baba補正）

目的: 馬場状態による勝率の変化をlog-odds空間で補正
学習: 目的変数 = logit(smoothed_actual) - logit(calibrated_pred_win)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.special import logit, expit
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class BabaAdjustmentModel:
    """
    馬場補正モデル
    
    アプローチ:
    1. ベース予測とのlog-odds差を学習
    2. 外れ値をクリップ
    3. データ量に応じたshrinkage
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        clip_sigma: float = 3.0,
        min_data_for_full_weight: int = 10,
        params: Optional[Dict] = None
    ):
        """
        Args:
            alpha: Laplace smoothing係数（0.5~1.0）
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
        calibrated_pred: np.ndarray,
        actual_win: np.ndarray,
        features: pd.DataFrame,
        horse_baba_race_counts: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        学習データを準備
        
        Args:
            calibrated_pred: 校正済みベース予測
            actual_win: 実際の勝利（1/0）
            features: 馬場関連の特徴量
            horse_baba_race_counts: 各馬の馬場条件別レース数
        
        Returns:
            (特徴量, ターゲット, サンプルウェイト)
        """
        
        # Smoothing
        # smoothed_actual = (win + α) / (1 + 2α)
        smoothed_actual = (actual_win + self.alpha) / (1 + 2 * self.alpha)
        
        # log-odds差を計算
        # target = logit(smoothed_actual) - logit(calibrated_pred)
        
        # logit計算（0と1を避けるためにclip）
        calibrated_pred_clipped = np.clip(calibrated_pred, 0.001, 0.999)
        smoothed_actual_clipped = np.clip(smoothed_actual, 0.001, 0.999)
        
        logit_pred = logit(calibrated_pred_clipped)
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
        
        # サンプルウェイト（データ量に応じた信頼度）
        if horse_baba_race_counts is not None:
            # confidence = min(1.0, count / min_data_for_full_weight)
            sample_weight = np.minimum(
                1.0,
                horse_baba_race_counts / self.min_data_for_full_weight
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
        
        print(f"\n馬場補正モデル - 最適イテレーション: {self.model.best_iteration}")
    
    def predict(
        self,
        X: pd.DataFrame,
        apply_shrinkage: bool = True,
        horse_baba_race_counts: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        log-odds補正を予測
        
        Args:
            X: 特徴量
            apply_shrinkage: データ量に応じた縮小を適用するか
            horse_baba_race_counts: 各馬の馬場条件別レース数
        
        Returns:
            log-odds補正（delta_baba）
        """
        
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        delta = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Shrinkage適用
        if apply_shrinkage and horse_baba_race_counts is not None:
            confidence = np.minimum(
                1.0,
                horse_baba_race_counts / self.min_data_for_full_weight
            )
            delta = delta * confidence
        
        return delta
    
    def apply_adjustment(
        self,
        calibrated_pred: np.ndarray,
        delta_baba: np.ndarray
    ) -> np.ndarray:
        """
        補正を適用して最終確率を計算
        
        final_prob = sigmoid(logit(calibrated_pred) + delta_baba)
        
        Args:
            calibrated_pred: 校正済みベース予測
            delta_baba: 馬場補正（log-odds）
        
        Returns:
            補正後の確率
        """
        
        # log-odds計算
        calibrated_pred_clipped = np.clip(calibrated_pred, 0.001, 0.999)
        logit_pred = logit(calibrated_pred_clipped)
        
        # 補正
        logit_final = logit_pred + delta_baba
        
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


class BabaFeatureExtractor:
    """
    馬場補正用の特徴量を抽出
    """
    
    def __init__(self, track_statistics: Dict):
        """
        Args:
            track_statistics: 競馬場別の統計
            例: {'東京': {'avg_cushion': 9.1, 'std_cushion': 0.5, ...}}
        """
        self.track_statistics = track_statistics
    
    def extract_features(
        self,
        race_data: pd.DataFrame,
        horse_history: pd.DataFrame
    ) -> pd.DataFrame:
        """
        馬場補正用の特徴量を抽出
        
        Args:
            race_data: レース情報
            horse_history: 各馬の過去成績
        
        Returns:
            特徴量DataFrame
        """
        
        features = []
        
        for _, race in race_data.iterrows():
            track = race['track_name']
            
            # 競馬場別の標準化
            if track in self.track_statistics:
                stats = self.track_statistics[track]
                
                normalized_cushion = (
                    race['cushion_value'] - stats['avg_cushion']
                ) / stats['std_cushion']
                
                normalized_moisture = (
                    race['moisture'] - stats['avg_moisture']
                ) / stats['std_moisture']
            else:
                normalized_cushion = 0
                normalized_moisture = 0
            
            # 馬の過去成績（馬場指数別）
            horse_id = race['horse_id']
            horse_past = horse_history[
                horse_history['horse_id'] == horse_id
            ]
            
            if len(horse_past) > 0:
                # 高速馬場での勝率（baba_index < -1.5）
                high_speed_races = horse_past[horse_past['baba_index'] < -1.5]
                high_speed_win_rate = (
                    high_speed_races['finish_position'] == 1
                ).mean() if len(high_speed_races) > 0 else 0.1
                
                # 時計かかる馬場での勝率（baba_index > 1.5）
                slow_races = horse_past[horse_past['baba_index'] > 1.5]
                slow_win_rate = (
                    slow_races['finish_position'] == 1
                ).mean() if len(slow_races) > 0 else 0.1
                
                # データ量
                high_speed_count = len(high_speed_races)
                slow_count = len(slow_races)
            else:
                high_speed_win_rate = 0.1
                slow_win_rate = 0.1
                high_speed_count = 0
                slow_count = 0
            
            features.append({
                'predicted_baba_index': race['predicted_baba_index'],
                'normalized_cushion': normalized_cushion,
                'normalized_moisture': normalized_moisture,
                'horse_high_speed_win_rate': high_speed_win_rate,
                'horse_slow_win_rate': slow_win_rate,
                'horse_high_speed_count': high_speed_count,
                'horse_slow_count': slow_count,
                'track_correlation': self.track_statistics.get(track, {}).get(
                    'cushion_correlation', 0
                )
            })
        
        return pd.DataFrame(features)


def example_usage():
    """使用例"""
    
    # ダミーデータ
    np.random.seed(42)
    n = 1000
    
    # 校正済み予測
    calibrated_pred = np.random.beta(2, 8, n)
    
    # 実際の結果
    actual_win = np.random.binomial(1, 0.1, n)
    
    # 特徴量（馬場関連）
    features = pd.DataFrame({
        'predicted_baba_index': np.random.normal(0, 1.5, n),
        'normalized_cushion': np.random.normal(0, 1, n),
        'normalized_moisture': np.random.normal(0, 1, n),
        'horse_high_speed_win_rate': np.random.beta(2, 8, n),
        'horse_slow_win_rate': np.random.beta(2, 8, n),
        'horse_high_speed_count': np.random.randint(0, 20, n),
        'horse_slow_count': np.random.randint(0, 20, n)
    })
    
    # 馬場レース数（shrinkage用）
    horse_baba_counts = features['horse_high_speed_count'] + \
                       features['horse_slow_count']
    
    # モデル訓練
    model = BabaAdjustmentModel(alpha=0.5)
    
    X, y, weights = model.prepare_training_data(
        calibrated_pred[:800],
        actual_win[:800],
        features.iloc[:800],
        horse_baba_counts[:800].values
    )
    
    model.train(X, y, sample_weight=weights)
    
    # 予測
    delta_baba = model.predict(
        features.iloc[800:],
        apply_shrinkage=True,
        horse_baba_race_counts=horse_baba_counts[800:].values
    )
    
    # 補正適用
    final_prob = model.apply_adjustment(
        calibrated_pred[800:],
        delta_baba
    )
    
    print("\n=== 馬場補正の例 ===")
    print("元の予測確率:", calibrated_pred[800:805])
    print("delta_baba:", delta_baba[:5])
    print("補正後確率:", final_prob[:5])
    
    # 特徴量重要度
    print("\n=== 特徴量重要度 ===")
    print(model.get_feature_importance())


if __name__ == "__main__":
    example_usage()
