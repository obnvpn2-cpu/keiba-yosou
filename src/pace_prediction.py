"""
ペース予測モデル

目的: 前半3F、後半3Fを連続値で予測
特徴量: 距離、逃げ馬数、馬場状態、スピード指数など
出力: 予測ペース → ペース偏差（想定 - 通常）
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error


class PacePredictionModel:
    """
    ペース予測モデル（回帰）
    
    予測対象:
    - 前半3F（秒）
    - 後半3F（秒）
    """
    
    def __init__(
        self,
        target: str = 'front_3f',  # 'front_3f' or 'last_3f'
        params: Optional[Dict] = None
    ):
        """
        Args:
            target: 予測対象（'front_3f' or 'last_3f'）
            params: LightGBMパラメータ
        """
        
        self.target = target
        
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
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
        
        # 距離別の基準ペース（参考値）
        self.baseline_pace = {
            1200: {'front_3f': 33.0, 'last_3f': 35.5},
            1400: {'front_3f': 33.5, 'last_3f': 35.0},
            1600: {'front_3f': 34.0, 'last_3f': 34.5},
            1800: {'front_3f': 34.5, 'last_3f': 35.0},
            2000: {'front_3f': 35.0, 'last_3f': 35.5},
            2400: {'front_3f': 36.0, 'last_3f': 36.0}
        }
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        モデルを訓練
        
        Args:
            X: 特徴量
            y: ターゲット（実際のペース、秒）
            X_val: 検証特徴量
            y_val: 検証ターゲット
        """
        
        self.feature_names = X.columns.tolist()
        
        # データセット作成
        train_data = lgb.Dataset(
            X,
            label=y,
            free_raw_data=False
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
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
        
        print(f"\n{self.target}予測モデル - 最適イテレーション: {self.model.best_iteration}")
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        ペースを予測
        
        Args:
            X: 特徴量
        
        Returns:
            予測ペース（秒）
        """
        
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        モデルを評価
        
        Args:
            X: 特徴量
            y: 実際のペース
        
        Returns:
            評価指標の辞書
        """
        
        y_pred = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }
        
        return metrics
    
    def calculate_pace_deviation(
        self,
        predicted_pace: np.ndarray,
        distances: np.ndarray
    ) -> np.ndarray:
        """
        ペース偏差を計算（予測 - 基準）
        
        Args:
            predicted_pace: 予測ペース
            distances: 距離
        
        Returns:
            ペース偏差（秒）
        """
        
        baseline = np.array([
            self._get_baseline_pace(dist)
            for dist in distances
        ])
        
        deviation = predicted_pace - baseline
        
        return deviation
    
    def _get_baseline_pace(self, distance: int) -> float:
        """距離に対応する基準ペースを取得"""
        
        if distance in self.baseline_pace:
            return self.baseline_pace[distance][self.target]
        
        # 最も近い距離の値を使用
        closest_distance = min(
            self.baseline_pace.keys(),
            key=lambda x: abs(x - distance)
        )
        
        return self.baseline_pace[closest_distance][self.target]
    
    def classify_pace(
        self,
        pace: float,
        distance: int
    ) -> str:
        """
        ペースを分類（ハイ/標準/スロー）
        
        Args:
            pace: 予測ペース
            distance: 距離
        
        Returns:
            'ハイペース', '標準ペース', 'スローペース'
        """
        
        baseline = self._get_baseline_pace(distance)
        
        if pace < baseline - 0.5:
            return 'ハイペース'
        elif pace > baseline + 0.5:
            return 'スローペース'
        else:
            return '標準ペース'


class PaceFeatureExtractor:
    """
    ペース予測用の特徴量を抽出
    """
    
    @staticmethod
    def extract_features(
        race_data: pd.DataFrame,
        horse_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ペース予測用の特徴量を抽出
        
        Args:
            race_data: レース情報
            horse_data: 各馬の情報
        
        Returns:
            特徴量DataFrame
        """
        
        features = []
        
        for race_id in race_data['race_id'].unique():
            race = race_data[race_data['race_id'] == race_id].iloc[0]
            horses = horse_data[horse_data['race_id'] == race_id]
            
            # 逃げ・先行馬の分析
            escape_horses = horses[horses['running_style'] == '逃げ']
            leading_horses = horses[horses['running_style'].isin(['逃げ', '先行'])]
            
            # 特徴量
            feature = {
                # レース条件
                'distance': race['distance'],
                'track_type': 1 if race['track_type'] == '芝' else 0,
                'track_condition': race['track_condition_encoded'],
                'track_name': race['track_name_encoded'],
                
                # 頭数
                'n_horses': len(horses),
                'n_escape': len(escape_horses),
                'n_leading': len(leading_horses),
                'escape_ratio': len(escape_horses) / len(horses) if len(horses) > 0 else 0,
                
                # 逃げ馬の能力
                'escape_avg_corner1': escape_horses['avg_corner1_position'].mean() if len(escape_horses) > 0 else 5,
                'escape_best_3f': escape_horses['best_3f_time'].mean() if len(escape_horses) > 0 else 40,
                'escape_avg_speed_index': escape_horses['speed_index'].mean() if len(escape_horses) > 0 else 50,
                
                # 先行馬の能力
                'leading_avg_corner1': leading_horses['avg_corner1_position'].mean() if len(leading_horses) > 0 else 6,
                'leading_best_3f': leading_horses['best_3f_time'].mean() if len(leading_horses) > 0 else 40,
                
                # 馬場状態
                'baba_index': race.get('baba_index', 0),
                'moisture': race.get('moisture', 15),
                'cushion_value': race.get('cushion_value', 9),
                
                # クラス
                'class_level': race.get('class_level', 2),  # 0:新馬, 1:未勝利, 2:1勝, ...
                
                # 枠順の偏り（内枠に強い馬が多いとペースアップしやすい）
                'avg_gate_number': horses['gate_number'].mean(),
                
                # 全体の能力水準
                'avg_speed_index': horses['speed_index'].mean() if 'speed_index' in horses.columns else 50
            }
            
            features.append(feature)
        
        return pd.DataFrame(features)


def example_usage():
    """使用例"""
    
    # ダミーデータ作成
    np.random.seed(42)
    n_races = 500
    
    # レース情報
    race_data = pd.DataFrame({
        'race_id': [f'race_{i}' for i in range(n_races)],
        'distance': np.random.choice([1600, 1800, 2000], n_races),
        'track_type': np.random.choice(['芝', 'ダート'], n_races),
        'track_condition_encoded': np.random.choice([0, 1, 2, 3], n_races),
        'track_name_encoded': np.random.choice([0, 1, 2, 3], n_races),
        'baba_index': np.random.normal(0, 1.5, n_races),
        'moisture': np.random.normal(15, 5, n_races),
        'cushion_value': np.random.normal(9, 1, n_races),
        'class_level': np.random.choice([0, 1, 2, 3], n_races)
    })
    
    # 馬情報（ダミー）
    horse_data = pd.DataFrame({
        'race_id': np.repeat([f'race_{i}' for i in range(n_races)], 18),
        'running_style': np.random.choice(['逃げ', '先行', '差し', '追込'], n_races * 18),
        'avg_corner1_position': np.random.uniform(1, 15, n_races * 18),
        'best_3f_time': np.random.uniform(32, 38, n_races * 18),
        'speed_index': np.random.normal(50, 10, n_races * 18),
        'gate_number': np.tile(range(1, 19), n_races)
    })
    
    # 実際のペース（ダミー）
    front_3f_actual = 34 + np.random.normal(0, 1, n_races)
    
    # 特徴量抽出
    X = PaceFeatureExtractor.extract_features(race_data, horse_data)
    y = front_3f_actual
    
    # モデル訓練
    model = PacePredictionModel(target='front_3f')
    model.train(X[:400], y[:400], X[400:], y[400:])
    
    # 評価
    metrics = model.evaluate(X[400:], y[400:])
    print("\n=== 評価結果 ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # ペース偏差の計算
    predicted = model.predict(X[400:])
    deviation = model.calculate_pace_deviation(
        predicted,
        race_data['distance'].values[400:]
    )
    
    print("\n=== ペース予測の例 ===")
    print("予測ペース:", predicted[:5])
    print("ペース偏差:", deviation[:5])
    
    # ペース分類
    for i in range(5):
        pace_type = model.classify_pace(
            predicted[i],
            race_data['distance'].values[400 + i]
        )
        print(f"レース{i}: {predicted[i]:.2f}秒 ({pace_type})")


if __name__ == "__main__":
    example_usage()
