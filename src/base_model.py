"""
ベースモデル: LightGBMによる勝率予測

目的: 各馬の勝率を確率で出す
評価: Brier, Logloss, NDCG
学習: ウォークフォワード方式
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    ndcg_score
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class BaseModel:
    """
    ベース勝率予測モデル
    
    特徴:
    - LightGBM binary classification
    - ウォークフォワードCV対応
    - 包括的評価指標（Brier, LogLoss, AUC, NDCG）
    """
    
    def __init__(
        self,
        params: Optional[Dict] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50
    ):
        """
        Args:
            params: LightGBMパラメータ
            n_estimators: 最大イテレーション数
            early_stopping_rounds: Early stopping
        """
        
        if params is None:
            params = {
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 7,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1
            }
        
        self.params = params
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None
    ):
        """
        モデルを訓練
        
        Args:
            X_train: 訓練特徴量
            y_train: 訓練ラベル（1: 勝利, 0: 敗北）
            X_val: 検証特徴量
            y_val: 検証ラベル
            categorical_features: カテゴリカル特徴量のリスト
        """
        
        self.feature_names = X_train.columns.tolist()
        
        # データセット作成
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features,
            free_raw_data=False
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                categorical_feature=categorical_features,
                free_raw_data=False
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # 訓練
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"\n最適イテレーション: {self.model.best_iteration}")
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        勝率を予測
        
        Args:
            X: 特徴量
        
        Returns:
            勝率（0~1の確率）
        """
        
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        race_ids: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        包括的な評価
        
        Args:
            X: 特徴量
            y: 正解ラベル
            race_ids: レースID（NDCG計算用）
        
        Returns:
            評価指標の辞書
        """
        
        y_pred = self.predict(X)
        
        metrics = {}
        
        # 1. Brierスコア（確率の精度）
        metrics['brier_score'] = brier_score_loss(y, y_pred)
        
        # 2. Log Loss
        metrics['log_loss'] = log_loss(y, y_pred)
        
        # 3. AUC（識別能力）
        metrics['auc'] = roc_auc_score(y, y_pred)
        
        # 4. NDCG（ランキング精度）
        if race_ids is not None:
            ndcg_scores = []
            
            for race_id in race_ids.unique():
                race_mask = race_ids == race_id
                race_y = y[race_mask]
                race_pred = y_pred[race_mask]
                
                if len(race_y) > 1:
                    # True relevance: 1位=3, 2位=2, 3位=1, それ以外=0
                    # (finish_positionから計算する場合の例)
                    # ここでは単純化して、勝利=1, その他=0とする
                    true_relevance = race_y.values.reshape(1, -1)
                    pred_scores = race_pred.reshape(1, -1)
                    
                    try:
                        ndcg = ndcg_score(true_relevance, pred_scores)
                        ndcg_scores.append(ndcg)
                    except:
                        pass
            
            if ndcg_scores:
                metrics['ndcg'] = np.mean(ndcg_scores)
        
        # 5. Top-1 Accuracy（最高予測の的中率）
        if race_ids is not None:
            top1_correct = 0
            total_races = 0
            
            for race_id in race_ids.unique():
                race_mask = race_ids == race_id
                race_y = y[race_mask].values
                race_pred = y_pred[race_mask]
                
                if len(race_y) > 1:
                    top1_idx = np.argmax(race_pred)
                    if race_y[top1_idx] == 1:
                        top1_correct += 1
                    total_races += 1
            
            if total_races > 0:
                metrics['top1_accuracy'] = top1_correct / total_races
        
        return metrics
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        特徴量の重要度を取得
        
        Args:
            importance_type: 'gain' or 'split'
            top_n: 上位N個
        
        Returns:
            重要度のDataFrame
        """
        
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return df
    
    def save(self, path: str):
        """モデルを保存"""
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        self.model.save_model(path)
        
        # メタデータも保存
        meta_path = Path(path).with_suffix('.json')
        meta = {
            'params': self.params,
            'n_estimators': self.n_estimators,
            'feature_names': self.feature_names
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def load(self, path: str):
        """モデルを読み込み"""
        self.model = lgb.Booster(model_file=path)
        
        # メタデータも読み込み
        meta_path = Path(path).with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.params = meta['params']
                self.n_estimators = meta['n_estimators']
                self.feature_names = meta['feature_names']


class WalkForwardValidator:
    """
    ウォークフォワードCVの実行と評価
    """
    
    def __init__(self, base_model_class=BaseModel):
        self.base_model_class = base_model_class
        self.fold_results = []
    
    def validate(
        self,
        timeline_manager,
        X: pd.DataFrame,
        y: pd.Series,
        race_ids: pd.Series,
        n_splits: int = 5
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        ウォークフォワードCVを実行
        
        Args:
            timeline_manager: TimelineManagerインスタンス
            X: 全特徴量
            y: 全ラベル
            race_ids: レースID
            n_splits: 分割数
        
        Returns:
            (平均評価指標, fold別結果リスト)
        """
        
        splits = timeline_manager.walk_forward_split(n_splits=n_splits)
        
        self.fold_results = []
        
        for split in splits:
            print(f"\n{'='*60}")
            print(f"Fold {split.fold}")
            print(f"Train: {split.train_start.date()} ~ {split.train_end.date()}")
            print(f"Test:  {split.test_start.date()} ~ {split.test_end.date()}")
            print(f"{'='*60}")
            
            # データ分割
            X_train = X.iloc[split.train_indices]
            y_train = y.iloc[split.train_indices]
            X_test = X.iloc[split.test_indices]
            y_test = y.iloc[split.test_indices]
            race_ids_test = race_ids.iloc[split.test_indices]
            
            # モデル訓練
            model = self.base_model_class()
            model.train(X_train, y_train)
            
            # 評価
            metrics = model.evaluate(X_test, y_test, race_ids_test)
            
            # 結果を保存
            fold_result = {
                'fold': split.fold,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': metrics
            }
            self.fold_results.append(fold_result)
            
            # 出力
            print(f"\n評価結果:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # 平均計算
        avg_metrics = self._calculate_average_metrics()
        
        print(f"\n{'='*60}")
        print(f"平均評価指標（{n_splits}-fold）")
        print(f"{'='*60}")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print(f"{'='*60}")
        
        return avg_metrics, self.fold_results
    
    def _calculate_average_metrics(self) -> Dict[str, float]:
        """fold別結果から平均を計算"""
        
        all_metrics = {}
        
        for fold_result in self.fold_results:
            for metric_name, value in fold_result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        avg_metrics = {
            name: np.mean(values)
            for name, values in all_metrics.items()
        }
        
        return avg_metrics


def example_usage():
    """使用例"""
    
    from timeline_manager import TimelineManager
    
    # ダミーデータ作成
    np.random.seed(42)
    n_samples = 10000
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='W')
    
    data = []
    for i, date in enumerate(dates[:n_samples // 18]):
        # 1レース18頭
        for horse_num in range(1, 19):
            data.append({
                'race_id': f'race_{i}',
                'race_date': date,
                'horse_id': f'horse_{np.random.randint(1, 1000)}',
                'track_name': np.random.choice(['東京', '中山', '京都']),
                'distance': np.random.choice([1600, 1800, 2000]),
                'horse_number': horse_num,
                'finish_position': horse_num  # ダミー
            })
    
    df = pd.DataFrame(data)
    
    # 特徴量とラベル作成
    X = pd.DataFrame({
        'horse_number': df['horse_number'],
        'distance': df['distance'],
        'track_encoded': df['track_name'].astype('category').cat.codes
    })
    
    y = (df['finish_position'] == 1).astype(int)
    race_ids = df['race_id']
    
    # TimelineManager
    tm = TimelineManager(df, date_column='race_date')
    
    # ウォークフォワードCV
    validator = WalkForwardValidator()
    avg_metrics, fold_results = validator.validate(
        tm, X, y, race_ids, n_splits=3
    )


if __name__ == "__main__":
    example_usage()
