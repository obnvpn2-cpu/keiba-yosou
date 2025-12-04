"""
MLflow管理

目的: 実験追跡、パラメータ管理、モデルバージョニング
機能: メトリクス記録、アーティファクト保存、モデル登録
"""

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime


class MLflowManager:
    """
    MLflow実験管理
    
    機能:
    - 実験作成
    - メトリクス記録
    - パラメータ記録
    - モデル保存
    - アーティファクト保存
    """
    
    def __init__(
        self,
        experiment_name: str = "keiba_v1",
        tracking_uri: Optional[str] = None
    ):
        """
        Args:
            experiment_name: 実験名
            tracking_uri: MLflow tracking URI
        """
        
        self.experiment_name = experiment_name
        
        # Tracking URI設定
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # 実験作成または取得
        try:
            self.experiment = mlflow.create_experiment(experiment_name)
        except:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient()
        self.run_id = None
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None
    ):
        """
        MLflow runを開始
        
        Args:
            run_name: run名
            tags: タグ辞書
        """
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if tags is None:
            tags = {}
        
        mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = mlflow.active_run().info.run_id
        
        print(f"MLflow run開始: {run_name}")
        print(f"Run ID: {self.run_id}")
    
    def log_params(
        self,
        params: Dict[str, Any]
    ):
        """
        パラメータを記録
        
        Args:
            params: パラメータ辞書
        """
        
        # MLflowは文字列、数値、boolのみサポート
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                mlflow.log_param(key, json.dumps(value))
            else:
                mlflow.log_param(key, value)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        メトリクスを記録
        
        Args:
            metrics: メトリクス辞書
            step: ステップ数（時系列データ用）
        """
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        model_type: str = "lightgbm"
    ):
        """
        モデルを記録
        
        Args:
            model: モデルオブジェクト
            artifact_path: 保存パス
            model_type: モデルタイプ
        """
        
        if model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path)
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")
        
        print(f"モデル記録完了: {artifact_path}")
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ):
        """
        アーティファクトを記録
        
        Args:
            local_path: ローカルファイルパス
            artifact_path: 保存先パス
        """
        
        mlflow.log_artifact(local_path, artifact_path)
        print(f"アーティファクト記録: {local_path}")
    
    def log_dataframe(
        self,
        df: pd.DataFrame,
        filename: str
    ):
        """
        DataFrameを記録
        
        Args:
            df: DataFrame
            filename: ファイル名
        """
        
        temp_path = f"/tmp/{filename}"
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path)
        print(f"DataFrame記録: {filename}")
    
    def end_run(self):
        """MLflow runを終了"""
        mlflow.end_run()
        print("MLflow run終了")


class ExperimentTracker:
    """
    実験トラッキングのラッパー
    
    使いやすいインターフェース
    """
    
    def __init__(
        self,
        experiment_name: str = "keiba_v1"
    ):
        self.manager = MLflowManager(experiment_name)
    
    def track_training(
        self,
        model,
        params: Dict,
        train_metrics: Dict,
        val_metrics: Dict,
        fold: Optional[int] = None
    ):
        """
        訓練をトラッキング
        
        Args:
            model: モデル
            params: パラメータ
            train_metrics: 訓練メトリクス
            val_metrics: 検証メトリクス
            fold: fold番号
        """
        
        # run名
        if fold is not None:
            run_name = f"fold_{fold}"
        else:
            run_name = "training"
        
        # run開始
        self.manager.start_run(
            run_name=run_name,
            tags={'fold': str(fold) if fold else 'none'}
        )
        
        # パラメータ記録
        self.manager.log_params(params)
        
        # メトリクス記録
        for name, value in train_metrics.items():
            self.manager.log_metrics({f'train_{name}': value})
        
        for name, value in val_metrics.items():
            self.manager.log_metrics({f'val_{name}': value})
        
        # モデル記録
        self.manager.log_model(model, artifact_path="model")
        
        # run終了
        self.manager.end_run()
    
    def track_backtest(
        self,
        backtest_results: Dict,
        config: Dict
    ):
        """
        バックテストをトラッキング
        
        Args:
            backtest_results: バックテスト結果
            config: バックテスト設定
        """
        
        self.manager.start_run(
            run_name="backtest",
            tags={'type': 'backtest'}
        )
        
        # 設定記録
        self.manager.log_params(config)
        
        # メトリクス記録
        for strategy, result in backtest_results['strategy_results'].items():
            for metric_name, value in result.items():
                if isinstance(value, (int, float)):
                    self.manager.log_metrics({
                        f'{strategy}_{metric_name}': value
                    })
        
        self.manager.end_run()
    
    def track_full_pipeline(
        self,
        config: Dict,
        cv_results: Dict,
        final_metrics: Dict,
        models: Dict
    ):
        """
        パイプライン全体をトラッキング
        
        Args:
            config: 全体設定
            cv_results: クロスバリデーション結果
            final_metrics: 最終評価指標
            models: モデル辞書
        """
        
        self.manager.start_run(
            run_name="full_pipeline",
            tags={'type': 'pipeline'}
        )
        
        # 設定
        self.manager.log_params(config)
        
        # CV結果
        for fold_idx, fold_result in enumerate(cv_results):
            for metric_name, value in fold_result.items():
                if isinstance(value, (int, float)):
                    self.manager.log_metrics(
                        {f'cv_fold{fold_idx}_{metric_name}': value}
                    )
        
        # 最終メトリクス
        self.manager.log_metrics(final_metrics)
        
        # モデル保存
        for model_name, model in models.items():
            self.manager.log_model(
                model,
                artifact_path=f"models/{model_name}"
            )
        
        self.manager.end_run()


def example_usage():
    """使用例"""
    
    # 実験トラッカー
    tracker = ExperimentTracker(experiment_name="keiba_v1")
    
    # ダミーモデル
    import lightgbm as lgb
    
    X = np.random.rand(1000, 10)
    y = np.random.binomial(1, 0.1, 1000)
    
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    model.fit(X, y)
    
    # 訓練をトラッキング
    params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 7
    }
    
    train_metrics = {
        'brier_score': 0.085,
        'auc': 0.72
    }
    
    val_metrics = {
        'brier_score': 0.092,
        'auc': 0.68
    }
    
    tracker.track_training(
        model.booster_,
        params,
        train_metrics,
        val_metrics,
        fold=1
    )
    
    print("実験トラッキング完了")
    print("MLflow UIで確認: mlflow ui")


if __name__ == "__main__":
    example_usage()
