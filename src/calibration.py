"""
確率キャリブレーション（Calibration）

目的: ベースモデルの確率を校正して信頼性を向上
方式: CalibratedClassifierCV (Platt Scaling / Isotonic Regression)
評価: ECE, MCE, Brier
"""

import pandas as pd
import numpy as np
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)
from sklearn.metrics import brier_score_loss
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


class ProbabilityCalibrator:
    """
    確率のキャリブレーション
    
    方式:
    - Platt Scaling: データが少ない場合に有効（シグモイド関数でフィット）
    - Isotonic Regression: データが多い場合に有効（非パラメトリック）
    """
    
    def __init__(
        self,
        method: str = 'auto',
        n_bins: int = 10
    ):
        """
        Args:
            method: 'sigmoid' (Platt) / 'isotonic' / 'auto'
            n_bins: Calibration Curve用のビン数
        """
        
        if method == 'auto':
            self.method = None  # データ量で自動判定
        else:
            self.method = method
        
        self.n_bins = n_bins
        self.calibrator = None
    
    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'ProbabilityCalibrator':
        """
        キャリブレーションを実行
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測確率（校正前）
            sample_weight: サンプルウェイト
        
        Returns:
            self
        """
        
        # データ量で方式を自動判定
        n_samples = len(y_true)
        
        if self.method is None:
            if n_samples < 1000:
                method = 'sigmoid'  # Platt Scaling
                print(f"自動選択: Platt Scaling (n={n_samples})")
            else:
                method = 'isotonic'  # Isotonic Regression
                print(f"自動選択: Isotonic Regression (n={n_samples})")
        else:
            method = self.method
        
        # ダミー分類器（予測値をそのまま返す）
        class DummyClassifier:
            def __init__(self, y_pred):
                self.y_pred = y_pred
            
            def predict_proba(self, X):
                # X は使わない（すでにy_predがある）
                return np.column_stack([1 - self.y_pred, self.y_pred])
        
        dummy = DummyClassifier(y_pred)
        
        # キャリブレーション
        self.calibrator = CalibratedClassifierCV(
            dummy,
            method=method,
            cv='prefit'
        )
        
        # ダミーのXを作成（使われないが必要）
        X_dummy = np.zeros((len(y_pred), 1))
        
        self.calibrator.fit(X_dummy, y_true, sample_weight=sample_weight)
        
        return self
    
    def transform(
        self,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        予測確率を校正
        
        Args:
            y_pred: 校正前の予測確率
        
        Returns:
            校正後の予測確率
        """
        
        if self.calibrator is None:
            raise ValueError("calibrate()を先に実行してください")
        
        # ダミーのX
        X_dummy = np.zeros((len(y_pred), 1))
        
        calibrated_probs = self.calibrator.predict_proba(X_dummy)[:, 1]
        
        return calibrated_probs
    
    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        キャリブレーションの品質を評価
        
        Args:
            y_true: 正解ラベル
            y_pred: 予測確率
        
        Returns:
            評価指標の辞書
        """
        
        metrics = {}
        
        # 1. Brierスコア
        metrics['brier_score'] = brier_score_loss(y_true, y_pred)
        
        # 2. ECE (Expected Calibration Error)
        ece = self._calculate_ece(y_true, y_pred)
        metrics['ece'] = ece
        
        # 3. MCE (Maximum Calibration Error)
        mce = self._calculate_mce(y_true, y_pred)
        metrics['mce'] = mce
        
        return metrics
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Expected Calibration Error (ECE)
        
        ECE = Σ (|bin内の平均予測確率 - bin内の実際の正例率|) * bin内のサンプル割合
        """
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=self.n_bins, strategy='uniform'
        )
        
        # 各binのサンプル数を計算
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        bin_counts = np.bincount(bin_indices, minlength=self.n_bins)
        bin_weights = bin_counts / len(y_pred)
        
        # ECE計算
        ece = np.sum(
            np.abs(fraction_of_positives - mean_predicted_value) * 
            bin_weights[:len(fraction_of_positives)]
        )
        
        return ece
    
    def _calculate_mce(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Maximum Calibration Error (MCE)
        
        MCE = max |bin内の平均予測確率 - bin内の実際の正例率|
        """
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=self.n_bins, strategy='uniform'
        )
        
        mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        
        return mce
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_before: np.ndarray,
        y_pred_after: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Calibration Curveをプロット
        
        Args:
            y_true: 正解ラベル
            y_pred_before: 校正前の予測確率
            y_pred_after: 校正後の予測確率
            save_path: 保存先パス（Noneの場合は表示のみ）
        """
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 校正前
        fraction_before, mean_before = calibration_curve(
            y_true, y_pred_before, n_bins=self.n_bins
        )
        ax.plot(mean_before, fraction_before, 's-', label='校正前', color='red')
        
        # 校正後
        fraction_after, mean_after = calibration_curve(
            y_true, y_pred_after, n_bins=self.n_bins
        )
        ax.plot(mean_after, fraction_after, 'o-', label='校正後', color='blue')
        
        # 理想的なキャリブレーション（対角線）
        ax.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
        
        ax.set_xlabel('Mean predicted probability', fontsize=12)
        ax.set_ylabel('Fraction of positives', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class CalibrationPipeline:
    """
    ベースモデル + キャリブレーションのパイプライン
    """
    
    def __init__(self, base_model):
        """
        Args:
            base_model: ベースモデル（.predict()メソッドを持つ）
        """
        self.base_model = base_model
        self.calibrator = None
    
    def fit_calibrator(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = 'auto'
    ):
        """
        キャリブレーターを訓練
        
        Args:
            X_cal: キャリブレーション用特徴量
            y_cal: キャリブレーション用ラベル
            method: キャリブレーション方式
        """
        
        # ベースモデルで予測
        y_pred_cal = self.base_model.predict(X_cal)
        
        # キャリブレーション
        self.calibrator = ProbabilityCalibrator(method=method)
        self.calibrator.calibrate(y_cal.values, y_pred_cal)
        
        # 評価
        y_pred_cal_calibrated = self.calibrator.transform(y_pred_cal)
        
        print("\n=== キャリブレーション評価（校正用データ） ===")
        
        metrics_before = self.calibrator.evaluate_calibration(
            y_cal.values, y_pred_cal
        )
        print("\n校正前:")
        for name, value in metrics_before.items():
            print(f"  {name}: {value:.4f}")
        
        metrics_after = self.calibrator.evaluate_calibration(
            y_cal.values, y_pred_cal_calibrated
        )
        print("\n校正後:")
        for name, value in metrics_after.items():
            print(f"  {name}: {value:.4f}")
    
    def predict(
        self,
        X: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """
        予測（オプションで校正）
        
        Args:
            X: 特徴量
            calibrated: 校正するか
        
        Returns:
            予測確率
        """
        
        y_pred = self.base_model.predict(X)
        
        if calibrated and self.calibrator is not None:
            y_pred = self.calibrator.transform(y_pred)
        
        return y_pred
    
    def evaluate_on_test(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        テストデータで評価
        
        Args:
            X_test: テスト特徴量
            y_test: テストラベル
            plot_path: Calibration Curveの保存先
        
        Returns:
            評価指標の辞書
        """
        
        y_pred_before = self.base_model.predict(X_test)
        y_pred_after = self.predict(X_test, calibrated=True)
        
        print("\n=== テストデータでの評価 ===")
        
        metrics_before = self.calibrator.evaluate_calibration(
            y_test.values, y_pred_before
        )
        print("\n校正前:")
        for name, value in metrics_before.items():
            print(f"  {name}: {value:.4f}")
        
        metrics_after = self.calibrator.evaluate_calibration(
            y_test.values, y_pred_after
        )
        print("\n校正後:")
        for name, value in metrics_after.items():
            print(f"  {name}: {value:.4f}")
        
        # Calibration Curveをプロット
        if plot_path:
            self.calibrator.plot_calibration_curve(
                y_test.values,
                y_pred_before,
                y_pred_after,
                save_path=plot_path
            )
        
        return {
            'before': metrics_before,
            'after': metrics_after
        }


def example_usage():
    """使用例"""
    
    # ダミーデータ
    np.random.seed(42)
    n = 1000
    
    # 校正前の予測確率（過信されている例）
    y_true = np.random.binomial(1, 0.3, n)
    y_pred_raw = np.random.beta(2, 5, n)  # 偏った分布
    
    # キャリブレーター
    calibrator = ProbabilityCalibrator(method='isotonic')
    calibrator.calibrate(y_true[:800], y_pred_raw[:800])
    
    # 校正後
    y_pred_calibrated = calibrator.transform(y_pred_raw[800:])
    
    # 評価
    print("=== テストデータ（校正前） ===")
    metrics_before = calibrator.evaluate_calibration(
        y_true[800:], y_pred_raw[800:]
    )
    for name, value in metrics_before.items():
        print(f"{name}: {value:.4f}")
    
    print("\n=== テストデータ（校正後） ===")
    metrics_after = calibrator.evaluate_calibration(
        y_true[800:], y_pred_calibrated
    )
    for name, value in metrics_after.items():
        print(f"{name}: {value:.4f}")
    
    # プロット
    calibrator.plot_calibration_curve(
        y_true[800:],
        y_pred_raw[800:],
        y_pred_calibrated
    )


if __name__ == "__main__":
    example_usage()
