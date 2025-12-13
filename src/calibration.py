# calibration.py (v3)
# v2 からの主な改善点:
# - fit() の再学習制御（force パラメータ）
# - Python/numpy バージョンの記録
# - ログレベルの設定機能
# - evaluate() の fit チェック追加
# - Isotonic の少サンプル警告
# - メタデータに統計情報追加
# - ECE のビン除外警告
# - 極端な分布の検出

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pickle

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# ロガー設定
logger = logging.getLogger(__name__)

# デフォルトのログハンドラ設定（使用側で上書き可能）
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@dataclass
class CalibrationConfig:
    """
    キャリブレーション設定。
    
    Attributes:
        method: キャリブレーション手法
            - "identity": 何もしない（デバッグ用）
            - "platt": Platt Scaling（ロジスティック回帰）
            - "isotonic": Isotonic Regression（単調増加制約）
        n_bins: ECE計算用のビン数（デフォルト: 15）
        min_samples_bin: あまり小さいビンを作らないための下限（デフォルト: 20）
        eps: 数値安定性のための最小値（デフォルト: 1e-7）
        min_samples_isotonic: Isotonic Regression の推奨最小サンプル数（デフォルト: 500）
        extreme_distribution_threshold: 極端な分布と判定する正例率の閾値（デフォルト: 0.01）
    """
    method: str = "platt"
    n_bins: int = 15
    min_samples_bin: int = 20
    eps: float = 1e-7
    min_samples_isotonic: int = 500
    extreme_distribution_threshold: float = 0.01

    def __post_init__(self):
        """設定値のバリデーション"""
        valid_methods = {"identity", "platt", "isotonic"}
        if self.method.lower() not in valid_methods:
            raise ValueError(
                f"method は {valid_methods} のいずれかを指定してください。"
                f"指定値: {self.method}"
            )
        
        if self.n_bins < 2:
            raise ValueError(f"n_bins は 2 以上を指定してください。指定値: {self.n_bins}")
        
        if self.min_samples_bin < 1:
            raise ValueError(
                f"min_samples_bin は 1 以上を指定してください。"
                f"指定値: {self.min_samples_bin}"
            )
        
        if not (0 < self.eps < 0.1):
            raise ValueError(
                f"eps は 0 < eps < 0.1 の範囲で指定してください。"
                f"指定値: {self.eps}"
            )
        
        if self.min_samples_isotonic < 10:
            raise ValueError(
                f"min_samples_isotonic は 10 以上を指定してください。"
                f"指定値: {self.min_samples_isotonic}"
            )
        
        if not (0 < self.extreme_distribution_threshold < 0.5):
            raise ValueError(
                f"extreme_distribution_threshold は 0 < x < 0.5 の範囲で指定してください。"
                f"指定値: {self.extreme_distribution_threshold}"
            )


class ProbabilityCalibrator:
    """
    1次元の確率出力をキャリブレーションするクラス。
    
    確率予測モデルの出力を実際の発生確率に合わせて調整（キャリブレーション）する。
    モデルの識別性能は変わらないが、確率の信頼性が向上する。
    
    対応手法:
        - identity: 何もしない（ベースライン比較用）
        - platt: Platt Scaling（シグモイド関数でフィッティング）
        - isotonic: Isotonic Regression（単調増加制約付き回帰）
    
    使用例:
        >>> config = CalibrationConfig(method="platt", n_bins=10)
        >>> calibrator = ProbabilityCalibrator(config)
        >>> calibrator.fit(y_pred_train, y_true_train)
        >>> y_calibrated = calibrator.predict(y_pred_test)
        >>> metrics = calibrator.evaluate(y_pred_val, y_true_val)
        >>> print(f"ECE改善: {metrics['ece_raw']:.4f} → {metrics['ece_calibrated']:.4f}")
    
    注意:
        - キャリブレーション用データは学習データとは別に用意すること
        - データ分布が大きく変わる場合は再キャリブレーションが必要
        - 一度学習したキャリブレータを再学習する場合は force=True が必要
    """
    
    # バージョン情報
    VERSION = "3.0.0"
    
    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        """
        Args:
            config: キャリブレーション設定。None の場合はデフォルト設定を使用。
        """
        self.config = config or CalibrationConfig()
        self._fitted: bool = False
        self._estimator: Any = None  # LogisticRegression or IsotonicRegression
        self._fit_metadata: Dict[str, Any] = {}

    # --------------------------------------------------------------
    # 公開API
    # --------------------------------------------------------------
    
    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        force: bool = False,
    ) -> None:
        """
        生の予測確率と正解ラベルからキャリブレーションモデルを学習する。
        
        Args:
            y_pred: shape (N,) の生確率 or スコア（0〜1を想定）
            y_true: shape (N,) のラベル（0 or 1）
            force: True の場合、既に学習済みでも再学習する（デフォルト: False）
        
        Raises:
            ValueError: 入力配列の形状が不正な場合、または既に学習済みの場合
        
        Examples:
            >>> calibrator = ProbabilityCalibrator()
            >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
            >>> y_true = np.array([0, 0, 1, 1])
            >>> calibrator.fit(y_pred, y_true)
            
            >>> # 再学習する場合
            >>> calibrator.fit(y_pred_new, y_true_new, force=True)
        """
        # 再学習チェック
        if self._fitted and not force:
            raise ValueError(
                "このキャリブレータは既に学習済みです。"
                "再学習する場合は force=True を指定するか、reset() を呼び出してください。"
            )
        
        # 入力検証
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"y_pred と y_true の長さが一致しません。"
                f"y_pred: {y_pred.shape[0]}, y_true: {y_true.shape[0]}"
            )
        
        if y_pred.shape[0] == 0:
            raise ValueError("空の配列が渡されました。")
        
        # 極端な分布の検出
        pos_rate = float(np.mean(y_true))
        if (pos_rate < self.config.extreme_distribution_threshold or
            pos_rate > 1.0 - self.config.extreme_distribution_threshold):
            warnings.warn(
                f"データが極端に偏っています（正例率: {pos_rate:.4f}）。"
                f"キャリブレーションが不安定になる可能性があります。"
                f"データの追加収集またはリサンプリングを検討してください。",
                category=UserWarning
            )
        
        # メタデータ記録（統計情報を追加）
        self._fit_metadata = {
            "n_samples": len(y_pred),
            "fit_timestamp": datetime.now().isoformat(),
            "method": self.config.method,
            "y_pred_stats": {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred)),
                "median": float(np.median(y_pred)),
                "q25": float(np.percentile(y_pred, 25)),
                "q75": float(np.percentile(y_pred, 75)),
            },
            "y_true_stats": {
                "positive_rate": pos_rate,
                "n_positive": int(np.sum(y_true)),
                "n_negative": int(len(y_true) - np.sum(y_true)),
            },
        }
        
        logger.info(
            f"Calibration fit 開始: method={self.config.method}, "
            f"n_samples={len(y_pred)}, positive_rate={pos_rate:.4f}"
        )

        method = self.config.method.lower()

        if method == "identity":
            # 何もしない
            self._estimator = None
            self._fitted = True
            logger.info("Identity calibration (何もしない) を設定しました。")
            return

        # 0〜1 にクリップ（数値安定性のため）
        y_pred_clipped = self._clip_probabilities(y_pred)
        
        # クリップによる変更を警告
        n_clipped = np.sum((y_pred < self.config.eps) | (y_pred > 1.0 - self.config.eps))
        if n_clipped > 0:
            warnings.warn(
                f"{n_clipped} 個のサンプル ({n_clipped/len(y_pred)*100:.2f}%) が "
                f"範囲外のためクリップされました。"
                f"モデルの出力が 0〜1 の範囲外になっている可能性があります。",
                category=UserWarning
            )

        if method == "platt":
            self._fit_platt(y_pred_clipped, y_true)
        elif method == "isotonic":
            self._fit_isotonic(y_pred_clipped, y_true)
        else:
            # post_init で検証済みのため、ここには来ないはず
            raise ValueError(f"未知の method: {self.config.method}")
        
        logger.info("Calibration fit 完了")

    def predict(
        self,
        y_pred: np.ndarray,
        check_distribution: bool = True,
    ) -> np.ndarray:
        """
        キャリブレート済みの確率を返す。
        
        Args:
            y_pred: shape (N,) の生確率 or スコア（0〜1を想定）
            check_distribution: True の場合、学習時と分布が大きく異なる場合に警告
        
        Returns:
            shape (N,) のキャリブレート済み確率（範囲: [eps, 1-eps]）
        
        Raises:
            ValueError: fit() 前に呼ばれた場合
        
        Examples:
            >>> y_cal = calibrator.predict(np.array([0.3, 0.7]))
            >>> print(y_cal)
            [0.25 0.75]  # 例
        """
        if not self._fitted:
            raise ValueError(
                "ProbabilityCalibrator はまだ fit() されていません。"
                "先に fit(y_pred, y_true) を実行してください。"
            )

        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        
        if len(y_pred) == 0:
            return np.array([])
        
        # 分布チェック（学習時と予測時の分布差を検出）
        if check_distribution and self._fit_metadata:
            self._check_prediction_distribution(y_pred)
        
        method = self.config.method.lower()

        if method == "identity" or self._estimator is None:
            return self._clip_probabilities(y_pred)

        y_pred_clipped = self._clip_probabilities(y_pred)

        if method == "platt":
            return self._predict_platt(y_pred_clipped)
        elif method == "isotonic":
            return self._predict_isotonic(y_pred_clipped)
        else:
            # 理論上ここには来ない
            return y_pred_clipped

    def reset(self) -> None:
        """
        学習状態をリセットし、未学習の状態に戻す。
        
        Examples:
            >>> calibrator.fit(y_pred1, y_true1)
            >>> # ...使用...
            >>> calibrator.reset()  # リセット
            >>> calibrator.fit(y_pred2, y_true2)  # 新たに学習
        """
        self._fitted = False
        self._estimator = None
        self._fit_metadata = {}
        logger.info("Calibrator をリセットしました")

    # --------------------------------------------------------------
    # 評価系
    # --------------------------------------------------------------
    
    def evaluate(
        self,
        y_pred_raw: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        キャリブレーション前後の指標をまとめて返す。
        
        Args:
            y_pred_raw: キャリブレーション前の予測確率
            y_true: 正解ラベル
        
        Returns:
            各種指標の辞書:
                - ece_raw: キャリブレーション前の ECE
                - ece_calibrated: キャリブレーション後の ECE
                - brier_raw: キャリブレーション前の Brier Score
                - brier_calibrated: キャリブレーション後の Brier Score
        
        Raises:
            ValueError: fit() 前に呼ばれた場合
        
        Examples:
            >>> metrics = calibrator.evaluate(y_pred_val, y_true_val)
            >>> print(f"ECE: {metrics['ece_raw']:.4f} → {metrics['ece_calibrated']:.4f}")
            >>> print(f"Brier: {metrics['brier_raw']:.4f} → {metrics['brier_calibrated']:.4f}")
        """
        if not self._fitted:
            raise ValueError(
                "ProbabilityCalibrator はまだ fit() されていません。"
                "先に fit(y_pred, y_true) を実行してください。"
            )
        
        y_pred_raw = np.asarray(y_pred_raw, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        if y_pred_raw.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"y_pred_raw と y_true の長さが一致しません。"
                f"y_pred_raw: {y_pred_raw.shape[0]}, y_true: {y_true.shape[0]}"
            )

        y_pred_raw_clipped = self._clip_probabilities(y_pred_raw)
        y_pred_cal = self.predict(y_pred_raw_clipped, check_distribution=False)

        ece_raw = self.compute_ece(
            y_true, y_pred_raw_clipped,
            n_bins=self.config.n_bins,
            min_samples_bin=self.config.min_samples_bin
        )
        ece_cal = self.compute_ece(
            y_true, y_pred_cal,
            n_bins=self.config.n_bins,
            min_samples_bin=self.config.min_samples_bin
        )

        brier_raw = brier_score_loss(y_true, y_pred_raw_clipped)
        brier_cal = brier_score_loss(y_true, y_pred_cal)

        return {
            "ece_raw": float(ece_raw),
            "ece_calibrated": float(ece_cal),
            "brier_raw": float(brier_raw),
            "brier_calibrated": float(brier_cal),
        }

    @staticmethod
    def compute_ece(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 15,
        min_samples_bin: int = 20,
    ) -> float:
        """
        Expected Calibration Error (ECE) を計算。
        
        予測確率を n_bins に分割し、各ビンで「平均予測確率」と
        「実際の正解率」の差を測定する。
        
        Args:
            y_true: 正解ラベル（0 or 1）
            y_prob: 予測確率（0〜1）
            n_bins: ビン数
            min_samples_bin: ビンに含まれるべき最小サンプル数
                （これより少ないビンは ECE 計算から除外）
        
        Returns:
            ECE 値（0〜1）。小さいほど良い。
        
        Note:
            - 理想的にはECE = 0（完全にキャリブレートされている）
            - ECE > 0.1 の場合、キャリブレーションの改善余地が大きい
            - min_samples_bin によって多くのビンが除外される場合、警告が出る
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

        if len(y_true) != len(y_prob):
            raise ValueError(
                f"配列長が一致しません。y_true: {len(y_true)}, y_prob: {len(y_prob)}"
            )

        # ビン境界
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total = len(y_true)
        
        if total == 0:
            return 0.0

        n_bins_excluded = 0
        n_bins_used = 0

        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]

            # 最後のビンだけ右端を含む
            if i < n_bins - 1:
                mask = (y_prob >= lo) & (y_prob < hi)
            else:
                mask = (y_prob >= lo) & (y_prob <= hi)
            
            if not np.any(mask):
                continue

            y_true_bin = y_true[mask]
            y_prob_bin = y_prob[mask]

            # 小サンプルビンはスキップ
            if len(y_true_bin) < min_samples_bin:
                n_bins_excluded += 1
                continue

            n_bins_used += 1
            avg_conf = float(np.mean(y_prob_bin))
            avg_acc = float(np.mean(y_true_bin))

            ece += (len(y_true_bin) / total) * abs(avg_conf - avg_acc)

        # ビン除外の警告
        if n_bins_excluded > n_bins / 2:
            warnings.warn(
                f"ECE 計算で {n_bins_excluded}/{n_bins} ビンがサンプル不足でスキップされました。"
                f"n_bins を減らすか、min_samples_bin を小さくすることを検討してください。",
                category=UserWarning
            )

        return float(ece)

    def get_reliability_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        リライアビリティカーブ用のデータを返す。
        
        Args:
            y_true: 正解ラベル
            y_prob: 予測確率
            n_bins: ビン数
        
        Returns:
            (bin_centers, bin_acc, bin_conf) のタプル:
                - bin_centers: 各ビンの中心値
                - bin_acc: 各ビンの実際の正解率
                - bin_conf: 各ビンの平均予測確率
                ※ サンプルのないビンは np.nan
        
        Note:
            完全にキャリブレートされている場合、bin_acc と bin_conf が
            対角線上（y=x）に並ぶ。
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        bin_acc = []
        bin_conf = []

        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            
            if i < n_bins - 1:
                mask = (y_prob >= lo) & (y_prob < hi)
            else:
                mask = (y_prob >= lo) & (y_prob <= hi)
            
            if not np.any(mask):
                bin_acc.append(np.nan)
                bin_conf.append(np.nan)
                continue

            y_true_bin = y_true[mask]
            y_prob_bin = y_prob[mask]

            bin_acc.append(float(np.mean(y_true_bin)))
            bin_conf.append(float(np.mean(y_prob_bin)))

        return bin_centers, np.array(bin_acc), np.array(bin_conf)

    # --------------------------------------------------------------
    # モデル保存/読み込み
    # --------------------------------------------------------------
    
    def save(self, path: str) -> None:
        """
        キャリブレータを pickle で保存。
        
        Args:
            path: 保存先パス
        
        Note:
            Python, sklearn, numpy のバージョン情報も保存されるため、
            異なる環境での読み込みには注意が必要。
        """
        import sklearn
        
        obj = {
            "version": self.VERSION,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "config": self.config,
            "fitted": self._fitted,
            "estimator": self._estimator,
            "fit_metadata": self._fit_metadata,
            "save_timestamp": datetime.now().isoformat(),
        }
        
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        
        logger.info(f"Calibrator を保存しました: {path}")

    def load(self, path: str) -> None:
        """
        キャリブレータを読み込み。
        
        Args:
            path: 読み込み元パス
        
        Warns:
            Python, sklearn, numpy のバージョンが異なる場合に警告
        """
        import sklearn
        
        with open(path, "rb") as f:
            obj = pickle.load(f)
        
        # バージョンチェック
        saved_version = obj.get("version", "1.0.0")
        if saved_version != self.VERSION:
            warnings.warn(
                f"保存時のバージョン ({saved_version}) と "
                f"現在のバージョン ({self.VERSION}) が異なります。",
                category=UserWarning
            )
        
        # Python バージョンチェック
        saved_python = obj.get("python_version", "unknown")
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if saved_python != current_python:
            warnings.warn(
                f"保存時の Python バージョン ({saved_python}) と "
                f"現在のバージョン ({current_python}) が異なります。"
                f"互換性問題が発生する可能性があります。",
                category=UserWarning
            )
        
        # sklearn バージョンチェック
        saved_sklearn = obj.get("sklearn_version", "unknown")
        if saved_sklearn != sklearn.__version__:
            warnings.warn(
                f"保存時の sklearn バージョン ({saved_sklearn}) と "
                f"現在のバージョン ({sklearn.__version__}) が異なります。"
                f"互換性問題が発生する可能性があります。",
                category=UserWarning
            )
        
        # numpy バージョンチェック
        saved_numpy = obj.get("numpy_version", "unknown")
        if saved_numpy != np.__version__:
            warnings.warn(
                f"保存時の numpy バージョン ({saved_numpy}) と "
                f"現在のバージョン ({np.__version__}) が異なります。",
                category=UserWarning
            )
        
        self.config = obj.get("config", CalibrationConfig())
        self._fitted = obj.get("fitted", False)
        self._estimator = obj.get("estimator", None)
        self._fit_metadata = obj.get("fit_metadata", {})
        
        logger.info(f"Calibrator を読み込みました: {path}")

    # --------------------------------------------------------------
    # 内部実装
    # --------------------------------------------------------------
    
    def _clip_probabilities(self, p: np.ndarray) -> np.ndarray:
        """確率値を [eps, 1-eps] にクリップ"""
        return np.clip(p, self.config.eps, 1.0 - self.config.eps)
    
    def _logit(self, p: np.ndarray) -> np.ndarray:
        """ロジット変換: logit(p) = log(p / (1-p))"""
        p = self._clip_probabilities(p)
        return np.log(p / (1.0 - p))
    
    def _check_prediction_distribution(self, y_pred: np.ndarray) -> None:
        """予測時の分布が学習時と大きく異なる場合に警告"""
        if not self._fit_metadata:
            return
        
        fit_stats = self._fit_metadata.get("y_pred_stats", {})
        if not fit_stats:
            return
        
        fit_mean = fit_stats.get("mean", 0.5)
        fit_std = fit_stats.get("std", 0.2)
        
        pred_mean = float(np.mean(y_pred))
        pred_std = float(np.std(y_pred))
        
        # 平均が2σ以上ずれている場合
        if abs(pred_mean - fit_mean) > 2 * fit_std:
            warnings.warn(
                f"予測分布が学習時と大きく異なります。"
                f"学習時平均: {fit_mean:.4f}, 予測時平均: {pred_mean:.4f}"
                f"再キャリブレーションを検討してください。",
                category=UserWarning
            )
    
    def _fit_platt(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Platt Scaling の学習"""
        X = self._logit(y_pred).reshape(-1, 1)
        
        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )
        lr.fit(X, y_true.astype(int))
        
        self._estimator = lr
        self._fitted = True
        
        logger.debug(
            f"Platt Scaling 完了: coef={lr.coef_[0]}, intercept={lr.intercept_}"
        )
    
    def _fit_isotonic(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Isotonic Regression の学習"""
        # 少サンプル警告
        if len(y_pred) < self.config.min_samples_isotonic:
            warnings.warn(
                f"Isotonic Regression のサンプル数 ({len(y_pred)}) が "
                f"推奨値 ({self.config.min_samples_isotonic}) を下回っています。"
                f"過適合のリスクがあるため、Platt Scaling の使用を検討してください。",
                category=UserWarning
            )
        
        ir = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        ir.fit(y_pred, y_true)
        
        self._estimator = ir
        self._fitted = True
        
        logger.debug("Isotonic Regression 完了")
    
    def _predict_platt(self, y_pred: np.ndarray) -> np.ndarray:
        """Platt Scaling による予測"""
        X = self._logit(y_pred).reshape(-1, 1)
        proba = self._estimator.predict_proba(X)[:, 1]
        return self._clip_probabilities(proba)
    
    def _predict_isotonic(self, y_pred: np.ndarray) -> np.ndarray:
        """Isotonic Regression による予測"""
        proba = self._estimator.transform(y_pred)
        return self._clip_probabilities(proba)
