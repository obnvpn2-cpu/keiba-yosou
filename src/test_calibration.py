# test_calibration_v3.py
# Calibration v3 の包括的な単体テスト
# v2 のテストを拡張し、新機能もカバー

import pytest
import numpy as np
import tempfile
import os
import warnings
from typing import Tuple

# テスト対象のインポート
# from calibration_v3 import CalibrationConfig, ProbabilityCalibrator


# ============================================================
# テスト用データ生成
# ============================================================

def generate_test_data(
    n: int = 1000,
    seed: int = 42,
    overconfident: bool = True,
    extreme: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    テスト用のサンプルデータを生成
    
    Args:
        n: サンプル数
        seed: 乱数シード
        overconfident: True の場合、過信された予測を生成
        extreme: True の場合、極端に偏ったデータを生成
    
    Returns:
        (y_pred, y_true) のタプル
    """
    np.random.seed(seed)
    
    if extreme:
        # 極端に偏ったデータ
        p_true = 0.005  # 0.5% の正例率
        y_true = np.random.binomial(1, p_true, n)
        y_pred = np.clip(np.random.beta(1, 100, n), 0.01, 0.99)
    else:
        # 通常のデータ
        p_true = 0.3
        y_true = np.random.binomial(1, p_true, n)
        
        if overconfident:
            # 過信されたモデル
            y_pred = np.clip(np.random.beta(2, 5, n) * 1.5, 0.01, 0.99)
        else:
            # 適切なモデル
            y_pred = np.clip(np.random.beta(3, 7, n), 0.01, 0.99)
    
    return y_pred, y_true


# ============================================================
# CalibrationConfig のテスト
# ============================================================

class TestCalibrationConfigV3:
    """CalibrationConfig v3 のテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        from calibration_v3 import CalibrationConfig
        
        config = CalibrationConfig()
        assert config.method == "platt"
        assert config.n_bins == 15
        assert config.min_samples_bin == 20
        assert config.eps == 1e-7
        assert config.min_samples_isotonic == 500
        assert config.extreme_distribution_threshold == 0.01
    
    def test_custom_config(self):
        """カスタム設定のテスト"""
        from calibration_v3 import CalibrationConfig
        
        config = CalibrationConfig(
            method="isotonic",
            n_bins=20,
            min_samples_bin=30,
            eps=1e-6,
            min_samples_isotonic=1000,
            extreme_distribution_threshold=0.02,
        )
        assert config.method == "isotonic"
        assert config.n_bins == 20
        assert config.min_samples_bin == 30
        assert config.eps == 1e-6
        assert config.min_samples_isotonic == 1000
        assert config.extreme_distribution_threshold == 0.02
    
    def test_invalid_min_samples_isotonic(self):
        """無効な min_samples_isotonic でエラー"""
        from calibration_v3 import CalibrationConfig
        
        with pytest.raises(ValueError, match="min_samples_isotonic は"):
            CalibrationConfig(min_samples_isotonic=5)
    
    def test_invalid_extreme_threshold(self):
        """無効な extreme_distribution_threshold でエラー"""
        from calibration_v3 import CalibrationConfig
        
        with pytest.raises(ValueError, match="extreme_distribution_threshold は"):
            CalibrationConfig(extreme_distribution_threshold=0.6)


# ============================================================
# ProbabilityCalibrator v3 の新機能テスト
# ============================================================

class TestProbabilityCalibratorV3NewFeatures:
    """v3 の新機能テスト"""
    
    def test_fit_with_force_parameter(self):
        """force パラメータによる再学習制御"""
        from calibration_v3 import ProbabilityCalibrator
        
        y_pred1, y_true1 = generate_test_data(n=500, seed=42)
        y_pred2, y_true2 = generate_test_data(n=500, seed=100)
        
        calibrator = ProbabilityCalibrator()
        
        # 最初の学習
        calibrator.fit(y_pred1, y_true1)
        assert calibrator._fitted
        
        # force=False で再学習しようとするとエラー
        with pytest.raises(ValueError, match="既に学習済み"):
            calibrator.fit(y_pred2, y_true2, force=False)
        
        # force=True で再学習可能
        calibrator.fit(y_pred2, y_true2, force=True)
        assert calibrator._fitted
    
    def test_reset_method(self):
        """reset() メソッドのテスト"""
        from calibration_v3 import ProbabilityCalibrator
        
        y_pred, y_true = generate_test_data(n=500)
        
        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_pred, y_true)
        assert calibrator._fitted
        
        # リセット
        calibrator.reset()
        assert not calibrator._fitted
        assert calibrator._estimator is None
        assert calibrator._fit_metadata == {}
        
        # リセット後は再学習可能（force 不要）
        calibrator.fit(y_pred, y_true)
        assert calibrator._fitted
    
    def test_predict_with_distribution_check(self):
        """分布チェック機能のテスト"""
        from calibration_v3 import ProbabilityCalibrator
        
        # 学習データ
        y_pred_train, y_true_train = generate_test_data(n=1000, seed=42)
        
        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_pred_train, y_true_train)
        
        # 同じ分布のテストデータ（警告なし）
        y_pred_test_same = generate_test_data(n=100, seed=43)[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_cal = calibrator.predict(y_pred_test_same, check_distribution=True)
            # 分布が似ているので警告は出ないはず
            assert len(y_cal) == len(y_pred_test_same)
        
        # 大きく異なる分布のテストデータ（警告あり）
        y_pred_test_diff = np.random.beta(10, 1, 100)  # 平均が高い
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_cal = calibrator.predict(y_pred_test_diff, check_distribution=True)
            # 分布が異なるので警告が出る可能性
            # （ただし、2σ以内なら警告なし）
    
    def test_extreme_distribution_warning(self):
        """極端な分布での警告"""
        from calibration_v3 import ProbabilityCalibrator
        
        # 極端に偏ったデータ
        y_pred, y_true = generate_test_data(n=1000, extreme=True)
        
        calibrator = ProbabilityCalibrator()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calibrator.fit(y_pred, y_true)
            
            # 極端な分布の警告が出ることを確認
            assert len(w) > 0
            assert any("極端に偏っています" in str(warning.message) for warning in w)
    
    def test_isotonic_small_sample_warning(self):
        """Isotonic の少サンプル警告"""
        from calibration_v3 import ProbabilityCalibrator, CalibrationConfig
        
        # 少サンプルデータ
        y_pred, y_true = generate_test_data(n=200)  # < 500
        
        config = CalibrationConfig(method="isotonic")
        calibrator = ProbabilityCalibrator(config)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calibrator.fit(y_pred, y_true)
            
            # Isotonic の少サンプル警告が出ることを確認
            assert len(w) > 0
            assert any("推奨値" in str(warning.message) for warning in w)
    
    def test_evaluate_requires_fit(self):
        """evaluate() が fit チェックすることを確認"""
        from calibration_v3 import ProbabilityCalibrator
        
        y_pred, y_true = generate_test_data(n=500)
        
        calibrator = ProbabilityCalibrator()
        
        # fit 前に evaluate するとエラー
        with pytest.raises(ValueError, match="まだ fit"):
            calibrator.evaluate(y_pred, y_true)
        
        # fit 後は OK
        calibrator.fit(y_pred, y_true)
        metrics = calibrator.evaluate(y_pred, y_true)
        assert "ece_raw" in metrics


# ============================================================
# メタデータのテスト
# ============================================================

class TestMetadataV3:
    """v3 の拡張メタデータのテスト"""
    
    def test_metadata_statistics(self):
        """統計情報がメタデータに含まれることを確認"""
        from calibration_v3 import ProbabilityCalibrator
        
        y_pred, y_true = generate_test_data(n=1000)
        
        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_pred, y_true)
        
        metadata = calibrator._fit_metadata
        
        # 基本情報
        assert "n_samples" in metadata
        assert "fit_timestamp" in metadata
        assert "method" in metadata
        
        # y_pred の統計情報
        assert "y_pred_stats" in metadata
        y_pred_stats = metadata["y_pred_stats"]
        assert "mean" in y_pred_stats
        assert "std" in y_pred_stats
        assert "min" in y_pred_stats
        assert "max" in y_pred_stats
        assert "median" in y_pred_stats
        assert "q25" in y_pred_stats
        assert "q75" in y_pred_stats
        
        # y_true の統計情報
        assert "y_true_stats" in metadata
        y_true_stats = metadata["y_true_stats"]
        assert "positive_rate" in y_true_stats
        assert "n_positive" in y_true_stats
        assert "n_negative" in y_true_stats
        
        # 値の妥当性チェック
        assert y_pred_stats["mean"] == pytest.approx(np.mean(y_pred), rel=1e-5)
        assert y_true_stats["positive_rate"] == pytest.approx(np.mean(y_true), rel=1e-5)


# ============================================================
# 保存/読み込みのテスト（v3 拡張）
# ============================================================

class TestSaveLoadV3:
    """v3 の拡張版保存/読み込みテスト"""
    
    def test_save_includes_version_info(self):
        """保存時にバージョン情報が含まれることを確認"""
        from calibration_v3 import ProbabilityCalibrator
        import pickle
        import sys
        
        y_pred, y_true = generate_test_data(n=500)
        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_pred, y_true)
        
        # 保存
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            calibrator.save(temp_path)
            
            # pickle で直接読み込んで内容確認
            with open(temp_path, "rb") as f:
                obj = pickle.load(f)
            
            assert "version" in obj
            assert "python_version" in obj
            assert "sklearn_version" in obj
            assert "numpy_version" in obj
            assert "save_timestamp" in obj
            
            # Python バージョンが正しいことを確認
            expected_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            assert obj["python_version"] == expected_python
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_load_version_warning(self):
        """異なるバージョンでの読み込み時に警告が出ることを確認"""
        from calibration_v3 import ProbabilityCalibrator
        import pickle
        
        y_pred, y_true = generate_test_data(n=500)
        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_pred, y_true)
        
        # 保存
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            calibrator.save(temp_path)
            
            # 保存データを改変（バージョンを変更）
            with open(temp_path, "rb") as f:
                obj = pickle.load(f)
            
            obj["version"] = "9.9.9"  # 存在しないバージョン
            obj["sklearn_version"] = "0.0.1"  # 古いバージョン
            
            with open(temp_path, "wb") as f:
                pickle.dump(obj, f)
            
            # 読み込み時に警告が出ることを確認
            calibrator2 = ProbabilityCalibrator()
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                calibrator2.load(temp_path)
                
                # バージョン警告が出ることを確認
                assert len(w) >= 2  # version と sklearn_version の警告
                
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================
# ECE 計算の v3 機能テスト
# ============================================================

class TestComputeECEV3:
    """v3 の ECE 計算拡張機能テスト"""
    
    def test_ece_bin_exclusion_warning(self):
        """ビン除外警告のテスト"""
        from calibration_v3 import ProbabilityCalibrator
        
        # 少サンプルデータで多くのビンを作る
        y_true = np.random.binomial(1, 0.3, 100)
        y_prob = np.random.uniform(0.1, 0.9, 100)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # n_bins を大きく、min_samples_bin も大きくすると警告
            ece = ProbabilityCalibrator.compute_ece(
                y_true, y_prob,
                n_bins=20,  # 多い
                min_samples_bin=50  # 大きい
            )
            
            # ビン除外の警告が出るはず
            assert len(w) > 0
            assert any("ビンがサンプル不足" in str(warning.message) for warning in w)


# ============================================================
# 統合テスト
# ============================================================

class TestIntegrationV3:
    """v3 の統合テスト"""
    
    def test_full_workflow_with_new_features(self):
        """v3 の新機能を使った完全なワークフロー"""
        from calibration_v3 import ProbabilityCalibrator, CalibrationConfig
        
        # データ生成
        y_pred_train, y_true_train = generate_test_data(n=2000, seed=42)
        y_pred_test, y_true_test = generate_test_data(n=500, seed=100)
        
        # 設定
        config = CalibrationConfig(
            method="platt",
            n_bins=10,
            min_samples_bin=30,
        )
        
        # キャリブレータ作成
        calibrator = ProbabilityCalibrator(config)
        
        # 学習
        calibrator.fit(y_pred_train, y_true_train)
        
        # メタデータ確認
        assert calibrator._fit_metadata["n_samples"] == 2000
        
        # 予測（分布チェック付き）
        y_cal = calibrator.predict(y_pred_test, check_distribution=True)
        assert len(y_cal) == len(y_pred_test)
        
        # 評価
        metrics = calibrator.evaluate(y_pred_test, y_true_test)
        assert "ece_calibrated" in metrics
        
        # 保存
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            calibrator.save(temp_path)
            
            # 読み込み
            calibrator2 = ProbabilityCalibrator()
            calibrator2.load(temp_path)
            
            # 同じ予測結果が得られることを確認
            y_cal2 = calibrator2.predict(y_pred_test, check_distribution=False)
            assert np.allclose(y_cal, y_cal2)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_reset_and_refit(self):
        """reset() して再学習するワークフロー"""
        from calibration_v3 import ProbabilityCalibrator
        
        y_pred1, y_true1 = generate_test_data(n=500, seed=42)
        y_pred2, y_true2 = generate_test_data(n=500, seed=100)
        
        calibrator = ProbabilityCalibrator()
        
        # 1回目の学習
        calibrator.fit(y_pred1, y_true1)
        metrics1 = calibrator.evaluate(y_pred1, y_true1)
        
        # リセット
        calibrator.reset()
        
        # 2回目の学習（force 不要）
        calibrator.fit(y_pred2, y_true2)
        metrics2 = calibrator.evaluate(y_pred2, y_true2)
        
        # メトリクスが異なることを確認（異なるデータなので）
        # （必ずしも異なるとは限らないが、確率的には異なるはず）


# ============================================================
# パフォーマンステスト
# ============================================================

class TestPerformanceV3:
    """v3 のパフォーマンステスト"""
    
    def test_large_dataset_with_metadata(self):
        """大規模データでのメタデータ記録パフォーマンス"""
        from calibration_v3 import ProbabilityCalibrator
        import time
        
        # 100万サンプル
        y_pred, y_true = generate_test_data(n=1_000_000)
        
        calibrator = ProbabilityCalibrator()
        
        start = time.time()
        calibrator.fit(y_pred, y_true)
        fit_time = time.time() - start
        
        # メタデータが正しく記録されていることを確認
        assert calibrator._fit_metadata["n_samples"] == 1_000_000
        assert "y_pred_stats" in calibrator._fit_metadata
        
        print(f"\n大規模データテスト (n=1,000,000):")
        print(f"  fit 時間: {fit_time:.2f} 秒")
        print(f"  メタデータサイズ: {len(str(calibrator._fit_metadata))} 文字")


# ============================================================
# pytest 実行用
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Calibration v3 単体テスト")
    print("=" * 60)
    print("\n推奨: pytest でテストを実行してください")
    print("コマンド: pytest test_calibration_v3.py -v")
    print("\n個別のテストクラスを実行する場合:")
    print("pytest test_calibration_v3.py::TestProbabilityCalibratorV3NewFeatures -v")
    print("\n新機能のテストのみ実行:")
    print("pytest test_calibration_v3.py -k 'V3' -v")
    print("=" * 60)
