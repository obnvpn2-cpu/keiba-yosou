# calibration_usage_examples.py
# Calibration v2 の実用的な使用例とベストプラクティス

import numpy as np
import pandas as pd
from typing import Tuple

# 実際のプロジェクトでは以下のようにインポート
# from calibration_v2 import CalibrationConfig, ProbabilityCalibrator
# from model_utils import fit_calibrated_base_model


# ============================================================
# Example 1: 基本的な使い方
# ============================================================

def example_1_basic_usage():
    """最もシンプルな使用例"""
    print("=" * 60)
    print("Example 1: 基本的な使い方")
    print("=" * 60)
    
    from calibration_v2 import ProbabilityCalibrator, CalibrationConfig
    
    # サンプルデータ生成
    np.random.seed(42)
    n = 1000
    
    # 過信されたモデル（実際の確率より高く出力）
    y_true = np.random.binomial(1, 0.3, n)
    y_pred_overconfident = np.clip(
        np.random.beta(2, 5, n) * 1.5,  # 意図的に過信
        0.01, 0.99
    )
    
    # キャリブレータの作成と学習
    config = CalibrationConfig(method="platt", n_bins=10)
    calibrator = ProbabilityCalibrator(config)
    calibrator.fit(y_pred_overconfident, y_true)
    
    # 予測
    y_calibrated = calibrator.predict(y_pred_overconfident)
    
    # 評価
    metrics = calibrator.evaluate(y_pred_overconfident, y_true)
    
    print(f"\n📊 キャリブレーション結果:")
    print(f"  ECE (Before): {metrics['ece_raw']:.6f}")
    print(f"  ECE (After):  {metrics['ece_calibrated']:.6f}")
    print(f"  改善率: {(1 - metrics['ece_calibrated']/metrics['ece_raw'])*100:.1f}%")
    print(f"\n  Brier (Before): {metrics['brier_raw']:.6f}")
    print(f"  Brier (After):  {metrics['brier_calibrated']:.6f}")
    print(f"  改善率: {(1 - metrics['brier_calibrated']/metrics['brier_raw'])*100:.1f}%")
    print()


# ============================================================
# Example 2: 3つの手法の比較
# ============================================================

def example_2_compare_methods():
    """Identity / Platt / Isotonic の比較"""
    print("=" * 60)
    print("Example 2: 3つの手法の比較")
    print("=" * 60)
    
    from calibration_v2 import ProbabilityCalibrator, CalibrationConfig
    
    # サンプルデータ
    np.random.seed(42)
    n = 2000
    y_true = np.random.binomial(1, 0.4, n)
    y_pred = np.clip(np.random.beta(3, 5, n) * 1.3, 0.01, 0.99)
    
    methods = ["identity", "platt", "isotonic"]
    results = {}
    
    for method in methods:
        config = CalibrationConfig(method=method, n_bins=15)
        calibrator = ProbabilityCalibrator(config)
        calibrator.fit(y_pred, y_true)
        
        metrics = calibrator.evaluate(y_pred, y_true)
        results[method] = metrics
    
    # 結果表示
    print(f"\n{'Method':<12} {'ECE (Raw)':<12} {'ECE (Cal)':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for method, metrics in results.items():
        improvement = (1 - metrics['ece_calibrated']/metrics['ece_raw']) * 100
        print(
            f"{method:<12} "
            f"{metrics['ece_raw']:<12.6f} "
            f"{metrics['ece_calibrated']:<12.6f} "
            f"{improvement:>10.1f}%"
        )
    
    print("\n💡 Tip:")
    print("  - Identity: ベースライン（何もしない）")
    print("  - Platt: パラメトリック（シグモイド関数でフィット）")
    print("  - Isotonic: ノンパラメトリック（より柔軟だがオーバーフィットしやすい）")
    print()


# ============================================================
# Example 3: 保存と読み込み
# ============================================================

def example_3_save_load():
    """モデルの保存と読み込み"""
    print("=" * 60)
    print("Example 3: 保存と読み込み")
    print("=" * 60)
    
    from calibration_v2 import ProbabilityCalibrator, CalibrationConfig
    import os
    import tempfile
    
    # 一時ファイル
    temp_file = os.path.join(tempfile.gettempdir(), "calibrator_temp.pkl")
    
    # 学習
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_pred = np.clip(np.random.beta(2, 5, 1000), 0.01, 0.99)
    
    calibrator = ProbabilityCalibrator(CalibrationConfig(method="platt"))
    calibrator.fit(y_pred, y_true)
    
    # 保存
    calibrator.save(temp_file)
    print(f"✅ 保存完了: {temp_file}")
    
    # 読み込み
    calibrator2 = ProbabilityCalibrator()
    calibrator2.load(temp_file)
    print(f"✅ 読み込み完了")
    
    # 同じ結果が得られることを確認
    y_cal_1 = calibrator.predict(y_pred[:10])
    y_cal_2 = calibrator2.predict(y_pred[:10])
    
    assert np.allclose(y_cal_1, y_cal_2), "読み込み後の予測が一致しません"
    print("✅ 検証OK: 保存前後で同じ予測結果")
    
    # クリーンアップ
    os.remove(temp_file)
    print()


# ============================================================
# Example 4: Reliability Curve の取得
# ============================================================

def example_4_reliability_curve():
    """リライアビリティカーブの取得と可視化（matplotlib使用）"""
    print("=" * 60)
    print("Example 4: Reliability Curve")
    print("=" * 60)
    
    from calibration_v2 import ProbabilityCalibrator, CalibrationConfig
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib が必要です。スキップします。")
        return
    
    # サンプルデータ
    np.random.seed(42)
    n = 5000
    y_true = np.random.binomial(1, 0.4, n)
    y_pred = np.clip(np.random.beta(3, 5, n) * 1.2, 0.01, 0.99)
    
    # キャリブレーション前後
    calibrator = ProbabilityCalibrator(CalibrationConfig(method="isotonic"))
    calibrator.fit(y_pred, y_true)
    y_cal = calibrator.predict(y_pred)
    
    # Reliability Curve
    bin_centers_raw, bin_acc_raw, bin_conf_raw = calibrator.get_reliability_curve(
        y_true, y_pred, n_bins=10
    )
    bin_centers_cal, bin_acc_cal, bin_conf_cal = calibrator.get_reliability_curve(
        y_true, y_cal, n_bins=10
    )
    
    # プロット
    plt.figure(figsize=(10, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    
    # Before calibration
    mask_raw = ~np.isnan(bin_acc_raw)
    plt.plot(
        bin_conf_raw[mask_raw], bin_acc_raw[mask_raw],
        'o-', label='Before Calibration', markersize=8
    )
    
    # After calibration
    mask_cal = ~np.isnan(bin_acc_cal)
    plt.plot(
        bin_conf_cal[mask_cal], bin_acc_cal[mask_cal],
        's-', label='After Calibration', markersize=8
    )
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Reliability Curve (Calibration Plot)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = "/mnt/user-data/outputs/reliability_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ プロット保存: {output_path}")
    plt.close()
    print()


# ============================================================
# Example 5: 競馬データでの実用例
# ============================================================

def example_5_horse_racing_calibration():
    """競馬の勝率予測モデルのキャリブレーション"""
    print("=" * 60)
    print("Example 5: 競馬データでの実用例（シミュレーション）")
    print("=" * 60)
    
    from calibration_v2 import ProbabilityCalibrator, CalibrationConfig
    
    # 競馬データをシミュレーション
    np.random.seed(42)
    
    # レース数とモデルの予測
    n_races = 500
    n_horses_per_race = 16
    n_total = n_races * n_horses_per_race
    
    # 勝率予測（多くの馬は低確率、一部が高確率）
    # 実際のモデル出力を模擬: 大半が 0.01-0.15、たまに 0.3-0.7
    y_pred = np.concatenate([
        np.random.beta(1, 20, int(n_total * 0.8)),  # 大半は低確率
        np.random.beta(3, 5, int(n_total * 0.2))   # 一部は中〜高確率
    ])
    np.random.shuffle(y_pred)
    y_pred = np.clip(y_pred, 0.001, 0.999)
    
    # 実際の勝ち（各レースで1頭のみ勝利）
    y_true = np.zeros(n_total)
    for i in range(n_races):
        race_probs = y_pred[i*n_horses_per_race:(i+1)*n_horses_per_race]
        # 確率に比例して勝者を決定
        winner_idx = np.random.choice(
            n_horses_per_race,
            p=race_probs / race_probs.sum()
        )
        y_true[i*n_horses_per_race + winner_idx] = 1
    
    # Train/Val 分割
    split_idx = int(n_total * 0.7)
    y_pred_train = y_pred[:split_idx]
    y_true_train = y_true[:split_idx]
    y_pred_val = y_pred[split_idx:]
    y_true_val = y_true[split_idx:]
    
    # キャリブレーション（Isotonic を使用 - 競馬は非線形性が高いため）
    config = CalibrationConfig(method="isotonic", n_bins=20, min_samples_bin=30)
    calibrator = ProbabilityCalibrator(config)
    calibrator.fit(y_pred_train, y_true_train)
    
    # 評価
    metrics = calibrator.evaluate(y_pred_val, y_true_val)
    
    print(f"\n📊 競馬勝率予測のキャリブレーション結果:")
    print(f"  検証データ: {len(y_true_val)} 頭分")
    print(f"  実際の勝馬: {int(y_true_val.sum())} 頭")
    print(f"\n  ECE (Before): {metrics['ece_raw']:.6f}")
    print(f"  ECE (After):  {metrics['ece_calibrated']:.6f}")
    print(f"  改善: {(metrics['ece_raw'] - metrics['ece_calibrated']):.6f}")
    print(f"\n  Brier (Before): {metrics['brier_raw']:.6f}")
    print(f"  Brier (After):  {metrics['brier_calibrated']:.6f}")
    
    # サンプル予測
    print(f"\n📝 サンプル予測（最初の5頭）:")
    print(f"{'Raw Prob':<12} {'Calibrated':<12} {'Actual':<8}")
    print("-" * 35)
    
    y_cal_val = calibrator.predict(y_pred_val)
    for i in range(min(5, len(y_pred_val))):
        print(
            f"{y_pred_val[i]:<12.4f} "
            f"{y_cal_val[i]:<12.4f} "
            f"{int(y_true_val[i]):<8}"
        )
    
    print("\n💡 実用上の注意:")
    print("  - キャリブレーションはランク順位を変えない（AUCは不変）")
    print("  - 確率の「絶対値」の信頼性が向上する")
    print("  - 賭け戦略（オッズとの比較等）で特に重要")
    print()


# ============================================================
# Example 6: ビン数と min_samples_bin の影響
# ============================================================

def example_6_hyperparameter_tuning():
    """ハイパーパラメータの影響を確認"""
    print("=" * 60)
    print("Example 6: ハイパーパラメータの影響")
    print("=" * 60)
    
    from calibration_v2 import ProbabilityCalibrator, CalibrationConfig
    
    # サンプルデータ
    np.random.seed(42)
    n = 5000
    y_true = np.random.binomial(1, 0.3, n)
    y_pred = np.clip(np.random.beta(2, 5, n) * 1.4, 0.01, 0.99)
    
    # 各設定でのECEを比較
    results = []
    
    for n_bins in [5, 10, 15, 20]:
        for min_samples in [10, 20, 50]:
            config = CalibrationConfig(
                method="platt",
                n_bins=n_bins,
                min_samples_bin=min_samples
            )
            calibrator = ProbabilityCalibrator(config)
            calibrator.fit(y_pred, y_true)
            
            metrics = calibrator.evaluate(y_pred, y_true)
            results.append({
                'n_bins': n_bins,
                'min_samples': min_samples,
                'ece': metrics['ece_calibrated']
            })
    
    # 結果表示
    print(f"\n{'n_bins':<10} {'min_samples':<15} {'ECE (Calibrated)':<20}")
    print("-" * 50)
    
    for r in results:
        print(
            f"{r['n_bins']:<10} "
            f"{r['min_samples']:<15} "
            f"{r['ece']:<20.6f}"
        )
    
    print("\n💡 Tip:")
    print("  - n_bins が多いほど細かく評価できるが、各ビンのサンプルが減る")
    print("  - min_samples_bin を大きくすると安定するが、ビンが少なくなる")
    print("  - データサイズに応じて調整が必要")
    print()


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Calibration v2 使用例集")
    print("=" * 60 + "\n")
    
    # 各例を実行
    example_1_basic_usage()
    example_2_compare_methods()
    example_3_save_load()
    example_4_reliability_curve()
    example_5_horse_racing_calibration()
    example_6_hyperparameter_tuning()
    
    print("=" * 60)
    print("全ての例の実行が完了しました！")
    print("=" * 60)
