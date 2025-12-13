# model_utils.py (v2)
# BaseWinModel と ProbabilityCalibrator の統合ユーティリティ
# 
# v1 からの改善点:
# - インポート修正（calibration → calibration_v3）
# - print → logger に変更
# - calib_df パラメータ追加（データリーク対策）
# - より詳細な警告とログ

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from calibration_v3 import CalibrationConfig, ProbabilityCalibrator

logger = logging.getLogger(__name__)

# デフォルトのログハンドラ設定
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def fit_calibrated_base_model(
    base_model: "BaseWinModel",
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    calib_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "win_flag",
    calibration_method: str = "platt",
    calibration_config: Optional[CalibrationConfig] = None,
) -> Tuple["BaseWinModel", ProbabilityCalibrator, Dict[str, float]]:
    """
    BaseWinModel と ProbabilityCalibrator をまとめて学習するヘルパー関数。
    
    ワークフロー:
        1. base_model を train_df で学習（validation として val_df を使用）
        2. キャリブレーション用データ（calib_df または val_df）で生予測確率を取得
        3. 生予測確率と正解ラベルで calibrator を学習
        4. キャリブレーション前後の評価指標を計算
    
    Args:
        base_model: BaseWinModel インスタンス（未学習の状態）
        train_df: 学習用データフレーム
        val_df: 検証用データフレーム（モデルの early stopping 等に使用）
        calib_df: キャリブレーション専用データフレーム。None の場合は val_df を使用（非推奨）
        feature_cols: 特徴量カラムのリスト。None の場合は自動検出。
        target_col: ターゲットカラム名（デフォルト: "win_flag"）
        calibration_method: キャリブレーション手法
            - "identity": 何もしない
            - "platt": Platt Scaling
            - "isotonic": Isotonic Regression
        calibration_config: CalibrationConfig インスタンス。
            None の場合は calibration_method から自動生成。
    
    Returns:
        (fitted_model, calibrator, metrics) のタプル:
            - fitted_model: 学習済み BaseWinModel
            - calibrator: 学習済み ProbabilityCalibrator
            - metrics: キャリブレーション前後の評価指標
                {
                    "ece_raw": float,
                    "ece_calibrated": float,
                    "brier_raw": float,
                    "brier_calibrated": float,
                }
    
    Examples:
        >>> from base_win_model import BaseWinModel
        >>> 
        >>> # 推奨: キャリブレーション専用データを用意
        >>> model = BaseWinModel()
        >>> fitted_model, calibrator, metrics = fit_calibrated_base_model(
        ...     base_model=model,
        ...     train_df=train_df,
        ...     val_df=val_df,
        ...     calib_df=calib_df,  # 別途用意
        ...     calibration_method="platt"
        ... )
        >>> 
        >>> # 非推奨: val_df を使い回す場合
        >>> fitted_model, calibrator, metrics = fit_calibrated_base_model(
        ...     base_model=model,
        ...     train_df=train_df,
        ...     val_df=val_df,
        ...     calib_df=None,  # データリークのリスクあり
        ...     calibration_method="platt"
        ... )
    
    Warning:
        - calib_df を指定しない場合、val_df がモデル学習とキャリブレーション学習の
          両方に使用されます。これは **データリークの可能性** があります。
        - 特に、BaseWinModel が early stopping 等で val_df を使用する場合、
          キャリブレーションの性能が過大評価される可能性があります。
        - 本番環境では、必ず calib_df を別途用意することを推奨します。
    
    Note:
        - キャリブレーション後は識別性能（AUC等）は変わらないが、
          確率の信頼性（ECE, Brier Score）が改善される
    """
    # 特徴量カラムの自動検出
    if feature_cols is None:
        # ID系・ラベル系を除いた列を自動採用
        exclude = {"horse_id", "race_id", target_col}
        feature_cols = [c for c in train_df.columns if c not in exclude]
        logger.info(f"特徴量を自動検出しました: {len(feature_cols)} 列")

    # 1. BaseWinModel 学習
    logger.info(f"BaseWinModel の学習を開始... (train={len(train_df)}, val={len(val_df)})")
    base_model.fit(
        train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        val_df=val_df,
    )
    logger.info("BaseWinModel の学習が完了しました。")

    # 2. キャリブレーション用データの決定
    if calib_df is None:
        warnings.warn(
            "calib_df が指定されていないため、val_df をキャリブレーション学習にも使用します。"
            "これは データリークのリスク があります。"
            "本番環境では、キャリブレーション専用データ（calib_df）の指定を強く推奨します。"
            "詳細: https://scikit-learn.org/stable/modules/calibration.html",
            category=UserWarning
        )
        calib_data = val_df
        logger.warning("データリークのリスク: val_df をキャリブレーションにも使用しています")
    else:
        calib_data = calib_df
        logger.info(f"キャリブレーション専用データを使用: {len(calib_df)} サンプル")

    # 3. キャリブレーション用データでの生予測確率を取得
    logger.info("キャリブレーション用データでの予測確率を取得...")
    y_calib = calib_data[target_col].astype(int).values
    raw_probs_calib = base_model.predict_proba(calib_data)
    logger.info(
        f"キャリブレーションデータ: {len(y_calib)} サンプル, "
        f"正例率: {y_calib.mean():.4f}"
    )

    # 4. キャリブレータ学習
    logger.info(f"キャリブレーション学習を開始 (method={calibration_method})...")
    
    if calibration_config is None:
        calibration_config = CalibrationConfig(method=calibration_method)
    
    calibrator = ProbabilityCalibrator(calibration_config)
    calibrator.fit(raw_probs_calib, y_calib)
    logger.info("キャリブレーション学習が完了しました。")

    # 5. キャリブレーション前後の指標を計算
    logger.info("評価指標を計算中...")
    metrics = calibrator.evaluate(raw_probs_calib, y_calib)
    
    # 結果サマリーを表示
    logger.info("\n" + "=" * 60)
    logger.info("キャリブレーション結果")
    logger.info("=" * 60)
    logger.info(f"ECE (Expected Calibration Error):")
    logger.info(f"  Before: {metrics['ece_raw']:.6f}")
    logger.info(f"  After:  {metrics['ece_calibrated']:.6f}")
    logger.info(f"  改善:   {metrics['ece_raw'] - metrics['ece_calibrated']:.6f}")
    logger.info("")
    logger.info(f"Brier Score:")
    logger.info(f"  Before: {metrics['brier_raw']:.6f}")
    logger.info(f"  After:  {metrics['brier_calibrated']:.6f}")
    logger.info(f"  改善:   {metrics['brier_raw'] - metrics['brier_calibrated']:.6f}")
    logger.info("=" * 60 + "\n")

    return base_model, calibrator, metrics


def apply_calibration_to_predictions(
    calibrator: ProbabilityCalibrator,
    predictions_df: pd.DataFrame,
    prob_col: str = "win_prob",
    output_col: str = "win_prob_calibrated",
    check_distribution: bool = True,
) -> pd.DataFrame:
    """
    予測結果データフレームにキャリブレーションを適用。
    
    Args:
        calibrator: 学習済み ProbabilityCalibrator
        predictions_df: 予測結果を含むデータフレーム
        prob_col: 生予測確率のカラム名
        output_col: キャリブレート後の確率を格納するカラム名
        check_distribution: True の場合、学習時と分布が大きく異なる場合に警告
    
    Returns:
        キャリブレート後の確率が追加されたデータフレーム
    
    Raises:
        ValueError: calibrator が未学習、または prob_col が存在しない場合
    
    Examples:
        >>> predictions = model.predict_proba(test_df)
        >>> result_df = pd.DataFrame({
        ...     'horse_id': test_df['horse_id'],
        ...     'win_prob': predictions
        ... })
        >>> result_df = apply_calibration_to_predictions(
        ...     calibrator, result_df
        ... )
        >>> print(result_df[['horse_id', 'win_prob', 'win_prob_calibrated']])
    """
    if not hasattr(calibrator, '_fitted') or not calibrator._fitted:
        raise ValueError(
            "calibrator は学習済みである必要があります。"
            "先に calibrator.fit() を実行してください。"
        )
    
    result_df = predictions_df.copy()
    
    if prob_col not in result_df.columns:
        raise ValueError(
            f"カラム '{prob_col}' が見つかりません。"
            f"利用可能なカラム: {list(result_df.columns)}"
        )
    
    logger.info(f"キャリブレーションを適用中... ({len(result_df)} 行)")
    
    raw_probs = result_df[prob_col].values
    calibrated_probs = calibrator.predict(raw_probs, check_distribution=check_distribution)
    
    result_df[output_col] = calibrated_probs
    
    logger.info(
        f"キャリブレーション完了: "
        f"平均確率 {raw_probs.mean():.4f} → {calibrated_probs.mean():.4f}"
    )
    
    return result_df


def compare_calibration_methods(
    y_pred: pd.Series,
    y_true: pd.Series,
    methods: Optional[List[str]] = None,
    n_bins: int = 15,
) -> pd.DataFrame:
    """
    複数のキャリブレーション手法を比較。
    
    Args:
        y_pred: 予測確率
        y_true: 正解ラベル
        methods: 比較する手法のリスト（デフォルト: ["identity", "platt", "isotonic"]）
        n_bins: ECE 計算用のビン数
    
    Returns:
        各手法の評価指標を含むデータフレーム
    
    Examples:
        >>> results = compare_calibration_methods(
        ...     y_pred=val_df['win_prob'],
        ...     y_true=val_df['win_flag'],
        ...     methods=["platt", "isotonic"]
        ... )
        >>> print(results)
               method  ece_raw  ece_calibrated  brier_raw  brier_calibrated
        0      platt   0.0234          0.0087     0.1523            0.1489
        1   isotonic   0.0234          0.0052     0.1523            0.1478
    """
    if methods is None:
        methods = ["identity", "platt", "isotonic"]
    
    logger.info(f"キャリブレーション手法を比較中: {methods}")
    
    results = []
    
    for method in methods:
        logger.info(f"  - {method} を評価中...")
        
        config = CalibrationConfig(method=method, n_bins=n_bins)
        calibrator = ProbabilityCalibrator(config)
        
        try:
            calibrator.fit(y_pred.values, y_true.values)
            metrics = calibrator.evaluate(y_pred.values, y_true.values)
            
            results.append({
                "method": method,
                "ece_raw": metrics["ece_raw"],
                "ece_calibrated": metrics["ece_calibrated"],
                "brier_raw": metrics["brier_raw"],
                "brier_calibrated": metrics["brier_calibrated"],
                "ece_improvement": metrics["ece_raw"] - metrics["ece_calibrated"],
                "brier_improvement": metrics["brier_raw"] - metrics["brier_calibrated"],
            })
        except Exception as e:
            logger.warning(f"  - {method} の評価に失敗: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # 改善度でソート
    if len(results_df) > 0:
        results_df = results_df.sort_values("ece_improvement", ascending=False)
    
    logger.info("\n比較結果:")
    logger.info(results_df.to_string(index=False))
    
    return results_df
