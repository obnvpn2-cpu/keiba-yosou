"""
BaseWinModel v1.1 - 統合使用例 + Calibration v3 (v2 - 改善版)

全モジュール連携:
1. HorseHistoryStore v2.0
2. RaceFeatureBuilder v5.0
3. BaseFeatureBuilder v2.0
4. BaseWinModel v1.1
5. Calibration v3 (Platt / Isotonic)

v1 からの改善点:
- データリーク修正（val でキャリブレーション）
- 特徴量構築の関数化
- 例外処理の強化
- ログ出力の改善
- 時系列分割の修正
- val_df を使った学習
- テストデータの評価追加
"""

# 標準ライブラリ
import logging
import traceback
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# サードパーティ
import pandas as pd
import numpy as np

# ローカルモジュール
from HorseHistoryStore import HorseHistoryStore
from race_feature_builder import RaceFeatureBuilder
from base_feature_builder import BaseFeatureBuilder
from base_model import BaseWinModel, create_win_labels
from calibration import CalibrationConfig, ProbabilityCalibrator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('base_model_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_dummy_data():
    """
    ダミーデータ作成（より現実的な相関を持たせる）
    """
    np.random.seed(42)

    # 馬の戦績データ（100レース分）
    n_races = 100
    horses_per_race = 16
    n_total = n_races * horses_per_race

    race_dates = pd.date_range("2024-01-01", periods=n_races, freq="D")

    performance_data = pd.DataFrame({
        "horse_id": [f"horse_{i % 50:03d}" for i in range(n_total)],
        "race_id": [f"race_{i // horses_per_race:04d}" for i in range(n_total)],
        "race_date": np.repeat(race_dates, horses_per_race),
        "race_datetime": np.repeat(
            pd.date_range("2024-01-01 14:00", periods=n_races, freq="D"),
            horses_per_race
        ),
        "track_code": np.random.choice(["東京", "中山", "阪神"], n_total),
        "course_type": np.random.choice(["芝", "ダート"], n_total),
        "distance": np.random.choice([1600, 1800, 2000], n_total),
        "field_size": np.repeat(np.random.randint(12, 19, n_races), horses_per_race),
        "corner1_pos": np.random.randint(1, 17, n_total),
        "final_3f_time": np.random.uniform(33, 38, n_total),
        "jockey_id": np.random.choice([f"jockey_{i}" for i in range(20)], n_total),
        "jockey_name": np.random.choice(["武豊", "ルメール", "デムーロ"], n_total),
        "jockey_weight": np.random.uniform(52, 58, n_total),
        "popularity": np.random.randint(1, 17, n_total),
        "remarks": np.random.choice(["", "", "", "", "出遅れ"], n_total),
        "age": np.random.randint(3, 8, n_total),
        "sex": np.random.choice(["牡", "牝", "騙"], n_total),
        "career_runs": np.random.randint(1, 30, n_total),
        "frame": [i % horses_per_race + 1 for i in range(n_total)],
        "horse_number": [i % horses_per_race + 1 for i in range(n_total)],
        "weight": np.random.uniform(52, 58, n_total),
        "rest_days": np.random.randint(7, 60, n_total),
    })

    # 人気と着順に相関を持たせる
    for race_id in performance_data['race_id'].unique():
        mask = performance_data['race_id'] == race_id
        race_df = performance_data[mask]
        
        # 人気順にソート
        sorted_indices = race_df.index[race_df['popularity'].argsort()]
        
        # 上位人気ほど好着順の確率を高く（ただしランダム性も残す）
        finish_probs = np.array([
            np.random.beta(2 + (16-i)*0.3, 5 + i*0.2)
            for i in range(1, len(sorted_indices) + 1)
        ])
        
        finish_positions = np.argsort(finish_probs) + 1
        performance_data.loc[sorted_indices, 'finish_position'] = finish_positions

    # オッズを人気から生成（人気が高いほどオッズが低い）
    for race_id in performance_data['race_id'].unique():
        mask = performance_data['race_id'] == race_id
        popularity = performance_data.loc[mask, 'popularity'].values
        
        # 人気から大まかなオッズを計算（1番人気=2倍程度、16番人気=50倍程度）
        base_odds = 1.5 + (popularity - 1) * 3.0
        noise = np.random.uniform(0.8, 1.2, len(popularity))
        odds = base_odds * noise
        
        performance_data.loc[mask, 'odds'] = odds

    # corner 位置に連続性を持たせる
    performance_data['corner2_pos'] = np.clip(
        performance_data['corner1_pos'] + np.random.randint(-2, 3, n_total),
        1, 16
    ).astype(int)
    
    performance_data['corner3_pos'] = np.clip(
        performance_data['corner2_pos'] + np.random.randint(-2, 3, n_total),
        1, 16
    ).astype(int)
    
    performance_data['corner4_pos'] = np.clip(
        performance_data['corner3_pos'] + np.random.randint(-2, 3, n_total),
        1, 16
    ).astype(int)

    # win_flagを追加
    performance_data["win_flag"] = create_win_labels(
        performance_data["finish_position"],
        positive_up_to=1
    )

    return performance_data


def build_features_for_races(
    race_ids: List[str],
    data: pd.DataFrame,
    history_store: HorseHistoryStore,
    race_builder: RaceFeatureBuilder,
    base_builder: BaseFeatureBuilder,
    dataset_name: str = "train",
) -> pd.DataFrame:
    """
    レースリストから特徴量を構築
    
    Args:
        race_ids: レースIDのリスト
        data: パフォーマンスデータ
        history_store: HorseHistoryStore インスタンス
        race_builder: RaceFeatureBuilder インスタンス
        base_builder: BaseFeatureBuilder インスタンス
        dataset_name: データセット名（ログ用）
    
    Returns:
        特徴量データフレーム
    """
    features_list = []
    failed_races = []
    
    logger.info(f"【{dataset_name}】特徴量構築を開始: {len(race_ids)} レース")
    start_time = time.time()
    
    for i, race_id in enumerate(race_ids):
        try:
            race_mask = data['race_id'] == race_id
            race_df = data[race_mask].copy()

            if len(race_df) == 0:
                logger.warning(f"レース {race_id} のデータが空です")
                continue
            
            # race_row の安全な構築
            first_row = race_df.iloc[0]
            
            race_row = {
                "race_id": race_id,
                "race_datetime": first_row["race_datetime"],
                "track_type": first_row["course_type"],
                "distance": first_row["distance"],
                "field_size": first_row["field_size"],
                "track_condition": first_row.get("track_condition", "良"),
                "course": first_row["track_code"],
                "turn_type": first_row.get("turn_type", "左回り"),
                "track_bias": first_row.get("track_bias", 0.0),
            }

            entries_df = race_df[[
                "horse_id", "jockey_id", "age", "sex", "career_runs",
                "frame", "horse_number", "weight", "rest_days", "odds"
            ]].copy()

            as_of = race_row["race_datetime"]

            # RaceFeatureBuilder
            race_feature_output = race_builder.build_for_race(
                race_row=race_row,
                entries_df=entries_df,
                as_of=as_of,
            )

            # BaseFeatureBuilder
            features_df = base_builder.build_features_for_race(
                entries_df=entries_df,
                race_row=race_row,
                as_of=as_of,
                race_feature_output=race_feature_output,
            )

            # ラベル追加
            features_df["race_id"] = race_id
            features_df["win_flag"] = race_df["win_flag"].values
            features_df["finish_position"] = race_df["finish_position"].values
            
            # odds カラムを追加（評価用）
            features_df["odds"] = race_df["odds"].values

            features_list.append(features_df)
            
            # プログレス表示（10レースごと）
            if (i + 1) % 10 == 0:
                logger.info(f"  進捗: {i+1}/{len(race_ids)} レース完了")
                
        except Exception as e:
            logger.error(
                f"レース {race_id} の特徴量構築に失敗: {e}\n"
                f"{traceback.format_exc()}"
            )
            failed_races.append({
                "race_id": race_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            continue

    if not features_list:
        raise ValueError(f"{dataset_name}: 全てのレースで特徴量構築に失敗しました")

    features = pd.concat(features_list, ignore_index=True)
    
    elapsed = time.time() - start_time
    logger.info(
        f"【{dataset_name}】特徴量構築完了: "
        f"{len(features)} 行, {len(features.columns)} 列, "
        f"{elapsed:.2f} 秒"
    )
    
    # 失敗レースの警告
    if failed_races:
        logger.warning(
            f"{len(failed_races)}/{len(race_ids)} レースで構築失敗 "
            f"({len(failed_races)/len(race_ids)*100:.1f}%)"
        )
        
        # 失敗が多い場合はファイルに保存
        if len(failed_races) > len(race_ids) * 0.1:
            import json
            error_log_path = f"failed_races_{dataset_name}.json"
            with open(error_log_path, "w") as f:
                json.dump(failed_races, f, indent=2, ensure_ascii=False)
            logger.warning(f"失敗レースの詳細を保存: {error_log_path}")
    
    return features


def main():
    """統合フロー実行"""
    
    overall_start = time.time()
    
    logger.info("=" * 80)
    logger.info("BaseWinModel v1.1 - 統合使用例 (+ Calibration v3) v2")
    logger.info("=" * 80)

    # ダミーデータ作成
    logger.info("\n【ステップ1】データ準備")
    performance_data = create_dummy_data()

    logger.info(f"  総レース数: {performance_data['race_id'].nunique()}")
    logger.info(f"  総走数: {len(performance_data)}")
    logger.info(f"  勝ち数: {performance_data['win_flag'].sum()}")

    # 学習・検証・テスト分割（時系列厳守）
    logger.info("\n【ステップ2】データ分割（時系列）")
    
    # race_date でソートしてから分割（Critical 修正）
    race_info = performance_data[['race_id', 'race_date']].drop_duplicates()
    race_info = race_info.sort_values('race_date')
    race_ids_sorted = race_info['race_id'].values
    
    n_races = len(race_ids_sorted)
    train_size = int(n_races * 0.6)
    val_size = int(n_races * 0.2)

    train_races = race_ids_sorted[:train_size]
    val_races = race_ids_sorted[train_size:train_size + val_size]
    test_races = race_ids_sorted[train_size + val_size:]

    train_data = performance_data[performance_data['race_id'].isin(train_races)]
    val_data = performance_data[performance_data['race_id'].isin(val_races)]
    test_data = performance_data[performance_data['race_id'].isin(test_races)]

    logger.info(f"  学習: {len(train_data)}走（{len(train_races)}レース）")
    logger.info(f"  検証: {len(val_data)}走（{len(val_races)}レース）")
    logger.info(f"  テスト: {len(test_data)}走（{len(test_races)}レース）")
    
    # 日付範囲の確認
    logger.info(f"  学習期間: {train_data['race_date'].min()} 〜 {train_data['race_date'].max()}")
    logger.info(f"  検証期間: {val_data['race_date'].min()} 〜 {val_data['race_date'].max()}")
    logger.info(f"  テスト期間: {test_data['race_date'].min()} 〜 {test_data['race_date'].max()}")

    # HorseHistoryStore初期化
    logger.info("\n【ステップ3】HorseHistoryStore初期化")
    history_store = HorseHistoryStore(performance_data)
    logger.info("  ✅ 完了")

    # RaceFeatureBuilder初期化
    logger.info("\n【ステップ4】RaceFeatureBuilder初期化")
    race_builder = RaceFeatureBuilder(history_store)
    logger.info("  ✅ 完了")

    # BaseFeatureBuilder初期化
    logger.info("\n【ステップ5】BaseFeatureBuilder初期化")
    base_builder = BaseFeatureBuilder(history_store)
    logger.info("  ✅ 完了")

    # 特徴量構築（関数化により重複削減）
    logger.info("\n【ステップ6】特徴量テーブル構築")
    
    train_features = build_features_for_races(
        race_ids=train_races,
        data=train_data,
        history_store=history_store,
        race_builder=race_builder,
        base_builder=base_builder,
        dataset_name="train",
    )
    
    val_features = build_features_for_races(
        race_ids=val_races,
        data=val_data,
        history_store=history_store,
        race_builder=race_builder,
        base_builder=base_builder,
        dataset_name="val",
    )

    # BaseWinModel初期化・学習
    logger.info("\n【ステップ7】BaseWinModel初期化・学習")
    model = BaseWinModel()

    # val_df を使って学習（Critical 修正: early stopping 有効化）
    model.fit(
        train_df=train_features,
        feature_cols=None,  # 自動推定
        target_col="win_flag",
        val_df=val_features,  # early stopping 有効化
    )

    logger.info(f"  ✅ 学習完了")
    logger.info(f"    - 使用特徴量数: {len(model.feature_cols)}")
    logger.info(f"    - カテゴリカル特徴量数: {len(model.categorical_features)}")

    # 特徴量重要度
    logger.info("\n【ステップ8】特徴量重要度（上位10件）")
    importance_df = model.get_feature_importance(top_n=10)
    for i, row in importance_df.iterrows():
        logger.info(f"    {i+1:2d}. {row['feature']:30s} {row['importance']:10.0f}")

    # 予測
    logger.info("\n【ステップ9】予測（学習・検証データ）")
    train_probs = model.predict_proba(train_features)
    val_probs = model.predict_proba(val_features)
    
    logger.info(f"  ✅ 予測完了")
    logger.info(f"    - 学習データ予測範囲: {train_probs.min():.4f} 〜 {train_probs.max():.4f}")
    logger.info(f"    - 検証データ予測範囲: {val_probs.min():.4f} 〜 {val_probs.max():.4f}")

    # odds カラムの存在チェック（Critical 修正）
    odds_col = "odds_raw" if "odds_raw" in train_features.columns else "odds"
    
    if odds_col not in train_features.columns:
        logger.warning(f"警告: {odds_col} カラムが存在しません。評価をスキップします。")
        odds_train = None
        odds_val = None
    else:
        odds_train = train_features[odds_col]
        odds_val = val_features[odds_col]

    # 評価
    logger.info("\n【ステップ10】評価（学習・検証データ）")
    
    if odds_train is not None:
        train_metrics = model.evaluate(
            df=train_features,
            y=train_features["win_flag"],
            race_ids=train_features["race_id"],
            finish_positions=train_features["finish_position"],
            odds=odds_train,
        )

        logger.info("  学習データ評価:")
        for key, value in train_metrics.items():
            logger.info(f"    - {key:20s}: {value:.4f}")
    
    if odds_val is not None:
        val_metrics = model.evaluate(
            df=val_features,
            y=val_features["win_flag"],
            race_ids=val_features["race_id"],
            finish_positions=val_features["finish_position"],
            odds=odds_val,
        )

        logger.info("  検証データ評価:")
        for key, value in val_metrics.items():
            logger.info(f"    - {key:20s}: {value:.4f}")

    # モデル保存
    logger.info("\n【ステップ11】モデル保存")
    
    # モデル保存ディレクトリの作成
    MODEL_DIR = Path("./models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"base_win_model_{timestamp}.txt"
    
    model.save(str(model_path))
    logger.info(f"  ✅ 保存完了: {model_path}")

    # モデル読み込み
    logger.info("\n【ステップ12】モデル読み込み")
    model2 = BaseWinModel()
    model2.load(str(model_path))
    logger.info("  ✅ 読み込み完了")

    # 読み込み後の予測
    train_probs2 = model2.predict_proba(train_features)
    diff = np.abs(train_probs - train_probs2).max()
    logger.info(f"  確認: 予測値の差（max）= {diff:.10f}")

    if diff < 1e-6:
        logger.info("  ✅ モデル保存・読み込みOK")
    else:
        logger.warning("  ⚠️ モデル保存・読み込みに問題あり")

    # 🔽【ステップ13】Calibration v3 によるキャリブレーション
    logger.info("\n【ステップ13】確率キャリブレーション (Calibration v3)")

    # Critical 修正: val でキャリブレーション
    calib_config = CalibrationConfig(method="platt", n_bins=15)
    calibrator = ProbabilityCalibrator(calib_config)

    logger.info("  検証データでキャリブレーターを学習...")
    calibrator.fit(val_probs, val_features["win_flag"].values)

    # 評価（検証データ）
    calib_metrics = calibrator.evaluate(val_probs, val_features["win_flag"].values)

    logger.info("  📊 Calibration 指標（検証データ）")
    logger.info(f"    - ECE  Before: {calib_metrics['ece_raw']:.6f}")
    logger.info(f"    - ECE  After : {calib_metrics['ece_calibrated']:.6f}")
    logger.info(f"    - 改善率: {(1 - calib_metrics['ece_calibrated']/calib_metrics['ece_raw'])*100:.1f}%")
    logger.info(f"    - Brier Before: {calib_metrics['brier_raw']:.6f}")
    logger.info(f"    - Brier After : {calib_metrics['brier_calibrated']:.6f}")

    # ECE が改善していない場合は警告
    if calib_metrics['ece_calibrated'] >= calib_metrics['ece_raw']:
        logger.warning("  ⚠️ キャリブレーションで ECE が改善していません")

    # キャリブレーション後の確率例
    calibrated_val_probs = calibrator.predict(val_probs)
    
    logger.info("\n  サンプル（検証データ先頭10件）:")
    for i in range(min(10, len(calibrated_val_probs))):
        logger.info(
            f"    raw={val_probs[i]:.4f}  "
            f"calib={calibrated_val_probs[i]:.4f}  "
            f"y={int(val_features['win_flag'].iloc[i])}"
        )

    # Reliability Curve の取得（オプション）
    bin_centers, bin_acc, bin_conf = calibrator.get_reliability_curve(
        val_features["win_flag"].values,
        val_probs,
        n_bins=10
    )
    
    logger.info("\n  Reliability Curve:")
    for i in range(len(bin_centers)):
        if not np.isnan(bin_acc[i]):
            logger.info(
                f"    Bin {i+1:2d} (中心={bin_centers[i]:.2f}): "
                f"予測={bin_conf[i]:.4f}, 実際={bin_acc[i]:.4f}"
            )

    # 🔽【ステップ14】テストデータでの最終評価
    logger.info("\n【ステップ14】テストデータでの最終評価")
    
    test_features = build_features_for_races(
        race_ids=test_races,
        data=test_data,
        history_store=history_store,
        race_builder=race_builder,
        base_builder=base_builder,
        dataset_name="test",
    )
    
    # 予測
    test_probs_raw = model.predict_proba(test_features)
    test_probs_calibrated = calibrator.predict(test_probs_raw)
    
    logger.info(f"  ✅ 予測完了")
    logger.info(f"    - 予測範囲（raw）: {test_probs_raw.min():.4f} 〜 {test_probs_raw.max():.4f}")
    logger.info(f"    - 予測範囲（calib）: {test_probs_calibrated.min():.4f} 〜 {test_probs_calibrated.max():.4f}")
    
    # モデル評価
    if odds_col in test_features.columns:
        test_metrics = model.evaluate(
            df=test_features,
            y=test_features["win_flag"],
            race_ids=test_features["race_id"],
            finish_positions=test_features["finish_position"],
            odds=test_features[odds_col],
        )

        logger.info("  テストデータ モデル評価:")
        for key, value in test_metrics.items():
            logger.info(f"    - {key:20s}: {value:.4f}")
    
    # キャリブレーション評価
    test_calib_metrics = calibrator.evaluate(
        test_probs_raw,
        test_features["win_flag"].values
    )
    
    logger.info("\n  テストデータ キャリブレーション:")
    logger.info(f"    - ECE (raw):        {test_calib_metrics['ece_raw']:.6f}")
    logger.info(f"    - ECE (calibrated): {test_calib_metrics['ece_calibrated']:.6f}")
    logger.info(f"    - Brier (raw):      {test_calib_metrics['brier_raw']:.6f}")
    logger.info(f"    - Brier (calibrated): {test_calib_metrics['brier_calibrated']:.6f}")

    # 処理時間
    overall_elapsed = time.time() - overall_start
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BaseWinModel v1.1 + Calibration v3 - 統合テスト完了")
    logger.info(f"総処理時間: {overall_elapsed:.2f} 秒")
    logger.info("=" * 80)

    logger.info("\n【次のステップ】")
    logger.info("1. ✅ データリーク修正完了（val でキャリブレーション）")
    logger.info("2. ✅ 時系列分割修正完了（race_date でソート）")
    logger.info("3. ✅ テストデータ評価追加完了")
    logger.info("4. 次は実データでの学習・評価・バックテストを行う。")
    logger.info("5. PaceAdjustment と統合した end-to-end パイプラインへ接続する。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        raise
