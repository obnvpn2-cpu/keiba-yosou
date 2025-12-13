# Phase 3 完成 - 統合使用例（全モジュール連携）

"""
競馬予想AI Phase 3 完全統合フロー

モジュール構成:
1. HorseHistoryStore v2.0
2. RaceFeatureBuilder v5.0
3. PaceInputBuilder v2.0
4. PaceModel v3.0
5. PaceAdjustment v2.0

このスクリプトは全モジュールを統合した使用例を示します。
"""

import pandas as pd
import numpy as np
from datetime import datetime

# モジュールインポート
from HorseHistoryStore import HorseHistoryStore
from race_feature_builder_v5 import RaceFeatureBuilder
from pace_input_builder import PaceInputBuilder
from pace_model import PaceModel
from pace_adjustment import PaceAdjustment


def create_dummy_data():
    """ダミーデータ作成（実運用ではnetkeiba等から取得）"""
    
    np.random.seed(42)
    
    # 馬の戦績データ
    performance_data = pd.DataFrame({
        "horse_id": np.repeat(["horse_A", "horse_B", "horse_C"], 30),
        "race_id": [f"race_{i}" for i in range(90)],
        "race_date": pd.date_range("2024-01-01", periods=90, freq="D"),
        "race_datetime": pd.date_range("2024-01-01 14:00", periods=90, freq="D"),
        "track_code": np.random.choice(["東京", "中山", "阪神"], 90),
        "course_type": np.random.choice(["芝", "ダート"], 90),
        "distance": np.random.choice([1600, 1800, 2000], 90),
        "field_size": np.random.randint(12, 19, 90),
        "corner1_pos": np.random.randint(1, 16, 90),
        "corner2_pos": np.random.randint(1, 16, 90),
        "corner3_pos": np.random.randint(1, 16, 90),
        "corner4_pos": np.random.randint(1, 16, 90),
        "final_3f_time": np.random.uniform(33, 38, 90),
        "finish_position": np.random.randint(1, 16, 90),
        "jockey_id": np.random.choice(["jockey_1", "jockey_2", "jockey_3"], 90),
        "jockey_name": np.random.choice(["武豊", "ルメール", "デムーロ"], 90),
        "jockey_weight": np.random.uniform(52, 58, 90),
        "odds": np.random.uniform(1.5, 50, 90),
        "popularity": np.random.randint(1, 16, 90),
        "remarks": np.random.choice(["", "", "", "", "出遅れ"], 90),
    })
    
    # レース情報
    race_data = pd.DataFrame({
        "race_id": ["race_2024_10_001"],
        "race_datetime": [datetime(2024, 10, 1, 14, 0)],
        "track_type": ["芝"],
        "distance": [1600],
        "field_size": [16],
        "track_condition": ["良"],
        "course": ["東京"],
        "turn_type": ["左回り"],
        "track_bias": [0.0],
    })
    
    # 出走馬情報
    entries_data = pd.DataFrame({
        "race_id": ["race_2024_10_001"] * 3,
        "horse_id": ["horse_A", "horse_B", "horse_C"],
        "jockey_id": ["jockey_1", "jockey_2", "jockey_3"],
        "jockey_name": ["武豊", "ルメール", "デムーロ"],
    })
    
    return performance_data, race_data, entries_data


def main():
    """統合フロー実行"""
    
    print("=" * 80)
    print("Phase 3 完成 - 統合使用例")
    print("=" * 80)
    
    # ダミーデータ作成
    performance_data, race_data, entries_data = create_dummy_data()
    
    print("\n【ステップ1】HorseHistoryStore初期化")
    history_store = HorseHistoryStore(performance_data)
    print(f"  ✅ 戦績データ読み込み: {len(performance_data)}件")
    
    print("\n【ステップ2】RaceFeatureBuilder初期化")
    race_builder = RaceFeatureBuilder(history_store)
    print("  ✅ RaceFeatureBuilder準備完了")
    
    print("\n【ステップ3】レース特徴量生成")
    race_row = race_data.iloc[0]
    entries_df = entries_data
    as_of = race_row["race_datetime"]
    
    # v5.0: 辞書を返す
    result = race_builder.build_for_race(race_row, entries_df, as_of)
    race_features = result["race_features"]
    horse_features = result["horse_features"]
    
    print(f"  ✅ レース特徴量生成完了")
    print(f"    - field_size: {race_features.get('field_size')}")
    print(f"    - num_nige: {race_features.get('num_nige')}")
    print(f"    - distance: {race_features.get('distance')}")
    
    print(f"\n  ✅ 馬ごと特徴量生成完了")
    for horse_id, features in horse_features.items():
        print(f"    - {horse_id}: {features.get('running_style')}")
    
    print("\n【ステップ4】PaceModel学習（ダミー）")
    pace_model = PaceModel()
    print("  ⚠️ 実際のデータで学習が必要")
    print("  ✅ PaceModel準備完了（未学習）")
    
    print("\n【ステップ5】ペース予測（ダミー）")
    # 実際には pace_model.predict_pace_vector(race_features)
    pace_vector = {
        "front_3f": 33.5,
        "last_3f": 36.0,
    }
    pace_balance = pace_vector["last_3f"] - pace_vector["front_3f"]
    print(f"  ✅ ペース予測:")
    print(f"    - front_3f: {pace_vector['front_3f']}秒")
    print(f"    - last_3f: {pace_vector['last_3f']}秒")
    print(f"    - pace_balance: {pace_balance:.1f}秒（{'ハイペース' if pace_balance > 0 else 'スローペース'}）")
    
    print("\n【ステップ6】BaseModel勝率予測（ダミー）")
    # 実際には BaseModel + Calibration の出力
    base_probs = {
        "horse_A": 0.15,  # 逃げ馬
        "horse_B": 0.10,  # 差し馬
        "horse_C": 0.08,  # 先行馬
    }
    print("  ✅ ベース勝率（補正前）:")
    for horse_id, prob in base_probs.items():
        style = horse_features[horse_id]["running_style"]
        print(f"    - {horse_id}（{style}）: {prob*100:.1f}%")
    
    print("\n【ステップ7】PaceAdjustment適用")
    pace_adjuster = PaceAdjustment()
    
    final_probs, debug_info = pace_adjuster.adjust_with_debug(
        base_probs,
        horse_features,
        pace_vector
    )
    
    print("  ✅ ペース補正後勝率:")
    for horse_id, final_prob in final_probs.items():
        base_prob = base_probs[horse_id]
        delta = final_prob - base_prob
        style = horse_features[horse_id]["running_style"]
        print(f"    - {horse_id}（{style}）: {base_prob*100:.1f}% → {final_prob*100:.1f}% ({delta*100:+.1f}%)")
    
    print("\n  📊 デバッグ情報:")
    for horse_id, info in debug_info.items():
        print(f"    {horse_id}:")
        print(f"      pace_balance: {info['pace_balance']:.2f}秒")
        print(f"      normalized_balance: {info['normalized_balance']:.4f}")
        print(f"      style_coef: {info['style_coef']:.2f}")
        print(f"      impact: {info['impact']:.4f}")
        print(f"      delta_logit: {info['delta_logit']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Phase 3 完成 - 全モジュール統合成功")
    print("=" * 80)
    
    print("\n【次のステップ】")
    print("1. 実際のnetkeibaデータでPaceModelを学習")
    print("2. BaseModelの実装とCalibration")
    print("3. TimelineManagerとの統合")
    print("4. バックテスト実施")


if __name__ == "__main__":
    main()
