#!/usr/bin/env python3
"""
scenario モジュールの使用例。

このスクリプトは、既存の predict_lgbm との接続方法を示すサンプルコード。
実際のプロジェクトでは、このパターンを参考に統合する。
"""

import json
from typing import Dict

# scenario モジュールのインポート
from scenario import (
    ScenarioSpec,
    ScenarioScore,
    ScenarioAdjuster,
    RaceContext,
    TrackMoisture,
    MoistureReading,
    AdjustmentConfig,
    build_scenario_ui_context,
)


def example_basic_usage():
    """基本的な使用例"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # 1. レースコンテキストの作成
    race_ctx = RaceContext(
        race_id="202305021211",
        race_name="東京優駿（日本ダービー）",
        course="東京",
        surface="turf",
        distance=2400,
        distance_cat="long",
        race_date="2023-05-28",
        race_class="G1",
    )
    
    # 2. シナリオ定義
    spec = ScenarioSpec(
        scenario_id="slow_inner_bias",
        race_context=race_ctx,
        pace="S",
        track_condition="良",
        bias="内",
        cushion_value=9.5,
        moisture=TrackMoisture(
            turf=MoistureReading(goal=12.5, corner_4=11.8),
            dirt=MoistureReading(goal=8.2, corner_4=7.9),
        ),
        front_runner_ids=["2020104123"],  # 逃げ想定
        stalker_ids=["2020104456", "2020104789"],  # 先行想定
        closer_ids=["2020104111"],  # 差し想定
        weak_horse_ids=["2020104999"],  # 状態不安
        notes="逃げ馬が1頭だけなのでスローペース濃厚。内枠有利の馬場。",
    )
    
    # 3. ベース予測値（実際は predict_lgbm から取得）
    base_predictions = {
        "2020104123": {"win": 0.15, "in3": 0.40},
        "2020104456": {"win": 0.20, "in3": 0.45},
        "2020104789": {"win": 0.12, "in3": 0.35},
        "2020104111": {"win": 0.18, "in3": 0.42},
        "2020104999": {"win": 0.08, "in3": 0.25},
        "2020104222": {"win": 0.10, "in3": 0.30},
        "2020104333": {"win": 0.07, "in3": 0.22},
        "2020104444": {"win": 0.10, "in3": 0.28},
    }
    
    # 馬名マッピング（実際は DB から取得）
    horse_names = {
        "2020104123": "スピードスター",
        "2020104456": "キングオブターフ",
        "2020104789": "ミドルランナー",
        "2020104111": "ラストスパート",
        "2020104999": "フラフラホース",
        "2020104222": "ノーマルワン",
        "2020104333": "ダークホース",
        "2020104444": "アウトサイダー",
    }
    
    # 馬ごとの特徴量（枠番・脚質など）
    # 枠番は frame_no キーで指定
    horse_features = {
        "2020104123": {"frame_no": 2, "run_style": "逃げ"},   # 内枠
        "2020104456": {"frame_no": 4, "run_style": "先行"},
        "2020104789": {"frame_no": 5, "run_style": "先行"},
        "2020104111": {"frame_no": 6, "run_style": "差し"},
        "2020104999": {"frame_no": 7, "run_style": "追込"},
        "2020104222": {"frame_no": 1, "run_style": "先行"},   # 最内
        "2020104333": {"frame_no": 8, "run_style": "差し"},   # 外枠
        "2020104444": {"frame_no": 3, "run_style": "先行"},
    }
    
    # 4. 補正適用
    adjuster = ScenarioAdjuster()
    score = adjuster.adjust(
        spec=spec,
        base_predictions=base_predictions,
        horse_features=horse_features,
        horse_names=horse_names,
    )
    
    # 5. 結果表示
    print("\nScenario Summary:")
    print(spec.to_llm_summary())
    print()
    
    print(score.generate_summary_text())
    print()
    
    # 6. プラス補正がかかった馬
    print("\nHorses with positive adjustment:")
    for h in score.horses_with_positive_adjustment():
        print(f"  {h.horse_name}: +{h.win_delta:.3f} ({', '.join(h.get_reasons_japanese())})")
    
    # 7. マイナス補正がかかった馬
    print("\nHorses with negative adjustment:")
    for h in score.horses_with_negative_adjustment():
        print(f"  {h.horse_name}: {h.win_delta:.3f} ({', '.join(h.get_reasons_japanese())})")


def example_scenario_comparison():
    """複数シナリオの比較例"""
    print("\n" + "=" * 60)
    print("Example 2: Scenario Comparison")
    print("=" * 60)
    
    race_ctx = RaceContext(
        race_id="202305021211",
        race_name="サンプルレース",
        course="東京",
        surface="turf",
        distance=1600,
    )
    
    # ベース予測値
    base_predictions = {
        "horse_A": {"win": 0.25, "in3": 0.50},
        "horse_B": {"win": 0.20, "in3": 0.45},
        "horse_C": {"win": 0.15, "in3": 0.40},
    }
    
    horse_names = {
        "horse_A": "逃げ馬",
        "horse_B": "先行馬",
        "horse_C": "差し馬",
    }
    
    # 馬ごとの特徴量（枠番・脚質）
    horse_features = {
        "horse_A": {"frame_no": 1, "run_style": "逃げ"},
        "horse_B": {"frame_no": 3, "run_style": "先行"},
        "horse_C": {"frame_no": 5, "run_style": "差し"},
    }
    
    # シナリオ1: スローペース
    spec_slow = ScenarioSpec(
        scenario_id="slow_pace",
        race_context=race_ctx,
        pace="S",
        track_condition="良",
        bias="フラット",
        front_runner_ids=["horse_A"],
        closer_ids=["horse_C"],
    )
    
    # シナリオ2: ハイペース
    spec_fast = ScenarioSpec(
        scenario_id="fast_pace",
        race_context=race_ctx,
        pace="H",
        track_condition="良",
        bias="フラット",
        front_runner_ids=["horse_A"],
        closer_ids=["horse_C"],
    )
    
    adjuster = ScenarioAdjuster()
    
    score_slow = adjuster.adjust(spec_slow, base_predictions, horse_features, horse_names)
    score_fast = adjuster.adjust(spec_fast, base_predictions, horse_features, horse_names)
    
    print("\nComparison: Slow Pace vs Fast Pace")
    print("-" * 50)
    print(f"{'Horse':<12} {'Base':>8} {'Slow':>8} {'Fast':>8} {'Diff':>8}")
    print("-" * 50)
    
    for horse_id in base_predictions.keys():
        base = base_predictions[horse_id]["win"]
        slow = score_slow.get_horse(horse_id).adj_win
        fast = score_fast.get_horse(horse_id).adj_win
        diff = slow - fast
        name = horse_names[horse_id]
        print(f"{name:<12} {base:>8.3f} {slow:>8.3f} {fast:>8.3f} {diff:>+8.3f}")


def example_llm_context():
    """LLM用コンテキスト生成例"""
    print("\n" + "=" * 60)
    print("Example 3: LLM Context Generation")
    print("=" * 60)
    
    race_ctx = RaceContext(
        race_id="202305021211",
        race_name="サンプルレース",
        course="東京",
        surface="turf",
        distance=2000,
    )
    
    spec = ScenarioSpec(
        scenario_id="test",
        race_context=race_ctx,
        pace="M",
        track_condition="稍重",
        bias="内",
        notes="雨上がりで内が有利になりそう",
    )
    
    base_predictions = {f"horse_{i}": {"win": 0.1, "in3": 0.3} for i in range(10)}
    horse_names = {f"horse_{i}": f"テスト馬{i+1}" for i in range(10)}
    
    # 馬ごとの特徴量（枠番・脚質）
    # 枠番は連番、脚質は適当に割り当て
    run_styles = ["逃げ", "先行", "先行", "差し", "差し", "差し", "追込", "追込", "先行", "差し"]
    horse_features = {
        f"horse_{i}": {"frame_no": i + 1, "run_style": run_styles[i]}
        for i in range(10)
    }
    
    adjuster = ScenarioAdjuster()
    score = adjuster.adjust(spec, base_predictions, horse_features, horse_names)
    
    llm_context = score.to_llm_context(top_n=5)
    
    print("\nLLM Context (JSON):")
    print(json.dumps(llm_context, ensure_ascii=False, indent=2))


def example_custom_config():
    """カスタム設定例"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Adjustment Config")
    print("=" * 60)
    
    # デフォルト設定を確認
    default_config = AdjustmentConfig.default()
    print(f"\nDefault front_runner_win['S']: {default_config.front_runner_win.get('S', 'N/A')}")
    print(f"Default weak_horse_win: {default_config.weak_horse_win}")
    
    # カスタム設定（より強い補正）
    # 脚質カテゴリは内部名（front_runner, pace_keeper, mid_chaser, deep_closer, other）を使用
    custom_pace_win = {
        "S": {
            "front_runner": 1.30,   # 逃げ: より強いブースト
            "pace_keeper":  1.15,   # 先行
            "mid_chaser":   0.90,   # 差し
            "deep_closer":  0.80,   # 追込: より強いペナルティ
            "other":        1.00,
        },
        "M": {
            "front_runner": 1.00,
            "pace_keeper":  1.00,
            "mid_chaser":   1.00,
            "deep_closer":  1.00,
            "other":        1.00,
        },
        "H": {
            "front_runner": 0.75,   # 逃げ: より強いペナルティ
            "pace_keeper":  0.85,   # 先行
            "mid_chaser":   1.15,   # 差し
            "deep_closer":  1.25,   # 追込: より強いブースト
            "other":        1.00,
        },
    }
    custom_config = AdjustmentConfig(
        pace_category_win=custom_pace_win,
        pace_category_in3=default_config.pace_category_in3,
        front_runner_win={"S": 1.10, "M": 1.05, "H": 0.90},  # より強いブースト
        front_runner_in3=default_config.front_runner_in3,
        weak_horse_win=0.75,  # より強いペナルティ
        weak_horse_in3=0.85,
    )
    
    race_ctx = RaceContext(race_id="test_race")
    spec = ScenarioSpec(
        scenario_id="test",
        race_context=race_ctx,
        pace="S",
        track_condition="良",
        bias="フラット",
        front_runner_ids=["horse_1"],
    )
    
    base_predictions = {"horse_1": {"win": 0.20, "in3": 0.40}}
    horse_features = {"horse_1": {"frame_no": 1, "run_style": "逃げ"}}
    
    # デフォルト設定での補正
    default_adjuster = ScenarioAdjuster()
    default_score = default_adjuster.adjust(spec, base_predictions, horse_features)
    
    # カスタム設定での補正
    custom_adjuster = ScenarioAdjuster(config=custom_config)
    custom_score = custom_adjuster.adjust(spec, base_predictions, horse_features)
    
    print(f"\nBase win rate: {base_predictions['horse_1']['win']:.3f}")
    print(f"Adjusted (default config): {default_score.get_horse('horse_1').adj_win:.3f}")
    print(f"Adjusted (custom config):  {custom_score.get_horse('horse_1').adj_win:.3f}")


def example_serialization():
    """シリアライズ/デシリアライズ例"""
    print("\n" + "=" * 60)
    print("Example 5: Serialization / Deserialization")
    print("=" * 60)
    
    race_ctx = RaceContext(
        race_id="202305021211",
        race_name="テストレース",
    )
    
    spec = ScenarioSpec(
        scenario_id="original",
        race_context=race_ctx,
        pace="S",
        track_condition="良",
        bias="内",
        notes="テスト用シナリオ",
    )
    
    # to_dict でシリアライズ
    spec_dict = spec.to_dict()
    print("\nSerialized ScenarioSpec:")
    print(json.dumps(spec_dict, ensure_ascii=False, indent=2))
    
    # from_dict でデシリアライズ
    restored_spec = ScenarioSpec.from_dict(spec_dict)
    print(f"\nRestored scenario_id: {restored_spec.scenario_id}")
    print(f"Restored pace: {restored_spec.pace}")
    print(f"Restored notes: {restored_spec.notes}")


def example_ui_context():
    """UI向けコンテキスト生成例（build_scenario_ui_context の使用例）"""
    print("\n" + "=" * 60)
    print("Example 6: UI Context Generation (build_scenario_ui_context)")
    print("=" * 60)
    
    # 1. レースコンテキストの作成
    race_ctx = RaceContext(
        race_id="202312241100",
        race_name="有馬記念",
        course="中山",
        surface="turf",
        distance=2500,
        distance_cat="long",
        race_date="2023-12-24",
        race_class="G1",
    )
    
    # 2. シナリオ定義（ユーザーがUIのスライダーで指定した想定）
    spec = ScenarioSpec(
        scenario_id="slow_inner",
        race_context=race_ctx,
        pace="S",
        track_condition="良",
        bias="内",
        cushion_value=9.2,
        front_runner_ids=["2019110042"],
        stalker_ids=["2020104567"],
        closer_ids=["2021103456"],
        weak_horse_ids=["2018109876"],
        notes="逃げ馬1頭でスロー濃厚、内伸びバイアス想定",
    )
    
    # 3. ベース予測値（実際は predict_lgbm から取得）
    base_predictions = {
        "2019110042": {"win": 0.1751, "in3": 0.4535},
        "2020104567": {"win": 0.1482, "in3": 0.4021},
        "2021103456": {"win": 0.1203, "in3": 0.3512},
        "2018109876": {"win": 0.0892, "in3": 0.2845},
        "2020105678": {"win": 0.1124, "in3": 0.3201},
        "2019108765": {"win": 0.0956, "in3": 0.2987},
        "2021104321": {"win": 0.0823, "in3": 0.2654},
        "2020106543": {"win": 0.0712, "in3": 0.2341},
        "2019107654": {"win": 0.0654, "in3": 0.2123},
        "2018108543": {"win": 0.0403, "in3": 0.1781},
    }
    
    # 4. 馬ごとの特徴量
    horse_features = {
        "2019110042": {"frame_no": 3, "run_style": "逃げ"},
        "2020104567": {"frame_no": 2, "run_style": "先行"},
        "2021103456": {"frame_no": 6, "run_style": "差し"},
        "2018109876": {"frame_no": 8, "run_style": "追込"},
        "2020105678": {"frame_no": 1, "run_style": "先行"},
        "2019108765": {"frame_no": 4, "run_style": "差し"},
        "2021104321": {"frame_no": 5, "run_style": "先行"},
        "2020106543": {"frame_no": 7, "run_style": "追込"},
        "2019107654": {"frame_no": 3, "run_style": None},
        "2018108543": {"frame_no": 2, "run_style": "先行"},
    }
    
    # 5. 馬名マッピング
    horse_names = {
        "2019110042": "スーリールダンジュ",
        "2020104567": "タスティエーラ",
        "2021103456": "ジャスティンパレス",
        "2018109876": "イクイノックス",
        "2020105678": "ソールオリエンス",
        "2019108765": "ドウデュース",
        "2021104321": "タイトルホルダー",
        "2020106543": "スターズオンアース",
        "2019107654": "ディープボンド",
        "2018108543": "アスクビクターモア",
    }
    
    # 6. ScenarioAdjuster で補正を適用
    adjuster = ScenarioAdjuster()
    score = adjuster.adjust(
        spec=spec,
        base_predictions=base_predictions,
        horse_features=horse_features,
        horse_names=horse_names,
    )
    
    # 7. build_scenario_ui_context で UI 向け JSON を生成
    ui_context = build_scenario_ui_context(
        score,
        sort_by="adj_win",
        top_n=10,
        include_summary=True,
    )
    
    # 8. JSON として出力
    print("\n--- UI Context (JSON) ---")
    print(json.dumps(ui_context, ensure_ascii=False, indent=2))
    
    # 9. 実用例: LLMに渡すプロンプトの例
    print("\n--- LLM Prompt Example ---")
    prompt = f"""以下のシナリオに基づいて、レース予想の解説を200文字程度で書いてください。

シナリオ情報:
{json.dumps(ui_context, ensure_ascii=False, indent=2)}

解説のポイント:
- このシナリオで有利になる馬とその理由
- 注意すべき馬（マイナス補正がかかった馬）
- 穴馬候補がいれば言及"""
    print(prompt[:500] + "...\n(truncated)")


def example_ui_context_minimal():
    """UI向けコンテキスト生成の最小例"""
    print("\n" + "=" * 60)
    print("Example 7: UI Context - Minimal Example")
    print("=" * 60)
    
    # 最小限のセットアップ
    race_ctx = RaceContext(race_id="202312241100", race_name="テストレース")
    spec = ScenarioSpec(
        scenario_id="test",
        race_context=race_ctx,
        pace="M",
        track_condition="良",
        bias="フラット",
    )
    
    base_predictions = {
        "horse_1": {"win": 0.30, "in3": 0.60},
        "horse_2": {"win": 0.25, "in3": 0.55},
        "horse_3": {"win": 0.20, "in3": 0.50},
    }
    
    horse_features = {
        "horse_1": {"frame_no": 1, "run_style": "逃げ"},
        "horse_2": {"frame_no": 4, "run_style": "先行"},
        "horse_3": {"frame_no": 7, "run_style": "差し"},
    }
    
    horse_names = {
        "horse_1": "テスト馬A",
        "horse_2": "テスト馬B",
        "horse_3": "テスト馬C",
    }
    
    adjuster = ScenarioAdjuster()
    score = adjuster.adjust(spec, base_predictions, horse_features, horse_names)
    
    # UI コンテキスト生成
    ui_context = build_scenario_ui_context(score, top_n=3)
    
    print("\n--- horses section only ---")
    for h in ui_context["horses"]:
        print(f"{h['name']}: base={h['win']['base']:.4f} -> adj={h['win']['adj']:.4f} ({h['reasons']})")


if __name__ == "__main__":
    example_basic_usage()
    example_scenario_comparison()
    example_llm_context()
    example_custom_config()
    example_serialization()
    example_ui_context()
    example_ui_context_minimal()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
