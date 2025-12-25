# FeaturePack v1 (features_v4)

200+ 特徴量を生成する ML パイプラインモジュール。

## 概要

- **200+ 特徴量**: 基本レース情報、馬成績、騎手・調教師統計、血統ハッシュ
- **リーク防止**: 全統計は as-of (race_date より前) で計算
- **高速処理**: SQL ベースの集計、Python ループ禁止

## クイックスタート

```bash
# 1. 特徴量テーブル構築
python scripts/build_feature_table_v4.py --db netkeiba.db

# 2. モデル学習・評価
python scripts/train_eval_v4.py --db netkeiba.db

# 3. データ品質レポート
python scripts/report_quality_v4.py --db netkeiba.db --output artifacts/
```

## 実行順序と依存関係

パイプラインは以下の順序で実行します:

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 0: Data Ingestion (前提)                                   │
│   scripts/fetch_masters.py → horses/jockeys/trainers/pedigree   │
│   src/ingestion/ → races/race_results/payouts                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Feature Table Build                                     │
│   scripts/build_feature_table_v4.py                             │
│   入力: races, race_results, horses, jockeys, trainers,         │
│         horse_pedigree                                          │
│   出力: feature_table_v4 (SQLite table)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Train & Evaluate                                        │
│   scripts/train_eval_v4.py                                      │
│   入力: feature_table_v4                                        │
│   出力: models/*.pkl, artifacts/eval_*.json                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Quality Report (任意)                                   │
│   scripts/report_quality_v4.py                                  │
│   入力: 全テーブル                                               │
│   出力: artifacts/quality_report_*.json/csv                     │
└─────────────────────────────────────────────────────────────────┘
```

### 依存テーブル一覧

| Step | 必須テーブル | オプショナル |
|------|-------------|-------------|
| 1    | races, race_results | horses, jockeys, trainers, horse_pedigree |
| 2    | feature_table_v4 | - |
| 3    | 全テーブル | - |

## 出力ファイル

### models/ (モデル成果物)
```
models/
├── lgbm_win_v4.pkl          # 勝利予測モデル
├── lgbm_in3_v4.pkl          # 3着内予測モデル
├── feature_cols_v4.json     # 特徴量カラム名リスト
└── model_meta_v4.json       # 学習パラメータ・評価指標
```

### artifacts/ (レポート・スナップショット)
```
artifacts/
├── eval_metrics_v4.json     # 評価指標 (AUC, LogLoss等)
├── roi_backtest_v4.json     # ROIバックテスト結果
├── quality_report_*.json    # データ品質レポート
├── quality_report_*.csv     # 品質サマリー
└── fetch_progress_*.json    # fetch_masters進捗スナップショット
```

## 特徴量グループ

| グループ         | 特徴量数 | 説明                         |
|-----------------|---------|------------------------------|
| base_race       | ~25     | 基本レース情報 (距離、馬場等)   |
| horse_form      | ~40     | 馬の過去成績・フォーム          |
| pace_position   | ~20     | ペース・位置取り               |
| class_prize     | ~15     | クラス・賞金                   |
| jockey_trainer  | ~40     | 騎手・調教師の as-of 成績      |
| ped_hash        | 512     | 5代血統ハッシュ (メイン)        |
| anc_hash        | 128     | 直系祖先ハッシュ (サブ)         |

## 血統ハッシュ仕様

### ped_hash (512次元)
- 5代血統の全祖先 (最大62頭) を Feature Hashing
- token = `gen{generation}:{position}:{ancestor_id or ancestor_name}`
- 例: `gen1:s:2019101234` (父), `gen2:sd:ディープインパクト` (母父)
- 世代による重み付け: `1/sqrt(generation)`

### anc_hash (128次元)
- 直系祖先のみ (父/母/母父)
- token = `{role}:{ancestor_name}`
- 例: `sire:ディープインパクト`, `dam:ウインドインハーヘア`

## 時系列分割

### year_based (デフォルト)
```
train: 2021-01 ~ 2023-12
val:   2023-10 ~ 2023-12 (train末尾を検証に流用)
test:  2024-01 ~ 現在
```

### date_based
```bash
python scripts/train_eval_v4.py --db netkeiba.db \
  --split-mode date_based \
  --train-end 2023-06-30 \
  --val-end 2023-12-31
```

## ファイル構成

```
src/features_v4/
├── __init__.py           # モジュール初期化
├── feature_table_v4.py   # DDL 定義
├── asof_aggregator.py    # As-Of 統計計算
├── feature_builder_v4.py # 特徴量生成エンジン
├── train_eval_v4.py      # 学習・評価パイプライン
├── quality_report.py     # データ品質レポート
└── README.md             # このファイル

scripts/
├── build_feature_table_v4.py  # 特徴量構築 CLI
├── train_eval_v4.py           # 学習・評価 CLI
└── report_quality_v4.py       # 品質レポート CLI
```

## テスト

```bash
# 全テスト実行
pytest tests/test_features_v4.py -v

# カバレッジ付き
pytest tests/test_features_v4.py --cov=src/features_v4
```

## リーク防止 (As-Of ルール)

### 基本原則

**「予測時点で知り得ない情報は使わない」**

すべての統計量は `race_date` より**厳密に前**のデータのみで計算します。

### As-Of レベル

| Level | 基準 | 安全性 | 現状 |
|-------|------|-------|------|
| Level 1 | race_date < 対象日 | ✅ 安全 | **実装済み** |
| Level 2 | start_time < 対象時刻 | ✅ より安全 | TODO |

Level 1 では同日の他レース結果は含まれません（安全側に倒した設計）。

### リーク防止の実装例

#### ✅ 正しい実装 (as-of)

```sql
-- 馬の過去勝率を計算 (対象レース日より前のみ)
SELECT
    horse_id,
    COUNT(*) as n_starts,
    SUM(CASE WHEN finish_order = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
FROM race_results rr
JOIN races r ON rr.race_id = r.race_id
WHERE rr.horse_id = :target_horse_id
  AND r.date < :target_race_date   -- ★ 厳密に「より前」
GROUP BY horse_id
```

#### ❌ リークする実装 (危険)

```sql
-- 危険: <= を使うと同日レースの結果が含まれる
WHERE r.date <= :target_race_date  -- ❌ 同日レース結果が混入

-- 危険: 全データで統計を取る
SELECT AVG(finish_order) FROM race_results  -- ❌ 未来データが混入
WHERE horse_id = :horse_id
```

### リークしやすいパターン

| パターン | リスク | 対策 |
|----------|-------|------|
| `<=` 演算子 | 同日レース混入 | `<` を使用 |
| 全体統計 (AVG, COUNT等) | 未来データ混入 | WHERE date < target |
| ランキング特徴量 | 未来成績で計算 | as-of スナップショット |
| 血統ハッシュ | 子孫の成績混入 | 血統は静的なのでOK |

### テストでの確認

```python
# テスト例: as-of制約の検証
def test_no_future_leak():
    """2023-06-01のレースで2023-06-01以降のデータを使っていないか確認"""
    target_date = "2023-06-01"
    features = build_features_for_race(race_id, target_date)

    # 統計に使われたレース日を取得
    used_dates = get_dates_used_for_stats(race_id)

    # すべて対象日より前であることを確認
    assert all(d < target_date for d in used_dates), "Future data leak detected!"
```

## 既知の制約

1. **同日複数レース**: 現在の as-of は日付ベース (Level 1)
   - 同日の他レース結果は含まれない (安全側)
   - TODO: 発走時刻ベース (Level 2) の実装

2. **血統データ依存**: horse_pedigree テーブルがない場合、血統ハッシュは全てゼロ

3. **メモリ使用量**: 全年データ処理時は ~4GB RAM 推奨

## 依存パッケージ

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 開発用
```

## ライセンス

Private - obn
