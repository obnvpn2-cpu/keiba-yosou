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
