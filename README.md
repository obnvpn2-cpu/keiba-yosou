# KEIBA SCENARIO AI - 実装完了版

**競馬予想AIシステム - Phase 1実装**

このプロジェクトは、ChatGPTからの実装タスクセットに基づいて完全実装されたシステムです。

## 📋 実装済みタスク

### ✅ タスク1: データパイプライン整備
- `timeline_manager.py`: データリーク防止 + ウォークフォワードCV

### ✅ タスク2: ベースモデル
- `base_model.py`: LightGBM勝率予測 + 包括的評価

### ✅ タスク3: 確率キャリブレーション
- `calibration.py`: Platt Scaling / Isotonic Regression + ECE評価

### ✅ タスク4: 馬場補正モデル
- `baba_adjustment.py`: log-odds空間での学習可能な補正

### ✅ タスク5: ペース予測モデル
- `pace_prediction.py`: 前半3F/後半3Fの連続値予測

### ✅ タスク6: ペース補正モデル
- `pace_adjustment.py`: log-odds空間でのペース影響補正

### ✅ タスク7: 確率統合
- `probability_integration.py`: log-odds加算 + Softmax正規化

### ✅ タスク8: 相性スコア
- `synergy_score.py`: SVD + Bayesian Shrinkage

### ✅ タスク9: バックテスト
- `backtest.py`: オッズタイミング別、控除率適用、包括的評価

### ✅ タスク10: SHAP解析 + 説明文生成
- `shap_explainer.py`: 予測理由の可視化とテキスト化

### ✅ タスク11: FastAPI
- `api.py`: 予測API + エンドポイント

### ✅ タスク12: MLflow管理
- `mlflow_manager.py`: 実験追跡、パラメータ管理

## 🏗️ アーキテクチャ

```
データ収集（netkeiba + JRA公式）
↓
TimelineManager（データリーク防止）
↓
ベース勝率モデル（LightGBM）
↓
確率キャリブレーション（Platt/Isotonic）
↓
シナリオ補正レイヤー
 ├─ 馬場補正（log-odds）
 └─ ペース補正（log-odds）
↓
確率統合（Softmax正規化）
↓
期待値計算 + 投票アドバイザー
↓
FastAPI（予測エンドポイント）
```

## 📂 ディレクトリ構造

```
keiba_ai/
├── src/
│   ├── timeline_manager.py      # タイムライン管理
│   ├── base_model.py             # ベースモデル
│   ├── calibration.py            # キャリブレーション
│   ├── baba_adjustment.py        # 馬場補正
│   ├── pace_prediction.py        # ペース予測
│   ├── pace_adjustment.py        # ペース補正
│   ├── probability_integration.py # 確率統合
│   ├── synergy_score.py          # 相性スコア
│   ├── backtest.py               # バックテスト
│   ├── shap_explainer.py         # SHAP解析
│   ├── api.py                    # FastAPI
│   └── mlflow_manager.py         # MLflow管理
├── data/                         # データ保存先
├── models/                       # モデル保存先
├── notebooks/                    # Jupyter notebooks
├── tests/                        # テストコード
├── config/                       # 設定ファイル
├── requirements.txt              # 依存パッケージ
└── README.md                     # このファイル
```

## 🚀 セットアップ

### 1. 環境構築

```bash
# Python 3.11+ 推奨
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージインストール
pip install -r requirements.txt
```

### 2. データ準備

```python
from timeline_manager import TimelineManager

# データ読み込み
df = pd.read_csv('race_data.csv')

# TimelineManager初期化
tm = TimelineManager(df, date_column='race_date')

# ウォークフォワードCV
splits = tm.walk_forward_split(n_splits=5)
```

### 3. モデル訓練

```python
from base_model import BaseModel, WalkForwardValidator
from calibration import CalibrationPipeline

# ベースモデル
base_model = BaseModel()
base_model.train(X_train, y_train)

# キャリブレーション
calibration_pipeline = CalibrationPipeline(base_model)
calibration_pipeline.fit_calibrator(X_cal, y_cal)

# ウォークフォワードCV
validator = WalkForwardValidator()
avg_metrics, fold_results = validator.validate(tm, X, y, race_ids)
```

### 4. 補正モデル訓練

```python
from baba_adjustment import BabaAdjustmentModel
from pace_adjustment import PaceAdjustmentModel

# 馬場補正
baba_model = BabaAdjustmentModel(alpha=0.5)
X_baba, y_baba, weights = baba_model.prepare_training_data(
    calibrated_pred, actual_win, features_baba
)
baba_model.train(X_baba, y_baba, sample_weight=weights)

# ペース補正
pace_model = PaceAdjustmentModel(alpha=0.5)
X_pace, y_pace, weights = pace_model.prepare_training_data(
    after_baba_pred, actual_win, features_pace
)
pace_model.train(X_pace, y_pace, sample_weight=weights)
```

### 5. 予測パイプライン

```python
from probability_integration import PredictionPipeline, ProbabilityIntegrator

# パイプライン構築
pipeline = PredictionPipeline(
    base_model=base_model,
    calibrator=calibration_pipeline.calibrator,
    baba_model=baba_model,
    pace_model=pace_model,
    integrator=ProbabilityIntegrator()
)

# 予測実行
results = pipeline.predict(
    X_base, X_baba, X_pace, race_ids,
    horse_baba_counts, horse_pace_counts
)
```

### 6. バックテスト

```python
from backtest import BacktestEngine, BacktestConfig

# 設定
config = BacktestConfig(
    odds_timing='前日',
    takeout_rate=0.25,
    strategies=['all', 'ev>1.0', 'top1']
)

# バックテスト実行
engine = BacktestEngine(config)
summary = engine.run(predictions_df, actual_df, odds_df)
```

### 7. API起動

```bash
# FastAPI起動
cd src
python api.py

# または
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

アクセス: http://localhost:8000/docs

### 8. MLflow UI

```bash
mlflow ui
```

アクセス: http://localhost:5000

## 📊 評価指標

### ベースモデル
- **Brier Score**: 確率精度（目標: < 0.15）
- **AUC**: 識別能力（目標: > 0.70）
- **NDCG**: ランキング精度（目標: > 0.75）

### キャリブレーション
- **ECE**: Expected Calibration Error（目標: < 0.05）
- **MCE**: Maximum Calibration Error

### バックテスト
- **ROI**: 回収率（目標: > 110% for EV>1.0）
- **的中率**: 単勝的中率
- **最大ドローダウン**: 資金管理

## 🔍 データリーク防止

### 重要原則
1. `race_date`より後の情報は絶対に参照しない
2. オッズは「実際の発売タイミング」と整合させる
3. ウォークフォワード方式で学習・評価
4. `TimelineManager`で厳格に管理

### 検証方法
```python
# データリークチェック
is_safe, issues = tm.validate_no_leakage(X, y)
if not is_safe:
    print("データリーク検出:", issues)
```

## 🎯 Phase 1目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| Brier Score | < 0.15 | 確率精度 |
| ECE | < 0.05 | キャリブレーション |
| NDCG | > 0.75 | ランキング精度 |
| ROI (EV>1.0) | > 110% | 回収率 |

## 🛠️ 次のステップ

### ChatGPTレビュー項目
1. データリーク検査
2. 補正モデルのターゲット品質チェック
3. キャリブレーション品質レビュー
4. ペース予測の妥当性
5. ウォークフォワード評価の正しさ

### Phase 2への拡張
- JRDBデータ追加（月額¥2,480）
- パドック評価モデル
- 調教分析
- 精度向上（目標ROI: 120%）

## 📝 ライセンス

このプロジェクトは個人利用のみを目的としています。

## 👤 作成者

obn + Claude (Anthropic) + ChatGPT (OpenAI)

---

**最終更新: 2025年12月4日**
**実装完了: Phase 1 全12タスク**
