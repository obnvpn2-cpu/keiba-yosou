# Scenario-Aware Racing Intelligence (keiba-scenario-ai)

> **人間と協業する競馬予想AI**  
> 「ベースモデル × シナリオ補正 × わかりやすいUI」で、  
> 競馬ファンが “自分の想定（ペース/バイアス等）” を反映した予想を作れることを目指すプロジェクトです。

- データソース：JRA（db.netkeiba.com）
- 期間：2021–2024 年の全レース（平地中心）
- DB：SQLite（リポジトリ直下 `netkeiba.db`）
- ベースモデル：複勝 in3（3着以内）確率予測（Logistic / LightGBM）
- シナリオ補正：ペース・バイアス・脚質など **人間入力**でベース予測を調整
- 学習用特徴量：**「前日までに確定している情報」＋「過去走履歴（今走を除外）」** を基本に設計（リーク防止）

---

## 0. まず押さえる方針（重要）

### 0.1 「前日までに確定している情報」で早出ししたい
実運用で **当日情報（馬体重・最終オッズ等）を揃える時間がない** 前提。

- ✅ 前日までに確定：枠順/馬番、距離、コース、クラス、出走頭数、ローテ（日数）など
- ❌ 当日（直前）情報：馬体重・馬体重増減、当日最終オッズ（市場情報）など  
  → **ベースモデルには入れない（再現不能）**

> 注：過去レースの odds/popularity は「履歴」として扱う余地があるが、  
> **今走の当日最終オッズは “使う場合でも” 列を分離し、デフォルトでは学習に入れない**方針。

### 0.2 `track_condition` はベースモデルの主入力にしない
当日の馬場状態・トラックバイアスは **シナリオ/UI で人間が指定して補正する領域**。  
ベースモデルは「ニュートラルな地力・近況中心の予測」を担当する。

---

## 1. 現在の構成（Road 1〜3 反映）

### 1.1 Road 1：DB安全化（idempotent & UPSERT）
- マイグレーション管理：`scripts/run_migrations.py`
- 統一 UPSERT：`src/db/upsert.py`
- `_migrations` テーブルで適用済みを追跡し、**何度回しても壊れない**設計

### 1.2 Road 2 / 2.5：マスタ拡充（horses/jockeys/trainers + 5代血統）
- `scripts/fetch_masters.py`：マスタ一括収集（中断・再開可能）
- `fetch_status` により **resume（途中再開）** を実現
- `/horse/ped/{horse_id}/` から **5代血統表**を取得し、正規化して保存（`horse_pedigree`）

### 1.3 Road 3：FeaturePack v1（feature_table_v4）+ 学習/評価 + 品質レポート
- 200+ 特徴量テーブル：`feature_table_v4`
- リークフリー as-of 集計：`src/features_v4/asof_aggregator.py`
- 生成：`scripts/build_feature_table_v4.py`
- 学習/評価/ROI：`scripts/train_eval_v4.py`
- 品質レポート：`scripts/report_quality_v4.py`

---

## 2. セットアップ

### 2.1 Python 環境
- Python 3.12 推奨（3.11+ 対応）

```bash
# 本番用
pip install -r requirements.txt

# 開発用（テスト・リンター含む）
pip install -r requirements-dev.txt
```

### 2.2 ローカル開発セットアップ

```bash
# 1. リポジトリクローン
git clone https://github.com/obnvpn2-cpu/keiba-yosou.git
cd keiba-yosou

# 2. 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 開発用パッケージインストール
pip install -r requirements-dev.txt

# 4. テスト実行
pytest tests/ -v

# 5. リンター実行
ruff check src/ scripts/ tests/
ruff format --check src/ scripts/ tests/

# 6. 型チェック（任意）
mypy src/ --ignore-missing-imports
```

### 2.3 CI（継続的インテグレーション）

GitHub Actions で PR/push 時に自動実行:
- **Lint**: ruff check + format check
- **Test**: pytest with coverage
- **Type Check**: mypy (advisory)

設定: `.github/workflows/ci.yml`

### 2.4 DB の位置（正）
- 正とする DB：リポジトリ直下

```text
netkeiba.db
```

> SQLite の `sqlite3` コマンドは必須ではありません（Python スクリプトで完結します）。

---

## 3. 推奨：最短で「学習→評価」まで回す（v4）

### 3.1 マイグレーション（Road 1）
```bash
python scripts/run_migrations.py --db netkeiba.db --status
python scripts/run_migrations.py --db netkeiba.db --check-duplicates
python scripts/run_migrations.py --db netkeiba.db
```

### 3.2 レース結果 ingestion（2021–2024）
```bash
cd src
python -m ingestion.ingest_runner \
  --start-year 2021 --end-year 2024 \
  --db ../netkeiba.db
cd ..
```

主なテーブル：
- `races`：レース条件（日時・コース・クラス・頭数など）
- `race_results`：各馬の着順・人気・騎手/厩舎・賞金 等
- `payouts`：払戻（本PJは主に複勝ROI評価で使用）
- `corners`, `lap_times`, `horse_laps`, `short_comments`：将来拡張（展開/説明/UI）

### 3.3 馬ごとの履歴テーブル（horse_results）
```bash
python scripts/build_horse_results.py --db netkeiba.db
```

- 履歴集計は基本 **shift(1)**（今走の情報を混ぜない）
- v4 ではさらに **as-of 集計器**を使用してリーク防止を強化

### 3.4 マスタ取得（Road 2 / 2.5）
まずは jockey / trainer を埋め切り、その後 horses を埋め切る想定。

```bash
# 進捗だけ見る
python scripts/fetch_masters.py --db netkeiba.db --report

# 全件取り切る（途中で落ちても resume 可能）
python scripts/fetch_masters.py --db netkeiba.db --entity jockey --run-until-empty
python scripts/fetch_masters.py --db netkeiba.db --entity trainer --run-until-empty
python scripts/fetch_masters.py --db netkeiba.db --entity horse   --run-until-empty

# 5代血統（horse_pedigree）
python scripts/fetch_masters.py --db netkeiba.db --entity horse_pedigree --run-until-empty
```

長時間運用向け（安全寄り）：
```bash
python scripts/fetch_masters.py --db netkeiba.db --entity horse --run-until-empty \
  --sleep-min 3.0 --sleep-max 5.0
```

> `--limit` のデフォルトが 100 なのは「安全に小刻みに回す」ためです。  
> 全回ししたいときは `--run-until-empty` を使います（リトライ/停止条件も内蔵）。

### 3.5 品質レポート（スクレイプが進むほど価値が上がる）
```bash
python scripts/report_quality_v4.py --db netkeiba.db
```

- マスタのカバレッジや欠損状況を `artifacts/` に JSON/CSV で出力（想定）

### 3.6 FeaturePack v1（feature_table_v4）生成
```bash
python scripts/build_feature_table_v4.py --db netkeiba.db
```

### 3.7 学習・評価・ROI（v4）
```bash
python scripts/train_eval_v4.py --db netkeiba.db
```

- デフォルト split：`year_based`
  - train = 2021–2023
  - val   = 2023Q4（サブセット）
  - test  = 2024
- `--split-mode year_based|date_based` に対応（実装側の引数名に追従）

### 3.8 Feature Diagnostics（特徴量診断）

特徴量が「効いているか/効いていないか」を診断するための機能です。

```bash
# 学習後に診断を実行
python scripts/train_eval_v4.py --db netkeiba.db --feature-diagnostics

# 既存モデルに対して診断のみ実行（学習をスキップ）
python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only

# 高速モード（Permutation Importance をスキップ）
python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only --no-permutation

# 旧モデル（v4以前）を使用（feature_columns ファイル名の自動フォールバック）
python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only \
  --model-path models/lgbm_target_win.txt

# 特定の特徴量を除外して診断
python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only \
  --exclude-features-file exclude_features.txt
```

#### 特徴量除外ファイルの形式

`--exclude-features-file` で指定するファイルは、1行1特徴量名の形式です。
`#` で始まる行はコメントとして無視されます。

```text
# 除外する特徴量リスト
horse_weight
horse_weight_diff
is_first_run
# 市場関連も除外
market_win_odds
market_popularity
```

#### Fail-Soft 設計

診断機能は「部分的な失敗があっても続行する」設計です：
- 一部の特徴量がデータに存在しなくても、利用可能な特徴量で診断を続行
- Permutation Importance が失敗しても、LightGBM 重要度は出力
- Segment Performance が計算できない場合でも、警告を記録して他の診断は完了
- **スキーマ不一致時**（`--diagnostics-only` で旧モデルを使用時等）は Segment Performance をスキップし、警告として記録（エラーではない）

診断結果の JSON には `warnings` と `errors` フィールドが含まれ、
どのような問題が発生したかを確認できます。

#### Windows 環境での注意事項

Windows で日本語パス（例：`C:\Users\ユーザー\デスクトップ\`）を含む環境では、
LightGBM の `save_model()` や `Booster(model_file=...)` が失敗する場合があります。

**1. モデル保存時のフォールバック**

`save_model()` が失敗した場合、自動的に `model_to_string()` + Python ファイル書き込みでフォールバック保存を試みます：
- 成功時：WARNING ログに `"Saved model via model_to_string fallback"` と出力
- 両方失敗時：モデルファイルは保存されず、in-memory モデルで評価・診断を続行

```
# 正常ログ例
Saved model to models/lgbm_target_win_v4.txt

# フォールバック時のログ例
save_model failed (...), trying model_to_string fallback...
Saved model via model_to_string fallback to models/lgbm_target_win_v4.txt
```

**2. --feature-diagnostics での in-memory モデル使用**

`--feature-diagnostics` オプション使用時は、学習後にディスクからモデルを再ロードせず、
in-memory のモデルをそのまま使用します。これにより日本語パス環境でもエラーなく診断が実行できます。

**3. --diagnostics-only でのモデルロードフォールバック**

`--diagnostics-only` モードでは `load_booster()` ヘルパーを使用してモデルをロードします。
`lgb.Booster(model_file=...)` が失敗した場合、Python の `read_text()` でファイルを読み込み、
`lgb.Booster(model_str=...)` 経由でロードを試みます：

```
# フォールバック時のログ例
lgb.Booster(model_file=...) failed (...), trying model_str fallback...
Loaded model via model_str fallback: models/lgbm_target_win_v4.txt
```

#### 出力内容

1. **LightGBM 標準重要度**（gain / split）
   - `feature_importance_target_win_test_v4.csv`

2. **Permutation Importance**（複数メトリクス）
   - AUC, LogLoss, Top1/3/5 Hit Rate, MRR
   - `permutation_importance_target_win_test_v4.csv`

3. **Feature Group Importance**（グループ別集計）
   - horse_form, jockey_trainer, pedigree, base_race 等
   - `group_importance_target_win_test_v4.csv`

4. **Segment Performance**（セグメント別パフォーマンス）
   - surface_id（芝/ダート）、distance_cat（距離カテゴリ）、track_condition_id（馬場状態）別
   - `segment_performance_target_win_test_v4.csv`

5. **診断レポート**（テキスト & JSON）
   - `diagnostics_report_target_win_test_v4.txt`
   - `diagnostics_summary_target_win_test_v4.json`（warnings/errors 含む）

---

### 3.9 前日運用モード（Pre-race Mode）

レース当日の馬体重が確定する前に予測を行いたい場合は、`--mode pre_race` を使用します。

```bash
# 前日運用モード（当日体重を除外して学習・評価）
python scripts/train_eval_v4.py --db netkeiba.db --mode pre_race

# デフォルトモード（全特徴量を使用）
python scripts/train_eval_v4.py --db netkeiba.db --mode default
```

#### 使い分け

| モード | 用途 | 除外される特徴量 |
|--------|------|------------------|
| `default` | レース当日（馬体重確定後）の予測 | なし |
| `pre_race` | レース前日〜当日朝（馬体重未確定）の予測 | h_body_weight, h_body_weight_diff, h_body_weight_dev, market_* |

#### 体重特徴量の意味

**前日安全版（Pre-race Safe）** - `--mode pre_race` でも使用可能：

| 特徴量 | 説明 |
|--------|------|
| `h_avg_body_weight` | 過去走の平均馬体重 |
| `h_last_body_weight` | 直近出走時の馬体重 |
| `h_last_body_weight_diff` | 直近出走時の馬体重増減 |
| `h_recent3_avg_body_weight` | 直近3走の平均馬体重 |
| `h_recent3_std_body_weight` | 直近3走の馬体重標準偏差（安定性指標） |
| `h_recent3_body_weight_trend` | 直近3走の体重トレンド（正=増量傾向、負=減量傾向） |
| `h_body_weight_z` | 馬体重 z-score（直近体重と平均体重の乖離度） |

**当日版（Race-day Only）** - `--mode pre_race` では除外：

| 特徴量 | 説明 |
|--------|------|
| `h_body_weight` | 今走の馬体重（当日計測） |
| `h_body_weight_diff` | 今走の馬体重増減（当日計測） |
| `h_body_weight_dev` | 馬体重偏差（平均との差） |

#### 除外特徴量の定義ファイル

`config/exclude_features/pre_race.txt` に除外対象が定義されています：

```text
# 当日体重（馬体重計測は当日朝に行われる）
h_body_weight
h_body_weight_diff
h_body_weight_dev

# 当日市場情報
market_win_odds
market_popularity
market_odds_rank
```

独自の除外リストを使用したい場合は `--exclude-features-file` オプションを使用してください。

---

### 3.10 前日締め運用（Pre-day Cutoff Operation）【実験的】

実運用では、レース前日の時点でオッズ・人気を取得して予測を確定させたい場合があります。
この「前日締め運用」をサポートするため、`odds_snapshots` テーブルとスナップショットベースの評価機能を実装しています。

> **注**: この機能は実験的です。まずは Feature Diagnostics でベースモデルの特徴量選抜を優先してください。

#### オッズスナップショットの取得

```bash
# 単一レースのオッズ取得（netkeiba API 経由）
python scripts/fetch_odds_snapshots.py --race-id 202406050811

# 日付指定で全レースのオッズ取得
python scripts/fetch_odds_snapshots.py --date 2024-12-28

# 明日のレースのオッズ取得 (cron用)
python scripts/fetch_odds_snapshots.py --tomorrow
```

#### スナップショットベースの評価

```bash
# decision_cutoff を指定してROI評価
python scripts/train_eval_v4.py --db netkeiba.db \
  --decision-cutoff "2024-12-27T21:00:00"

# スナップショットを使わない場合（race_results の最終人気を使用）
python scripts/train_eval_v4.py --db netkeiba.db --no-snapshots
```

#### 運用フロー例

1. **前日 21:00**: `fetch_odds_snapshots.py --tomorrow` でスナップショット取得
2. **当日朝**: モデル予測 → 賭け対象選定（decision_cutoff = 前日 21:00）
3. **レース後**: 結果反映 → ROI 評価（スナップショットの人気を使用）

> **注**: `odds_snapshots` テーブルが存在しない/データがない場合は、自動的に `race_results` の人気にフォールバックします。

---

## 4. FeaturePack v1（feature_table_v4）の考え方

### 4.1 特徴量グループ（目安）
- **base_race**：レース基本情報（場所/サーフェス/距離/クラス等）
- **horse_form**：馬の as-of 履歴統計（近況・安定性など）
- **pace_position**：コーナー/位置取り系（欠損率も品質で監視）
- **class_prize**：クラス推移、賞金（as-of 集計で安全に）
- **jockey_trainer**：騎手/調教師の as-of 統計
- **pedigree**：血統特徴（5代血統をハッシュ化してベクトル化）
- **market（任意）**：当日最終オッズ等は “列分離” し、デフォルト学習では外す運用を想定

### 4.2 血統特徴（5代血統）
- `horse_pedigree` に正規化保存（horse_id / generation / position をキー）
- 5代血統（最大 62 祖先）を **トークン化 → ハッシュ化 → 固定次元ベクトル** に変換
- 直系（sire/dam/bms）と 5代全体で別ベクトルを持つ設計（例：512 + 128 dims）

---

## 5. リーク防止（fail-fast）と再現性

### 5.1 as-of 集計を “唯一の正” にする
- **今走より未来の情報を混ぜない**ことが最重要
- masters の `career_*`（総賞金など）は便利だが、時点が曖昧になりがち  
  → **学習特徴としては使用禁止**（as-of 集計で代替する）

### 5.2 禁止列・分離列
- `target_*`, `finish_*`, `payout`, `race_id`, `horse_id` は当然除外
- 当日情報（馬体重・当日最終オッズ等）は **列として存在しても “デフォルト学習では除外”**
- 「当日馬場」はシナリオ/UI 領域（ベースモデルへは基本入れない）

### 5.3 artifacts
- 使った特徴量一覧、分割設定、モデル、重要度、品質レポートなどを `artifacts/` へ集約（想定）

---

## 6. 旧テーブル（v2/v3）について（レガシー）

- `feature_table_v2`：素材テーブル（初期）
- `feature_table_v3`：9軸（ax1〜ax9）テーブル

> 現在の主戦力は v4 です。  
> v2/v3 は比較・回帰テスト・検証のために残してOK（ただし新規改善は v4 優先）。

---

## 7. シナリオ補正 & UI（概要）
- 目的：当日要素（馬場・バイアス・ペース・隊列）を人間が入力し、ベース確率を補正
- 入力：ベース確率（win/in3）＋人間指定（ペース/バイアス/当日馬場など）
- 出力：補正後確率＋「得する馬/損する馬」の差分

---

## 8. 開発

### 8.1 テスト
```bash
python -m pytest -q
# 例：v4 だけ
python -m pytest tests/test_features_v4.py -v
```

### 8.2 便利コマンド（任意）
```bash
# マイグレーション状況
python scripts/run_migrations.py --db netkeiba.db --status

# マスタ進捗
python scripts/fetch_masters.py --db netkeiba.db --report
```

---

## 9. Roadmap（ざっくり）
- ✅ Road 1：DB安全化（idempotent migrations + UPSERT）
- ✅ Road 2 / 2.5：マスタ（horses/jockeys/trainers）+ 5代血統
- ✅ Road 3：FeaturePack v1（feature_table_v4）+ 学習/評価/ROI + 品質レポート
- ✅ Road 3.5：Feature Diagnostics（特徴量診断機能）
  - LightGBM gain/split 重要度
  - Permutation Importance（AUC, LogLoss, Top1/3/5, MRR）
  - Feature Group Importance
  - Segment Performance（芝/ダート、距離、馬場等）
- ✅ Road 3.6：Pre-race Mode（前日運用モード）
  - 前日安全版体重特徴量（h_last_body_weight, h_recent3_* 等）
  - `--mode pre_race` オプション（当日体重を除外して学習）
  - BodyWeightContext（LLM 説明用の体重コンテキスト）
- 🔄 現在の優先事項：
  - Feature Diagnostics を使った特徴量選抜・改善
  - ベースモデルの精度向上（market特徴量は使わない方針）
- ⏭ 次：
  - ROI 改善ループ（calibration / 閾値戦略 / 予測の説明可能性）
  - UI/シナリオ補正の精度改善（人間の "想定" をより再現可能に）
  - 前日締め運用の本格化（odds_snapshots の活用）

## Chat Handoff（新チャット貼り付け用の現在地）

- 今日の日付：2025-12-29（JST）
- 前提：Road1〜Road3.6（Pre-race Mode）までpull済み
- 環境：
  - sqlite3 コマンドは未導入（Pythonスクリプトで運用）
  - dev依存は導入済み：`pip install -r requirements-dev.txt`
- DB：`netkeiba.db` をそのまま使用（DBコピーはしない方針）
- 最新の実装：
  - Feature Diagnostics：`--feature-diagnostics` / `--diagnostics-only` フラグ
    - Fail-soft 設計：部分エラーでも続行、warnings/errors を記録
    - `--exclude-features-file` オプション追加
    - feature_columns ファイル名フォールバック（v4 → legacy）
  - odds_snapshots：netkeiba API 経由でオッズ取得（8馬制限バグ修正済み）
  - Snapshot-based market features：`--include-snapshots` / `--decision-cutoff` オプション
- 次にやること（推奨順）：
  1) Feature Diagnostics 実行：`python scripts/train_eval_v4.py --db netkeiba.db --diagnostics-only --no-permutation`
  2) 診断結果を確認して、不要な特徴量を特定
  3) 特徴量選抜後に再学習・評価
  4) 閾値戦略の最適化（`--roi-sweep`）
