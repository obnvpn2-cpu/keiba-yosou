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
- ⏭ 次：
  - ROI 改善ループ（特徴量選抜 / calibration / 閾値戦略 / 予測の説明可能性）
  - UI/シナリオ補正の精度改善（人間の “想定” をより再現可能に）

## Chat Handoff（新チャット貼り付け用の現在地）

- 今日の日付：2025-12-24（JST）
- 前提：Road1〜Road3（feature_table_v4 / v4パイプライン / 品質レポート）までpull済み
- 環境：
  - sqlite3 コマンドは未導入（Pythonスクリプトで運用）
  - dev依存は導入済み：`pip install -r requirements-dev.txt`
- DB：`netkeiba.db` をそのまま使用（DBコピーはしない方針）
- スクレイピング進捗：
  - horse_pedigree：100% 完了（26,128/26,128）
  - jockey / trainer：全件取得を進行中（`--run-until-empty`）
  - horses：全件取得を進行中（`--run-until-empty`）
- いま回している/直近で回すコマンド：
  - 進捗確認：`python scripts/fetch_masters.py --db netkeiba.db --report`
  - jockey：`python scripts/fetch_masters.py --db netkeiba.db --entity jockey --run-until-empty --sleep-min 3.0 --sleep-max 5.0`
  - trainer：`python scripts/fetch_masters.py --db netkeiba.db --entity trainer --run-until-empty --sleep-min 3.0 --sleep-max 5.0`
  - horse：`python scripts/fetch_masters.py --db netkeiba.db --entity horse --run-until-empty --sleep-min 3.0 --sleep-max 5.0`
- 次にやること（最短）：
  1) masters を取り切る（reportで Pending=0 を目指す）
  2) 品質レポート：`python scripts/report_quality_v4.py --db netkeiba.db`
  3) v4特徴量生成：`python scripts/build_feature_table_v4.py --db netkeiba.db`
  4) 学習/評価/ROI：`python scripts/train_eval_v4.py --db netkeiba.db`
