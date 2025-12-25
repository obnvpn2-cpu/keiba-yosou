# Scenario-Aware Racing Intelligence (keiba-scenario-ai)

> **人間と協業する競馬予想AI**
> 「ベースモデル × シナリオ補正 × わかりやすいUI」で、
> 競馬ファンが “自分の想定（ペース/バイアス等）” を反映した予想を作れることを目指すプロジェクトです。

- データソース：JRA（db.netkeiba.com）
- 期間：2021–2024 年の全レース（平地中心）
- DB：SQLite（リポジトリ直下 `netkeiba.db`）
- ベースモデル：複勝 in3（3着以内）確率予測
  - Logistic Regression（ベンチマーク）
  - LightGBM（主力・比較対象）
- シナリオ補正：ペース・バイアス・脚質など人間の入力でベース予測を調整
- 学習用特徴量：**「前日までに確定している情報」＋「過去走履歴（今走を除外）」** を基本に設計

---

## 0. まず押さえる方針（重要）

### 0.1 「前日までに確定している情報」で早出ししたい
実運用で **当日情報（馬体重・最終オッズ等）を揃える時間がない**前提。

- ✅ 前日までに確定：枠順/馬番、距離、コース、クラス、出走頭数、ローテ（日数）など
- ❌ 当日（直前）情報：馬体重・馬体重増減、最終オッズ（市場情報）など  
  → **ベースモデルには入れない（再現不能）**

> ※「過去レースの odds/popularity」は “履歴” として扱う（ax9）  
> ただし **今走の最終オッズ** は入れない。

### 0.2 `track_condition` / `track_condition_id` はベースモデルに入れない
当日の馬場状態・トラックバイアスは **シナリオ/UI で人間が指定して補正する領域**。
ベースモデルは「ニュートラルな地力・近況中心の予測」を担当する。

- `track_condition(_id)` は “リーク防止/スコープ分離” として **禁止列** 扱い（後述）

---

## 1. リポジトリ概要（3レイヤ）

1) **Data / Features**
   - `src/ingestion/`：スクレイピング～DB格納
   - `scripts/build_horse_results.py`：馬の履歴テーブル（horse_results）
   - `scripts/build_feature_table_v2.py`：学習素材テーブル（v2）
   - `scripts/build_feature_table_v3.py`：**9軸特徴量テーブル（v3）**
2) **Base Model**
   - `scripts/train_eval_logistic.py`：ロジスティック回帰（複勝 in3）
   - `scripts/train_eval_lgbm.py`：**LightGBM（複勝 in3）**
3) **Scenario & UI**
   - `scenario/`：ペース・バイアス・脚質を反映する補正レイヤ
   - `ui/`：JSON を読んで展開シナリオを可視化する簡易フロント

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

- スクリプトは基本 `--db` で受け取る  
- 古いパス表記（`data/keiba.db` 等）が残っている場合があるが、**今は `netkeiba.db` を優先**

---

## 3. データパイプライン

### 3.1 JRA レース情報の ingestion

```bash
cd src
python -m ingestion.ingest_runner \
  --start-year 2021 --end-year 2024 \
  --db ../netkeiba.db
```

主なテーブル：
- `races`：レース条件（日時・コース・クラス・頭数など）
- `race_results`：各馬の着順・人気・騎手/厩舎（ある場合）など
- `payouts`：馬券種別ごとの払戻（本PJは主に「複勝」を利用）
- `corners`, `lap_times`, `horse_laps`, `short_comments`：将来拡張（シナリオ/説明/UI）

### 3.2 馬ごとの履歴テーブル：`horse_results`

```bash
python scripts/build_horse_results.py --db netkeiba.db
```

- 各馬の出走履歴を `race_date` でソートし、`start_index` を付与
- ここから履歴特徴量（`hr_*` や ax 系の一部）を計算
- **リーク防止**：履歴集計は基本 **shift(1)**（今走の情報を混ぜない）

> 注意：DBスキーマにより `horse_results` 側に `jockey_id/trainer_id` が無い場合がある。  
> その場合は `race_results` から JOIN して拾う実装（build_feature_table_v3 側で吸収する）。

### 3.3 学習素材テーブル：`feature_table_v2`

```bash
python scripts/build_feature_table_v2.py --db netkeiba.db
```

- v2 は “素材”：
  - レース条件（前日確定）＋簡易履歴集計＋履歴特徴 `hr_*`
- 当日情報（馬体重など）や、シナリオ領域（当日馬場など）は
  - **テーブルに存在してもベースモデルでは使わない**（学習側で禁止列排除）

### 3.4 学習用テーブル（9軸完成版）：`feature_table_v3`

```bash
python scripts/build_feature_table_v3.py --db netkeiba.db
# 開発用（行数制限）
python scripts/build_feature_table_v3.py --db netkeiba.db --limit 10000
```

- v3 は **v2 をベースに、9軸の ax* 特徴量を追加した学習テーブル**
- 生成時に corners 由来の欠損率（例：c1_pos/c4_pos）をログで出す（欠損が多いと ax6 系の有効行が減る）

---

## 4. 9つの軸（実装済み：feature_table_v3）

キムラヨウヘイ / のれん / メシ馬 の G1・重賞見解を整理し、共通して重視する視点を 9軸に分解。
**ベースモデルは「前日確定情報＋過去履歴」で作る**（当日要素はシナリオ側）。

> 命名規則：`ax{軸番号}_*`（feature_table_v3 に格納）

### 軸1：近況・成長（ax1_）… 9列
直近のフォーム、改善/悪化、トレンド。
- 例：`ax1_last_finish`, `ax1_recent3_avg_finish`, `ax1_recent3_finish_slope`, `ax1_best_finish_recent5` など

### 軸2：距離・コース条件適性（ax2_）… 17列
同サーフェス/同コース/同距離カテゴリ、距離替わり・コース替わり。
- 例：`ax2_same_surface_in3_rate`, `ax2_same_course_starts`, `ax2_dist_change_prev`, `ax2_is_surface_change` など

### 軸3：地力・レースレベル（ax3_）… 6列
クラス経験/賞金などで “格” を表現。
- 例：`ax3_best_class_recent5`, `ax3_total_prize_recent5`, `ax3_curr_vs_best_class` など

### 軸4：ローテーション・臨戦過程（ax4_）… 7列
休み明け/詰め/昇級降級など。
- 例：`ax4_rest_days`, `ax4_is_short_turnaround`, `ax4_class_diff`, `ax4_is_class_up` など

### 軸5：枠・位置取り（ax5_）… 5列
前日確定の枠順/馬番を正規化し、内外を表現。
- 例：`ax5_waku_pct`, `ax5_is_inner`, `ax5_gate_distance_to_edge_norm` など

### 軸6：脚質・位置取り × 展開（ax6_）… 9列
corners 由来の位置割合/位置取り変動、レース内の先行勢密度など（※展開の最終反映はシナリオ側）。
- 例：`ax6_recent3_c1_pos_pct_mean`, `ax6_recent5_pos_gain_mean`, `ax6_race_front_runner_count` など

### 軸7：安定性・ブレ（ax7_）… 4列
着順/上がり/人気/位置変動の分散で “安定感” を表す。
- 例：`ax7_finish_std_recent5`, `ax7_last3f_std_recent5` など

### 軸8：騎手・厩舎（ax8_）… 2列
騎手/厩舎の累積 in3 率（as-of：今走より前まで）。
- 例：`ax8_jockey_in3_rate_total_asof`, `ax8_trainer_in3_rate_total_asof`
- 注意：DBによっては `jockey_id/trainer_id` の所在が `race_results` 側のみの場合あり（build_v3 側で JOIN して吸収）

### 軸9：オッズ/人気 proxy（ax9_）… 4列
“市場とのズレ” を **過去走の odds/popularity** から proxy 化（今走の最終オッズは使わない）。
- 例：`ax9_avg_win_odds_recent5`, `ax9_finish_minus_popularity_mean_recent5` など

---

## 5. ベースモデル評価（Logistic / LightGBM）

### 5.1 ロジスティック回帰（ベンチマーク）

```bash
python scripts/train_eval_logistic.py --db netkeiba.db
```

- データ：`feature_table_v3` → 無ければ v2 → 無ければ v1（スクリプト内優先順）
- split：
  - train: 2021–2023
  - test : 2024
- 出力：Global / Ranking / Strategy / Calibration / Debug

### 5.2 LightGBM（主力）

```bash
# 基本
python scripts/train_eval_lgbm.py --db netkeiba.db

# rr_（レース内相対）を追加
python scripts/train_eval_lgbm.py --db netkeiba.db --add-rr --rr-kind both

# 特徴量デバッグ（使用列の保存）
python scripts/train_eval_lgbm.py --db netkeiba.db --debug-features
```

- 保存物（例）：
  - `models/lgbm_in3.txt`
  - `models/lgbm_in3_features.txt` / `.json`
  - `models/lgbm_in3_importance.csv`

---

## 6. リーク防止（fail-fast）と再現性

学習スクリプトは **禁止列（リーク/スコープ分離/再現不能）** を持ち、数値列でも feature に入れない。

- 例：`target_*`, `finish_*`, `payout`, `race_id`, `horse_id`
- 早出し方針：`horse_weight`, `horse_weight_diff`（当日情報）は禁止
- スコープ分離：`track_condition(_id)` は禁止（シナリオ側で扱う）

> 禁止列は `PROHIBITED_EXACT` / `PROHIBITED_PATTERNS` で管理。  
> もし混入したら **fail-fast で例外**（安全側）。

### artifacts（再現性）
- `artifacts/features_used.txt`：実際に使った feature 列一覧
- rr を使った場合：rr 列の一覧も出力

---

## 7. 評価出力（共通）

- Global Metrics：Accuracy / ROC-AUC / LogLoss / Brier
- Ranking：Top1/2/3 カバー率、平均 rank など
- Strategy：
  - Top1 全レース
  - 閾値付き
  - （追加）top確率ギャップ診断（gap12 等）とギャップ閾値戦略テーブル（スクリプト対応時）
- Calibration：ビン集計＋ ECE
- Debug：払戻欠損、レースあたり払戻数、欠損除外の状況など

---

## 8. シナリオ補正 & UI（概要）

- 目的：当日要素（馬場・バイアス・ペース・隊列）を人間が入力し、ベース予測を補正
- 入力：ベース確率（win/in3）＋人間指定（ペース/バイアス/当日馬場など）
- 出力：補正後確率＋「得する馬/損する馬」の差分

---

## 9. 開発メモ（衝突しにくい PR 作成手順）

AIエージェント（Codex/Claude）併用では、同一ファイルを同時に触ってコンフリクトしがち。
以下テンプレで衝突確率を下げる。

1) 必ず最新 main から開始
```bash
git checkout main
git fetch origin
git pull --ff-only origin main
```

2) 1PR=1目的で作業ブランチ
```bash
git checkout -b feature/<short-name>
```

3) 変更範囲は最小化（整形だけ変更を混ぜない）
- 触るファイルは 1〜3 本を目標
- import並び替え等の “意味のない差分” を入れない

4) push 前に rebase（作業が長い場合）
```bash
git fetch origin
git rebase origin/main
```

5) push & PR
```bash
git push -u origin feature/<short-name>
```

※ README 更新は衝突源になりやすいので、必要なら docs-only PR と分離する