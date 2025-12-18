# Scenario-Aware Racing Intelligence (keiba-scenario-ai)

> **人間と協業する競馬予想AI**  
> 「ベースモデル × シナリオ補正 × わかりやすい UI」で、  
> 競馬ファンが自分のシナリオを反映した予想を作れることを目指すプロジェクトです。

- データソース：**JRA（db.netkeiba.com）**
- 期間：**2021–2024 年の全レース（平地中心）**
- DB：SQLite（リポジトリ直下 `netkeiba.db`）
- ベースモデル：複勝 in3 確率予測（ロジスティック回帰 → 将来 LightGBM 等に発展）
- シナリオ補正：ペース・バイアス・脚質など人間の入力でベース予測を調整

---

## 1. リポジトリ概要

大きく 3 レイヤ構成です。

1. **Data / Features**
   - `src/ingestion/`：スクレイピング～DB格納
   - `scripts/build_horse_results.py`：馬の履歴テーブル
   - `scripts/build_feature_table_v2.py`：学習用テーブル
2. **Base Model**
   - `scripts/train_eval_logistic.py`：ロジスティック回帰による複勝 in3 予測
   - 今後 LightGBM などのモデルを追加予定
3. **Scenario & UI**
   - `scenario/`：ペース・バイアス・脚質を反映する補正レイヤ
   - `ui/`：JSON を読んで展開シナリオを可視化する簡易フロント

---

## 2. セットアップ

### 2.1 Python 環境

- Python 3.11+ 推奨
- 必要ライブラリ（例）：

```bash
pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn lightgbm requests beautifulsoup4
```

（実際の requirements はリポジトリ内のファイルを参照）

### 2.2 DB の位置

- プロジェクトの **正とする DB** はリポジトリ直下の：

```text
netkeiba.db
```

- ingestion・スクリプトはすべて `--db` オプションでパスを受け取る形に統一しています。
- 古いコードには `src/netkeiba.db` や `data/keiba.db` などの記述が残っている場合がありますが、  
  **現在は `netkeiba.db`（リポジトリ直下）を優先**してください。

---

## 3. データパイプライン

### 3.1 JRA レース情報の ingestion

レース一覧 → 詳細ページ → DB 保存までを一括で行います。

```bash
cd src
python -m ingestion.ingest_runner \
  --start-year 2021 --end-year 2024 \
  --db ../netkeiba.db
```

- 必要に応じて年を分割して実行することもできます。

```bash
# 例：2021 年だけ
cd src
python -m ingestion.ingest_runner \
  --start-year 2021 --end-year 2021 \
  --db ../netkeiba.db
```

- 主なテーブル：
  - `races`：レース条件（日時・コース・クラス・頭数など）
  - `race_results`：各馬の着順・タイム・人気など
  - `payouts`：馬券種別ごとの払戻（本プロジェクトは主に `複勝` を利用）
  - `corners`, `lap_times`, `horse_laps`, `short_comments`：  
    コーナー通過順・ラップ・馬ごとのラップ・コメント（シナリオ／将来拡張用）

### 3.2 馬ごとの履歴テーブル：`horse_results`

```bash
python scripts/build_horse_results.py --db netkeiba.db
```

- `horse_results(horse_id, race_id, race_date, start_index, finish_order, last3f, odds, popularity, prize, ...)`
- 各馬の出走履歴を `race_date` でソートし、`start_index` を付与。
- ここから **履歴特徴量 `hr_*`** を計算します（リーク防止のため「今走を除外」）。

### 3.3 学習用テーブル：`feature_table_v2`

```bash
python scripts/build_feature_table_v2.py --db netkeiba.db
```

- 役割：
  - 旧 `feature_table`（v1）のレース条件・簡易履歴集計に加え、
  - `horse_results` から計算した `hr_*` 履歴特徴を横持ちで追加。
- 主な列：
  - キー・ターゲット：
    - `race_id, horse_id, target_win, target_in3, target_value`
  - レース条件：
    - `course, surface, surface_id, distance, distance_cat`
    - `track_condition, track_condition_id, field_size, race_class`
    - `race_year, race_month, waku, umaban, horse_weight, horse_weight_diff`
  - v1 由来の履歴集計：
    - `n_starts_total, win_rate_total, in3_rate_total`
    - `n_starts_dist_cat, win_rate_dist_cat, in3_rate_dist_cat` など
  - 履歴特徴 `hr_*`（近況・成長軸）
    - 生涯成績系（start数・勝率・平均着順・賞金など）
    - 直近3走系（平均着順・ベスト着順・上がり・人気など）
    - 直近5走系（平均着順・大敗回数など）
    - トレンド系（成績の上向き／下り坂指標）
    - 間隔（`hr_days_since_prev` など）

> 補足：  
> 以前は個別ラップから作る `hlap_*` 特徴を学習テーブルに入れていましたが、  
> データの制約などにより、**現行の `feature_table_v2` では `hlap_*` 系は一度外しています**。  
> ラップ情報はテーブル（`horse_laps` 等）としては残っており、  
> シナリオレイヤや将来モデルで再利用する想定です。

---

## 4. ベースモデル（ロジスティック回帰）の評価

### 4.1 学習・評価スクリプト

```bash
python scripts/train_eval_logistic.py --db netkeiba.db
```

- データ：
  - `feature_table_v2`（なければ `feature_table` をフォールバック）
  - `race_results`（`horse_no`）
  - `payouts`（`bet_type = '複勝'`）
- スプリット：
  - train: race_year ∈ {2021, 2022, 2023}
  - test : race_year = 2024
- モデル：
  - `LogisticRegression(max_iter=1000)`（複勝 in3 の二値分類）

### 4.2 出力される指標（現状）

- **Global Metrics**
  - Accuracy（0.5 閾値）
  - ROC-AUC
  - Log Loss
  - Brier Score
- **Ranking Evaluation**
  - テストレース数
  - Top1 / Top2 / Top3 カバー率
  - 複勝圏内の馬のうち、最も高く予測された馬の平均順位
- **Strategy: Top1 All Races**
  - 全レースで「予測確率最大の 1頭」を複勝 1点買いした場合の
    - ベット数 / 的中数 / 的中率 / 投資 / 払戻 / 回収率
- **Strategy: Top1 with Thresholds**
  - 閾値（0.25 / 0.30 / 0.35 / 0.40）以上のレースだけベットした場合の
    - Bets / Hits / Hit% / ROI% / AvgProb
- **Calibration Evaluation**
  - 10 ビンでのキャリブレーションと ECE（Expected Calibration Error）
- **Debug Information**
  - 複勝払戻の欠損率
  - 1レースあたりの払戻件数（平均・最小・最大）
  - 払戻件数が 3 未満のレース数 など

> 現時点のログ例（2021–23 学習 / 2024 テスト、特徴量 46 本）では、  
> Accuracy ≒ 0.79, ROC-AUC ≒ 0.74, 複勝 top1 全レース買いの回収率 ≒ 79% 程度。  
> まだ「控除率を超える水準」には届いておらず、  
> **「現状の実力を測るためのベンチマーク」** として位置づけています。

---

## 5. シナリオ補正レイヤ & UI（概要）

### 5.1 シナリオ補正レイヤ（scenario/）

- 入力：
  - ベースモデルの win / in3 確率
  - 人間が指定する：
    - 予想ペース（スロー / ミドル / ハイ）
    - 馬場バイアス（内/外・前/後）
    - 想定隊列（逃げ・先行・差し・追い込み） など
- 出力：
  - 補正後の win / in3 確率
  - シナリオ前後で「どの馬がどれだけ得／損しているか」の差分情報

### 5.2 UI（ui/）

- `run_scenario_prediction.py` で JSON を生成し、
- `run_scenario_ui.py` で Web UI を立ち上げ、
- ブラウザ上でシナリオごとの変化を可視化する構成になっています。

（詳細な使い方は各スクリプト内の docstring を参照）

---

## 6. ヒューマン予想家からのインスピレーション

プロジェクトでは、

- **キムラヨウヘイ**
- **のれん**
- **メシ馬**

といった予想家の G1 / 重賞の見解を md で整理し、  
そこから **「共通して重視している 9 つの視点（軸）」** を抽出しています。

この 9 軸は、

- 近況・成長
- 条件適性（距離 / コース / 馬場）
- 地力・レースレベル
- 位置取り・脚質 × 展開
- 馬場バイアス・枠順
- ローテーション・仕上がり
- メンタル・レースセンス
- 陣営・騎手意図
- オッズ・期待値バランス

といった要素で構成されており、  
**「人間の予想ロジックをどう特徴量に落とすか」** を設計するためのフレームとして使っています。

今後、`feature_table_v2` 以降の拡張では、  
この 9 軸をガイドラインにして特徴量を 100 本以上に増やしていく予定です。

---

## 7. 今後のロードマップ（ざっくり）

1. **評価ロジックの安定化**
   - `train_eval_logistic.py` を、今後も全モデル共通で使える「物差し」として維持。
   - 評価時に必要なログ・サマリを整える。
2. **特徴量の拡張（100+ 本へ）**
   - `horse_results` ベースの hr_* をさらに強化（近況・成長軸）。
   - `horse_profiles`（馬のプロフィール）、`pedigree`（血統）、`jockey_stats` / `trainer_stats` などのテーブルを追加。
   - レースレベル・オッズ系の特徴量も充実させる。
3. **モデル側の強化**
   - LightGBM / XGBoost などツリー系モデルでの評価。
   - オッズを目的関数に組み込んだ期待値志向モデルの検討。
   - キャリブレーションレイヤ（Platt / Isotonic 等）の追加。
4. **シナリオレイヤとの統合強化**
   - ベースモデル出力のキャリブレーションを前提に、  
     シナリオ補正後の確率も一貫性を維持する設計へ。

---

## 8. 開発者・エージェント向けメモ

- DB スキーマを変更する場合は、
  - 影響するスクリプトを確認した上で、
  - README / AGENT に仕様と意図を追記してください。
- 新しい特徴量を追加したら、
  - 欠損率・分布の sanity check
  - 「未来の情報を混ぜていないか」のチェック
  を忘れずに。
- `AGENT.md` はコードを書くエージェント向けの詳細ガイドラインです。  
  エージェント側の修正方針はそちらを参照してください。

---

最終更新日: **2025-12-18**
