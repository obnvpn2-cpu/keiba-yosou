# Scenario-Aware Racing Intelligence (keiba-scenario-ai)

> **人間と協業する競馬予想AI**  
> 「ベースモデル × シナリオ補正 × わかりやすい UI」で、  
> 競馬ファンが“自分の想定（ペース/バイアス等）”を反映した予想を作れることを目指すプロジェクトです。

- データソース：**JRA（db.netkeiba.com）**
- 期間：**2021–2024 年の全レース（平地中心）**
- DB：SQLite（リポジトリ直下 `netkeiba.db`）
- ベースモデル：複勝 in3 確率予測（ロジスティック回帰 → 将来 LightGBM 等に発展）
- シナリオ補正：ペース・バイアス・脚質など人間の入力でベース予測を調整

---

## 0. まず押さえる方針（重要）

### 0.1 「前日までに確定している情報」で早出ししたい
実運用で **当日情報（馬体重・最終オッズ等）を揃える時間がない**前提。

- ✅ 前日までに確定：枠順/馬番、距離、コース、クラス、出走頭数、ローテ（日数）など
- ❌ 当日（直前）情報：馬体重・馬体重増減、最終オッズ（＝市場情報）など → **ベースモデルには入れない（再現不能）**

### 0.2 `track_condition` / `track_condition_id` はベースモデルに入れない
当日の馬場状態・バイアスは **シナリオ/UI で人間が指定して補正する領域**とし、
ベースモデルは「ニュートラルな地力・近況中心の予測」を担当します。

> そのため、現状の `train_eval_logistic.py` でも  
> `track_condition(_id)` は **リーク防止/スコープ分離の禁止列**として扱っています（後述）。

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
- 重要：`feature_table_v2` は “素材” です  
  当日情報（例：馬体重）や、シナリオ領域（例：当日馬場）は **テーブルに存在しても、ベースモデルには入れません**  
  （`train_eval_logistic.py` 側で禁止列として排除）。

#### 3.3.1 主な列（概念整理）

- キー・ターゲット：
  - `race_id, horse_id, target_win, target_in3, target_value`
- レース条件（前日までに確定するもの中心）：
  - `course, surface, surface_id, distance, distance_cat`
  - `field_size, race_class`
  - `race_year, race_month, waku, umaban`
- 当日情報（※ベースモデルでは使わない）：
  - `horse_weight, horse_weight_diff`（直前情報）
- シナリオ領域（※ベースモデルでは使わない）：
  - `track_condition, track_condition_id`
- v1 由来の履歴集計（前走までの集計）：
  - `n_starts_total, win_rate_total, in3_rate_total, avg_finish_total, std_finish_total`
  - `n_starts_dist_cat, win_rate_dist_cat, in3_rate_dist_cat, avg_finish_dist_cat, avg_last3f_dist_cat`
  - `days_since_last_run` など
- 履歴特徴 `hr_*`（近況・成長を強化）
  - 生涯成績系（start数・勝率・平均着順・賞金など）
  - 直近3走系（平均着順・ベスト着順・上がり・人気など）
  - 直近5走系（平均着順・大敗回数など）
  - トレンド系（成績の上向き／下り坂指標）
  - 間隔（`hr_days_since_prev` など）

> 補足：  
> 以前は個別ラップから作る `hlap_*` 特徴を学習テーブルに入れていましたが、  
> データ制約などにより、**現行の `feature_table_v2` では `hlap_*` 系は一度外しています**。  
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
  - `race_results`（`horse_no` / `finish_order`）
  - `payouts`（`bet_type = '複勝'` → `fukusho_payout`）
- スプリット：
  - train: race_year ∈ {2021, 2022, 2023}
  - test : race_year = 2024
- モデル：
  - `LogisticRegression(max_iter=1000)`（複勝 in3 の二値分類）

### 4.2 重要オプション（リーク防止 / 再現性 / デバッグ）

#### 4.2.1 当日情報・シナリオ領域を禁止列として扱う（デフォルト）
`train_eval_logistic.py` は **禁止列（リーク/スコープ分離）** を持ち、数値列でも feature に入れません。

- 例：`target_*`, `finish_order`, `fukusho_payout` は当然禁止
- スコープ分離：`track_condition`, `track_condition_id` は **シナリオ側**で扱うので禁止
- 早出し：`horse_weight`, `horse_weight_diff` は **当日情報**なので禁止

> 禁止列はスクリプト内の `PROHIBITED_EXACT` / `PROHIBITED_PATTERNS` で管理。  
> もし禁止列が feature に混入したら **fail-fast で落ちます**（安全側）。

#### 4.2.2 相対（rr_）特徴量：レース内での順位/標準化
“そのレースの中で相対的に強いか” を入れるため、`rr_` を生成できます。

```bash
# rr を足す（rank_pct / zscore / both）
python scripts/train_eval_logistic.py --db netkeiba.db --add-rr --rr-kind both
```

- `--rr-kind`：
  - `rank_pct`：レース内 percentile（小さいほど良い列は反転）
  - `zscore`：レース内 z-score（小さいほど良い列は符号反転）
  - `both`：両方作る
- `--rr-cols` で rr の元にする列を上書き可能  
  未指定ならスクリプト内 `DEFAULT_RR_COLS` を使用（主に `hr_*`）

#### 4.2.3 払戻欠損の扱い（回収率評価の品質）
`fukusho_payout` は欠損が存在しうるため、評価時に “本来払戻があるはずなのに NULL” を除外するロジックがあります。

- デフォルト：除外 ON（安全）
- 解除したいとき：`--no-exclude-missing-payout`

#### 4.2.4 artifacts 出力（再現性・確認用）
- `artifacts/features_used.txt`：実際に使った feature 列一覧
- `artifacts/rr_cols_used.txt`：作成された rr 列一覧（rr を有効にした場合）

### 4.3 出力される指標（現状）

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
  - 閾値以上のレースだけベットした場合の
    - Bets / Hits / Hit% / ROI% / AvgProb
- **Calibration Evaluation**
  - 10 ビンでのキャリブレーションと ECE（Expected Calibration Error）
- **Debug Information**
  - 複勝払戻の欠損率
  - 1レースあたりの払戻件数（平均・最小・最大）
  - 払戻件数が 3 未満のレース数 など

> ここは現時点では **「現状の実力を測るためのベンチマーク」** です。  
> まずはリークなし・再現可能な入力で安定して評価できる状態を優先します。

---

## 5. シナリオ補正レイヤ & UI（概要）

### 5.1 シナリオ補正レイヤ（scenario/）

- 入力：
  - ベースモデルの win / in3 確率
  - 人間が指定する：
    - 予想ペース（スロー / ミドル / ハイ）
    - 馬場バイアス（内/外・前/後）
    - 想定隊列（逃げ・先行・差し・追い込み） など
    - 当日馬場（`track_condition` 相当） ※UIから指定する想定
- 出力：
  - 補正後の win / in3 確率
  - シナリオ前後で「どの馬がどれだけ得／損しているか」の差分情報

### 5.2 UI（ui/）

- `run_scenario_prediction.py` で JSON を生成し、
- `run_scenario_ui.py` で Web UI を立ち上げ、
- ブラウザ上でシナリオごとの変化を可視化する構成になっています。

（詳細は各スクリプト内 docstring を参照）

---

## 6. ヒューマン予想家からのインスピレーション（9つの軸）

キムラヨウヘイ / のれん / メシ馬 の G1・重賞見解を整理し、  
**共通して重視している視点**を 9 つの軸に分解しました。  
ベースモデルの特徴量は「前日確定情報」で作り、当日要素（馬場/バイアス/ペース等）はシナリオ側で扱います。

> ここでは「何を特徴量に落とすか」を具体化します。  
> まだ DB に無い列も含めて “設計案” として列挙しています。

### 軸1：近況・成長（フォームの良し悪し）
狙い：いま上向きか、安定しているか、底を見せていないか。

- 既にある/入れやすい
  - 直近3/5走の平均着順・ベスト着順
  - 直近3/5走の平均上がり（last3f）
  - 大敗回数（例：直近5走で二桁着順カウント）
  - トレンド（直近の着順改善/悪化傾向）
  - 生涯 in3 率 / 勝率 / 平均着順 / 分散（ブレ）
- 今後追加したい
  - “同格条件” でのフォーム（クラス別の近況）
  - 末脚の再現性（上がりの平均だけでなく分散・上位率）
  - 近況の「指数化」（直近x走のスコアリング）

### 軸2：条件適性（距離・コース・右左・芝ダ適性）
狙い：ベースは “条件が変わっても崩れにくい地力” を見たい。  
当日の馬場状態はシナリオで上書きする。

- 既にある/入れやすい
  - 距離カテゴリ別の成績（n_starts / win_rate / in3_rate）
  - コース別の成績（n_starts_course / win_rate_course）
  - 表面（芝/ダ）別の成績
- 今後追加したい
  - 右回り/左回り（コース特性のカテゴリ化）
  - 坂の有無・直線の長さ等（コースプロファイル）
  - “条件替わり” への強さ（距離延長/短縮での傾向）

### 軸3：地力・レースレベル（相手関係）
狙い：「このメンバーで通用するか」を数値化する。

- 既にある/入れやすい
  - race_class（クラス）
  - field_size（頭数）
  - `rr_`（レース内相対）  
    → 同レース内の相対位置で “メンバー比較” を補助
- 今後追加したい
  - “メンバー強度” 指標（出走馬の直近成績平均など）
  - 上位クラス経験/好走経験（昇級耐性）
  - レースレベル推定（賞金/指数の平均など）

### 軸4：位置取り・脚質×展開（前後・脚の質）
狙い：展開はシナリオ側で触るが、ベースでも「脚質の輪郭」は持つ。

- 既にある/入れやすい
  - 直近の着順と上がりから “差し届く/前残り” の気配を推定
  - 近走の安定性（展開に左右されやすい/されにくい）
- 今後追加したい（シナリオ補正の土台）
  - コーナー通過順（corners）から脚質を推定（逃げ/先行/差し/追込）
  - 前半/後半のラップ耐性（horse_laps）
  - “出遅れ癖/位置取りのブレ” 指標

### 軸5：枠順・バイアス相性（内外・前後）
狙い：当日のバイアスは人間が指定。枠順は前日確定で入れられる。

- 既にある/入れやすい
  - waku / umaban
  - コース×枠の傾向（統計的に作る）
- 今後追加したい
  - コース別「枠有利/不利」テーブル（過去データから事前構築）
  - バイアス入力に応じた補正係数（scenario layer）

### 軸6：ローテーション・仕上がり（間隔/使い詰め）
狙い：前走からの間隔・連戦で走りが変わる。

- 既にある/入れやすい
  - days_since_last_run / `hr_days_since_prev`
  - 直近n走の密度（60/90日で何走）
- 今後追加したい
  - 「叩き2走目」「休み明け何走目」フラグ
  - 間隔×成績（その馬がどのローテで走るタイプか）

### 軸7：メンタル・レースセンス（勝負所の安定感）
狙い：勝ち切れない/詰めの甘さ、逆に崩れない強さ。

- 既にある/入れやすい
  - 着順の分散（std_finish）
  - “大敗しない” 指標（big_loss_count）
  - in3率と勝率のギャップ（勝ち切り力 vs 堅実さ）
- 今後追加したい
  - 接戦の強さ（着差が取れるなら）
  - “人気より走る/人気ほど走れない” の癖（市場軸と絡む）

### 軸8：陣営・騎手意図（人間側の戦略）
狙い：騎手・厩舎・乗替りなどの「意思決定」を反映（ただし前日確定で拾える範囲）。

- 既にある/入れやすい（※DBにあれば）
  - 騎手ID/厩舎ID（なければ将来追加）
  - 乗替りフラグ
- 今後追加したい
  - 騎手×コース/距離の得意（過去統計）
  - 厩舎の得意条件（過去統計）
  - “勝負気配” proxy（同厩舎の使い分け等）

### 軸9：オッズ・期待値バランス（市場とのズレ）
狙い：回収率を上げるには市場とのズレが鍵。ただし **最終オッズは当日情報**で早出しと相性が悪い。

- 今は「モデル特徴量」ではなく「評価/買い方」で扱う
  - 予測確率と回収率の関係（閾値戦略・ギャップ戦略）
  - “買う/買わない” の判断材料として gap/確率を活用
- 将来（当日運用が可能になったら）
  - オッズを入れた別モデル（with_odds）を分岐して検証
  - 価値（Value）ベット最適化

---

## 7. ロードマップ（更新版）

### Phase 1（完了/進行中）：ベンチマーク整備
- feature_table_v2 + hr_* によるベースライン
- リーク防止（禁止列チェック）と artifacts 出力
- rr_ など「レース内相対」拡張と効果測定

### Phase 2：ベースモデル強化（再現可能な前日予想）
- LightGBM 等に移行（同じ split/評価で比較）
- 9軸のうち「前日確定で取れる」領域を優先して特徴量追加
  - ローテーション、条件適性、地力、枠、騎手/厩舎（取れる範囲）
- 予測の “確信度” 指標（例：上位確率差・ギャップ）を整備

### Phase 3：シナリオ補正 & UI を本格化
- UI で入力（ペース/バイアス/当日馬場）→ 補正結果を返す
- “得する馬/損する馬” の説明を安定生成

---

## 8. 開発メモ（衝突しにくい PR 作成手順）

複数の AI エージェント（Codex/Claude）を併用すると、**同じファイルの同じ周辺行**を触ってコンフリクトが起きがちです。  
下のテンプレで “衝突しにくい” 作業に寄せます。

### 8.1 PR 作成テンプレ（コピペ用）

1) **必ず最新 main から開始**
```bash
git checkout main
git fetch origin
git pull --ff-only origin main
```

2) **作業ブランチ作成（1PR=1目的）**
```bash
git checkout -b feature/<short-name>
```

3) **変更範囲を最小化**
- 触るファイルは必要最小限（できれば 1〜3 ファイル）
- “整形だけ” の変更（import 並べ替え等）を混ぜない

4) **テスト/実行ログを残す**
- 例：`python scripts/train_eval_logistic.py --db netkeiba.db --sanity-only`
- 主要コマンドは PR 本文に貼る

5) **push 前に rebase（長時間作業した場合）**
```bash
git fetch origin
git rebase origin/main
```

6) **push & PR 作成**
```bash
git push -u origin feature/<short-name>
```

---

（README は必要に応じて更新するが、衝突を避けたい場合は “docs-only PR” と分離する）
