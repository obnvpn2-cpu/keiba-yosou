## 0. あなたの役割

このリポジトリに対してコードを書く「エージェント」（Claude Code など）は、  
**実装担当エンジニア** という位置づけです。

- 目的は「人間と協業する競馬予想AI（Scenario-Aware Racing Intelligence）」を、  
  **壊さず・リークさせず・段階的に強化すること**。
- この文書は「エージェント向けの開発ガイドライン」です。
- 人間のオーナーは
  - 設計方針／優先度決め
  - どこまでやるかのスコープ決め
  を担当し、**実際のコード修正・追加はエージェントに任せる**想定です。

---

## 1. プロジェクト全体像（ざっくり）

### 1.1 コンセプト

- ベースモデル：  
  JRA 全レース（2021–2024）のデータから  
  「**各馬の複勝圏内（in3）に入る確率**」を出す機械学習モデル。
- シナリオ補正レイヤ：  
  人間が指定する
  - ペース（H/M/S）
  - 馬場バイアス（内/外・前/後）
  - 逃げ・先行・差し脚質
  などを入力として、**ベース確率を加減して最終予測を出すレイヤ**。
- UI：  
  ベース予測＋シナリオ補正結果を JSON にし、  
  Web UI で「どの馬が得しているか／損しているか」を可視化する。

この AGENT.md では、特に

- **データパイプライン（ingestion → DB → feature_table_v2）**
- **ベースモデル＆評価ロジック（`train_eval_logistic.py` など）**

を壊さず拡張するためのルールを定義します。

---

## 2. ディレクトリ & 主なコンポーネント

### 2.1 ディレクトリ構成のイメージ

- `src/ingestion/`
  - `scraper.py`：db.netkeiba.com からのスクレイピング（Cookie セッション）
  - `race_list.py`：年／月／場 から `race_id` 一覧を取得
  - `parser.py`：レース詳細 HTML のパース
  - `db.py`：SQLite への保存ラッパ（`Database` クラス）
  - `ingest_runner.py`：CLI から一括 ingestion を走らせるエントリポイント
- `scripts/`
  - `build_horse_results.py`：馬ごとの過去走テーブル `horse_results` 構築
  - `build_feature_table_v2.py`：学習用テーブル `feature_table_v2` 構築
  - `train_eval_logistic.py`：ベースライン（logistic）学習＋評価スクリプト
- `scenario/`：シナリオ補正レイヤ（別チャットで進行中）
- `ui/`：JSON を読んでシナリオ結果を可視化する簡易フロント

---

## 3. DB とテーブル仕様（2021–2024）

### 3.1 DB ファイル

- **正とする DB パス：リポジトリ直下の `netkeiba.db`**
  - ingestion 時は原則：

    ```bash
    cd src
    python -m ingestion.ingest_runner \
      --start-year 2021 --end-year 2024 \
      --db ../netkeiba.db
    ```

  - `--db` オプションでパスを差し替え可能だが、  
    README / スクリプトのデフォルト前提は **`../netkeiba.db`** に統一。

### 3.2 主なテーブル

（詳細な CREATE TABLE は `ingestion/db.py` を参照）

- `races`
  - `race_id (TEXT, PK)`
  - `date (TEXT, YYYY-MM-DD)`
  - `place, kai, nichime, race_no, name, grade, race_class`
  - `course_type (芝/ダ), distance, course_turn (右/左), course_inout`
  - `weather, track_condition`
  - `head_count` など
- `race_results`
  - `race_id, horse_id, horse_no, finish_order, time, last3f, odds, popularity` …
- `payouts`
  - `race_id, bet_type, combination, payout`
  - 本プロジェクトでは主に `bet_type = '複勝'` を使用
- `corners`, `lap_times`, `horse_laps`, `short_comments`
  - コーナー通過順・ラップ・馬ごとのラップ・コメントなど
  - **現行ベースモデルでは直接は使っていない**（シナリオレイヤ／将来拡張用）

### 3.3 学習用の派生テーブル

- `horse_results`
  - `horse_id, race_id, race_date, start_index, finish_order, last3f, odds, popularity, prize, ...`
  - 各馬の出走履歴を `race_date` ソートし、`start_index` を振ったテーブル。
  - **履歴特徴量 `hr_*` を作るための元データ。**
- `feature_table`（v1）
  - 旧来の学習テーブル。レース条件＋簡単な履歴集計。
  - ここにあった個別ラップ由来の `hlap_*` 系特徴は、  
    データ制約により **現在のメインパイプラインからは削除済み**。
- `feature_table_v2`
  - v1 に `hr_*` 履歴特徴を横持ちで追加したテーブル。
  - `build_feature_table_v2.py` で再構築する。
  - 現在は「ベースモデルのメイン入力」として利用。

---

## 4. ベースモデル & 評価ロジック

### 4.1 train_eval_logistic.py の役割

`scripts/train_eval_logistic.py` は

- 2021–2023 を学習
- 2024 をテスト

として、

- **ターゲット：`target_in3`（複勝圏フラグ）**
- **モデル：ロジスティック回帰（現時点のベースライン）**

を学習・評価するスクリプトです。

使用するデータは：

1. `feature_table_v2`（なければ `feature_table` にフォールバック）
2. `race_results`（`horse_no`）
3. `payouts`（複勝払戻）

を SQL で join したもの。

### 4.2 評価ロジック（現状仕様）

Claude によるリファクタ済みで、評価は以下に分割されています：

- **Global Metrics**
  - Accuracy（0.5 閾値）
  - ROC-AUC
  - Log Loss
  - Brier Score
- **Ranking Evaluation**
  - テストレース数
  - Top1 / Top2 / Top3 カバー率  
    （各レースで、複勝圏内の馬が上位何番目までに含まれているか）
  - `best_in3`（複勝圏内で最も高く予測された馬）の平均順位
- **Strategy: Top1 All Races**
  - 全レースで「予測確率最大の 1頭」を複勝 1点買いした場合の
    - ベット数 / 的中数 / 的中率 / ROI
- **Strategy: Top1 with Thresholds**
  - 閾値（デフォルト 0.25 / 0.30 / 0.35 / 0.40）以上のレースだけベットする戦略。
  - 各閾値ごとに Bets / Hits / Hit% / ROI% / AvgProb を出力。
- **Calibration Evaluation**
  - 10 ビンで予測確率と実際の的中率を比較。
  - Expected Calibration Error (ECE) を算出。
- **Debug Information**
  - `fukusho_payout` 欠損率
  - 1レースあたりの複勝払戻件数（平均・最小・最大）
  - 複勝払戻件数 < 3 のレース数 など

> ⚠️ 重要：  
> ロジスティック回帰＋現行特徴量（~46 次元）は、  
> 現時点では **「控除率に若干負ける程度のベンチマーク」** です。  
> 「ガチで戦えるモデル」ではなく、**全世代のモデルを比較するための共通物差し** として扱ってください。

---

## 5. hr_* 履歴特徴量（近況・成長軸）

### 5.1 現在の hr_* 構成（ざっくり）

`build_feature_table_v2.py` では、`horse_results` を元に **リーク無し** で hr_* を計算しています。

- 生涯成績系
  - `hr_career_starts, hr_career_wins, hr_career_in3`
  - `hr_career_win_rate, hr_career_in3_rate`
  - `hr_career_avg_finish, hr_career_std_finish`
  - `hr_career_avg_last3f, hr_career_avg_popularity`
  - `hr_career_total_prize, hr_career_avg_prize`
- 直近3走系
  - `hr_recent3_starts`
  - `hr_recent3_avg_finish, hr_recent3_best_finish`
  - `hr_recent3_avg_last3f`
  - `hr_recent3_avg_win_odds, hr_recent3_avg_popularity`
  - `hr_recent3_big_loss_count`
- 直近5走系（Claude による拡張）
  - `hr_recent5_starts`
  - `hr_recent5_avg_finish, hr_recent5_best_finish`
  - `hr_recent5_avg_last3f`
  - `hr_recent5_avg_popularity`
  - `hr_recent5_big_loss_count`
- トレンド系
  - `hr_recent3_finish_trend`
  - `hr_recent3_last_vs_best_diff`
- 間隔
  - `hr_days_since_prev`

**すべて「今走を除外した履歴のみ」から集計**しており、`shift(1)` ベースでリークを防止しています。

### 5.2 今後の拡張

- 近況・成長以外にも、以下のようなカテゴリの特徴量を増やしていく計画です：
  - 馬の静的プロフィール（性別・年齢・生産者など）
  - 血統（sire / dam / 母父ベースの集計・クラスタ）
  - 騎手・調教師成績（距離別・馬場別・脚質別など）
  - レースレベル／フィールド強度
  - オッズ・マーケット指標　…など
- さらに、**人間予想家 3人（キムラヨウヘイ / のれん / メシ馬）の見解**から抽出した  
  「共通して重視している 9つの視点」を軸に、  
  **意味のある特徴量を最終的に 100 本以上** に増やす方針です。

---

## 6. シナリオ補正レイヤ（触るときの注意）

- シナリオ補正は別チャットで設計済みの独立レイヤです：
  - 入力：  
    - ベースモデルの予測確率（win / in3）  
    - 人間が指定するペース・バイアス・脚質シナリオ
  - 出力：  
    - 補正後の win / in3 確率
    - 「どの馬がどれだけ得／損しているか」の説明用 JSON
- このレイヤは **「ベースモデルの出力を前提にした後段」** なので、
  - ベースモデル側のインターフェース（列名や意味）を安易に壊さないこと。
  - どうしても変える場合は **JSON スキーマや README / AGENT にも反映** すること。

---

## 7. 絶対に避けてほしいこと

1. **データリーク**
   - レース日より後の情報（後続の成績、次走のオッズなど）を  
     feature_table_v2 に混ぜない。
   - `horse_results` から集計するときは、**必ず「今走より前のみ」を対象** にする。
2. **DBスキーマの破壊的変更**
   - 既存カラムを勝手に削除・リネームしない。
   - どうしても変える場合は
     - 影響範囲（どのスクリプトが参照しているか）を洗い出し、
     - README / AGENT に変更理由と新仕様を追記すること。
3. **ラップ・シナリオ系の一括削除／大リファクタ**
   - `lap_times`, `horse_laps`, `corners` などは  
     現行ベースモデルでは直接使っていなくても、
     シナリオレイヤや将来の高機能モデルで利用予定です。
   - これらを無効化する・テーブル構造を変えるときは、  
     **必ずコメントとドキュメントで意図を残すこと**。
4. **巨大なリファクタを一気に行うこと**
   - ベースモデル・シナリオレイヤ・UI を一度に書き換えない。
   - 1 PR / 1 コミットのスコープはできるだけ狭く、  
     「いま必要な変更」に絞る。

---

## 8. 推奨する開発スタイル

- ログを必ず仕込む
  - `logging` モジュールを使い、DEBUG/INFO レベルを使い分ける。
  - 「どの DB を見ているか」「何件ロードしたか」「何列の特徴量か」は INFO で出す。
- 小さく試す
  - まずは 1 レース / 数レースだけを対象に動かして、SQL / join / ロジックが合っているかを確認。
- Notebook / スクリプトで確認
  - 新しい特徴量を追加したら、簡単な集計（平均・分布・欠損率）を Notebook 等で確認してから本番パイプラインに組み込む。
- 迷ったらコメントを残す
  - 「こういう前提で実装した」というメモをコードコメント or AGENT / README に残す。
  - 後続のエージェントや人間開発者が前提をトレースしやすくする。

---

## 9. 最後に

- **最優先タスクは「壊さないこと」と「前に進めること」。**
- ロジスティック回帰＋現行46特徴はあくまでベンチマークであり、  
  目標は「人間予想家の 9軸＋100本超の特徴量＋より強いモデル（LightGBM など）」です。
- その道筋を見失わないように、
  - データパイプライン
  - `feature_table_v2` / `horse_results`
  - 評価ロジック
  を丁寧に育てていってください。

このガイドラインに従って、`ingestion/`・`scripts/`・`scenario/` 周りのコード修正・追加を行ってください。
