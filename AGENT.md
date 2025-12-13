# AGENT.md

## このリポジトリでのエージェントの役割

あなたは **「人間と協業する競馬予想AI」** プロジェクトのための開発エージェントです。  
最大の仕事は、

- 既存コードベースを壊さずに、
- スクレイピング・特徴量生成・予測モデル・シナリオ補正レイヤを拡張・改善すること

です。

抽象的な議論ではなく、**実際に動くコード** と **最小限で筋の通った設計** を出してください。

---

## プロジェクト概要（前提知識）

- テーマ: **人間と協業する競馬予想AI（JRA専門）**
- 全体構成は 3 レイヤ:
  1. ベース予測（LightGBM）
  2. シナリオ補正レイヤ（ペース・バイアス・展開など人間入力による補正）
  3. 可視化 UI（Next.js / keiba-ui、まだ開発途中）

- データソース:
  - `https://db.netkeiba.com`（JRA レース）
  - ユーザーは netkeiba スーパープレミアム会員
  - Cookie は `.env` から読み込む（値はリポジトリに含めない）

---

## 技術スタックと前提

- 言語: Python 3.12
- ライブラリ:
  - スクレイピング: `requests`, `BeautifulSoup4`
  - ML: `lightgbm`, `numpy`, `pandas`, `scikit-learn`
  - 補助: `logging`, `dataclasses`
- DB: SQLite (`src/netkeiba.db` / `data/*.db`)
  - **DB ファイルは Git 管理対象にしない**（`.gitignore` で除外）

---

## ディレクトリ構成（重要部分）

プロジェクトルート: `keiba-scenario-ai/`

```text
src/
  ingestion/
    scraper.py          # netkeiba スクレイパ（Cookie 認証 + JSONP など）
    parser.py           # レース詳細ページの HTML / JSONP パーサ
    models.py           # Race / RaceResult / Payout / LapTime / HorseLap など dataclass
    db.py               # SQLite への保存ロジック
    ingest_runner.py    # CLI エントリポイント

  features/
    __init__.py         # build_feature_table(conn) エントリ
    feature_builder.py  # 特徴量生成ロジック（ラップ系もここ）
    sqlite_store_feature.py  # feature_table DDL & INSERT

  base_model.py         # LightGBM ベースモデル
  calibration.py        # 確率キャリブレーション
  baba_adjustment.py    # 馬場補正
  pace_prediction.py    # ペース予測
  pace_adjustment.py    # ペース補正
  probability_integration.py
  backtest.py
  shap_explainer.py
  api.py                # FastAPI API
  scenario/             # シナリオ補正レイヤ UI / 型定義 等

keiba-ui/               # Next.js UI（今は高優先度ではない）
data/                   # ローカル DB 等（Git 管理外）
models/                 # 学習済みモデル
````

---

## 重要なドメイン知識（最低限）

### レース・ラップ構造

* `races` テーブル

  * `race_id`, `date`, `place`, `name`, `grade`, `race_class`, `course_type`, `distance`, `track_condition` など
* `race_results` テーブル

  * `race_id`, `horse_id`, `horse_name`, `finish_order`, `frame_no`, `horse_no`, `time_sec`, `last_3f`, `win_odds`, `popularity`, `body_weight`, `body_weight_diff` など
* `lap_times`（レース全体ラップ）

  * `race_id`, `lap_index`, `distance_m`, `time_sec`
* `horse_laps`（個別ラップ）

  * `race_id`, `horse_id`, `section_m`, `time_sec`, `position`
  * 距離が偶数（例: 1400m, 1000m） → `section_m`: 200, 400, ..., 距離
  * 距離が奇数（例: 1300m, 2500m） → `section_m`: 100, 300, ..., 距離

### ラップ系特徴量（hlap_*）

`feature_table` に持たせたいカラム（すでに設計済み）:

* `hlap_overall_vs_race`
* `hlap_early_vs_race`（総距離の 0〜40%）
* `hlap_mid_vs_race`（40〜80%）
* `hlap_late_vs_race`（80〜100%）
* `hlap_last600_vs_race`（ゴール前 600m）

意味:

* 各区間で `delta = 馬のラップ秒 - レース平均ラップ秒`
* 各ゾーンごとに `delta` を平均したもの

  * マイナス → 平均より速い
  * プラス → 平均より遅い

**この設計は変えないこと。** 実装を整理するのは良いが、意味が変わる変更は避ける。

---

## 現時点の「課題」と優先順位

### 1. feature_table が空のままになる問題

* 問題の原因:

  * `features/feature_builder.py` の `build_features_for_race` が **`horse_past_runs` テーブルありき**の設計
  * 現在の DB には `horse_past_runs` テーブルが存在しない
  * `load_base_tables` は `horse_past_runs` 非存在時に「空 DataFrame」を入れる
  * その状態で `tables["horse_past_runs"]["horse_id"]` などを参照し、`KeyError('horse_id')` を起こしている
* 方針:

  * **過去走テーブルは現時点では無い前提で OK**
  * 過去走ベース特徴量は一旦 **全スキップ** or `NaN` で埋める
  * その代わり、「現レースの情報＋ラップ比特徴量（`hlap_*`）」だけで `feature_table` を組めるようにする

### 2. ラップ系特徴量の実装の安定化

* `compute_lap_ratio_features_for_race(...)`（関数名は近いもの）を軸に、

  * `horse_laps` + `lap_times` → `hlap_*` の計算
  * それを `build_features_for_race` の馬ごとの特徴量にマージ
* ここでの責務を明確にし、**ゾーン分割ロジック・距離設定・欠損時の扱い**を統一する。

---

## エージェントの基本方針

### コードスタイル / 実装ポリシー

* 既存スタイルに合わせることを最優先

  * ログ → `logging` モジュール (`logger = logging.getLogger(__name__)`)
  * 型 → 可能な範囲で `typing` / `dataclasses` を利用
* 「なんとなくきれいにしたくなったから全体を書き換える」は禁止

  * 目的と関係ないリファクタはやらない
* 依存パッケージはむやみに増やさない

  * 特にスクレイピング周りは `requests + bs4` で十分

### DB / スキーマ関連

* **既存テーブルのスキーマを勝手に変えない**

  * どうしても変える必要がある場合は

    * 変更理由
    * 影響範囲
    * マイグレーション手順（既存 DB をどう扱うか）
      をコメントなりドキュメントなりで明示すること
* DB ファイル（`src/netkeiba.db`, `data/*.db`）は Git 対象外である前提でコードを書く

### Git / ファイル運用

* 絶対にコミットしてはいけないもの:

  * `keiba-ui/node_modules/`
  * `src/netkeiba.db`
  * `data/keiba.db` および `data/*.db`
  * `.env`, Cookie 情報
* これらは `.gitignore` 前提でコードを書くこと。

### 変更提案の粒度

* 基本は **関数単位** で提案する

  * 「この関数をこのコードに置き換えてください」という形が望ましい
* ファイル全体を書き換える場合は、

  * 「ファイル全体を差し替え」の旨をはっきり書き、
  * 変更意図をコメントで残す

---

## エージェントにやってほしい具体的なタスク例

### A. feature_table 周り

1. `features/feature_builder.py` の `build_features_for_race` を修正し、

   * `horse_past_runs` が空でもエラーにならず、
   * 過去走由来の特徴量はスキップ or `NaN` にし、
   * `hlap_*` を含む現レース由来の特徴量だけで `feature_table` を構築できるようにする。
2. `load_base_tables` と `race_attrs_cache` の整合性を確認し、

   * surface / distance / track_condition が常に取得できるようにする。
3. ラップ系特徴量の計算関数を整理し、テストしやすい形に分割する。

### B. ingestion / スクレイピング

* 必要に応じて:

  * `ingestion/parser.py` の DOM セレクタ修正
  * JSONP パースの堅牢化
  * HTTP リトライ・スロットリングの調整
* ただし、「現状動いている部分」はむやみに触らず、**明確なバグや要件追加があるときだけ触ること。**

### C. モデル・補正レイヤ

* `base_model.py`, `calibration.py`, `baba_adjustment.py`, `pace_adjustment.py` などの改善

  * ただし、先に `feature_table` が安定してから着手すること

---

## 実行・検証に使うコマンド（参考）

### ingestion

```bash
cd src

# 2024年 JRA 全レース
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024

# 特定レース（例: 有馬記念, 202406050811）
python -m ingestion.ingest_runner --race-ids 202406050811 -v
```

### feature_table 再生成

```bash
cd src
python -c "import sqlite3; conn = sqlite3.connect('netkeiba.db'); conn.execute('DROP TABLE IF EXISTS feature_table'); conn.commit(); conn.close()"

python -c "import sqlite3, logging; logging.basicConfig(level=logging.INFO); from features import build_feature_table; conn = sqlite3.connect('netkeiba.db'); build_feature_table(conn)"
```

### ラップ特徴量の確認

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("netkeiba.db")

df = pd.read_sql_query(
    """
    SELECT
        race_id,
        horse_id,
        hlap_overall_vs_race,
        hlap_early_vs_race,
        hlap_mid_vs_race,
        hlap_late_vs_race,
        hlap_last600_vs_race
    FROM feature_table
    WHERE race_id IN ('202406050811', '202408070108')
    ORDER BY race_id, horse_id
    LIMIT 20
    """,
    conn,
)
print(df.to_string(index=False))
```

---

## 最後に

* あなたの最優先タスクは **「壊さないこと」と「実用品として前に進めること」**。
* 大きなリファクタより、**「今必要なところだけ、筋の通った最小変更で直す」** こと。
* 仕様や前提に迷いがある場合は、コード上のコメントや README/AGENT への追記として「こう解釈して実装した」と残すこと。

このガイドラインに沿って、`features/feature_builder.py` や `ingestion/` 周りのコード修正・追加を行ってください。