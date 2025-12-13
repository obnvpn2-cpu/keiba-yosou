# KEIBA SCENARIO AI

**人間と協業する競馬予想AI**

このプロジェクトは、  
「機械学習モデルのベース予測」と「人間が考えるレースシナリオ（ペース・バイアス・展開）」を組み合わせて  
**“シナリオ補正後の期待値” を出すこと**を目的とした競馬予想システムです。

---

## 🔧 全体アーキテクチャ

大きく3レイヤ構成です。

1. **ベース予測レイヤ**
   - LightGBM を使った勝率予測モデル
   - 入力は SQLite の `feature_table`
2. **シナリオ補正レイヤ**
   - 人間が指定するシナリオ（ペース・馬場バイアス・隊列など）を受け取り、
   - ベース予測に対して log-odds 空間で補正をかける
3. **UI / API レイヤ**
   - Python + FastAPI による API
   - Next.js ベースのフロントエンド（`keiba-ui/`）で可視化予定

---

## 📂 ディレクトリ構成（主要）

リポジトリルート直下：

```text
keiba-scenario-ai/
├── src/
│   ├── ingestion/                 # netkeiba スクレイピング & DB 保存
│   │   ├── scraper.py             # Cookie 認証付き HTTP クライアント
│   │   ├── parser.py              # レース詳細 HTML / JSONP パーサ
│   │   ├── models.py              # dataclass 群
│   │   ├── db.py                  # SQLite への保存ロジック
│   │   └── ingest_runner.py       # CLI エントリポイント
│   │
│   ├── features/
│   │   ├── __init__.py            # build_feature_table エントリ
│   │   ├── feature_builder.py     # 特徴量生成（ラップ比特徴量含む）
│   │   └── sqlite_store_feature.py# feature_table 作成 & INSERT
│   │
│   ├── base_model.py              # LightGBM ベース勝率モデル
│   ├── base_feature_builder.py    # 旧 feature builder 等（必要に応じて移行）
│   ├── calibration.py             # 確率キャリブレーション (Platt / Isotonic)
│   ├── baba_adjustment.py         # 馬場補正モデル (log-odds)
│   ├── pace_prediction.py         # ペース予測モデル
│   ├── pace_adjustment.py         # ペース補正モデル
│   ├── probability_integration.py # 補正結果の統合
│   ├── synergy_score.py           # 相性スコア
│   ├── backtest.py                # バックテスト
│   ├── shap_explainer.py          # SHAP 解析
│   ├── api.py                     # FastAPI エンドポイント
│   ├── timeline_manager.py        # 時系列分割・リーク防止
│   └── ...                        # テストやユーティリティ
│
├── keiba-ui/                      # Next.js UI プロジェクト（WIP）
├── data/                          # ローカル用データ格納（*.db など）※ Git 管理外推奨
├── models/                        # 学習済みモデル格納
├── requirements.txt               # Python 依存パッケージ
└── README.md                      # このファイル
````

※ `src/netkeiba.db` や `data/*.db` はローカル生成用で、Git 管理から外す想定です。

---

## 🧲 データ取得（netkeiba ingestion）

### 1. Cookie 設定

`src/ingestion/.env.example` を基に `.env` を作成し、
netkeiba の Cookie 値を環境変数として設定します。

```bash
cd src
cp ingestion/.env.example .env
nano .env  # 各 NETKEIBA_COOKIE_XXX にブラウザから取得した値を入れる
```

`.env` の中身イメージ：

```dotenv
NETKEIBA_COOKIE_netkeiba=...
NETKEIBA_COOKIE_nkauth=...
NETKEIBA_COOKIE_ga_netkeiba_member=...
# など、scraper.py が読む Cookie 群
```

### 2. 2024 年 JRA 全レースを取得する

```bash
cd src

# ドライラン（DB 書き込みなしでレース ID などの流れだけ確認）
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024 --dry-run

# 本番 ingestion（netkeiba.db に書き込み）
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024
```

### 3. 特定レースだけ再取得する（デバッグ用）

```bash
cd src

# 例: 有馬記念（2024-12-22, race_id=202406050811）だけ再取得
python -m ingestion.ingest_runner --race-ids 202406050811 -v
```

ingestion の結果は `src/netkeiba.db` に保存されます（`ingestion/db.py` の `DEFAULT_DB_PATH`）。

主に以下のテーブルが作られます：

* `races`：レース基本情報
* `race_results`：各馬の成績
* `payouts`：払戻情報
* `lap_times`：レース全体のラップ
* `horse_laps`：各馬の 200m ごとの個別ラップ
* `corners`：通過順位 など

---

## 🏇 horse_laps とラップ系特徴量

### horse_laps テーブル

netkeiba の「個別ラップ」API (`ajax_race_result_horse_laptime.html`) から JSONP を取得し、
HTML テーブルに変換した上で、以下の形式で保存しています：

* `race_id TEXT`
* `horse_id TEXT`
* `section_m INTEGER`

  * コース距離が偶数（例: 1400m, 1000m） → 200, 400, ..., 距離
  * コース距離が奇数（例: 1300m, 2500m） → 100, 300, ..., 距離
* `time_sec REAL`

  * 区間ラップ（秒）
* `position INTEGER`

  * 位置情報（現時点では主に `NULL`。将来拡張用）

### ラップ比特徴量（hlap_*）

`features/feature_builder.py` 内で、`horse_laps` と `lap_times` を用いて
**「その馬のラップがレース平均と比べてどれだけ速い/遅いか」** を特徴量にしています。

`feature_table` に追加されるカラム：

* `hlap_overall_vs_race`

  * レース全区間の平均差分
* `hlap_early_vs_race`

  * 0〜40% 区間の平均差分
* `hlap_mid_vs_race`

  * 40〜80% 区間の平均差分
* `hlap_late_vs_race`

  * 80〜100% 区間の平均差分
* `hlap_last600_vs_race`

  * ゴール前 600m（距離 600m 分）の平均差分

計算イメージ：

* 各区間で
  `delta = 馬のラップ秒数 - レース平均ラップ秒数`
* 指定したゾーンごとに `delta` を平均したものを特徴量として使う
  → マイナスなら「そのゾーンで平均より速い」、プラスなら「遅い」。

---

## 📊 feature_table の生成

### 1. feature_table を作り直す（必要なら一度 DROP）

```bash
cd src
python
```

```python
import sqlite3
conn = sqlite3.connect("netkeiba.db")
conn.execute("DROP TABLE IF EXISTS feature_table")
conn.commit()
conn.close()
exit()
```

### 2. feature_table を再構築

```bash
cd src

python -c "import sqlite3, logging; logging.basicConfig(level=logging.INFO); from features import build_feature_table; conn = sqlite3.connect('netkeiba.db'); build_feature_table(conn)"
```

ログ上で

* `Loaded races: ...`
* `Loaded race_results: ...`
* `Loaded lap_times: ...`
* `Loaded horse_laps: ...`
* `Feature table built: N rows`

と出ていれば OK。

### 3. ラップ特徴量の確認例

```bash
cd src
python
```

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

ここで `hlap_*` が `NULL` ではなく数値として埋まっていればラップ系特徴量の生成は成功しています。

---

## 🤖 モデル・補正レイヤ（概要）

Phase 1 で実装済みのモジュール（詳細コードは各ファイル参照）：

* `base_model.py`
  LightGBM によるベース勝率モデル
* `calibration.py`
  Platt Scaling / Isotonic Regression による確率キャリブレーション
* `baba_adjustment.py`
  馬場状態に応じた log-odds 補正
* `pace_prediction.py`
  前半・後半ラップの連続値予測
* `pace_adjustment.py`
  ペースによる有利・不利を log-odds で補正
* `probability_integration.py`
  補正済み log-odds を統合して最終的な勝率に変換
* `backtest.py`
  オッズ・控除率を考慮したバックテスト
* `shap_explainer.py`
  SHAP による特徴量重要度と説明テキスト生成
* `api.py`
  FastAPI による推論 API

モデルの訓練〜推論フロー自体は、旧 README の記述と大枠は変わっていません。
ベースモデルの入力として、**新たにラップ比特徴量（`hlap_*`）を含む `feature_table`** が使えるようになった、というのが今回のアップデートです。

---

## ⚙️ セットアップ（再掲）

```bash
# ルートで
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

その後、

1. `src/ingestion/.env` に Cookie を設定
2. `python -m ingestion.ingest_runner ...` で `netkeiba.db` を作成
3. `build_feature_table(conn)` で `feature_table` を構築
4. ベースモデル学習・バックテスト・API 起動…という流れで利用します。

---

## 🛑 Git 運用上の注意

* **コミットしないもの**

  * `src/netkeiba.db`
  * `data/*.db`
  * `keiba-ui/node_modules/` など

`.gitignore` に

```gitignore
keiba-ui/node_modules/
src/netkeiba.db
data/*.db
```

を入れて、**大きい DB や依存パッケージは Git 管理から外す**運用を前提としています。

---

## 👤 作成者

* obn
* Claude (Anthropic)
* ChatGPT (OpenAI)

---

**最終更新: 2025-12-14**

* netkeiba ingestion パイプライン（JRA 2024 全レース）
* 個別ラップ (`horse_laps`) 取得
* ラップ比特徴量 (`hlap_*`) を feature_table に追加

````