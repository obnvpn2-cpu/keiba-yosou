# netkeiba Ingestion パイプライン

db.netkeiba.com からJRAのレース結果をスクレイピングし、SQLite に保存する ingestion パイプラインです。

## 特徴

- **Cookie認証対応**: プレミアム/マスター会員限定データも取得可能
- **ページネーション対応**: 大量のレースを自動で全ページ取得
- **堅牢な設計**:
  - User-Agentを12種類からランダム選択（ボット検知回避）
  - 400 Bad Request時はUser-Agentを変更して自動リトライ
  - 指数バックオフ付きリトライ（500系エラー対応）
  - 2〜3.5秒のランダムスリープによるレート制限対策
  - 個別エラーでもパイプライン続行
- **Idempotent**: 同じレースを何度再実行しても安全
- **正規化されたスキーマ**: 後続の分析に適したテーブル構造

## インストール

```bash
# 依存パッケージのインストール
pip install requests beautifulsoup4 python-dotenv

# または requirements.txt から
pip install -r requirements.txt
```

## Cookie設定

1. `.env.example` を `.env` にコピー
2. netkeiba にログインした状態でブラウザの開発者ツールを開く
3. Application > Cookies > netkeiba.com から各Cookieの値をコピー
4. `.env` に貼り付け

```bash
cp ingestion/.env.example .env
# .env を編集して Cookie 値を設定
```

**重要な Cookie:**
- `netkeiba`: セッション識別
- `nkauth`: 認証トークン

## 使い方

### 基本的な使い方

```bash
# 2024年の JRA 全レースを取得
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024
```

### オプション

```bash
# 特定の月だけ取得
python -m ingestion.ingest_runner \
    --start-year 2024 --end-year 2024 \
    --start-mon 6 --end-mon 6

# 特定の競馬場だけ取得（東京・中山）
python -m ingestion.ingest_runner \
    --start-year 2024 --end-year 2024 \
    --jyo 05 06

# 既存レースをスキップ（差分取得）
python -m ingestion.ingest_runner \
    --start-year 2024 --end-year 2024 \
    --skip-existing

# 特定のレースIDを直接指定
python -m ingestion.ingest_runner \
    --race-ids 202406050901 202406050902

# ドライラン（DBに保存せず確認だけ）
python -m ingestion.ingest_runner \
    --start-year 2024 --end-year 2024 \
    --dry-run
```

### コマンドライン引数

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--start-year` | 開始年 | 今年 |
| `--end-year` | 終了年 | 今年 |
| `--start-mon` | 開始月 | 1 |
| `--end-mon` | 終了月 | 12 |
| `--jyo` | 場コード（複数指定可） | JRA全場 |
| `--db` | データベースファイルパス | `netkeiba.db` |
| `--skip-existing` | 既存レースをスキップ | False |
| `--race-ids` | 直接レースID指定 | - |
| `--dry-run` | 保存せずログのみ | False |
| `-v, --verbose` | 詳細ログ | False |

### 場コード

| コード | 競馬場 |
|--------|--------|
| 01 | 札幌 |
| 02 | 函館 |
| 03 | 福島 |
| 04 | 新潟 |
| 05 | 東京 |
| 06 | 中山 |
| 07 | 中京 |
| 08 | 京都 |
| 09 | 阪神 |
| 10 | 小倉 |

## データベーススキーマ

### races（レース基本情報）

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | TEXT PK | 12桁レースID |
| date | TEXT | 開催日 |
| place | TEXT | 開催場（中山、東京 など） |
| kai | INTEGER | 回次 |
| nichime | INTEGER | 日目 |
| race_no | INTEGER | R番号 |
| name | TEXT | レース名 |
| grade | TEXT | G1, G2, G3, OP, Listed |
| race_class | TEXT | クラス条件 |
| course_type | TEXT | turf, dirt, steeple |
| distance | INTEGER | 距離（m） |
| course_turn | TEXT | right, left, straight |
| course_inout | TEXT | inner, outer |
| weather | TEXT | 天候 |
| track_condition | TEXT | 馬場状態 |
| start_time | TEXT | 発走時刻 |
| baba_index | INTEGER | 馬場指数 |
| baba_comment | TEXT | 馬場コメント |
| analysis_comment | TEXT | 分析コメント |
| head_count | INTEGER | 出走頭数 |

### race_results（出走馬成績）

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | TEXT PK | レースID |
| horse_id | TEXT PK | 馬ID |
| finish_order | INTEGER | 着順 |
| finish_status | TEXT | 取消、除外、中止、失格 |
| frame_no | INTEGER | 枠番 |
| horse_no | INTEGER | 馬番 |
| horse_name | TEXT | 馬名 |
| sex | TEXT | 性別（牡、牝、セ） |
| age | INTEGER | 年齢 |
| weight | REAL | 斤量 |
| jockey_id | TEXT | 騎手ID |
| jockey_name | TEXT | 騎手名 |
| time_str | TEXT | タイム文字列 |
| time_sec | REAL | タイム（秒） |
| margin | TEXT | 着差 |
| passing_order | TEXT | 通過順 |
| last_3f | REAL | 上り3F |
| win_odds | REAL | 単勝オッズ |
| popularity | INTEGER | 人気 |
| body_weight | INTEGER | 馬体重 |
| body_weight_diff | INTEGER | 馬体重増減 |
| time_index | INTEGER | タイム指数 |
| trainer_id | TEXT | 調教師ID |
| trainer_name | TEXT | 調教師名 |
| trainer_region | TEXT | 東/西 |
| owner_id | TEXT | 馬主ID |
| owner_name | TEXT | 馬主名 |
| prize_money | REAL | 賞金（万円） |
| remark_text | TEXT | 備考 |

### payouts（払い戻し）

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | TEXT PK | レースID |
| bet_type | TEXT PK | 券種 |
| combination | TEXT PK | 組番 |
| payout | INTEGER | 配当 |
| popularity | INTEGER | 人気 |

### corners（コーナー通過順位）

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | TEXT PK | レースID |
| corner_1 | TEXT | 1コーナー |
| corner_2 | TEXT | 2コーナー |
| corner_3 | TEXT | 3コーナー |
| corner_4 | TEXT | 4コーナー |

### lap_times（レース全体ラップ）

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | TEXT PK | レースID |
| lap_index | INTEGER PK | ラップ番号 |
| distance_m | INTEGER | 累積距離 |
| time_sec | REAL | ラップタイム |

### horse_laps（個別馬ラップ、マスター限定）

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | TEXT PK | レースID |
| horse_id | TEXT PK | 馬ID |
| section_m | INTEGER PK | 区間距離 |
| time_sec | REAL | ラップタイム |
| position | INTEGER | 順位 |

### short_comments（短評、マスター限定）

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | TEXT PK | レースID |
| horse_id | TEXT PK | 馬ID |
| horse_name | TEXT | 馬名 |
| finish_order | INTEGER | 着順 |
| comment | TEXT | 短評 |

## Pythonから使用

```python
from ingestion import (
    NetkeibaClient,
    fetch_race_ids,
    parse_race_page,
    init_database,
)

# クライアント作成
client = NetkeibaClient()

# レースID取得
race_ids = fetch_race_ids(
    start_year=2024,
    end_year=2024,
    start_mon=6,
    end_mon=6,
    client=client,
)

# DB初期化
db = init_database()

# 各レースを処理
for race_id in race_ids[:10]:  # 最初の10件
    html = client.get_race_page(race_id)
    data = parse_race_page(html, race_id)
    db.save_race_data(data)
    
    print(f"{data.race.name}: {len(data.results)} horses")

db.close()
client.close()
```

## ファイル構成

```
ingestion/
├── __init__.py        # パッケージ初期化
├── models.py          # データモデル (dataclass)
├── scraper.py         # HTTPクライアント（User-Agentローテーション・リトライ機能付き）
├── race_list.py       # レース一覧取得（ページネーション対応）
├── parser.py          # HTMLパーサー（BeautifulSoup + CSSセレクタ）
├── db.py              # SQLite操作（Upsert・Idempotent設計）
├── ingest_runner.py   # メイン実行スクリプト（CLI対応）
├── .env.example       # Cookie設定例
├── requirements.txt   # 依存パッケージ
└── README.md          # このファイル
```

## 注意事項

- netkeiba の利用規約を遵守してください
- 過度なアクセスは避けてください（デフォルトで2〜3.5秒のスリープあり）
- Cookie は定期的に更新が必要です
- 会員限定データは非ログイン時は空になります

## アクセス設定（scraper.py）

以下の設定値はカスタマイズ可能です：

```python
# User-Agentリスト（12種類からランダム選択）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",  # Chrome
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...",  # Safari
    # ...
]

# リクエスト間隔（2〜3.5秒）
MIN_SLEEP = 2.0
MAX_SLEEP = 3.5

# 400 Bad Request時のリトライ
MAX_400_RETRIES = 3
RETRY_400_SLEEP_MIN = 1.0
RETRY_400_SLEEP_MAX = 2.0

# 500系エラー時のリトライ
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.0  # 1, 2, 4, 8, 16秒と増加
```

### 400 Bad Requestが発生した場合

1. User-Agentを変更して自動リトライ（最大3回）
2. リトライ前に1〜2秒の追加スリープ
3. それでも失敗する場合は、1日程度間隔を空けて再実行してください

## ライセンス

個人利用のみ。netkeiba のデータの再配布は禁止です。
