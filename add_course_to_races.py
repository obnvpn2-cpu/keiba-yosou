import sqlite3

DB_PATH = "data/keiba.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("🔧 races テーブルに course カラムを追加します...")

# 1. カラム追加
try:
    cur.execute("ALTER TABLE races ADD COLUMN course TEXT;")
    print("✔ course カラム追加 OK")
except Exception as e:
    print("※ 既に course カラムが存在するかもしれません → 続行します")
    print(e)

print("🔧 course の中身を race_results から補完します...")

# 2. race_results から place（競馬場）を races.course に反映
# horse_past_runs の place は競馬場名なので、それを使う
cur.execute("""
    UPDATE races
    SET course = (
        SELECT place
        FROM horse_past_runs
        WHERE horse_past_runs.race_id = races.race_id
        LIMIT 1
    );
""")

conn.commit()
conn.close()

print("✨ 完了！ races テーブルに course が追加されました。")
