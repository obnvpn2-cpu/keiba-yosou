# diagnose_db.py
"""
DBの構造を診断して、race_date がどのテーブルにあるかを確認
"""
import sqlite3
import pandas as pd

DB_PATH = "data/keiba.db"

print("=" * 80)
print("🔍 DB Schema Diagnosis")
print("=" * 80)

conn = sqlite3.connect(DB_PATH)

# ========================================
# 1. races テーブルの構造
# ========================================
print("\n📋 1. races テーブルのカラム:")
cur = conn.cursor()
cur.execute("PRAGMA table_info(races)")
races_columns = cur.fetchall()
for col in races_columns:
    print(f"  - {col[1]} ({col[2]})")

races_has_race_date = any(col[1] == "race_date" for col in races_columns)
print(f"\n  ✅ race_date カラム: {'存在する' if races_has_race_date else '存在しない'}")

# racesテーブルのサンプルデータを確認
print("\n  サンプルデータ（最初の1行）:")
df_races = pd.read_sql_query("SELECT * FROM races LIMIT 1", conn)
print(df_races.to_string())

# ========================================
# 2. race_results テーブルの構造
# ========================================
print("\n" + "=" * 80)
print("📋 2. race_results テーブルのカラム:")
cur.execute("PRAGMA table_info(race_results)")
results_columns = cur.fetchall()
for col in results_columns:
    print(f"  - {col[1]} ({col[2]})")

results_has_race_date = any(col[1] == "race_date" for col in results_columns)
print(f"\n  ✅ race_date カラム: {'存在する' if results_has_race_date else '存在しない'}")

# race_resultsテーブルのサンプルデータを確認
print("\n  サンプルデータ（最初の1行）:")
df_results = pd.read_sql_query("SELECT * FROM race_results LIMIT 1", conn)
print(df_results.to_string())

# ========================================
# 3. horse_past_runs テーブルの構造
# ========================================
print("\n" + "=" * 80)
print("📋 3. horse_past_runs テーブルのカラム:")
cur.execute("PRAGMA table_info(horse_past_runs)")
past_columns = cur.fetchall()
for col in past_columns:
    print(f"  - {col[1]} ({col[2]})")

past_has_race_date = any(col[1] == "race_date" for col in past_columns)
print(f"\n  ✅ race_date カラム: {'存在する' if past_has_race_date else '存在しない'}")

# horse_past_runsテーブルのサンプルデータを確認
print("\n  サンプルデータ（最初の1行）:")
df_past = pd.read_sql_query("SELECT * FROM horse_past_runs LIMIT 1", conn)
print(df_past.to_string())

# ========================================
# 4. race_id の取得テスト
# ========================================
print("\n" + "=" * 80)
print("🔍 4. race_id の取得テスト:")

# race_results から race_id を取得
df_test = pd.read_sql_query("SELECT DISTINCT race_id FROM race_results LIMIT 1", conn)
if len(df_test) > 0:
    test_race_id = df_test.iloc[0]["race_id"]
    print(f"\n  テスト用 race_id: {test_race_id}")
    
    # races から race_date を取得
    if races_has_race_date:
        df = pd.read_sql_query(f"SELECT race_date FROM races WHERE race_id = '{test_race_id}'", conn)
        if len(df) > 0:
            print(f"  ✅ races.race_date = {df.iloc[0]['race_date']}")
        else:
            print(f"  ⚠️ races にこの race_id は存在しません")
    
    # race_results から race_date を取得
    if results_has_race_date:
        df = pd.read_sql_query(f"SELECT race_date FROM race_results WHERE race_id = '{test_race_id}' LIMIT 1", conn)
        if len(df) > 0:
            print(f"  ✅ race_results.race_date = {df.iloc[0]['race_date']}")
        else:
            print(f"  ⚠️ race_results にこの race_id は存在しません")
    
    # horse_past_runs から race_date を取得
    if past_has_race_date:
        df = pd.read_sql_query(f"SELECT race_date FROM horse_past_runs WHERE race_id = '{test_race_id}' LIMIT 1", conn)
        if len(df) > 0:
            print(f"  ✅ horse_past_runs.race_date = {df.iloc[0]['race_date']}")
        else:
            print(f"  ⚠️ horse_past_runs にこの race_id は存在しません")

# ========================================
# 5. サマリー
# ========================================
print("\n" + "=" * 80)
print("📊 サマリー:")
print(f"  races.race_date: {'✅ 存在' if races_has_race_date else '❌ 存在しない'}")
print(f"  race_results.race_date: {'✅ 存在' if results_has_race_date else '❌ 存在しない'}")
print(f"  horse_past_runs.race_date: {'✅ 存在' if past_has_race_date else '❌ 存在しない'}")

if not any([races_has_race_date, results_has_race_date, past_has_race_date]):
    print("\n  ⚠️ 警告: どのテーブルにも race_date カラムが存在しません！")
    print("  → feature_builder が動作するには、少なくとも1つのテーブルに race_date が必要です")
else:
    print("\n  ✅ 少なくとも1つのテーブルに race_date が存在します")

print("=" * 80)

conn.close()
