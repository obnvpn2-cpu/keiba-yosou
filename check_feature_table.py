# check_feature_table.py
"""
feature_table のデータ品質を確認
"""
import sqlite3
import pandas as pd

DB_PATH = "data/keiba.db"

print("=" * 80)
print("📊 feature_table データ品質チェック")
print("=" * 80)

conn = sqlite3.connect(DB_PATH)

# ========================================
# 1. 基本統計
# ========================================
print("\n✅ 1. 基本統計:")

df = pd.read_sql_query("""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT race_id) as unique_races,
        COUNT(DISTINCT horse_id) as unique_horses
    FROM feature_table
""", conn)

print(f"  総行数: {df['total_rows'].iloc[0]} 行")
print(f"  ユニークレース数: {df['unique_races'].iloc[0]} レース")
print(f"  ユニーク馬数: {df['unique_horses'].iloc[0]} 頭")

# ========================================
# 2. サンプルデータ（最初の3行）
# ========================================
print("\n" + "=" * 80)
print("📋 2. サンプルデータ（最初の3行）:")

df = pd.read_sql_query("""
    SELECT 
        race_id,
        horse_id,
        target_win,
        target_in3,
        course,
        surface,
        distance,
        race_year,
        race_month,
        n_starts_total,
        win_rate_total,
        is_first_run
    FROM feature_table
    LIMIT 3
""", conn)

print(df.to_string(index=False))

# ========================================
# 3. カラムのNull率
# ========================================
print("\n" + "=" * 80)
print("📊 3. カラムのNull率（上位10）:")

# 全カラムを取得
df_all = pd.read_sql_query("SELECT * FROM feature_table", conn)

null_rates = (df_all.isnull().sum() / len(df_all) * 100).sort_values(ascending=False)
null_rates_top = null_rates.head(10)

for col, rate in null_rates_top.items():
    print(f"  {col}: {rate:.1f}%")

# ========================================
# 4. 重要カラムの充填率
# ========================================
print("\n" + "=" * 80)
print("✅ 4. 重要カラムの充填率:")

important_cols = [
    "course",
    "surface",
    "distance",
    "track_condition",
    "race_class",
    "n_starts_total",
    "win_rate_total",
]

for col in important_cols:
    if col in df_all.columns:
        filled = df_all[col].notna().sum()
        total = len(df_all)
        rate = (filled / total * 100) if total > 0 else 0
        print(f"  {col}: {filled}/{total} ({rate:.1f}%)")

# ========================================
# 5. ターゲット変数の分布
# ========================================
print("\n" + "=" * 80)
print("🎯 5. ターゲット変数の分布:")

df = pd.read_sql_query("""
    SELECT 
        target_win,
        COUNT(*) as count
    FROM feature_table
    GROUP BY target_win
""", conn)
print("\n  target_win:")
print(df.to_string(index=False))

df = pd.read_sql_query("""
    SELECT 
        target_in3,
        COUNT(*) as count
    FROM feature_table
    GROUP BY target_in3
""", conn)
print("\n  target_in3:")
print(df.to_string(index=False))

df = pd.read_sql_query("""
    SELECT 
        target_value,
        COUNT(*) as count
    FROM feature_table
    GROUP BY target_value
""", conn)
print("\n  target_value:")
print(df.to_string(index=False))

# ========================================
# 6. レース年の分布
# ========================================
print("\n" + "=" * 80)
print("📅 6. レース年の分布:")

df = pd.read_sql_query("""
    SELECT 
        race_year,
        COUNT(*) as count
    FROM feature_table
    GROUP BY race_year
    ORDER BY race_year DESC
""", conn)
print(df.to_string(index=False))

# ========================================
# 7. 競馬場の分布
# ========================================
print("\n" + "=" * 80)
print("🏇 7. 競馬場の分布:")

df = pd.read_sql_query("""
    SELECT 
        course,
        COUNT(*) as count
    FROM feature_table
    GROUP BY course
    ORDER BY count DESC
    LIMIT 10
""", conn)
print(df.to_string(index=False))

# ========================================
# 8. 新馬フラグの分布
# ========================================
print("\n" + "=" * 80)
print("🐴 8. 新馬フラグの分布:")

df = pd.read_sql_query("""
    SELECT 
        is_first_run,
        COUNT(*) as count,
        ROUND(AVG(target_win), 3) as avg_win_rate,
        ROUND(AVG(target_in3), 3) as avg_in3_rate
    FROM feature_table
    GROUP BY is_first_run
""", conn)
print(df.to_string(index=False))

# ========================================
# 9. 距離カテゴリの分布
# ========================================
print("\n" + "=" * 80)
print("📏 9. 距離カテゴリの分布:")

df = pd.read_sql_query("""
    SELECT 
        distance_cat,
        COUNT(*) as count
    FROM feature_table
    GROUP BY distance_cat
    ORDER BY distance_cat
""", conn)
print(df.to_string(index=False))

# ========================================
# 10. データ品質サマリー
# ========================================
print("\n" + "=" * 80)
print("📊 10. データ品質サマリー:")

total_rows = len(df_all)
complete_rows = df_all.dropna().shape[0]
completeness = (complete_rows / total_rows * 100) if total_rows > 0 else 0

print(f"  総行数: {total_rows}")
print(f"  完全行数（Nullなし）: {complete_rows}")
print(f"  完全率: {completeness:.1f}%")

# 重要カラムの平均充填率
important_cols_filled = []
for col in important_cols:
    if col in df_all.columns:
        rate = (df_all[col].notna().sum() / total_rows * 100) if total_rows > 0 else 0
        important_cols_filled.append(rate)

avg_fill_rate = sum(important_cols_filled) / len(important_cols_filled) if important_cols_filled else 0
print(f"  重要カラムの平均充填率: {avg_fill_rate:.1f}%")

print("\n" + "=" * 80)
print("✅ データ品質チェック完了！")
print("=" * 80)

conn.close()
