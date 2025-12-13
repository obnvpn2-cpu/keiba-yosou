import sqlite3
import pandas as pd

DB = "data/keiba.db"

conn = sqlite3.connect(DB)

df = pd.read_sql_query("""
    SELECT race_id, race_name, race_date
    FROM races
    ORDER BY race_id DESC
    LIMIT 20;
""", conn)

print(df)

missing = df[df["race_date"].isna()]
print("\n❌ race_date が NULL のレース:")
print(missing)

conn.close()
