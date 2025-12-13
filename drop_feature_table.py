# drop_feature_table.py
import sqlite3
conn = sqlite3.connect("data/keiba.db")
conn.execute("DROP TABLE IF EXISTS feature_table")
conn.commit()
conn.close()
print("✅ Dropped feature_table")