# db/view_db.py
import sqlite3
import sys

DB_PATH = "visitors.db"

def view_table(table_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT * FROM {table_name}")
    rows = c.fetchall()
    col_names = [desc[0] for desc in c.description]
    conn.close()

    print(f"\n=== {table_name.upper()} ===")
    print(" | ".join(col_names))
    for r in rows:
        print(r)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        table = sys.argv[1]
        view_table(table)
    else:
        view_table("visitors")
        view_table("events")