# outputs/report_generator.py
import sqlite3
import os
import csv
import json

def export_reports(db_path="db/visitors.db", out_dir="outputs/reports"):
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Export visitors
    c.execute("SELECT * FROM visitors")
    visitors = c.fetchall()
    visitor_cols = [desc[0] for desc in c.description]
    with open(os.path.join(out_dir, "visitors.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(visitor_cols)
        writer.writerows(visitors)

    # Export events
    c.execute("SELECT * FROM events")
    events = c.fetchall()
    event_cols = [desc[0] for desc in c.description]
    with open(os.path.join(out_dir, "events.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(event_cols)
        writer.writerows(events)

    # Export summary JSON
    summary = {
        "unique_visitors": len(visitors),
        "total_events": len(events),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    conn.close()
    print(f"✅ Reports exported to {out_dir}")
