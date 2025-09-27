# modules/database.py
import os
import sqlite3
import json
import numpy as np

def init_db(db_path='db/visitors.db'):
    """Create DB and tables if not exist."""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS visitors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding TEXT,
        first_seen TEXT,
        image_path TEXT
      )
    ''')
    c.execute('''
      CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_id INTEGER,
        event_type TEXT,
        timestamp TEXT,
        image_path TEXT
      )
    ''')
    conn.commit()
    conn.close()

class Database:
    def __init__(self, db_path='db/visitors.db'):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def register_face(self, embedding, image_path, timestamp):
        """Store new face embedding and return face_id."""
        emb_text = json.dumps(np.array(embedding).tolist())
        self.cursor.execute(
            'INSERT INTO visitors (embedding, first_seen, image_path) VALUES (?, ?, ?)',
            (emb_text, timestamp, image_path)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def find_match(self, embedding, threshold=0.6):
        """Return (face_id, similarity) if a match >= threshold else (None, best_sim)."""
        self.cursor.execute('SELECT id, embedding FROM visitors')
        rows = self.cursor.fetchall()
        if not rows:
            return (None, 0.0)
        emb = np.array(embedding, dtype=float)
        best_sim = -1.0
        best_id = None
        for rid, emb_text in rows:
            try:
                ref = np.array(json.loads(emb_text), dtype=float)
            except Exception:
                continue
            denom = (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-10)
            sim = float(np.dot(emb, ref) / denom)
            if sim > best_sim:
                best_sim = sim
                best_id = rid
        if best_sim >= threshold:
            return (best_id, best_sim)
        return (None, best_sim)

    def insert_event(self, face_id, event_type, timestamp, image_path):
        self.cursor.execute(
            'INSERT INTO events (face_id, event_type, timestamp, image_path) VALUES (?, ?, ?, ?)',
            (face_id, event_type, timestamp, image_path)
        )
        self.conn.commit()

    def get_unique_count(self):
        self.cursor.execute('SELECT COUNT(*) FROM visitors')
        return int(self.cursor.fetchone()[0])

    def close(self):
        self.conn.close()
