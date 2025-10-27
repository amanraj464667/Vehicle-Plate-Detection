
# Simple SQLite storage for detected plates
import sqlite3
import os
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'plates.db')

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS plates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plate TEXT,
                  timestamp TEXT,
                  speed REAL,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

def insert_plate(plate, timestamp, speed=None, image_path=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO plates (plate, timestamp, speed, image_path) VALUES (?,?,?,?)',
              (plate, timestamp, speed, image_path))
    conn.commit()
    conn.close()
