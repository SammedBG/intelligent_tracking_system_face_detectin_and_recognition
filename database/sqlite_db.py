import sqlite3
from datetime import datetime
import os

DB_PATH = "data/recognition_data.db"

class SQLiteDB:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        query = """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            similarity REAL,
            age INTEGER,
            gender TEXT,
            glasses BOOLEAN,
            face_mask BOOLEAN,
            shirt_color TEXT,
            clothing_type TEXT,
            -- hair_length TEXT,  -- Uncomment if hair length is enabled in extractor
            uniqueness_signature TEXT,
            bbox TEXT,
            event_type TEXT
        );
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_detection(self, name, score, attributes, bbox, event_type):
        query = """
        INSERT INTO detections (
            timestamp, name, similarity, age, gender,
            glasses, face_mask, shirt_color, clothing_type, -- hair_length,
            uniqueness_signature, bbox, event_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        values = (
            attributes.get("timestamp", datetime.now().isoformat()),
            name,
            score,
            attributes.get("age"),
            attributes.get("gender"),
            int(attributes.get("glasses", False)),
            int(attributes.get("face_mask", False)),
            attributes.get("shirt_color"),
            attributes.get("clothing_type"),
            # attributes.get("hair_length"),  # Uncomment if hair length is enabled in extractor
            attributes.get("uniqueness_signature"),
            str(bbox),
            event_type
        )
        self.conn.execute(query, values)
        self.conn.commit()

    def update_detection(self, name, score, attributes, bbox, event_type):
        # Update the latest detection for this name
        query = """
        UPDATE detections SET
            timestamp = ?,
            similarity = ?,
            age = ?,
            gender = ?,
            glasses = ?,
            face_mask = ?,
            shirt_color = ?,
            clothing_type = ?,
            -- hair_length = ?,
            uniqueness_signature = ?,
            bbox = ?,
            event_type = ?
        WHERE id = (
            SELECT id FROM detections WHERE name = ? ORDER BY id DESC LIMIT 1
        );
        """
        values = (
            attributes.get("timestamp", datetime.now().isoformat()),
            score,
            attributes.get("age"),
            attributes.get("gender"),
            int(attributes.get("glasses", False)),
            int(attributes.get("face_mask", False)),
            attributes.get("shirt_color"),
            attributes.get("clothing_type"),
            # attributes.get("hair_length"),
            attributes.get("uniqueness_signature"),
            str(bbox),
            event_type,
            name
        )
        self.conn.execute(query, values)
        self.conn.commit()

    def get_last_events(self):
        # Returns a dict of {name: last_event_type} for all names
        query = "SELECT name, event_type FROM detections WHERE id IN (SELECT MAX(id) FROM detections GROUP BY name);"
        cur = self.conn.execute(query)
        return {row[0]: row[1] for row in cur.fetchall()}

    def close(self):
        self.conn.close()
