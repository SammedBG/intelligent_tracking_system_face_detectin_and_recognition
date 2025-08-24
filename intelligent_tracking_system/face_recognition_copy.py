import json
import os
import numpy as np
from insightface.app import FaceAnalysis
from intelligent_tracking_system.utils import cosine_similarity

class FaceRecognizer:
    def __init__(self, embeddings_path, insightface_root, insightface_provider, similarity_threshold=0.7, logger=None):
        self.logger = logger
        self.similarity_threshold = similarity_threshold

        with open(embeddings_path, "r") as f:
            profiles = json.load(f)

        self.names = [p["name"] for p in profiles]
        self.embeddings = np.array([p["embedding"] for p in profiles], dtype=np.float32)

        self.app = FaceAnalysis(name="antelopev2", root=insightface_root, providers=[insightface_provider])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.unknown_path = "data/unknown_embeddings.json"
        self.unknown_db = {}
        self.unknown_counter = 1

        if os.path.exists(self.unknown_path):
            with open(self.unknown_path, "r") as f:
                self.unknown_db = json.load(f)
            self.unknown_counter = len(self.unknown_db) + 1

    def match_embedding(self, emb):
        similarities = np.dot(self.embeddings, emb)
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        if best_score >= self.similarity_threshold:
            return self.names[best_idx], best_score
        return "Unknown", best_score

    def recognize_or_track_unknown(self, emb):
        # Step 1: check if it's a known person
        name, score = self.match_embedding(emb)
        if name != "Unknown":
            return name, score

        # Step 2: check existing unknowns
        for uid, saved_emb in self.unknown_db.items():
            sim = cosine_similarity(emb, np.array(saved_emb))
            if sim >= self.similarity_threshold:
                return uid, sim

        # Step 3: new unknown
        new_id = f"Unknown#{self.unknown_counter}"
        self.unknown_db[new_id] = emb.tolist()
        self.unknown_counter += 1
        with open(self.unknown_path, "w") as f:
            json.dump(self.unknown_db, f, indent=2)

        return new_id, 0.0
