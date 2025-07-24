import json
import numpy as np
from insightface.app import FaceAnalysis
from intelligent_tracking_system.utils import cosine_similarity

class FaceRecognizer:
    def __init__(self, embeddings_path, insightface_root, insightface_provider, similarity_threshold=0.5, logger=None):
        self.logger = logger
        self.similarity_threshold = similarity_threshold

        with open(embeddings_path, "r") as f:
            profiles = json.load(f)

        self.names = [p["name"] for p in profiles]
        self.embeddings = np.array([p["embedding"] for p in profiles], dtype=np.float32)

        self.app = FaceAnalysis(name="antelopev2", root=insightface_root, providers=[insightface_provider])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    def match_embedding(self, emb):
        similarities = np.dot(self.embeddings, emb)
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        if best_score >= self.similarity_threshold:
            return self.names[best_idx], best_score
        return "Unknown", best_score


    def recognize(self, frame):
        faces = self.app.get(frame)
        if not faces:
            return "Unknown", 0.0

        face = faces[0]
        emb = getattr(face, "normed_embedding", face.embedding)

        similarities = np.dot(self.embeddings, emb)  # Efficient cosine (since both are normalized)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= self.similarity_threshold:
            return self.names[best_idx], best_score
        return "Unknown", best_score

