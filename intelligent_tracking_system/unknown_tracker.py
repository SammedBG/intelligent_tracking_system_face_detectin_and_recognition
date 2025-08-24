import json
import numpy as np
import os
from typing import Tuple, Optional

class UnknownFaceTracker:
    def __init__(self, unknown_embeddings_path: str = "data/unknown_embeddings.json", similarity_threshold: float = 0.6):
        self.unknown_embeddings_path = unknown_embeddings_path
        self.similarity_threshold = similarity_threshold
        self.unknown_faces = []
        self.next_id = 1
        self._load_unknown_faces()
    
    def _load_unknown_faces(self):
        """Load previously seen unknown faces from file"""
        if os.path.exists(self.unknown_embeddings_path):
            try:
                with open(self.unknown_embeddings_path, 'r') as f:
                    data = json.load(f)
                    self.unknown_faces = data.get('unknown_faces', [])
                    self.next_id = data.get('next_id', 1)
            except Exception as e:
                print(f"Warning: Could not load unknown faces: {e}")
                self.unknown_faces = []
                self.next_id = 1
    
    def _save_unknown_faces(self):
        """Save unknown faces to file"""
        try:
            os.makedirs(os.path.dirname(self.unknown_embeddings_path), exist_ok=True)
            data = {
                'unknown_faces': self.unknown_faces,
                'next_id': self.next_id
            }
            with open(self.unknown_embeddings_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save unknown faces: {e}")
    
    def find_or_create_unknown_id(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Find existing unknown face or create new one.
        Returns: (unknown_id, similarity_score)
        """
        if len(self.unknown_faces) == 0:
            # First unknown face
            unknown_id = f"Unknown#{self.next_id}"
            self.unknown_faces.append({
                'id': unknown_id,
                'embedding': embedding.tolist()
            })
            self.next_id += 1
            self._save_unknown_faces()
            return unknown_id, 0.0
        
        # Compare with existing unknown faces
        embedding_np = np.array(embedding)
        best_similarity = 0.0
        best_match = None
        
        for unknown_face in self.unknown_faces:
            stored_embedding = np.array(unknown_face['embedding'])
            # Ensure both embeddings are normalized for cosine similarity
            embedding_norm = embedding_np / np.linalg.norm(embedding_np)
            stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
            similarity = np.dot(embedding_norm, stored_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = unknown_face
        
        # If similarity is high enough, reuse existing ID
        if best_similarity >= self.similarity_threshold:
            return best_match['id'], best_similarity
        
        # Create new unknown face
        unknown_id = f"Unknown#{self.next_id}"
        self.unknown_faces.append({
            'id': unknown_id,
            'embedding': embedding.tolist()
        })
        self.next_id += 1
        self._save_unknown_faces()
        return unknown_id, 0.0
    
    def get_unknown_count(self) -> int:
        """Get the number of unique unknown faces tracked"""
        return len(self.unknown_faces)
    
    def clear_unknown_faces(self):
        """Clear all unknown faces (useful for testing)"""
        self.unknown_faces = []
        self.next_id = 1
        self._save_unknown_faces() 