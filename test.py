import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis
from face_alignment.face_alignment import FaceAligner
from intelligent_tracking_system.utils import cosine_similarity

# Load stored embeddings
with open("data/embeddings.json", "r") as f:
    db = json.load(f)

person_name = db[0]["name"]
stored_emb = np.array(db[0]["embedding"], dtype=np.float32)
stored_emb /= np.linalg.norm(stored_emb)

# Load same training image (change the path to match your actual image)
img_path = "data/employee_images/reshma/reshma_1.jpg"
img = cv2.imread(img_path)

aligner = FaceAligner()
aligned = aligner.align(img)

if aligned is None:
    print("[ERROR] Alignment failed.")
    exit()

app = FaceAnalysis(name="antelopev2", root="arcface_model", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

faces = app.get(aligned)
if not faces:
    print("[ERROR] No face detected in test image.")
    exit()

live_emb = faces[0].embedding
live_emb /= np.linalg.norm(live_emb)

sim = cosine_similarity(stored_emb, live_emb)
print(f"Similarity score (same person): {sim:.4f}")

if sim >= 0.5:
    print("✅ Match likely. Recognition should succeed.")
else:
    print("❌ Embeddings don't match. There's a training-recognition mismatch.")
