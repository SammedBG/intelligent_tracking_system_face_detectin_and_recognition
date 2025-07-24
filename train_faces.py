import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

EMP_DIR = "data/employee_images"
PROFILE_JSON = "data/embeddings.json"

def main():
    app = FaceAnalysis(name='antelopev2', root='arcface_model', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings_data = []

    for person_name in os.listdir(EMP_DIR):
        person_path = os.path.join(EMP_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Processing: {person_name}")
        embeddings = []

        for file in tqdm(os.listdir(person_path), desc=person_name):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if not faces:
                print(f"[WARN] No face detected in {file}")
                continue

            face = faces[0]
            emb = getattr(face, "normed_embedding", face.embedding)
            embeddings.append(emb)

        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)  # Normalize mean embedding
            embeddings_data.append({
                "name": person_name,
                "embedding": mean_emb.tolist()
            })
        else:
            print(f"[WARN] No embeddings for {person_name}")

    with open(PROFILE_JSON, "w") as f:
        json.dump(embeddings_data, f, indent=4)

    print(f"[SUCCESS] Training complete. {len(embeddings_data)} identities saved to {PROFILE_JSON}")

if __name__ == "__main__":
    main()
