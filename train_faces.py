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

    # Load existing embeddings if file exists
    if os.path.exists(PROFILE_JSON):
        with open(PROFILE_JSON, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    existing_dict = {entry["name"]: entry for entry in existing_data}

    embeddings_data = existing_data.copy()
    updated_names = set()

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
            new_entry = {
                "name": person_name,
                "embedding": mean_emb.tolist()
            }
            # Update if exists, else append
            if person_name in existing_dict:
                for i, entry in enumerate(embeddings_data):
                    if entry["name"] == person_name:
                        embeddings_data[i] = new_entry
                        break
            else:
                embeddings_data.append(new_entry)
            updated_names.add(person_name)
        else:
            print(f"[WARN] No embeddings for {person_name}")

    # Optionally, keep entries for people not in EMP_DIR
    # embeddings_data = [entry for entry in embeddings_data if entry["name"] in updated_names]

    with open(PROFILE_JSON, "w") as f:
        json.dump(embeddings_data, f, indent=4)

    print(f"[SUCCESS] Training complete. {len(embeddings_data)} identities saved to {PROFILE_JSON}")

if __name__ == "__main__":
    main()
