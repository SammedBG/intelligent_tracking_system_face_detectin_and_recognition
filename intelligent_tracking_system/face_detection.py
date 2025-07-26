import cv2
import mediapipe as mp
from typing import List, Tuple

class FaceDetector:
    def __init__(self, conf_threshold: float = 0.5, max_num_faces: int = 5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            min_detection_confidence=conf_threshold
        )

    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        boxes = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                x1 = int(min(xs) * w)
                y1 = int(min(ys) * h)
                x2 = int(max(xs) * w)
                y2 = int(max(ys) * h)
                boxes.append((x1, y1, x2, y2))
        return boxes

    def close(self):
        self.face_mesh.close()
