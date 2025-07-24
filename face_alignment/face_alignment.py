import cv2
import numpy as np
import mediapipe as mp

class FaceAligner:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.LEFT_EYE_INDEXES = [33, 133]
        self.RIGHT_EYE_INDEXES = [362, 263]

    def align(self, image):
        h, w = image.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in self.LEFT_EYE_INDEXES], axis=0)
        right_eye = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in self.RIGHT_EYE_INDEXES], axis=0)

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned
