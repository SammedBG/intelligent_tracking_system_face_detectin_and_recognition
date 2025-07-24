import numpy as np
import cv2
import mediapipe as mp

class AccessoryDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                          refine_landmarks=True, min_detection_confidence=0.5)

    def predict(self, image, bbox):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, False  # no face, assume no glasses/mask

        # Get landmarks for first face
        landmarks = results.multi_face_landmarks[0].landmark
        nose = self._point(landmarks[4], w, h)
        left_eye = self._point(landmarks[33], w, h)
        right_eye = self._point(landmarks[263], w, h)
        mouth = self._point(landmarks[13], w, h)

        # Glasses detection heuristic: dark region around eyes
        eye_box = self._make_box([left_eye, right_eye], pad=15, img_shape=(h, w))
        eye_crop = image[eye_box[1]:eye_box[3], eye_box[0]:eye_box[2]]
        glasses = self._is_dark(eye_crop)

        # Mask detection heuristic: occluded mouth/nose
        mask_box = self._make_box([nose, mouth], pad=10, img_shape=(h, w))
        mask_crop = image[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]]
        face_mask = self._is_plain(mask_crop)

        return glasses, face_mask

    def _point(self, landmark, w, h):
        return int(landmark.x * w), int(landmark.y * h)

    def _make_box(self, points, pad=10, img_shape=(720,1280)):
        xs, ys = zip(*points)
        x1, y1 = max(min(xs)-pad,0), max(min(ys)-pad,0)
        x2, y2 = min(max(xs)+pad,img_shape[1]), min(max(ys)+pad,img_shape[0])
        return (x1, y1, x2, y2)

    def _is_dark(self, crop):
        if crop is None or crop.size == 0: return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < 60

    def _is_plain(self, crop):
        if crop is None or crop.size == 0: return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:,:,1]
        stddev = np.std(sat)
        return stddev < 18
