# modules/face_alignment.py

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class FaceAligner:
    """
    Uses MediaPipe Face Mesh to detect facial landmarks
    and perform similarity‐transform alignment based on eyes and jaw contour.
    """

    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        # Indices for left and right eye outer corners (Mediapipe face_mesh):
        self._LEFT_EYE_IDX = 33
        self._RIGHT_EYE_IDX = 263

    def align(self,
              image: np.ndarray,
              bbox: Tuple[int,int,int,int],
              output_size: int = 112) -> Optional[np.ndarray]:
        """
        Aligns the face inside bbox using eye contour points.
        Returns a square aligned face of size (output_size x output_size),
        or None if landmarks not found.
        """
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return None

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = face.shape

        # Get eye coordinates in face-crop space
        left_eye = lm[self._LEFT_EYE_IDX]
        right_eye = lm[self._RIGHT_EYE_IDX]
        left = np.array((left_eye.x * w, left_eye.y * h))
        right = np.array((right_eye.x * w, right_eye.y * h))

        # Compute center between eyes
        eyes_center = (left + right) / 2.0

        # Desired placement of eyes in output
        desired_left = (0.3 * output_size, 0.3 * output_size)
        desired_right = (0.7 * output_size, 0.3 * output_size)
        desired_center = ( (desired_left[0] + desired_right[0]) / 2,
                           (desired_left[1] + desired_right[1]) / 2 )

        # Compute similarity transform
        src_pts = np.vstack([left, right, eyes_center]).astype(np.float32)
        dst_pts = np.vstack([desired_left, desired_right, desired_center]).astype(np.float32)

        M = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(face, M, (output_size, output_size),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        return aligned
