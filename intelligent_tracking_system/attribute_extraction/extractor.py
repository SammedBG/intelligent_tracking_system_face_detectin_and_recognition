import numpy as np
import cv2
from datetime import datetime

class AttributeExtractor:
    def __init__(self):
        pass

    def extract(self, frame, bbox, face_obj=None):
        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]
        
        # Default fallback values
        age = 30
        gender = "Unknown"
        glasses = False
        face_mask = False
        shirt_color = "Unknown"
        clothing_type = "Unknown"

        # Extract age and gender from InsightFace face object
        if face_obj:
            try:
                age = int(face_obj.age)
                gender = "Male" if face_obj.gender == 1 else "Female"
            except:
                pass

        # Basic logic for glasses/mask (placeholder)
        glasses = False
        face_mask = False

        # Extract shirt region: below face bbox
        h, w, _ = frame.shape
        torso_top = y2
        torso_bottom = min(h, y2 + (y2 - y1) * 2)
        torso_left = max(0, x1)
        torso_right = min(w, x2)
        shirt_region = frame[torso_top:torso_bottom, torso_left:torso_right]

        if shirt_region.size > 0:
            shirt_color = self.extract_dominant_color(shirt_region)

        # Uniqueness signature
        age_group = "Senior" if age >= 50 else "Adult" if age >= 25 else "Youth"
        signature = f"{shirt_color}_{gender}_{age_group}_{clothing_type}"

        return {
            "age": age,
            "gender": gender,
            "glasses": glasses,
            "face_mask": face_mask,
            "shirt_color": shirt_color,
            "clothing_type": clothing_type,
            "uniqueness_signature": signature,
            "timestamp": datetime.now().isoformat()
        }

    def extract_dominant_color(self, image):
        # Resize to speed up
        image = cv2.resize(image, (50, 50))
        image = image.reshape(-1, 3)

        # Remove low contrast pixels
        image = np.array([c for c in image if np.all(c > 30) and np.all(c < 220)])
        if len(image) == 0:
            return "Unknown"

        # Convert BGR to HSV and get average hue
        hsv = cv2.cvtColor(np.uint8([image]), cv2.COLOR_BGR2HSV)[0]
        hue = np.mean(hsv[:, 0])

        return self.hue_to_color(hue)

    def hue_to_color(self, hue):
        if hue < 15 or hue >= 165:
            return "Red"
        elif 15 <= hue < 30:
            return "Orange"
        elif 30 <= hue < 45:
            return "Yellow"
        elif 45 <= hue < 85:
            return "Green"
        elif 85 <= hue < 135:
            return "Blue"
        elif 135 <= hue < 165:
            return "Purple"
        else:
            return "Unknown"
