import numpy as np
import cv2
from sklearn.cluster import KMeans

class ShirtColorExtractor:
    def __init__(self):
        self.color_labels = {
            "Black": [0, 0, 0],
            "White": [255, 255, 255],
            "Gray": [128, 128, 128],
            "Red": [220, 20, 60],
            "Orange": [255, 140, 0],
            "Yellow": [255, 215, 0],
            "Green": [0, 128, 0],
            "Blue": [0, 0, 255],
            "Purple": [128, 0, 128],
            "Pink": [255, 182, 193],
            "Brown": [139, 69, 19],
            "Beige": [245, 245, 220],
            "Cream": [255, 253, 208]
        }
        self.lab_colors = {name: self._rgb2lab(np.array(rgb)) for name, rgb in self.color_labels.items()}

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return "Unknown"

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape((-1, 3))

        try:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
            dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
            return self._map_to_label(dominant)
        except Exception:
            return "Unknown"

    def _map_to_label(self, lab_color):
        distances = {label: np.linalg.norm(lab_color - np.array(lab))
                     for label, lab in self.lab_colors.items()}
        return min(distances, key=distances.get)

    def _rgb2lab(self, rgb):
        rgb = np.uint8([[rgb]])
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0][0]
        return lab
