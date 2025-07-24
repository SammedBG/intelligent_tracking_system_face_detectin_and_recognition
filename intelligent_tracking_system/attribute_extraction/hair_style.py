import onnxruntime as ort
import numpy as np
import cv2

class HairStyleClassifier:
    def __init__(self, model_path="models/hair_length_classifier.onnx", labels=None):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels or ["short", "long"]

    def preprocess(self, crop):
        crop = cv2.resize(crop, (224, 224))
        img = crop.astype("float32") / 255.0
        img = img.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(img, axis=0)

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return "Unknown"
        try:
            inp = self.preprocess(crop)
            logits = self.session.run(None, {self.input_name: inp})[0]
            label_idx = int(np.argmax(logits))
            return self.labels[label_idx]
        except Exception:
            return "Unknown"
