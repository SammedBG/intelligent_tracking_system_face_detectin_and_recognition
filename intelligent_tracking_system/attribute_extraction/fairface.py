import numpy as np
import onnxruntime as ort
import cv2

class FairFaceAgeGender:
    def __init__(self, model_path="models/fairface_age_gender.onnx"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)
        return img.astype("float32")  # Ensure float32 for ONNX

    def predict(self, img):
        if img is None or img.size == 0:
            return 30, "Unknown"
        try:
            input_tensor = self.preprocess(img)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            age_probs, gender_probs = outputs[0], outputs[1]
            age = int(np.argmax(age_probs)) + 1  # Classes 1-70
            gender = "Male" if np.argmax(gender_probs) == 1 else "Female"
            return age, gender
        except Exception as e:
            print(f"[ERROR] FairFace prediction failed: {e}")
            return 30, "Unknown"
