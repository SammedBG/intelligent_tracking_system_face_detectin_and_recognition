import mediapipe as mp
import os
import cv2
import time
from datetime import datetime
import threading
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# --- Configuration Variables ---
SAVE_FOLDER = "data/employee_images/Reshma"
video_link = 0  # 0 = webcam
roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 500, 500
time_diff_threshold = 2.0
padding = 10
config_reload_interval = 10
min_detection_confidence_threshold = 0.5
brightness_increase = 0
show_roi_box = True
show_face_box = True
show_resolution_text = True
last_config_reload_time = time.time()

# Ensure the save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# --- Config reload placeholder (can be expanded) ---
def reload_config():
    return (
        video_link, roi_x1, roi_y1, roi_x2, roi_y2, time_diff_threshold,
        padding, config_reload_interval, min_detection_confidence_threshold,
        brightness_increase, show_roi_box, show_face_box, show_resolution_text
    )

# --- Safely calculate the next face counter ---
existing_faces = os.listdir(SAVE_FOLDER)
existing_numbers = []
for filename in existing_faces:
    try:
        if filename.startswith("face_") and filename.endswith(".jpg"):
            num = int(filename.split('_')[2].split('.')[0])
            existing_numbers.append(num)
    except (IndexError, ValueError):
        continue

face_counter = max(existing_numbers) + 1 if existing_numbers else 0

# --- Save face to file ---
def save_face_image(face, face_counter):
    ftimestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    face_filename = f"face_{ftimestamp}_{face_counter}.jpg"
    save_path = os.path.join(SAVE_FOLDER, face_filename)
    cv2.imwrite(save_path, face)

# --- Start Video Capture ---
cap = cv2.VideoCapture(video_link)

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_detection_confidence_threshold) as face_detection:

    last_captured_time = time.time()
    target_aspect_ratio = 16 / 9

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Ignoring empty camera frame.")
            break

        # Reload config periodically
        if time.time() - last_config_reload_time > config_reload_interval:
            video_link, roi_x1, roi_y1, roi_x2, roi_y2, time_diff_threshold, padding, config_reload_interval, min_detection_confidence_threshold, brightness_increase, show_roi_box, show_face_box, show_resolution_text = reload_config()
            last_config_reload_time = time.time()

        # Adjust brightness
        brightness_factor = 1 + (brightness_increase / 100)
        image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        # Resize to maintain aspect ratio
        original_height, original_width, _ = image.shape
        if show_resolution_text:
            resolution_text = f"Resolution: {original_width}x{original_height}"
            cv2.putText(image, resolution_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        target_width = int(original_height * target_aspect_ratio)
        image = cv2.resize(image, (target_width, original_height))

        current_time = time.time()
        if current_time - last_captured_time >= time_diff_threshold:
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                int(bbox.width * iw), int(bbox.height * ih)

                    x -= padding
                    y -= padding
                    w += 2 * padding
                    h += 2 * padding

                    x = max(0, x)
                    y = max(0, y)
                    w = min(iw - x, w)
                    h = min(ih - y, h)

                    # Check if face is inside the ROI
                    if (x + w > roi_x1 and x < roi_x2 and y + h > roi_y1 and y < roi_y2):
                        if show_face_box:
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        face = image[y:y + h, x:x + w]
                        threading.Thread(target=save_face_image, args=(face, face_counter)).start()
                        face_counter += 1
                        last_captured_time = current_time

        if show_roi_box:
            cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
