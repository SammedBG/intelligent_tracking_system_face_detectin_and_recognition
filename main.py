import cv2
import time
import json
import yaml
import signal
import logging
from intelligent_tracking_system.logger import setup_logger
from intelligent_tracking_system.face_detection import FaceDetector
from intelligent_tracking_system.face_recognition_copy import FaceRecognizer
from intelligent_tracking_system.attribute_extraction.extractor import AttributeExtractor
from database.sqlite_db import SQLiteDB
from intelligent_tracking_system.utils import iou
from intelligent_tracking_system.unknown_tracker import UnknownFaceTracker

LOG_PATH = "logs/app.log"
DETECTIONS_PATH = "data/detections.json"
CONFIG_PATH = "config.yaml"

class TrackingApp:
    def __init__(self, config: dict):
        self.logger = setup_logger(LOG_PATH, config.get("logging", {}).get("level", "INFO"))
        self.logger.info("Starting TrackingApp")

        fd_cfg = config.get("face_detector", {})
        fr_cfg = config.get("face_recognition", {})

        self.detector = FaceDetector(
            conf_threshold=fd_cfg.get("confidence_threshold", 0.5)
        )

        self.recognizer = FaceRecognizer(
            embeddings_path=fr_cfg.get("embeddings_path", "data/embeddings.json"),
            insightface_root=fr_cfg.get("insightface_root", "arcface_model"),
            insightface_provider=fr_cfg.get("insightface_provider", "CPUExecutionProvider"),
            similarity_threshold=fr_cfg.get("similarity_threshold", 0.6),
            logger=self.logger
        )

        self.attribute_extractor = AttributeExtractor()
        self.db = SQLiteDB()
        self.unknown_tracker = UnknownFaceTracker()
        self.detections = []
        self._running = True
        self.tracked_people = {} # Stores data for recognized people being tracked
        self.TRACK_IOU_THRESHOLD = 0.4
        self.TRACK_DISAPPEAR_SECONDS = 5.0


        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, signum, frame):
        self.logger.info("Signal received, stopping...")
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(0)
        self.logger.info("Camera started")

        inserted_names = set()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read frame from camera")
                break

            current_time = time.time()
            detected_boxes = self.detector.detect_faces(frame)

            # --- Tracking Logic ---
            unmatched_boxes = list(detected_boxes)
            updated_tracks = {}

            # 1. Try to match existing tracks with new detections
            for name, track_data in self.tracked_people.items():
                best_iou = self.TRACK_IOU_THRESHOLD
                best_match_box = None

                for box in unmatched_boxes:
                    iou_score = iou(track_data['bbox'], box)
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_match_box = box

                if best_match_box:
                    # Update the track with the new bbox and timestamp
                    track_data['bbox'] = best_match_box
                    track_data['timestamp'] = current_time
                    updated_tracks[name] = track_data
                    unmatched_boxes.remove(best_match_box)

            # 2. Process unmatched boxes as potential new faces
            if unmatched_boxes:
                insight_faces = self.recognizer.app.get(frame)  # Heavy call, only when needed
                for box in unmatched_boxes:
                    best_match = None
                    for face in insight_faces:
                        ibox = face.bbox.astype(int)
                        if iou(box, ibox) > 0.3:
                            emb = getattr(face, "normed_embedding", face.embedding)
                            name, score = self.recognizer.recognize_or_track_unknown(emb)
                            
                            # If not recognized, use unknown tracker
                            if "Unknown" in name:
                                unknown_id, unknown_score = self.unknown_tracker.find_or_create_unknown_id(emb)
                                name = unknown_id
                                score = unknown_score
                            
                            # Track both known and unknown faces
                            attributes = self.attribute_extractor.extract(frame, ibox, face_obj=face)
                            updated_tracks[name] = {
                                'bbox': ibox,
                                'name': name,
                                'score': score,
                                'attributes': attributes,
                                'timestamp': current_time
                            }
                            best_match = (ibox, name, score, face)
                            break

            # 3. Update master tracker, filtering out stale tracks
            self.tracked_people = {
                name: data for name, data in updated_tracks.items()
                if current_time - data['timestamp'] < self.TRACK_DISAPPEAR_SECONDS
            }

            # 4. Draw all active tracks and update database
            for name, track_data in self.tracked_people.items():
                ibox = track_data['bbox']
                score = track_data.get('score', 0.0)
                attributes = track_data['attributes']

                # Draw with different colors for known vs unknown
                if "Unknown" in name:
                    color = (0, 0, 255)  # Red for unknown
                    event_type = "unknown"
                else:
                    color = (0, 255, 0)  # Green for known
                    event_type = "recognized"
                
                cv2.rectangle(frame, (ibox[0], ibox[1]), (ibox[2], ibox[3]), color, 2)
                cv2.putText(frame, f"{name} {score:.2f}", (ibox[0], ibox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Update DB
                if name not in inserted_names:
                    self.db.insert_detection(name, score, attributes, ibox, event_type)
                    inserted_names.add(name)
                else:
                    self.db.update_detection(name, score, attributes, ibox, event_type)

            cv2.imshow("Office Personnel Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.logger.info("ESC pressed, exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Saving {len(self.detections)} detections to {DETECTIONS_PATH}")
        with open(DETECTIONS_PATH, "w") as f:
            json.dump(self.detections, f, indent=2)

        self.detector.close()
        self.db.close()
        self.logger.info("Shutdown complete.")

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config(CONFIG_PATH)
    app = TrackingApp(config)
    app.run()

if __name__ == "__main__":
    main()
