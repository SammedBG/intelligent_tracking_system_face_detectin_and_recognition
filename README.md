# Intelligent Tracking System

## Overview
A modular face recognition and tracking system using deep learning models for detection, recognition, and attribute extraction. Supports real-time video processing, GUI-based data collection, and extensible data management.

## Features
- Real-time face detection (YOLOv8)
- Face recognition (ArcFace/InsightFace)
- Attribute extraction (age, gender, glasses, mask, shirt color)
- Face alignment for improved accuracy
- GUI for capturing and labeling new faces
- Data storage in JSON and SQLite
- Logging and configuration via YAML

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/SammedBG/intelligent_tracking_system_face_detectin_and_recognition.git
   cd intelligent_tracking_system
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download models:**
   - Place YOLOv8 and ONNX models in the `models/` and `arcface_model/models/antelopev2/` directories as needed.
4. **Configure settings:**
   - Edit `config.yaml` to adjust thresholds, paths, and options.

## Usage
### 1. Training (Add new faces)
- Use the GUI to capture images:
  ```bash
  python gui_face.py
  ```
- After capturing, run:
  ```bash
  python train_faces.py
  ```
  This updates `data/embeddings.json` with new identities.

### 2. Running the Main Application
- Start the real-time tracking system:
  ```bash
  python main.py
  ```
- The system will display a video window with recognized faces and attributes.

### 3. Testing
- Run the test script to validate recognition:
  ```bash
  python test.py
  ```

## Data Structure
- `data/employee_images/`: Raw images for each person
- `data/embeddings.json`: Face embeddings database
- `data/detections.json`: Detection logs
- `data/unknown_embeddings.json`: Embeddings for unknown faces
- `data/label_map.pkl`: Label mapping (if used)
- `logs/`: Log files

## Configuration
- All main parameters are in `config.yaml` (model paths, thresholds, logging, etc.)

## Contributing
1. Fork the repository and create a new branch for your feature or fix.
2. Write clear, well-documented code and add tests where appropriate.
3. Ensure all dependencies are listed in `requirements.txt`.
4. Submit a pull request with a clear description of your changes.

## License
Specify your license here.

## Acknowledgements
- [InsightFace](https://github.com/deepinsight/insightface)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google/mediapipe) 
