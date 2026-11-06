import os
import threading
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = {
    "words": ("converted_keras/keras_model.h5", "converted_keras/labels.txt"),
    "letters": ("allLetters/keras_model.h5", "allLetters/labels.txt"),
    "numbers": ("numbers_converted_keras/keras_model.h5", "numbers_converted_keras/labels.txt"),
}

def read_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                labels.append(parts[1])
    return labels

app = Flask(__name__)
detector = HandDetector(maxHands=2)
classifiers = {}
labels_by_mode = {}

for mode, (model_path, labels_path) in MODEL_DIRS.items():
    model_file = os.path.join(BASE_DIR, model_path)
    labels_file = os.path.join(BASE_DIR, labels_path)
    if not os.path.isfile(model_file) or not os.path.isfile(labels_file):
        raise FileNotFoundError(f"Missing model files for {mode}")
    classifiers[mode] = Classifier(model_file, labels_file)
    labels_by_mode[mode] = read_labels(labels_file)

OFFSET = 20
IMAGE_SIZE = 300
prediction_lock = threading.Lock()

def prepare_hand_image(frame, bbox):
    x, y, w, h = [int(value) for value in bbox]
    if w <= 0 or h <= 0:
        return None

    height, width = frame.shape[:2]
    x1, y1 = max(0, x - OFFSET), max(0, y - OFFSET)
    x2, y2 = min(width, x + w + OFFSET), min(height, y + h + OFFSET)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    white = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
    aspect_ratio = h / w

    if aspect_ratio > 1:
        new_width = max(1, int(IMAGE_SIZE * w / h))
        resized = cv2.resize(crop, (new_width, IMAGE_SIZE))
        gap = (IMAGE_SIZE - new_width) // 2
        white[:, gap:gap + new_width] = resized
    else:
        new_height = max(1, int(IMAGE_SIZE * h / w))
        resized = cv2.resize(crop, (IMAGE_SIZE, new_height))
        gap = (IMAGE_SIZE - new_height) // 2
        white[gap:gap + new_height, :] = resized
    return white

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "models": list(classifiers.keys())})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "words")
    image_data = payload.get("image", "")

    if mode not in classifiers:
        return jsonify({"error": "Choose words, letters, or numbers."}), 400
    if not image_data:
        return jsonify({"error": "No camera frame was received."}), 400

    # Browser sends a data URL: data:image/jpeg;base64,...
    import base64
    try:
        encoded = image_data.split(",", 1)[-1]
        raw = base64.b64decode(encoded)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        frame = None

    if frame is None:
        return jsonify({"error": "The camera frame could not be read."}), 400

    # Match the original desktop app's mirrored-camera behavior.
    frame = cv2.flip(frame, 1)
    with prediction_lock:
        hands, _ = detector.findHands(frame, draw=False)
        if not hands:
            return jsonify({"detected": False, "message": "Show one hand inside the camera."})

        hand_image = prepare_hand_image(frame, hands[0]["bbox"])
        if hand_image is None:
            return jsonify({"detected": False, "message": "Move your hand fully into view."})

        prediction, index = classifiers[mode].getPrediction(hand_image, draw=False)
    labels = labels_by_mode[mode]
    if index < 0 or index >= len(labels):
        return jsonify({"detected": False, "message": "Could not classify this sign."})

    confidence = float(prediction[index]) if len(prediction) > index else 0.0
    return jsonify({
        "detected": True,
        "label": labels[index],
        "confidence": round(confidence * 100, 1),
        "mode": mode
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
