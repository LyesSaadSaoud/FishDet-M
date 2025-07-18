from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@app.route("/")
def home():
    return "FishDet-M API is running."

@app.route("/detect", methods=["POST"])
@app.route("/detect", methods=["POST"])
def detect():
    try:
        print("Received /detect request")

        if 'image' not in request.files or 'model' not in request.form:
            print("Missing image or model")
            return jsonify({"error": "Missing image or model selection"}), 400

        image_file = request.files['image']
        model_name = request.form['model']
        confidence = float(request.form.get("confidence", 50)) / 100
        print(f"Model: {model_name}, Confidence: {confidence}")

        # === Path to model ===
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.isfile(model_path):
            print(f"Model not found: {model_path}")
            return jsonify({"error": f"Model not found: {model_path}"}), 400

        # === Load model ===
        print("Loading model...")
        model = YOLO(model_path)

        # === Load image ===
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("Image decoding failed")
            return jsonify({"error": "Image decoding failed"}), 400
        print("Image decoded successfully")

        # === Run detection ===
        print("Running inference...")
        results = model(img, conf=confidence)
        boxes = results[0].boxes
        print(f"Detections: {boxes}")

        if boxes is None or len(boxes) == 0:
            print("No objects detected")
            return '', 204

        # === Annotate and return ===
        annotated = results[0].plot()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, annotated)
        print(f"Detection completed. Returning file: {temp_file.name}")
        return send_file(temp_file.name, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({"error": str(e)}), 500

    # === Run detection ===
    try:
        results = model(img, conf=confidence)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return '', 204  # No detections
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # === Save annotated result ===
    annotated = results[0].plot()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, annotated)
    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
