from flask import Flask, request, jsonify, send_file, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Change this to your actual model directory path in deployment (relative if possible)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "Missing image or model parameter"}), 400

    image_file = request.files['image']
    model_name = request.form['model']
    confidence = float(request.form.get("confidence", 50)) / 100

    # Path to selected model
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(model_path):
        return jsonify({"error": f"Model '{model_name}' not found."}), 400

    # Load selected YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    # Decode uploaded image
    try:
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # Perform detection
    try:
        results = model(img, conf=confidence)
        boxes = results[0].boxes

        if boxes is None or boxes.shape[0] == 0:
            return '', 204  # No objects detected

        # Annotate and return image
        annotated = results[0].plot()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, annotated)

        return send_file(temp_file.name, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
