from flask import Flask, request, jsonify, send_file, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
MODEL_DIR = "C:/Users/saads/PycharmProjects/pythonProject/Grounded-Segment-Anything/project"  # Replace with the actual path to your .pt files

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "Missing image or model selection"}), 400

    image_file = request.files['image']
    model_name = request.form['model']
    confidence = float(request.form.get("confidence", 50)) / 100

    # Path to selected model
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(model_path):
        return jsonify({"error": "Model not found"}), 400

    # Load selected model
    model = YOLO(model_path)

    # Decode image
    img_array = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Run detection
    results = model(img, conf=confidence)
    boxes = results[0].boxes

    if boxes is None or boxes.shape[0] == 0:
        return '', 204  # No objects detected

    # Annotate and return image
    annotated = results[0].plot()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, annotated)

    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
