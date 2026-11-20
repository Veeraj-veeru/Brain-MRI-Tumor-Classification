
import os
import json
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_mri_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

IMAGE_SIZE = 128
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class names
model = load_model(MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_tumor(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(img, dtype=np.float32)

    # Must match the preprocessing used when training the model.
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    index = int(np.argmax(predictions))

    return class_names[index], float(predictions[index])


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    file_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            error = "Please select an MRI image."
        elif not allowed_file(file.filename):
            error = "Please upload a JPG, JPEG, or PNG image."
        else:
            filename = secure_filename(file.filename)

            # Generate a unique name to avoid overwriting existing uploads.
            extension = filename.rsplit(".", 1)[1].lower()
            saved_name = f"{uuid.uuid4().hex}.{extension}"
            save_path = os.path.join(UPLOAD_FOLDER, saved_name)

            try:
                file.save(save_path)

                predicted_class, score = predict_tumor(save_path)

                result = (
                    "No Tumor"
                    if predicted_class.lower() == "notumor"
                    else predicted_class
                )
                confidence = f"{score * 100:.2f}%"
                file_path = url_for("uploaded_file", filename=saved_name)

            except (UnidentifiedImageError, OSError, ValueError):
                error = "The uploaded file could not be read as an image."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        file_path=file_path,
        error=error
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return app.send_static_file("nonexistent") if False else (
        __import__("flask").send_from_directory(
            app.config["UPLOAD_FOLDER"], filename
        )
    )


if __name__ == "__main__":
    app.run(debug=True)
