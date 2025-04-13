import os
import pickle
import numpy as np
import tensorflow as tf
from transformers import CLIPProcessor, TFCLIPModel
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import logging
import hashlib

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(
    filename="app.log",  # Will be in Railway's filesystem
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

UPLOAD_FOLDER = "one_piece_images_300"
CLIP_EMBEDDINGS_PATH = "clip_tf_embeddings.pkl"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created directory: {UPLOAD_FOLDER}")

# Cache for image embeddings
embedding_cache = {}

# Load CLIP model and processor
model = None
processor = None
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    try:
        model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logging.info("CLIP model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load CLIP model: {str(e)}")

# Load precomputed embeddings from the pickle file
try:
    with open(CLIP_EMBEDDINGS_PATH, "rb") as f:
        embedding_data = pickle.load(f)
    embeddings = embedding_data["embeddings"]
    image_paths = embedding_data["image_paths"]
    labels = embedding_data["labels"]
    CLASS_NAMES = embedding_data["characters"]
    logging.info("CLIP embeddings loaded successfully")
    logging.info(f"Sample image paths: {image_paths[:5]}")
    logging.info(f"Sample labels: {labels[:5]}")
    # Log Shinobu paths if present
    shinobu_paths = [
        path for path, label in zip(image_paths, labels) if label == "Shinobu"
    ]
    logging.info(f"Shinobu image paths: {shinobu_paths[:5]}")
except Exception as e:
    embeddings = None
    image_paths = None
    labels = None
    CLASS_NAMES = []
    logging.error(f"Failed to load embeddings: {str(e)}")


# Allowed file type checker
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Image preprocessing
def preprocess_image(image):
    # Process the image using CLIP's processor
    inputs = processor(images=image, return_tensors="tf")
    return inputs


# Cosine similarity calculation
def compute_cosine_similarity(emb1, emb2):
    emb1 = tf.nn.l2_normalize(emb1, axis=1)
    emb2 = tf.nn.l2_normalize(emb2, axis=1)
    return tf.reduce_sum(tf.multiply(emb1, emb2), axis=1).numpy()


@app.route("/")
def index():
    logging.info("Serving index page")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Prediction request received")

    if "predict_file" not in request.files:
        logging.error("No file part in prediction request")
        return jsonify({"error": "No file part"}), 400

    file = request.files["predict_file"]
    if not file or file.filename == "":
        logging.error("No file selected for prediction")
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        logging.error(f"Invalid file type: {file.filename}")
        return jsonify({"error": f"Invalid file type: {file.filename}"}), 400

    try:
        # Compute a hash of the uploaded image to use as a cache key
        file.seek(0)
        image_data = file.read()
        image_hash = hashlib.md5(image_data).hexdigest()
        file.seek(0)

        # Check if the embedding is cached
        if image_hash in embedding_cache:
            logging.info("Using cached embedding")
            image_embedding = embedding_cache[image_hash]
        else:
            # Process uploaded image using CLIP
            img = Image.open(file.stream).convert("RGB")
            inputs = preprocess_image(img)
            image_embedding = model.get_image_features(**inputs).numpy()
            # Cache the embedding
            embedding_cache[image_hash] = image_embedding
            logging.info("Computed and cached new embedding")

        if embeddings is None or model is None:
            logging.error("Model or embeddings not loaded")
            return jsonify({"error": "Model or embeddings not loaded"}), 503

        # Compute similarities using cosine similarity
        similarities = compute_cosine_similarity(image_embedding, embeddings)
        predicted_idx = np.argmax(similarities)
        predicted_character = labels[predicted_idx]

        # Get representative image path
        local_image_path = os.path.normpath(image_paths[predicted_idx])
        logging.info(f"Image path for {predicted_character}: {local_image_path}")

        # Validate image path
        if not os.path.isfile(local_image_path):
            logging.error(f"Image file missing: {local_image_path}")
            return (
                jsonify(
                    {
                        "predicted_character": predicted_character,
                        "representative_image": None,
                        "error": f"Image not found for {predicted_character}: {local_image_path}",
                    }
                ),
                200,
            )

        # Construct URL
        try:
            relative_path = os.path.relpath(local_image_path, UPLOAD_FOLDER).replace(
                "\\", "/"
            )
            representative_image = f"/dataset/{relative_path}"
            logging.info(f"Generated image URL: {representative_image}")
        except ValueError as e:
            logging.error(f"Failed to compute relative path: {str(e)}")
            return (
                jsonify(
                    {
                        "predicted_character": predicted_character,
                        "representative_image": None,
                        "error": f"Invalid image path for {predicted_character}: {local_image_path}",
                    }
                ),
                200,
            )

        return jsonify(
            {
                "predicted_character": predicted_character,
                "representative_image": representative_image,
            }
        )

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/dataset/<path:filename>")
def serve_dataset_image(filename):
    try:
        logging.info(f"Request to serve image: {filename}")
        full_path = os.path.normpath(
            os.path.join(app.config["UPLOAD_FOLDER"], filename)
        )
        logging.info(f"Resolved full path: {full_path}")
        if not os.path.isfile(full_path):
            logging.error(f"Image file not found: {full_path}")
            return jsonify({"error": f"Image not found: {filename}"}), 404
        logging.info(f"Serving image: {full_path}")
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        logging.error(f"Failed to serve image {filename}: {str(e)}")
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 500


if __name__ == "__main__":
    import os

    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
