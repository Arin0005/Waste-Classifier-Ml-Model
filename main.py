from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import io
import os
import gdown  # For downloading from Google Drive

app = Flask(__name__, static_folder='static')

# Model loading configuration
MODEL_PATH = r'models\waste_classification_mobilnet (1).h5'

print("Initializing model...")
# download_model()
import os

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)
img_size = (224, 224)  # Adjust according to your model's requirements


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Start timing
        start_time = time.time()

        # Read and preprocess the image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize(img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = "Organic" if prediction[0] < 0.5 else "Recyclable"
        confidence = float(prediction[0]) if predicted_class == "Recyclable" else float(1 - prediction[0])

        # Calculate processing time
        processing_time = time.time() - start_time

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence * 100:.2f}%",
            'processing_time': f"{processing_time:.2f} seconds"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)