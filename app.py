from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import io
import os

app = Flask(__name__, static_folder='static')

# Model loading configuration - Fixed path for Linux/Render
MODEL_PATH = 'models/waste_classification_mobilnet (1).h5'  # Changed backslashes to forward slashes

print("Initializing model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load model with compatibility fix
try:
    print("Loading model with custom options...")
    # Try loading with compile=False to avoid optimizer issues
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully without compilation!")
    
    # Recompile the model with basic settings
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    print("Model recompiled successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    # Alternative loading method
    try:
        print("Trying alternative loading method...")
        import h5py
        
        # Load model architecture and weights separately
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded with alternative method!")
        
    except Exception as e2:
        print(f"Both loading methods failed: {e2}")
        raise e2

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
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = img.resize(img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction with error handling
        try:
            prediction = model.predict(img_array, verbose=0)  # Added verbose=0 to reduce logs
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            return jsonify({'error': f'Prediction failed: {str(pred_error)}'}), 500
            
        predicted_class = "Organic" if prediction[0][0] < 0.5 else "Recyclable"
        confidence = float(prediction[0][0]) if predicted_class == "Recyclable" else float(1 - prediction[0][0])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence * 100:.2f}%",
            'processing_time': f"{processing_time:.2f} seconds"
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

# Add a health check endpoint for Render
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

# Add model info endpoint for debugging
@app.route('/model-info')
def model_info():
    try:
        return jsonify({
            'model_loaded': model is not None,
            'tensorflow_version': tf.__version__,
            'keras_version': tf.keras.__version__,
            'model_input_shape': str(model.input_shape) if model else 'No model',
            'model_output_shape': str(model.output_shape) if model else 'No model'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Use environment variables for production
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)