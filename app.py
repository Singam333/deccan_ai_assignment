import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB

# Setup basic authentication
auth = HTTPBasicAuth()

# You should change these credentials and ideally store them securely
users = {
    "admin": generate_password_hash("password123")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None

# Load the trained model
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/cifar10_mobilenet_final')
model = tf.keras.models.load_model(MODEL_PATH)

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the image
def preprocess_image(image):
    # Resize to 32x32 (CIFAR-10 size)
    image = image.resize((32, 32))
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "CIFAR-10 classifier"})

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    # Check if an image was sent in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    try:
        # Read and preprocess the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Return prediction results
        return jsonify({
            "class": predicted_class,
            "class_index": int(predicted_class_index),
            "confidence": confidence,
            "all_probabilities": {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to provide information about the API
@app.route('/', methods=['GET'])
def info():
    return jsonify({
        "name": "CIFAR-10 Image Classification API",
        "description": "API for classifying images into one of ten CIFAR-10 classes",
        "endpoints": {
            "/predict": "POST - Submit an image for classification (requires authentication)",
            "/health": "GET - Check API health status"
        },
        "classes": class_names
    })

if __name__ == '__main__':
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
