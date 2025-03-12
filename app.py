import numpy as np
import tensorflow as tf
import pickle
import io
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load trained model
with open("image_classification_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Basic authentication
USERNAME = "admin"
PASSWORD = "password"

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return jsonify({"message": "Authentication required"}), 401

@app.before_request
def require_auth():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    img = image.load_img(io.BytesIO(file.read()), target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return jsonify({"class": predicted_class, "confidence": confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
