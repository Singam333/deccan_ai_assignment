import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from io import BytesIO  # Import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
try:
    model = tf.keras.models.load_model("cifar10_mobilenet_final.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define class labels (Ensure these match your training labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get image from request
            file = request.files['file']

            # Use BytesIO to read the file content
            img_bytes = BytesIO(file.read())

            # Load image from bytes
            img = image.load_img(img_bytes, target_size=(32, 32))

            # Convert image to array
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions)

            # Return result
            return render_template('index.html', prediction={'class': predicted_class, 'confidence': float(confidence)})

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    print("Current Working Directory:", os.getcwd())
    app.run(debug=True)