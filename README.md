**Image Classification with CIFAR-10**
This project provides an image classification system using deep learning (TensorFlow/Keras) with the CIFAR-10 dataset.

It includes implementations in:
Flask (backend API)
Streamlit (interactive UI)
Both above applications use separate requirements.txt files.
Jupyter Notebook (.ipynb) for training & testing the model

**Project Structure**

The structure of the project is as follows 

├── flask_app/                     # Flask-based backend API
│   ├── templates/                  # Contains HTML files for UI
│   │   ├── index.html               # Simple web interface
│   ├── app.py                       # Flask app for inference
│   ├── requirements.txt              # Dependencies for Flask app
│
├── streamlit_app/                   # Streamlit-based frontend
│   ├── app.py                       # Streamlit UI for predictions
│   ├── requirements.txt              # Dependencies for Streamlit app
│
├── cifar10_mobilenet_final.h5       # Model saved outside for easy access
├── code.ipynb              # Jupyter Notebook for training
├── README.md                        # Read me file


**Setup & Installation**
**Clone the Repository**
git clone https://github.com/yourusername/your-repo.git
cd your-repo
**Install Dependencies**
Each app has different requirements

**Run the flask app**

cd flask_app
pip install -r requirements.txt
The Flask app will be available at: http://127.0.0.1:5000/

**Run the streamlit app**

cd ../streamlit_app
streamlit run app.py

The Streamlit app will be available at: http://localhost:8501/

NOTE: Ensure the cifar10_mobilenet_final.h5 (model) path is given in the flask and streamlit apps correctly 
