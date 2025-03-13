# Image Classification with CIFAR-10

This project provides an image classification system using deep learning (**TensorFlow/Keras**) with the CIFAR-10 dataset.

It includes implementations in:
- **Flask** (Backend API)
- **Streamlit** (Interactive UI)
- **Jupyter Notebook** (`.ipynb`) for training & testing the model.

## Project Structure
```bash
Deccan_ai_assignment/
│── flask_app/              # Flask-based backend API
│   ├── templates/          # Contains HTML files for UI
│   │   ├── index.html      # Simple web interface
│   ├── app.py              # Flask app for inference
│   ├── requirements.txt    # Dependencies for Flask app
│
│── streamlit_app/          # Streamlit-based frontend
│   ├── app.py              # Streamlit UI for predictions
│   ├── requirements.txt    # Dependencies for Streamlit app
│
│── cifar10_mobilenet_final.h5   # Model saved outside for easy access
│
│── code.ipynb               # Notebook for training & testing
│
│── README.md               # Documentation

```

## Setup & Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo

```

## Install Dependencies:
Each app has different requirements:

## Flask App:
```bash
cd flask_app
pip install -r requirements.txt
```

## Streamlit App:
```bash
cd streamlit_app
pip install -r requirements.txt
```

## Run the Flask App
```bash
python app.py
```
The Flask app will be available at:(http://127.0.0.1:5000/)

## Run the streamlit App
```bash
streamlit run app.py
```
The Streamlit app will be available at:(http://localhost:8501/)

## NOTE: 
Ensure the cifar10_mobilenet_final.h5 (model) path is given in the flask and streamlit apps correctly before running them.

