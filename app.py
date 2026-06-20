import os
import json
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
from config import Config
from models import db, User, Prediction

import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'best_cnn_model.h5'
model = None

def load_trained_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

CLASS_LABELS = ['anemia', 'normal', 'sickle', 'thalas']
CLASS_DESCRIPTIONS = {
    'anemia': {
        'name': 'Anemia',
        'description': 'A condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body\'s tissues.',
        'symptoms': ['Fatigue', 'Weakness', 'Pale skin', 'Shortness of breath', 'Dizziness'],
        'color': '#e74c3c'
    },
    'normal': {
        'name': 'Normal',
        'description': 'The blood cells appear healthy and normal with no signs of any blood disorders.',
        'symptoms': ['No symptoms - Healthy blood cells'],
        'color': '#27ae60'
    },
    'sickle': {
        'name': 'Sickle Cell Disease',
        'description': 'A group of inherited red blood cell disorders where red blood cells become hard and sticky and look like a C-shaped farm tool called a "sickle".',
        'symptoms': ['Anemia', 'Pain episodes', 'Swelling', 'Frequent infections', 'Delayed growth'],
        'color': '#9b59b6'
    },
    'thalas': {
        'name': 'Thalassemia',
        'description': 'An inherited blood disorder that causes your body to have less hemoglobin than normal.',
        'symptoms': ['Fatigue', 'Weakness', 'Pale or yellowish skin', 'Bone deformities', 'Slow growth'],
        'color': '#f39c12'
    }
}



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img