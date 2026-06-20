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

def predict_image(image_path):
    if model is None:
        return None, None, None
    
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100
    predicted_class = CLASS_LABELS[predicted_class_idx]
    
    all_probs = {CLASS_LABELS[i]: float(predictions[0][i]) * 100 for i in range(len(CLASS_LABELS))}
    
    return predicted_class, confidence, all_probs

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('register.html')
        
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

###
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.created_at.desc()).limit(5).all()
    
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    
    return render_template('dashboard.html', 
                         recent_predictions=recent_predictions,
                         total_predictions=total_predictions,
                         class_info=CLASS_DESCRIPTIONS)

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{current_user.id}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            predicted_class, confidence, all_probs = predict_image(filepath)
            
            if predicted_class is None:
                flash('Error making prediction. Please try again.', 'danger')
                return redirect(request.url)
            
            prediction = Prediction(
                user_id=current_user.id,
                image_path=filename,
                prediction=predicted_class,
                confidence=confidence,
                all_probabilities=json.dumps(all_probs)
            )
            db.session.add(prediction)
            db.session.commit()
            
            return redirect(url_for('results', prediction_id=prediction.id))
        else:
            flash('Invalid file type. Please upload an image file.', 'danger')
    
    return render_template('detect.html')


@app.route('/results/<int:prediction_id>')
@login_required
def results(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    if prediction.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    all_probs = json.loads(prediction.all_probabilities) if prediction.all_probabilities else {}
    class_info = CLASS_DESCRIPTIONS.get(prediction.prediction, {})
    
    return render_template('results.html', 
                         prediction=prediction, 
                         all_probs=all_probs,
                         class_info=class_info,
                         class_descriptions=CLASS_DESCRIPTIONS)

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.created_at.desc())\
        .paginate(page=page, per_page=10)
    
    return render_template('history.html', 
                         predictions=predictions,
                         class_descriptions=CLASS_DESCRIPTIONS)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name')
        current_user.last_name = request.form.get('last_name')
        current_user.email = request.form.get('email')
        
        new_password = request.form.get('new_password')
        if new_password:
            current_password = request.form.get('current_password')
            if current_user.check_password(current_password):
                current_user.set_password(new_password)
            else:
                flash('Current password is incorrect', 'danger')
                return redirect(url_for('profile'))
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
    
    return render_template('profile.html')





