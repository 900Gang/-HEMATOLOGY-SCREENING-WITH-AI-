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