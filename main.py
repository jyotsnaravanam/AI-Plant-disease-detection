from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'


import tensorflow as tf

# Load the entire model (architecture + weights) directly from the .h5 file
loaded_model = tf.keras.models.load_model("plant_disease_model.h5")
users = {}


suggestions = {
    "Apple Apple_scab": (
        "🍎 Apple Scab detected.\n"
        "• Disease may spread rapidly in cool, wet weather.\n"
        "• Apply recommended fungicides early.\n"
        "• Remove infected leaves to reduce future outbreaks."
    ),
    "Apple Black_rot": (
        "🍎 Apple Black Rot detected.\n"
        "• High risk of fruit decay during storage.\n"
        "• Prune infected branches immediately.\n"
        "• Use proper orchard sanitation."
    ),
    "Apple healthy": (
        "✅ Apple crop is healthy.\n"
        "• Maintain regular monitoring.\n"
        "• Follow balanced fertilization and irrigation."
    ),
    "Cherry healthy": (
        "✅ Cherry crop is healthy.\n"
        "• Low disease risk forecast.\n"
        "• Continue routine crop care practices."
    ),
    "Cherry Powdery_mildew": (
        "🍒 Powdery Mildew detected.\n"
        "• Disease may increase in dry, warm conditions.\n"
        "• Improve air circulation.\n"
        "• Apply sulfur-based fungicides if required."
    ),
    "Corn healthy": (
        "🌽 Corn crop is healthy.\n"
        "• Favorable growth conditions detected.\n"
        "• Maintain nutrient and water management."
    ),
    "Corn Northern_Leaf_Blight": (
        "🌽 Northern Leaf Blight detected.\n"
        "• Disease may spread in humid conditions.\n"
        "• Use resistant hybrids.\n"
        "• Apply fungicide if severity increases."
    ),
    "Potato Early blight": (
        "🥔 Early Blight detected.\n"
        "• Risk of yield reduction if untreated.\n"
        "• Apply preventive fungicides.\n"
        "• Avoid overhead irrigation."
    ),
    "Potato healthy": (
        "✅ Potato crop is healthy.\n"
        "• Continue good crop rotation practices.\n"
        "• Monitor fields weekly."
    ),
    "Potato Late_blight": (
        "🚨 Late Blight detected.\n"
        "• Highly destructive disease forecast.\n"
        "• Immediate fungicide application required.\n"
        "• Remove infected plants to prevent spread."
    ),
    "Rice brown spot": (
        "🌾 Rice Brown Spot detected.\n"
        "• Disease may worsen with nutrient deficiency.\n"
        "• Improve soil fertility.\n"
        "• Apply suitable fungicide."
    ),
    "Ricecrop healthy": (
        "✅ Rice crop is healthy.\n"
        "• Low disease probability.\n"
        "• Maintain proper water management."
    ),
    "Tomato Early_blight": (
        "🍅 Tomato Early Blight detected.\n"
        "• Disease may spread rapidly in warm, humid climate.\n"
        "• Remove infected leaves.\n"
        "• Apply fungicide at early stage."
    ),
    "Tomato healthy": (
        "✅ Tomato crop is healthy.\n"
        "• Favorable growth forecast.\n"
        "• Continue balanced nutrition and irrigation."
    )
}

@app.route('/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return "User already exists. Please sign in."
        users[username] = password
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return "Invalid credentials. Please try again."
    return render_template('signin.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('signin'))
    return render_template('home.html')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if 'username' not in session:
        return redirect(url_for('signin'))

    predicted_class = None
    suggestion_text = None
    image_path = None

    if request.method == 'POST':
        image_file = request.files['image']
        image_path = os.path.join('static', image_file.filename)
        image_file.save(image_path)

        test_image = image.load_img(image_path, target_size=(150, 150))
        test_image_array = image.img_to_array(test_image) / 255.0
        test_image_array = np.expand_dims(test_image_array, axis=0)

        predictions = loaded_model.predict(test_image_array)
        pred_label = np.argmax(predictions, axis=1)

        class_names = [
            'Apple Apple_scab', 'Apple Black_rot', 'Apple healthy',
            'Cherry healthy', 'Cherry Powdery_mildew',
            'Corn healthy', 'Corn Northern_Leaf_Blight',
            'Potato Early blight', 'Potato healthy', 'Potato Late_blight',
            'Rice brown spot', 'Ricecrop healthy',
            'Tomato Early_blight', 'Tomato healthy'
        ]

        predicted_class = class_names[pred_label[0]]
        suggestion_text = suggestions.get(predicted_class, "No suggestion available.")

    return render_template(
        'detection.html',
        predicted_class=predicted_class,
        suggestion_text=suggestion_text,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run()