from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('trained_model/trained_global_model.h5')

from flask import Flask, render_template, request, redirect, url_for

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # You can process this data or store it if you want
        return render_template('signup_success.html')
    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Here you can validate login — for now we just show success page
        return render_template('login_success.html')
    return render_template('login.html')



@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/')
def main_home():
    return render_template('index.html')

@app.route('/home')
def home_page():
    return render_template('home.html')




@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(64, 64), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.
    img_array = img_array.reshape(1, 64, 64, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = round(float(np.max(prediction)) * 100, 2)

    result = 'Tumor Detected ❌' if predicted_class == 1 else 'No Tumor ✅'
    result_class = 'yes' if predicted_class == 1 else 'no'

    return render_template('result.html', result=result, confidence=confidence, image_filename=file.filename)

@app.route('/faq')
def faq():
    return render_template('faq.html')


if __name__ == '__main__':
    app.run(debug=True)
