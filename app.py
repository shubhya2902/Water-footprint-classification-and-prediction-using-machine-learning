from flask import Flask, render_template, flash, redirect, url_for, session, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd


app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            flash('Login successful!', 'success')
            session['user_id'] = user.id
            return redirect(url_for('user_home'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if username or email already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists. Please choose a different one.', 'danger')
            return render_template('register.html')

        if password == confirm_password:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match.', 'danger')

    return render_template('register.html')


@app.route('/logout/')
def logout():
    session.pop('user_id', None)
    flash('Logout successful ... ', 'danger')
    return redirect(url_for('base'))


@app.route('/')
def base():
    return render_template('new_main.html')


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os


model_path = "SoilNet1.h5"
SoilNet = load_model(model_path)

# Allowed image types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Define soil classes and recommended crops
classes = {
    0: "Alluvial Soil: { Rice, Wheat, Sugarcane, Maize, Cotton, Soyabean, Jute }",
    1: "Black Soil: { Virginia, Wheat, Jowar, Millets, Linseed, Castor, Sunflower }",
    2: "Clay Soil: { Rice, Lettuce, Chard, Broccoli, Cabbage, Snap Beans }",
    3: "Red Soil: { Cotton, Wheat, Pulses, Millets, Oilseeds, Potatoes }"
}

# Function to predict soil type
def model_predict(image_path, model):
    image = load_img(image_path, target_size=(150, 150))  # Resize image
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    
    prediction_index = np.argmax(model.predict(image))  # Get prediction index
    soil_info = classes.get(prediction_index, "Unknown Soil Type")
    
    # Extract soil type and crops
    soil_name, crops_info = soil_info.split(":")
    crops_list = crops_info.strip(" {}").split(", ")

    output_pages = ["Alluvial.html", "Black.html", "Clay.html", "Red.html"]
    output_page = output_pages[prediction_index]

    return soil_name.strip(), crops_list, output_page



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/user_uploaded', filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        
        # Predict soil type and crops
        soil_name, crops_list, output_page = model_predict(file_path, SoilNet)

        def get_random_soil_row(soil_type):

            # Load and clean the dataset
            df = pd.read_csv("soil_color_composition.csv")
            df.columns = df.columns.str.strip()  # Clean any column name spacing

            if 'Soil_Type' not in df.columns:
                return "Error: 'Soil_Type' column not found in the dataset."

            # Filter by soil type
            filtered_df = df[df['Soil_Type'].str.lower() == soil_type.lower()]

            if filtered_df.empty:
                return f"No data found for soil type: {soil_type}"

            # Randomly pick one row
            random_row = filtered_df.sample(n=1)

            # Drop the 'Soil_Type' column and convert the rest to a list of values
            numeric_values = random_row.drop(columns=['Soil_Type']).values.flatten().tolist()

            return numeric_values

        predicted_composition = get_random_soil_row(soil_name) #get values from DF


        list = [
            'Nitrogen (%)',
            'Phosphorus (mg/kg)',
            'Potassium (mg/kg)',
            'Organic_Carbon (%)',
            'pH',
            'Moisture (%)']
        
        return render_template('result.html', 
                               output_page=output_page, 
                               pred_output=soil_name, 
                               crops_list=crops_list, 
                               user_image=file_path,
                               zipped_data=zip(list, predicted_composition)
                               )


@app.route('/pre')
def pre():
	return render_template('predict.html')


import pickle

# Load models once
MODEL_PATH_HECTARES = "dataset/DecisionTree_hectares_90_33_accuracy.pkl"
MODEL_PATH_ACRES = "dataset/DecisionTree_acres.pkl"

with open(MODEL_PATH_HECTARES, "rb") as f:
    model_hectares = pickle.load(f)

with open(MODEL_PATH_ACRES, "rb") as f:
    model_acres = pickle.load(f)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Get data from form
            crop_name = int(request.form['crop_name'])  # already value index
            crop_area = float(request.form['crop_area'])
            area_unit = int(request.form['area_unit'])  # 1 = hectares, 0 = acres
            irrigation_method = int(request.form['irrigation_method'])  # value index
            rainfall_mm = float(request.form['rainfall_mm'])
            soil_type = int(request.form['soil_type'])  # value index

            # Form feature array
            features = np.array([[crop_name, crop_area, area_unit, irrigation_method, rainfall_mm, soil_type]])

            print(features)

            # Choose model
            if area_unit == 1:
                model = model_hectares
                print("model_hectares is used")
                unit = "hectares"
            else:
                model = model_acres
                print("model_acres is used")
                unit="acres"

            # Predict
            prediction = model.predict(features)[0]

            print(prediction)
            print(area_unit)
            print(unit)

            return render_template('result_new.html', prediction=prediction, unit=unit)

        except Exception as e:
            return render_template('prediction.html', error=f"Error: {e}")

    return render_template('prediction.html')


@app.route('/user_home')
def user_home():
	return render_template('user_home.html')


@app.route('/contact')
def contact():
	return render_template('contact.html')





if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
