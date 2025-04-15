from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib


app = Flask(__name__)

# Load your trained model
model = joblib.load('model/systum.pkl')

# Encoding maps
gender_map = {'male': 0, 'female': 1}
experience_map = {'0-9y': 0, '10-19y': 1, '20-29y': 2, '30+ y': 3}
education_map = {'none': 0, 'high school': 1, 'university': 2}
income_map = {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3}
vehicle_ownership_map = {'0': 0, '1': 1}
vehicle_year_map = {'before 2015': 0, 'after 2015': 1}
married_map = {'0': 0, '1': 1}
children_map = {'0': 0, '1': 1}

age_map = {'16-25': 0, '26-39': 1, '40-64': 2, '65+': 3}

# Binning function
def bin_violations_or_accidents(value):
    value = int(value)
    if value == 0:
        return 0
    elif value == 1:
        return 1
    elif 2 <= value <= 5:
        return 2
    else:
        return 3

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        age = age_map[request.form['AGE']]
        gender = gender_map[request.form['GENDER']]
        experience = experience_map[request.form['DRIVING_EXPERIENCE']]
        education = education_map[request.form['EDUCATION']]
        income = income_map[request.form['INCOME']]
        vehicle = vehicle_ownership_map[request.form['VEHICLE_OWNERSHIP']]
        year = vehicle_year_map[request.form['VEHICLE_YEAR']]
        married = married_map[request.form['MARRIED']]
        children = children_map[request.form['CHILDREN']]
        duis = bin_violations_or_accidents(request.form['DUIS'])
        license_status = int(request.form['LICENSE'])  # assuming it's 0 or 1

        
        speeding = bin_violations_or_accidents(request.form['SPEEDING_VIOLATIONS'])
        accidents = bin_violations_or_accidents(request.form['PAST_ACCIDENTS'])

        # Create feature vector
        features = np.array([[age, gender, experience, education, income, vehicle,
                              year, married, children, speeding, accidents, duis, license_status]])

        prediction = model.predict(features)
        result = 'Accepted' if prediction[0] == 1 else 'Rejected'

        return render_template('index.html', prediction_text=f'Claim is likely: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
