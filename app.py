from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import sys
import sqlite3

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- DATABASE FUNCTIONS ---
def get_driver_data(license_id):
    """Get driver data from database using license ID"""
    try:
        conn = sqlite3.connect('database/dl_data.db')
        cursor = conn.cursor()
        
        # Query the database
        cursor.execute('''
            SELECT 
                age, gender, driving_experience, education, income,
                vehicle_ownership, vehicle_year, married, children,
                speeding_violations, past_accidents
            FROM driver_licenses 
            WHERE license_id = ?
        ''', (license_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"No driver found with license ID: {license_id}")
            
        # Convert to dictionary
        driver_data = {
            'AGE': result[0],
            'GENDER': result[1],
            'DRIVING_EXPERIENCE': result[2],
            'EDUCATION': result[3],
            'INCOME': result[4],
            'VEHICLE_OWNERSHIP': result[5],
            'VEHICLE_YEAR': result[6],
            'MARRIED': result[7],
            'CHILDREN': result[8],
            'SPEEDING_VIOLATIONS': result[9],
            'PAST_ACCIDENTS': result[10]
        }
        
        return driver_data
        
    except Exception as e:
        print(f"\n‼️ DATABASE ERROR: {str(e)}", file=sys.stderr)
        raise

# --- REQUIRED CUSTOM FUNCTION ---
def bin_values(X):
    """Custom binning function for SPEEDING_VIOLATIONS and PAST_ACCIDENTS"""
    def map_val(val):
        if val == 0:
            return 0
        elif val == 1:
            return 1
        elif 2 <= val <= 5:
            return 2
        else:
            return 3
    return pd.DataFrame(X).applymap(map_val)

# --- MODEL LOADING ---
def load_model():
    """Load model and transformer with custom functions"""
    try:
        model_path = r'systum.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at: {model_path}")
            
        print(f"Loading model from: {model_path}")
        transformer, model = joblib.load(model_path)
        print("✓ Model components loaded:")
        print(f" - Transformer: {type(transformer).__name__}")
        print(f" - Model: {type(model).__name__}")
        return transformer, model
        
    except Exception as e:
        print(f"\n‼️ CRITICAL ERROR: {str(e)}", file=sys.stderr)
        return None, None

transformer, model = load_model()

# --- REST OF THE APPLICATION ---
def preprocess_input(form_data):
    """Convert form data to model-ready format"""
    try:
        return {
            'AGE': form_data['AGE'],
            'GENDER': form_data['GENDER'],
            'DRIVING_EXPERIENCE': form_data['DRIVING_EXPERIENCE'],
            'EDUCATION': form_data['EDUCATION'],
            'INCOME': form_data['INCOME'],
            'VEHICLE_OWNERSHIP': int(form_data['VEHICLE_OWNERSHIP']),
            'VEHICLE_YEAR': form_data['VEHICLE_YEAR'],
            'MARRIED': int(form_data['MARRIED']),
            'CHILDREN': int(form_data['CHILDREN']),
            'SPEEDING_VIOLATIONS': int(form_data['SPEEDING_VIOLATIONS'].split('-')[0]) 
                                if '-' in str(form_data['SPEEDING_VIOLATIONS']) 
                                else int(form_data['SPEEDING_VIOLATIONS']),
            'PAST_ACCIDENTS': int(form_data['PAST_ACCIDENTS'].split('-')[0]) 
                            if '-' in str(form_data['PAST_ACCIDENTS']) 
                            else int(form_data['PAST_ACCIDENTS'])
        }
    except KeyError as e:
        raise ValueError(f"Missing form field: {str(e)}") from e
    except ValueError as e:
        raise ValueError(f"Invalid input value: {str(e)}") from e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not transformer or not model:
        return render_template('result.html', 
                            prediction_text="System Error: Model not initialized")

    try:
        # Check if prediction is based on license ID or manual entry
        submit_method = request.form.get('submit_method', 'manual')
        
        if submit_method == 'license':
            # Get license ID from form
            license_id = request.form.get('license_id')
            if not license_id:
                return render_template('result.html', 
                                     prediction_text="Error: No license ID provided")
            
            # Get driver data from database
            input_data = get_driver_data(license_id)
        else:
            # Use manual form data
            input_data = preprocess_input(request.form)
        
        columns = [
            'AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME',
            'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN',
            'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS'
        ]
        input_df = pd.DataFrame([input_data], columns=columns)
        transformed_data = transformer.transform(input_df)
        prediction = model.predict(transformed_data)
        result = 'Approved' if prediction[0] == 1 else 'Rejected'
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        print(f"\n‼️ PREDICTION ERROR: {str(e)}", file=sys.stderr)
        return render_template('result.html',
                            prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)