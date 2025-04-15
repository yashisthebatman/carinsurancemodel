from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import sys

app = Flask(__name__, static_folder='static', template_folder='templates')

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
        model_path = r'C:\Users\yvcha\Desktop\Car Insurance\systum.pkl'
        
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
                                if '-' in form_data['SPEEDING_VIOLATIONS'] 
                                else int(form_data['SPEEDING_VIOLATIONS']),
            'PAST_ACCIDENTS': int(form_data['PAST_ACCIDENTS'].split('-')[0]) 
                            if '-' in form_data['PAST_ACCIDENTS'] 
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