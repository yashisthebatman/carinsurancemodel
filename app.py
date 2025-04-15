import flask
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np # For handling potential type issues

app = Flask(__name__)

# --- Configuration ---
MODEL_FILE = 'car_insurance_claim_rf_pipeline.joblib'
# --- IMPORTANT: Define the exact feature names the model was trained on ---
# --- (excluding ID and OUTCOME) ---
# --- Make sure the order matches if your pipeline relies on it (ColumnTransformer usually handles order) ---
EXPECTED_FEATURES = [
    'AGE', 'GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME',
    'CREDIT_SCORE', 'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN',
    'POSTAL_CODE', 'ANNUAL_MILEAGE', 'VEHICLE_TYPE', 'SPEEDING_VIOLATIONS',
    'DUIS', 'PAST_ACCIDENTS'
]

# --- Load the trained pipeline ---
try:
    print(f"Loading model from {MODEL_FILE}...")
    pipeline = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found.")
    pipeline = None # Set to None if loading fails
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

# --- Routes ---

@app.route('/')
def home():
    """Renders the main HTML page with the input form."""
    if pipeline is None:
         # Optionally render an error page or message if model loading failed
         return "Error: Model could not be loaded. Please check the server logs.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the form."""
    if pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if not request.is_json:
         # Fallback for potential standard form submission (though JS uses JSON)
        form_data = request.form
        if not form_data:
             return jsonify({'error': 'No form data received'}), 400
        print("Received data via form submission (fallback)")
        input_data = {feature: form_data.get(feature) for feature in EXPECTED_FEATURES}

    else:
         # Preferred: Receive data as JSON from fetch request
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'No JSON data received'}), 400
        print("Received data via JSON submission")
        # Ensure we only take expected features from JSON
        input_data = {feature: json_data.get(feature) for feature in EXPECTED_FEATURES}


    print(f"Received input data: {input_data}")

    # --- Data Validation and Conversion ---
    processed_data = {}
    missing_fields = []
    type_errors = {}

    for feature in EXPECTED_FEATURES:
        value = input_data.get(feature)
        if value is None or value == '':
            missing_fields.append(feature)
            continue # Skip conversion if missing

        # Attempt type conversion based on typical feature types
        try:
            if feature in ['CREDIT_SCORE']:
                processed_data[feature] = float(value)
            elif feature in ['VEHICLE_OWNERSHIP', 'MARRIED', 'CHILDREN',
                           'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'DUIS',
                           'PAST_ACCIDENTS', 'POSTAL_CODE']: # Treat POSTAL_CODE as numerical if needed, else keep string
                 # If POSTAL_CODE was one-hot encoded, keep it as string/object
                 if feature == 'POSTAL_CODE' and not isinstance(value, (int, float)):
                     processed_data[feature] = str(value) # Keep as string if needed for OHE
                 else:
                    processed_data[feature] = int(value)
            else: # Assume others are categorical (strings)
                processed_data[feature] = str(value)
        except ValueError:
            type_errors[feature] = f"Invalid value '{value}' - expected number"

    if missing_fields:
        return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
    if type_errors:
        return jsonify({'error': 'Invalid data types', 'details': type_errors}), 400

    # --- Create DataFrame for Prediction ---
    try:
        # Create a DataFrame with the correct column order
        input_df = pd.DataFrame([processed_data], columns=EXPECTED_FEATURES)
        print(f"DataFrame for prediction:\n{input_df.to_string()}")
    except Exception as e:
         print(f"Error creating DataFrame: {e}")
         return jsonify({'error': 'Error processing input data'}), 500


    # --- Make Prediction ---
    try:
        # Predict probability (more informative)
        # predict_proba returns probabilities for [class 0, class 1]
        probabilities = pipeline.predict_proba(input_df)[0]
        claim_probability = probabilities[1] # Probability of outcome being 1 (claim)

        # You could also get the direct prediction:
        # prediction = pipeline.predict(input_df)[0]

        print(f"Prediction probabilities: {probabilities}")
        print(f"Claim probability (Class 1): {claim_probability:.4f}")

        return jsonify({
            'claim_probability': round(claim_probability * 100, 2) # Return as percentage
            # 'prediction_outcome': int(prediction) # Optionally return 0 or 1
            })

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Provide a more generic error to the user
        return jsonify({'error': 'Prediction failed'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network
    # Remove debug=True for production
    app.run(debug=True, host='0.0.0.0', port=5000)
