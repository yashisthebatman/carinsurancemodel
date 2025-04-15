from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import sys
# Make sure dotenv is imported
from dotenv import load_dotenv, find_dotenv
# Import genai library
import google.generativeai as genai

print("--- Script Start ---")

# --- Load Environment Variables ---
dotenv_path = find_dotenv() # Searches current dir and parents for .env
if dotenv_path:
    print(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    print("WARNING: .env file not found.", file=sys.stderr)

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- REQUIRED CUSTOM FUNCTION for ML Pipeline ---
def bin_values(X):
    """Custom binning function for SPEEDING_VIOLATIONS and PAST_ACCIDENTS"""
    # print("bin_values function called.") # Keep this commented unless debugging bin_values
    def map_val(val):
        if val == 0:
            return 0
        elif val == 1:
            return 1
        elif 2 <= val <= 5:
            return 2
        else: # 6+ or maybe other values
            return 3

    # Use modern .map for Series/DataFrame
    if isinstance(X, pd.DataFrame):
        # Assuming FunctionTransformer passes one column at a time
        if X.shape[1] == 1:
             # Apply map to the single Series, return as Series (usually expected by FT)
            return X.iloc[:, 0].map(map_val)
        else:
            # Fallback if multiple columns are somehow passed (less likely with ColumnTransformer)
            print("Warning: bin_values applying map to multiple columns.", file=sys.stderr)
            return X.map(map_val)
    elif isinstance(X, pd.Series):
         return X.map(map_val)
    else:
         # Handle unexpected input types if necessary
         print(f"Warning: bin_values received unexpected type: {type(X)}. Attempting Series conversion.", file=sys.stderr)
         try:
             # Attempt conversion, return Series
             return pd.Series(X).map(map_val)
         except Exception as map_e:
             print(f"Error applying map in bin_values: {map_e}", file=sys.stderr)
             return X # Return original if map fails


# --- MODEL LOADING ---
def load_model():
    """Load ML model and transformer."""
    print("--- Loading ML Model Components ---")
    try:
        model_path = 'systum.pkl'
        # Check relative path first, then script directory as fallback
        effective_path = model_path
        if not os.path.exists(effective_path):
            print(f"Model not found at relative path: '{effective_path}'. Trying script directory...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            effective_path = os.path.join(script_dir, model_path)
            if not os.path.exists(effective_path):
                 # If you have another absolute path as fallback, add it here
                 # user_abs_path = r'C:\...'
                 # if os.path.exists(user_abs_path): effective_path = user_abs_path else: ...
                 raise FileNotFoundError(f"Model file missing. Checked relative path and script directory: {script_dir}")

        print(f"Attempting to load model from: {effective_path}")
        transformer, model = joblib.load(effective_path)
        print("✓ ML Model components loaded successfully:")
        print(f" - Transformer Type: {type(transformer).__name__}")
        print(f" - Model Type: {type(model).__name__}")
        return transformer, model

    except FileNotFoundError as e:
         print(f"‼️ CRITICAL ERROR: {str(e)}", file=sys.stderr)
         return None, None
    except ModuleNotFoundError as e:
         # Handle cases where custom functions like bin_values might be needed during load
         if 'bin_values' in str(e):
             print("\n‼️ CRITICAL ERROR: 'bin_values' function needed but not found during model loading.", file=sys.stderr)
             print("   Ensure it's defined globally *before* calling load_model().", file=sys.stderr)
         else:
            print(f"\n‼️ CRITICAL ERROR: A required library is missing - {str(e)}", file=sys.stderr)
         return None, None
    except Exception as e:
        print(f"\n‼️ CRITICAL ERROR loading ML model from '{effective_path}': {str(e)} ({type(e).__name__})", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        return None, None

transformer, model = load_model()

# --- GEMINI API SETUP ---
def initialize_gemini():
    """Initializes the Gemini API client"""
    print("--- Initializing Gemini API ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("‼️ WARNING: GEMINI_API_KEY environment variable not found. LLM features disabled.", file=sys.stderr)
        return None
    else:
        print("API Key found. Configuring Gemini...")
        try:
            genai.configure(api_key=api_key)

            # *** Set the target model name explicitly ***
            target_model_name = 'models/gemini-1.5-flash-latest'
            print(f"Attempting to initialize Gemini Model ('{target_model_name}')...")

            # Initialize the GenerativeModel
            llm_model = genai.GenerativeModel(target_model_name)
            print(f"✓ Gemini Model '{target_model_name}' initialized successfully.")
            return llm_model

        except Exception as e:
            print(f"\n‼️ CRITICAL ERROR initializing Gemini SDK or Model: {str(e)} ({type(e).__name__})", file=sys.stderr)
            return None

gemini_model = initialize_gemini() # Attempt to initialize the model

# --- DATA PREPROCESSING ---
def preprocess_input(form_data_dict):
    """Converts form data dict for ML, preserving originals for LLM."""
    # print(f"Preprocessing raw data: {form_data_dict}") # Uncomment for debug
    processed_data = {}
    try:
        # Extract using .get() with defaults for safety
        processed_data['AGE'] = form_data_dict.get('AGE')
        processed_data['GENDER'] = form_data_dict.get('GENDER')
        processed_data['DRIVING_EXPERIENCE'] = form_data_dict.get('DRIVING_EXPERIENCE')
        processed_data['EDUCATION'] = form_data_dict.get('EDUCATION')
        processed_data['INCOME'] = form_data_dict.get('INCOME')
        processed_data['VEHICLE_OWNERSHIP'] = int(form_data_dict.get('VEHICLE_OWNERSHIP', 0))
        processed_data['VEHICLE_YEAR'] = form_data_dict.get('VEHICLE_YEAR')
        processed_data['MARRIED'] = int(form_data_dict.get('MARRIED', 0))
        processed_data['CHILDREN'] = int(form_data_dict.get('CHILDREN', 0))
        processed_data['SPEEDING_VIOLATIONS'] = form_data_dict.get('SPEEDING_VIOLATIONS') # Keep original string
        processed_data['PAST_ACCIDENTS'] = form_data_dict.get('PAST_ACCIDENTS')       # Keep original string
        return processed_data

    except (ValueError, TypeError) as e:
        print(f"ERROR during input conversion: {e}", file=sys.stderr)
        raise ValueError(f"Invalid input value detected: {str(e)}") from e

# --- LLM ADVICE GENERATION ---
def get_llm_advice(user_data_dict, prediction_result):
    """Generates advice using the Gemini API."""
    # print("--- Inside get_llm_advice ---") # Uncomment for debug
    if not gemini_model:
        print("LLM advice skipped: Gemini model not initialized.")
        return "LLM Advisor is unavailable due to an issue during startup."

    print(f"Generating LLM advice for prediction: {prediction_result}")
    # Create input summary for the LLM prompt
    input_summary = f"""
    User Profile Summary:
    - Age Group: {user_data_dict.get('AGE', 'N/A')}
    - Gender: {user_data_dict.get('GENDER', 'N/A')}
    - Driving Experience: {user_data_dict.get('DRIVING_EXPERIENCE', 'N/A')}
    - Education: {user_data_dict.get('EDUCATION', 'N/A')}
    - Income Level: {user_data_dict.get('INCOME', 'N/A')}
    - Owns Vehicle: {'Yes' if str(user_data_dict.get('VEHICLE_OWNERSHIP', '0')) == '1' else 'No'}
    - Vehicle Year: {user_data_dict.get('VEHICLE_YEAR', 'N/A')}
    - Marital Status: {'Married' if str(user_data_dict.get('MARRIED', '0')) == '1' else 'Single/Other'}
    - Has Children: {'Yes' if str(user_data_dict.get('CHILDREN', '0')) == '1' else 'No'}
    - Speeding Violations (Reported Range): {user_data_dict.get('SPEEDING_VIOLATIONS', 'N/A')}
    - Past Accidents (Reported Range): {user_data_dict.get('PAST_ACCIDENTS', 'N/A')}
    """

    # Construct the prompt
    prompt = f"""
    You are an expert car insurance advisor providing helpful feedback.
    A user submitted the information summarized below for a car insurance claim prediction:
    {input_summary}

    Our prediction model indicates the likely outcome for this profile is: **{prediction_result}**.

    Based *only* on the user profile information provided, give concise (2-4 sentences), constructive, and personalized advice on factors the user could potentially influence to improve their insurance profile or maintain a good one for future claims or renewals. Focus on actionable advice related to driving behavior (violations, accidents) or factors that change over time (like experience). Do not invent reasons or mention factors not listed in the profile. Be encouraging. Start the advice directly, without preamble like "Here is some advice:".

    Example for Rejected: "Focusing on maintaining a clean driving record with fewer speeding violations and accidents is key. Over time, gaining more driving experience can also positively impact future assessments."
    Example for Approved: "Maintaining your excellent driving record with minimal speeding violations or accidents is crucial. Continued safe driving habits are the best way to keep your insurance profile positive."
    """

    try:
        print(f"Attempting to call Gemini API model '{gemini_model.model_name}'...")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)

        # Process the response robustly
        advice = None
        if hasattr(response, 'text') and response.text:
             advice = response.text
        elif hasattr(response, 'parts') and response.parts:
             advice = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        if advice and advice.strip():
            print("✓ Gemini API call successful.")
            return advice.strip()
        else:
            # Handle blocked or empty responses
            blocking_reason = "Unknown"
            if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                 blocking_reason = response.prompt_feedback.block_reason
            print(f"‼️ WARNING: Gemini response was empty or blocked. Reason: {blocking_reason}", file=sys.stderr)
            return f"Could not generate specific advice (API response empty/blocked - Reason: {blocking_reason})."

    except Exception as e:
        print(f"\n‼️ ERROR calling Gemini API ({type(e).__name__}): {str(e)}", file=sys.stderr)
        # Consider logging the full traceback for persistent errors
        # import traceback; traceback.print_exc()
        return f"Could not retrieve advice due to a technical issue during the API call ({type(e).__name__})."


# --- Flask Routes ---
@app.route('/')
def home():
    """Serves the main input form page."""
    print("Serving index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, ML prediction, LLM advice, and renders result."""
    print("--- Received POST request on /predict ---")
    advice_text = "LLM advice generation skipped or failed." # Default
    ml_prediction_result = "Error" # Default

    # --- ML Model Availability Check ---
    if not transformer or not model:
        print("ERROR: ML model components not loaded.", file=sys.stderr)
        return render_template('result.html',
                            prediction_text="System Error: ML Model Not Initialized",
                            advice_text="System Error: ML Model Not Initialized.")

    try:
        # --- 1. Get and Preprocess Form Data ---
        form_data_dict = request.form.to_dict()
        print(f"Received form data: {form_data_dict}")
        initial_processed_data = preprocess_input(form_data_dict)

        # --- 2. Prepare Data Specifically for ML DataFrame ---
        ml_data_for_df = initial_processed_data.copy()
        try:
            # Convert Speeding Violations string to number for ML
            speeding_str = ml_data_for_df.get('SPEEDING_VIOLATIONS', '0')
            ml_data_for_df['SPEEDING_VIOLATIONS'] = int(speeding_str.split('-')[0]) if '-' in speeding_str else int(speeding_str.replace('+', ''))
            # Convert Past Accidents string to number for ML
            accidents_str = ml_data_for_df.get('PAST_ACCIDENTS', '0')
            ml_data_for_df['PAST_ACCIDENTS'] = int(accidents_str.split('-')[0]) if '-' in accidents_str else int(accidents_str.replace('+', ''))
            # print(f"Data prepared for ML DataFrame: {ml_data_for_df}") # Uncomment for debug
        except (ValueError, TypeError) as e:
             raise ValueError(f"Invalid format for Violations ('{speeding_str}') or Accidents ('{accidents_str}').") from e

        # --- 3. Create DataFrame and Predict ---
        # Ensure columns match training order
        columns_for_ml = [
            'AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME',
            'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN',
            'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS'
        ]
        input_df = pd.DataFrame([ml_data_for_df], columns=columns_for_ml)
        # print(f"ML Input DataFrame:\n{input_df.to_string()}") # Uncomment for debug

        print("Transforming data and predicting with ML model...")
        transformed_data = transformer.transform(input_df)
        prediction = model.predict(transformed_data)
        ml_prediction_result = 'Approved' if int(prediction[0]) == 1 else 'Rejected'
        print(f"✓ ML Prediction: {ml_prediction_result}")

        # --- 4. Get LLM Advice ---
        if gemini_model:
            # Pass the initially processed data with original strings to LLM function
            advice_text = get_llm_advice(initial_processed_data, ml_prediction_result)
        else:
            advice_text = "LLM Advisor is unavailable (initialization failed)."

        # --- 5. Render Result ---
        print("Rendering result page...")
        return render_template('result.html',
                               prediction_text=ml_prediction_result,
                               advice_text=advice_text)

    # --- Error Handling for the /predict Route ---
    except ValueError as e: # Catches input conversion errors
        print(f"ERROR in /predict (Input Validation): {str(e)}", file=sys.stderr)
        return render_template('result.html',
                            prediction_text="Error",
                            advice_text=f"Input Error: {str(e)}. Please check your entries.")
    except Exception as e: # Catches ML errors or other unexpected issues
        print(f"ERROR in /predict (Processing): {str(e)} ({type(e).__name__})", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return render_template('result.html',
                            prediction_text="Prediction Failed",
                            advice_text=f"An unexpected error occurred ({type(e).__name__}). Please try again later.")


# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Starting Flask Application ---")
    # host='0.0.0.0' makes the server accessible on your network
    # debug=True enables auto-reloading and debugger (disable in production)
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("--- Flask App Has Stopped ---")
