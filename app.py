from flask import Flask, request, jsonify
from datetime import datetime
import time
from flask_cors import CORS
import logging
import pandas as pd
from joblib import load
import sklearn
import numpy as np
from typing import Dict, Any, Union
import os
from dotenv import load_dotenv  # Commented out - not needed without Gemini
import google.generativeai as genai  # Commented out - not needed without Gemini

load_dotenv()  # Commented out - not needed without Gemini

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Using scikit-learn version: {sklearn.__version__}")

# Gemini model - COMMENTED OUT - not using AI recommendations
# gemini_model = None
# try:
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         logger.warning("GEMINI_API_KEY environment variable not set. AI recommendations will be disabled, falling back to static lists.")
#     else:
#         genai.configure(api_key=api_key)
#         gemini_model = genai.GenerativeModel("gemini-2.0-flash")
#         logger.info("Gemini AI model 'gemini-2.0-flash' initialized successfully.")
# except Exception as e:
#     logger.error(f"Failed to initialize Gemini AI model: {e}")
#     gemini_model = None


# Gemini model setup
gemini_model = None

try:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Falling back to static recommendations.")
    else:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("✅ Gemini 2.5 Flash initialized successfully")

except Exception as e:
    logger.error(f"❌ Gemini initialization failed: {e}")
    gemini_model = None

# Model and metadata initialization
model = None
selector = None
scaler = None
selected_features = []
all_features = []
model_performance = {}

# Define feature columns (from your training notebook)
FEATURE_COLUMNS = [
    'AGE',
    'SEX_ENCODED',
    'BODY MASS INDEX',
    'YEARS ON ART',
    'BP HISTORY',
    'EXERCISES',
    'BMI_CAT_ENCODED',
    'AGE_GROUP_ENCODED',
    'TENOFOVIR',
    'LAMIVUDINE',
    'DOLUTEGRAVIR',
    'DARUNAVIR',
    'ZIDOVUDINE',
    'ABACAVIR'
]

# Static recommendations based on risk level (no AI)
RECOMMENDATIONS = {
    'High': [
        "Initiate or intensify antihypertensive therapy",
        "Monitor blood pressure weekly",
        "Consider cardiology referral",
        "Review ART regimen for potential interactions",
        "Strict dietary sodium restriction",
        "Regular cardiovascular assessment"
    ],
    'Moderate': [
        "Lifestyle modifications recommended",
        "Monitor blood pressure monthly",
        "Consider starting low-dose antihypertensive if other risk factors present",
        "Encourage regular physical activity",
        "Dietary counseling for weight management"
    ],
    'Low': [
        "Continue current management",
        "Monitor blood pressure every 3-6 months",
        "Reinforce healthy lifestyle habits",
        "Maintain regular ART adherence",
        "Annual cardiovascular check-up"
    ]
}

def load_model_artifacts(filepath):
    """Load the complete model artifacts saved during training"""
    try:
        artifacts = load(filepath)
        
        # Extract all components
        # model_obj = artifacts.get('model')
        model_obj = artifacts.get('binary_model')
        selector_obj = artifacts.get('selector')
        scaler_obj = artifacts.get('scaler')
        selected_feats = artifacts.get('selected_features')
        all_feats = artifacts.get('all_features', FEATURE_COLUMNS)
        performance = artifacts.get('model_performance', {})
        
        logger.info(f"Model loaded successfully. Best model: {performance.get('best_model', 'Unknown')}")
        logger.info(f"Model performance - AUC: {performance.get('test_auc', 'N/A')}, Accuracy: {performance.get('test_accuracy', 'N/A')}")
        
        return model_obj, selector_obj, scaler_obj, selected_feats, all_feats, performance
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        return None, None, None, None, FEATURE_COLUMNS, {}

# Load the specific model file
# MODEL_FILENAME = 'hypertension_model_20260308_181557 (1).pkl'
# Load the specific model file
# MODEL_FILENAME = 'hypertension_model_enhanced_current (2).pkl'
# Load the specific model file
MODEL_FILENAME = 'hypertension_model_enhanced_20260619_033515.pkl'

# Try different locations to find the model
model_paths = [
    f'model/{MODEL_FILENAME}',
    f'models/{MODEL_FILENAME}',
    MODEL_FILENAME,
    f'./{MODEL_FILENAME}'
]

loaded = False
for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            logger.info(f"Attempting to load model from: {model_path}")
            model, selector, scaler, selected_features, all_features, model_performance = load_model_artifacts(model_path)
            if model is not None:
                logger.info(f"✅ Successfully loaded model from {model_path}")
                loaded = True
                break
    except Exception as e:
        logger.warning(f"Failed to load from {model_path}: {str(e)}")
        continue

if not loaded:
    logger.error(f"❌ CRITICAL: Could not load model from any path. API cannot start without model.")
    # Exit or raise exception - we don't want to run without the model
    raise RuntimeError(f"Model file {MODEL_FILENAME} not found. Please ensure the model file is in the correct location.")

logger.info(f"✅ Model loaded successfully. Ready to accept requests.")

def validate_input(data: Dict[str, Any]) -> Union[None, Dict[str, str]]:
    """Validate input data for ART patient hypertension prediction"""
    # Use the loaded selected features if available, otherwise use all features
    features_to_validate = selected_features if selected_features is not None else all_features
    required_fields = set(features_to_validate)
    missing_fields = required_fields - set(data.keys())
    
    if missing_fields:
        return {'error': f'Missing required fields: {", ".join(missing_fields)}'}

    type_errors = []
    
    # Numeric fields validation
    numeric_fields = ['AGE', 'BODY MASS INDEX', 'YEARS ON ART']
    binary_fields = ['SEX_ENCODED', 'BP HISTORY', 'EXERCISES', 
                     'TENOFOVIR', 'LAMIVUDINE', 'DOLUTEGRAVIR', 
                     'DARUNAVIR', 'ZIDOVUDINE', 'ABACAVIR']
    categorical_fields = ['BMI_CAT_ENCODED', 'AGE_GROUP_ENCODED']
    
    # Validate numeric fields
    for field in numeric_fields:
        if field in data:
            try:
                val = float(data[field])
                # if field == 'AGE' and (val < 18 or val > 100):
                #     type_errors.append(f"{field} must be between 18 and 100 years")
                if field == 'AGE' and (val < 0 or val > 120):
                    type_errors.append(f"{field} must be between 0 and 120 years")
                # if field == 'BODY MASS INDEX' and (val < 10 or val > 60):
                #     type_errors.append(f"{field} must be between 10 and 60")
                if field == 'BODY MASS INDEX' and (val < 7 or val > 200):
                    type_errors.append(f"{field} must be between 7 and 200")
                if field == 'YEARS ON ART' and (val < 0 or val > 120):
                    type_errors.append(f"{field} must be between 0 and 120 years")
            except (ValueError, TypeError):
                type_errors.append(f"{field} must be a number")
    
    # Validate binary fields (should be 0 or 1)
    for field in binary_fields:
        if field in data:
            try:
                val = int(data[field])
                if val not in [0, 1]:
                    type_errors.append(f"{field} must be 0 or 1")
            except (ValueError, TypeError):
                type_errors.append(f"{field} must be 0 or 1")
    
    # Validate categorical fields
    for field in categorical_fields:
        if field in data:
            try:
                val = int(data[field])
                if field == 'BMI_CAT_ENCODED' and val not in [0, 1, 2, 3]:
                    type_errors.append(f"{field} must be 0 (Underweight), 1 (Normal), 2 (Overweight), or 3 (Obese)")
                if field == 'AGE_GROUP_ENCODED' and val not in [0, 1, 2, 3, 4]:
                    type_errors.append(f"{field} must be 0 (<30), 1 (30-40), 2 (40-50), 3 (50-60), or 4 (60+)")
            except (ValueError, TypeError):
                type_errors.append(f"{field} must be an integer")

    if type_errors:
        return {'error': " | ".join(type_errors)}

    return None

def prepare_input_for_model(api_data: Dict[str, Any]):
    """Convert API input to model-ready format"""
    if model is None:
        return None
    
    # Create DataFrame with all features
    input_dict = {}
    for feature in all_features:
        input_dict[feature] = [api_data.get(feature, 0)]
    
    input_df = pd.DataFrame(input_dict)
    
    # Apply scaling if scaler exists
    if scaler is not None:
        input_scaled = scaler.transform(input_df[all_features])
        input_df_scaled = pd.DataFrame(input_scaled, columns=all_features)
    else:
        input_df_scaled = input_df
    
    # Apply feature selection if selector exists
    if selector is not None:
        input_selected = selector.transform(input_df_scaled[all_features])
        return input_selected
    else:
        # If no selector, use selected_features list to filter
        if selected_features is not None:
            return input_df_scaled[selected_features].values
        else:
            return input_df_scaled[all_features].values

def calculate_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability >= 0.6:
        return "High"
    elif probability >= 0.3:
        return "Moderate"
    else:
        return "Low"

# REMOVED Gemini AI function - using static recommendations only
# def get_recommendations(risk_level: str) -> list:
#     """Return static recommendations based on risk level"""
#     return RECOMMENDATIONS[risk_level]

def get_recommendations(risk_level: str, patient_data: dict) -> list:
    """
    Generate recommendations using Gemini AI with fallback to static
    """

    # Fallback if Gemini not available
    if gemini_model is None:
        return RECOMMENDATIONS[risk_level]

    try:
        prompt = f"""
        You are a clinical assistant.

        A patient has {risk_level} risk of hypertension.

        Key patient data:
        - Age: {patient_data.get('AGE')}
        - BMI: {patient_data.get('BODY MASS INDEX')}
        - Years on ART: {patient_data.get('YEARS ON ART')}
        - BP History: {patient_data.get('BP HISTORY')}
        - Exercises: {patient_data.get('EXERCISES')}

        Provide ONLY 4-6 recommendations.

        Rules:
        - No bullet points
        - No stars
        - No numbering
        - No introduction or explanation
        - Each recommendation must be on a new line
        """

        response = gemini_model.generate_content(prompt)
        # response = gemini_model.generate_content(prompt, request_options={"timeout": 5})

        # Clean response into list
        # text = response.text.strip()
        # recommendations = [line.strip("-• ") for line in text.split("\n") if line.strip()]
        text = response.text.strip()

        recommendations = []
        for line in text.split("\n"):
            line = line.strip("-• ").strip()
            if line:
                recommendations.append(line)

        # Limit to 6 max
        recommendations = recommendations[:6]

        return recommendations if recommendations else RECOMMENDATIONS[risk_level]

    except Exception as e:
        logger.error(f"Gemini failed: {e}")
        return RECOMMENDATIONS[risk_level]

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "service": "Hypertension Risk Prediction API for ART Patients",
        "model_loaded": model is not None,
        "model_file": MODEL_FILENAME if model is not None else None,
        "selector_loaded": selector is not None,
        "scaler_loaded": scaler is not None,
        "features_count": len(selected_features) if selected_features is not None else 0,
        "model_performance": model_performance if model_performance else {},
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def assess_risk():
    """Endpoint for assessing hypertension risk in ART patients - using only trained model"""
    start_time = time.time()

    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received.")
            return jsonify({'error': 'No JSON data received'}), 400

        logger.info(f"Received request data: {data}")

        validation_error = validate_input(data)
        if validation_error:
            logger.error(f"Validation error: {validation_error['error']}")
            return jsonify(validation_error), 400

        # Check if model is loaded (should be, but double-check)
        if model is None:
            logger.error("Model not loaded - this should not happen")
            return jsonify({
                'error': True,
                'message': 'Model not available. Please contact administrator.',
                'status': 'model_unavailable'
            }), 503

        # Prepare input for model
        input_processed = prepare_input_for_model(data)
        
        if input_processed is None:
            return jsonify({'error': 'Failed to process input data'}), 500

        # Make prediction using ONLY the trained model
        try:
            predicted_class = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0][1]
            
            risk_level = calculate_risk_level(prediction_proba)
            
            logger.info(f"Model predicted: Class = {predicted_class}, Probability = {prediction_proba:.4f}, Risk Level = {risk_level}")
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}", exc_info=True)
            return jsonify({
                'error': True,
                'message': 'Model prediction failed',
                'details': str(e)
            }), 500

        # Get static recommendations (no AI)
        # recommendations = get_recommendations(risk_level)
        recommendations = get_recommendations(risk_level, data)

        response = {
            "patientId": data.get('patientId', f"patient-{int(time.time() * 1000)}"),
            "patientName": data.get('name', 'N/A'),
            "riskLevel": risk_level,
            "probability": round(prediction_proba, 4),
            "riskScore": f"{prediction_proba:.1%}",
            "prediction": "Hypertension" if predicted_class == 1 else "No Hypertension",
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "inputFeatures": {k: v for k, v in data.items() if k in (selected_features or [])},
            "modelUsed": type(model).__name__,
            "processingTimeMs": int((time.time() - start_time) * 1000)
        }

        logger.info(f"Final assessment response sent")
        return jsonify(response)

    except Exception as e:
        logger.error(f"An unexpected error occurred during risk assessment: {str(e)}", exc_info=True)
        return jsonify({
            'error': True,
            'message': 'An internal server error occurred',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'processingTimeMs': int((time.time() - start_time) * 1000)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_file': MODEL_FILENAME if model is not None else None,
        # 'gemini_model_loaded': gemini_model is not None,  # Commented out
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'unavailable',
            'model_file': MODEL_FILENAME
        }), 404
    
    return jsonify({
        'model_type': type(model).__name__,
        'model_file': MODEL_FILENAME,
        'features_used': len(selected_features) if selected_features else 0,
        'selected_features': selected_features if selected_features else [],
        'all_features': all_features,
        'performance': model_performance if model_performance else {},
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

if __name__ == '__main__':
    logger.info(f"Starting Hypertension Risk Prediction API - MODEL ONLY mode")
    logger.info(f"Looking for model file: {MODEL_FILENAME}")
    logger.info("Gemini AI is ENABLED with dynamic recommendations")
    # logger.info("Gemini AI is DISABLED - using static recommendations only")
    
    # The app will only start if model loads successfully
    # If model fails to load, it will raise an exception
    
    app.run(host='0.0.0.0', port=5001, debug=True)
