# app.py
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# --- Configuration ---
MODELS_DIR = 'models'
SCALER_FILENAME = 'scaler.pkl'

# These should match the display names used in train_models.py
# and the filenames created by it.
MODEL_CONFIG = {
    'Random Forest Regressor': 'random_forest_regressor.pkl',
    'Decision Tree Regressor': 'decision_tree_regressor.pkl',
    'Gradient Boosting Regressor': 'gradient_boosting_regressor.pkl',
    'Hist Gradient Boosting Regressor': 'hist_gradient_boosting_regressor.pkl',
    'Linear Regression': 'linear_regression.pkl',
    'Elastic Net': 'elastic_net.pkl',
    'Ridge Regression': 'ridge_regression.pkl',
    'Lasso Regression': 'lasso_regression.pkl',
    'Huber Regressor': 'huber_regressor.pkl',
    'Passive Aggressive Regressor': 'passive_aggressive_regressor.pkl',
    'Support Vector Regressor': 'support_vector_regressor.pkl',
    'K-Nearest Neighbors Regressor': 'k-nearest_neighbors_regressor.pkl',
    'ARD Regression': 'ard_regression.pkl',
    'Bayesian Ridge': 'bayesian_ridge.pkl',
    'SGD Regressor': 'sgd_regressor.pkl'
}

SCALED_MODELS_SET = {
    'Support Vector Regressor',
    'K-Nearest Neighbors Regressor',
    'SGD Regressor'
}

# --- Load Models and Scaler ---
models_cache = {}
scaler = None

def load_resources():
    global scaler
    print("--- Loading Resources ---")
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME)
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Scaler '{SCALER_FILENAME}' loaded successfully.")
        except Exception as e:
            print(f"ERROR loading scaler '{scaler_path}': {e}")
    else:
        print(f"ERROR: Scaler file '{scaler_path}' not found. Scaled predictions will fail.")

    for display_name, filename in MODEL_CONFIG.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            try:
                models_cache[display_name] = joblib.load(model_path)
                print(f"Model '{display_name}' ({filename}) loaded.")
            except Exception as e:
                print(f"ERROR loading model '{display_name}' from '{model_path}': {e}")
        else:
            print(f"WARNING: Model file '{model_path}' for '{display_name}' not found.")
    print("--- Resource Loading Complete ---")

load_resources() # Load when app starts

@app.route('/', methods=['GET'])
def index():
    available_models = sorted(list(models_cache.keys())) # Sort for consistent order
    return render_template('index.html', models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    available_models = sorted(list(models_cache.keys()))
    try:
        data = request.form
        open_price = float(data['open'])
        high_price = float(data['high'])
        low_price = float(data['low'])
        volume = float(data['volume'])
        selected_model_name = data['model']

        if selected_model_name not in models_cache:
            return render_template('index.html', models=available_models,
                                   error_message=f"Model '{selected_model_name}' is not available.",
                                   open_val=open_price, high_val=high_price, low_val=low_price, volume_val=volume)

        model_to_use = models_cache[selected_model_name]
        
        features = np.array([[open_price, high_price, low_price, volume]])
        
        if selected_model_name in SCALED_MODELS_SET:
            if scaler:
                features = scaler.transform(features)
            else:
                return render_template('index.html', models=available_models,
                                   error_message="Scaler not loaded. Cannot use this model.",
                                   open_val=open_price, high_val=high_price, low_val=low_price, volume_val=volume)

        prediction = model_to_use.predict(features)
        predicted_price = round(prediction[0], 2)

        return render_template('index.html',
                               models=available_models,
                               selected_model=selected_model_name,
                               prediction_text=f'Predicted Gold Price: {predicted_price}',
                               open_val=open_price, high_val=high_price, low_val=low_price, volume_val=volume)

    except ValueError:
        return render_template('index.html', models=available_models,
                               error_message="Invalid input. Please enter numbers only.",
                               open_val=data.get('open', ''), high_val=data.get('high', ''),
                               low_val=data.get('low', ''), volume_val=data.get('volume', ''))
    except Exception as e:
        print(f"Prediction error: {e}") # Important for debugging on Render
        return render_template('index.html', models=available_models,
                               error_message=f"An error occurred: {str(e)}",
                               open_val=data.get('open', ''), high_val=data.get('high', ''),
                               low_val=data.get('low', ''), volume_val=data.get('volume', ''))

if __name__ == '__main__':
    # Port for Render, or 5000 for local
    port = int(os.environ.get('PORT', 5000))
    # For Render, Gunicorn runs it. For local, Flask dev server.
    # Host '0.0.0.0' makes it accessible on your network if needed locally.
    app.run(host='0.0.0.0', port=port, debug=False) # debug=False for production/Render