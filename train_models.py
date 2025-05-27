# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (LinearRegression, ElasticNet, Ridge, Lasso, HuberRegressor, 
                                  PassiveAggressiveRegressor, ARDRegression, BayesianRidge, SGDRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

print("--- Starting Model Training Script ---")

# --- Configuration ---
DATA_FILE = 'Gold Price.csv'
MODELS_DIR = 'models' # Folder to save models

# --- Create models directory if it doesn't exist ---
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Created directory: {MODELS_DIR}")

# --- Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: Data file '{DATA_FILE}' not found. Make sure it's in the same directory as the script.")
    exit()

print("Data loaded successfully.")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Open', 'High', 'Low', 'Volume', 'Price'])
print(f"Data shape after cleaning: {df.shape}")

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# --- SCALING ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled.")

# Save the scaler
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# --- Define Models ---
models_to_train = {
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    'Hist Gradient Boosting Regressor': HistGradientBoostingRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Elastic Net': ElasticNet(random_state=42),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'Huber Regressor': HuberRegressor(),
    'Passive Aggressive Regressor': PassiveAggressiveRegressor(random_state=42, max_iter=1000, tol=1e-3),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'ARD Regression': ARDRegression(),
    'Bayesian Ridge': BayesianRidge(),
    'SGD Regressor': SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
}

scaled_models_set = {
    'Support Vector Regressor',
    'K-Nearest Neighbors Regressor',
    'SGD Regressor'
}

print("\n--- Training and Saving Models ---")
results = {}
for model_name, model_instance in models_to_train.items():
    print(f"Training {model_name}...")
    if model_name in scaled_models_set:
        model_instance.fit(X_train_scaled, y_train)
        y_pred = model_instance.predict(X_test_scaled)
    else:
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MSE': mse, 'R2': r2}
    print(f"  R²: {r2:.4f}, MSE: {mse:.4f}")

    model_filename = model_name.replace(" ", "_").lower() + '.pkl'
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model_instance, model_path)
    print(f"  Saved to {model_path}")
    print('-----------------------------------')

print("\n--- Model Training and Saving Complete ---")
results_df = pd.DataFrame.from_dict(results, orient='index')
print("Model Performance Summary:")
print(results_df.sort_values('R2', ascending=False))
print("--- End of Model Training Script ---")