from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import tensorflow as tf

app = Flask(__name__)

# Load data
data = pd.read_csv(r'/home/muhammad/Education/Data_Mining_Lab/Project/data/preprocessed_and_cleaned_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Get unique cities
cities = data['city'].unique().tolist()

# Model paths
MODEL_DIR = r'/home/muhammad/Education/Data_Mining_Lab/Project/model'
LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, 'best_linear_regression.pkl')
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'LSTM_model.h5')

# Model performance metrics
model_metrics = {
    'linear': {
        'name': 'Linear Regression',
        'mae': 0.1295,
        'rmse': 0.1816,
        'r2': 0.8763
    },
    'xgboost': {
        'name': 'XGBoost',
        'mae': 0.0729,
        'rmse': 0.1307,
        'r2': 0.9360
    },
    'lstm': {
        'name': 'LSTM',
        'mae': 0.05318,
        'rmse': 0.08040, # Square root of MSE (0.0064639962593503845)
        'r2': 0.9824
    }
}

# SARIMA model metrics by city
sarima_metrics = {
    'phoenix': {'mae': 155.3445, 'rmse': 338.9352, 'r2': 0.8528},
    'la': {'mae': 359.3463, 'rmse': 800.1504, 'r2': 0.8469},
    'nyc': {'mae': 255.6791, 'rmse': 553.7811, 'r2': 0.7405},
    'dallas': {'mae': 909.8572, 'rmse': 1534.1677, 'r2': 0.6983},
    'houston': {'mae': 609.8086, 'rmse': 1083.1707, 'r2': 0.7394},
    'philadelphia': {'mae': 241.3158, 'rmse': 488.3987, 'r2': 0.4736},
    'san antonio': {'mae': 435.5215, 'rmse': 732.9877, 'r2': 0.7227},
    'san jose': {'mae': 381.1277, 'rmse': 826.9324, 'r2': 0.7633},
    'san diego': {'mae': 82.1875, 'rmse': 138.7749, 'r2': 0.8221},
    'seattle': {'mae': 35.7224, 'rmse': 67.4989, 'r2': 0.7832}
}

# Load LSTM model
def load_lstm_model():
    try:
        import tensorflow as tf
        # Define custom_objects to handle the 'mse' function
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError()
        }
        return tf.keras.models.load_model(LSTM_MODEL_PATH, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None

# Load models
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

# Load SARIMA model for a specific city
def load_sarima_model(city):
    city_name = city.lower().replace(' ', '_')
    model_path = os.path.join(MODEL_DIR, f'sarima_{city_name}_daily.pkl')
    return load_model(model_path)

# Perform clustering
def perform_clustering(k=3):
    # Features for clustering
    features = ['temperature', 'humidity', 'windSpeed', 'pressure', 'dewPoint', 'demand_mwh']
    X = data[features]
    
    # Apply RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title(f"Cluster Visualization (k={k})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    encoded = base64.b64encode(image_png).decode('utf-8')
    return encoded

# Generate forecast
def generate_forecast(city, start_date, end_date, model_type):
    # Filter data for the selected city
    city_data = data[data['city'] == city]
    
    if model_type == 'sarima':
        # Load SARIMA model for the selected city
        sarima_model = load_sarima_model(city)
        if sarima_model is None:
            return None, "SARIMA model not found for this city"
        
        # Convert to daily data
        daily_data = city_data.set_index('timestamp')['demand_mwh'].resample('D').mean()
        
        # Generate forecast
        forecast_steps = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        forecast = sarima_model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Get metrics for the city
        city_key = city.lower()
        metrics = sarima_metrics.get(city_key, {'mae': 'N/A', 'rmse': 'N/A', 'r2': 'N/A'})
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(daily_data.index, daily_data, label='Historical Data')
        plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
        plt.fill_between(
            forecast_mean.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='red',
            alpha=0.2
        )
        plt.title(f'SARIMA Forecast for {city}')
        plt.xlabel('Date')
        plt.ylabel('Demand (MWh)')
        plt.legend()
        plt.grid(True)
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        encoded = base64.b64encode(image_png).decode('utf-8')
        return encoded, metrics, None
    
    else:
        return None, None, "Unsupported model type"

# Compare models (actual vs predicted)
def compare_models(model_type):
    try:
        # Get metrics for the model
        metrics = model_metrics.get(model_type, {'name': 'Unknown', 'mae': 'N/A', 'rmse': 'N/A', 'r2': 'N/A'})
        
        # Set image path based on model type
        image_folder = r'/home/muhammad/Education/Data_Mining_Lab/Project/image_folder'
        
        if model_type == 'linear':
            image_path = os.path.join(image_folder, 'linear_regression.png')
            
        elif model_type == 'xgboost':
            image_path = os.path.join(image_folder, 'xgboost.png')
            
        elif model_type == 'lstm':
            image_path = os.path.join(image_folder, 'lstm.png')
            
        else:
            return None, metrics, "Unsupported model type"
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            return None, metrics, f"{metrics['name']} image not found"
        
        # Read and encode the image file
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
            encoded = base64.b64encode(image_data).decode('utf-8')
            
        return encoded, metrics, None
        
    except Exception as e:
        return None, None, f"Error loading image: {str(e)}"


@app.route('/')
def index():
    # Get min and max dates from the dataset
    min_date = data['timestamp'].min().strftime('%Y-%m-%d')
    max_date = data['timestamp'].max().strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                           cities=cities,
                           min_date=min_date,
                           max_date=max_date)

@app.route('/cluster', methods=['POST'])
def cluster():
    k = int(request.form.get('k', 3))
    cluster_image = perform_clustering(k)
    return jsonify({'image': cluster_image})

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        city = request.form.get('city')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        model_type = request.form.get('model_type')
        
        forecast_image, metrics, error = generate_forecast(city, start_date, end_date, model_type)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify({
            'image': forecast_image,
            'metrics': metrics,
            'model_name': 'SARIMA'  # Add model name for display
        })
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"})

# In the compare_models route handler
@app.route('/compare_models', methods=['POST'])
def compare_models_route():
    try:
        model_type = request.form.get('model_type')
        
        comparison_image, metrics, error = compare_models(model_type)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify({
            'image': comparison_image,
            'metrics': metrics,
            'model_name': model_metrics.get(model_type, {}).get('name', 'Unknown Model')
        })
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"})
if __name__ == '__main__':
    app.run(debug=True)
