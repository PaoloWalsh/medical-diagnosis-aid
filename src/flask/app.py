import pickle
from flask import Flask, request, jsonify
import numpy as np
import os
import onnxruntime as ort
# Initialize the Flask application
app = Flask(__name__)

# Define the path to your pickled model file
#MODEL_PATH = 'models/'
MODEL_PATH = os.getenv('MODEL_PATH', 'models/')  
MODEL_FILES = {
    'K-Nearest Neighbors': 'best_knn_model.onnx',
    'Logistic Regression': 'best_log_model.onnx',
}

MODEL_STATS = 'tuned_model_performance.csv'
# Global variable to store the loaded model
loaded_models = {}
# --- Model Loading Function ---
def load_models():
    """
    Loads the scikit-learn model from the specified pickle file.
    This function will be called once when the Flask app starts.
    """
    
    for save_name, model_name in MODEL_FILES.items():
        if os.path.exists(MODEL_PATH + model_name):
            try:
                with open(MODEL_PATH + model_name, 'rb') as _:
                    loaded_models[save_name] = ort.InferenceSession(MODEL_PATH + model_name)
                print(f"Model {save_name} successfully loaded from {model_name}")
            except (FileNotFoundError, IOError) as e:
                print(f"Error loading model '{save_name}' from {model_name}: {e}")
                # Do not stop the app, but log the error for this specific model
        else:
            print(f"Model file for '{save_name}' not found at {MODEL_PATH + model_name}. Skipping.")
    
    if not loaded_models:
        print("WARNING: No models were loaded successfully. Prediction endpoint will fail.")
    else:
        print(f"Successfully loaded models: {list(loaded_models.keys())}")

# --- Flask Routes ---

@app.route('/')
def home():
    """
    Basic home route to confirm the server is running.
    """
    available_models = list(loaded_models.keys())
    return (
        f"Flask multi-model prediction backend is running!<br>"
        f"Available models: {', '.join(available_models) if available_models else 'None'}<br>"
        f"Use /predict to make predictions, specifying 'model_name' in your JSON request."
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive data, make a prediction using the loaded model,
    and return the prediction.

    Expected request body:
    {
        "model_name": string  # or "K-Nearest Neighbors", "Logistic Regression", "SVC"
        "data": [[feature1_val1, feature2_val1, ...]]
    }
    """
    if not loaded_models:
        return jsonify({"error": "No models are loaded. Server misconfiguration."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    request_data = request.get_json()

    # Get the model name from the request
    model_name = request_data.get('model_name')
    if not model_name:
        return jsonify({"error": "Missing 'model_name' field in request body."}), 400

    # Retrieve the chosen model
    chosen_model = loaded_models.get(model_name)
    if chosen_model is None:
        return jsonify({
            "error": f"Model '{model_name}' not found or not loaded. Available models: {list(loaded_models.keys())}"
        }), 404 

    # Get the data for prediction
    input_data = request_data.get('data')
    if input_data is None: # Check explicitly for None, as get() returns None if key is missing
        return jsonify({"error": "Missing 'data' field in request body."}), 400

    try:
        # Convert the input list of lists to a NumPy array
        input_array = np.array(input_data)
        print(input_array)

        # Make prediction using the chosen model
        input_name = chosen_model.get_inputs()[0].name
        label_name = chosen_model.get_outputs()[0].name
        predictions = chosen_model.run([label_name], {input_name: input_array.astype(np.float32)})[0]
        probabilities = None  # ONNX models may not always provide probabilities
        
        try:
            # Check if the model provides probabilities
            prob_output_name = chosen_model.get_outputs()[1].name  # Assuming the second output is probabilities
            probabilities = chosen_model.run([prob_output_name], {input_name: input_array.astype(np.float32)})[0]
        except IndexError:
            # If the model does not have a second output for probabilities, skip
            probabilities = None
        response = {
            "model_used": model_name,
            "predictions": predictions.tolist()  # Convert ndarray to list
        }
        if probabilities is not None:
            response["probabilities"] = probabilities  # Convert ndarray to list
        return jsonify(response), 200

    except ValueError as ve:
        # Handle cases where input data dimensions don't match model's expected features
        return jsonify({"error": f"Invalid input data format or dimensions for model '{model_name}': {ve}"}), 400
    
@app.route('/models', methods=['POST'])
def get_model_performace():
    """
    API endpoint to return a chosen model's accuracy, recall and roc_auc.

    Expected request body:
    {
        "model_name": string  # or "K-Nearest Neighbors", "Logistic Regression", "SVC"
      
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    request_data = request.get_json()
    model_name = request_data.get('model_name')

    if not model_name:
        return jsonify({"error": "Missing 'model_name' field in request body."}), 400

    # load MODEL_PATH + tunded_model_performance.csv
    performance_file = MODEL_PATH + MODEL_STATS
    if not os.path.exists(performance_file):
        return jsonify({"error": f"Model performace file '{MODEL_STATS}' not found."}), 404
    try:
        with open(performance_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Find the line for the requested model
            for line in lines:
                if line.startswith(model_name):
                    stats = line.strip().split(',')
                    return jsonify({
                        "model_name": model_name,
                        "accuracy": stats[1],
                        "recall": stats[2],
                        "roc_auc": stats[3]
                    }), 200
            return jsonify({"error": f"Model '{model_name}' performace file not found."}), 404
    except IOError as e:
        return jsonify({"error": f"Error reading model performance file: {e}"}), 500
    
@app.route('/model_list', methods=['POST'])
def return_model_list():
    """
    API endpoint to return the list of available models.
    """
    if not loaded_models:
        return jsonify({"error": "No models are loaded. Server misconfiguration."}), 500

    return jsonify({"available_models": list(loaded_models.keys())}), 200


# --- Main execution block ---
if __name__ == '__main__':
    # Load the model when the application starts
    load_models()
    
    app.run(debug=False, host='0.0.0.0', port=5000)

