"""
Amazon SageMaker Inference Script for Heart Disease Prediction
This script defines how the model should handle prediction requests
"""

import json
import numpy as np
import pickle
import os

def model_fn(model_dir):
    # Load model artifacts
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # Extract features in correct order
        features = [
            data.get('Age', 0),
            data.get('Cholesterol', 0),
            data.get('FBS_over_120', 0),
            data.get('Max_HR', 0),
            data.get('ST_depression', 0),
            data.get('Vessels_fluro', 0)
        ]
        
        return np.array(features).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    # Extract model components
    w = model['weights']
    b = model['bias']
    mu = model['mean']
    sigma = model['std']
    
    # Normalize input using training statistics
    X_normalized = (input_data - mu) / sigma
    
    # Compute prediction
    z = np.dot(X_normalized, w) + b
    probability = sigmoid(z)[0]
    prediction = 1 if probability >= 0.5 else 0
    
    # Determine risk level
    if probability >= 0.7:
        risk_level = "HIGH RISK"
    elif probability >= 0.4:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': risk_level,
        'confidence': f"{probability*100:.1f}%" if prediction == 1 else f"{(1-probability)*100:.1f}%"
    }


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Example usage for local testing
if __name__ == "__main__":
    # Test with high-risk patient
    test_input = {
        "Age": 60,
        "Cholesterol": 300,
        "FBS_over_120": 1,
        "Max_HR": 120,
        "ST_depression": 2.5,
        "Vessels_fluro": 2
    }
    
    print("Test Input:", test_input)
    print("\nNote: This script requires model.pkl to be loaded for actual predictions.")
    print("Deploy to SageMaker for full functionality.")
