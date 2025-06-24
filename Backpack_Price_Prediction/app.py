from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained ML model
model = joblib.load("Model/trained_model.pkl")

# Home route - Serves the HTML page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction API - Handles form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form

        # Convert categorical inputs to numeric (you should use your own encoding logic)
        brand = hash(form_data["Brand"]) % 100  # Example encoding
        material = hash(form_data["Material"]) % 100
        size = hash(form_data["Size"]) % 100
        laptop_compartment = 1 if form_data["Laptop Compartment"].lower() == "yes" else 0
        waterproof = 1 if form_data["Waterproof"].lower() == "yes" else 0
        style = hash(form_data["Style"]) % 100
        color = hash(form_data["Color"]) % 100
        compartments = int(form_data["Compartments"])
        weight_capacity = float(form_data["Weight Capacity (kg)"])

        # Prepare input for model (Ensure correct order as per training)
        input_features = np.array([brand, material, size, laptop_compartment, waterproof,
                                   style, color, compartments, weight_capacity]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Return JSON response
        return jsonify({"predicted_price": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
