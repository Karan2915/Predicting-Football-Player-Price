from flask import Flask, request, jsonify
import joblib
import numpy as np

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = joblib.load("football_model.pkl")

# -----------------------------------
# CREATE FLASK APP
# -----------------------------------
app = Flask(__name__)

# -----------------------------------
# HOME ROUTE
# -----------------------------------
@app.route('/')
def home():

    return "Football Player Market Value Prediction API"

# -----------------------------------
# PREDICTION ROUTE
# -----------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({

        'predicted_market_value': float(prediction[0])

    })

# -----------------------------------
# RUN APP
# -----------------------------------
if __name__ == '__main__':

    app.run(debug=True)