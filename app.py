from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "CTR Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Example: [0.2, 0.8, 1.0, ...]
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'ctr_prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
