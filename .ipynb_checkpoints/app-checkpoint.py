import joblib
from flask import Flask, request,json, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained XGBoost model (make sure this file exists)
model = joblib.load("sales_Revenue_without discount_prediction_model.pkl")
model1 = joblib.load("sales_Revenue_with discount_prediction_model.pkl")
model2 = joblib.load("sales_Quantity_without discount_prediction_model.pkl")
model3 = joblib.load("sales_Quantity_with discount_prediction_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features', None)
    
        # return jsonify({"received": features})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

    if features is None or len(features) != 7:
        return jsonify({'error': 'Please provide 7 numeric features.'}), 400

    try:
         # cutting to 6 feature for prediction without discount
        features_six = np.array(features, dtype=float)[:-1].reshape(1, -1)
        features = np.array(features, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({'error': 'Invalid feature values. Make sure all are numbers.'}), 400
        
  
   
    prediction = model.predict(features_six)[0].astype(np.float64)
    prediction1 = model1.predict(features)[0].astype(np.float64)
    prediction2 = model2.predict(features_six)[0].astype(np.float64)
    prediction3 = model3.predict(features)[0].astype(np.float64)
   

    return jsonify({
            "prediction": prediction,
            "prediction1": prediction1,
            "prediction2": prediction2,
            "prediction3": prediction3
            # "probability": round(float(proba), 4)
        })

if __name__ == '__main__':
    app.run( port=3000, debug=True)
    # app.run(debug=True)
