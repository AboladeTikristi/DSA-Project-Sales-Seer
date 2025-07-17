# import joblib
# from flask import Flask, request,json, jsonify, render_template
# from flask_cors import CORS
# import numpy as np
# import os

# app = Flask(__name__)
# CORS(app)

# model = None
# try:
#     if os.path.exists("sales_Quantity_without discount_prediction_model.pkl"):
#         model = joblib.load("sales_Quantity_without discount_prediction_model.pkl")
#         print("✅ Model loaded successfully")
#     else:
#         print("❌ Model file not found!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
    
#     try:
#         data = request.json
#         features = data.get('features', None)
#         # Load your trained XGBoost model (make sure this file exists)
#         # model = joblib.load("sales_Revenue_without discount_prediction_model.pkl")
#         model1 = joblib.load("sales_Revenue_with discount_prediction_model.pkl")
#         model2 = joblib.load("sales_Quantity_without discount_prediction_model.pkl")
#         model3 = joblib.load("sales_Quantity_with discount_prediction_model.pkl")

#         if model is None:
#             return jsonify({"error": "Model not loaded"}), 500
            
#         # return jsonify({"received": features})
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"error": str(e)}), 400

#     if features is None or len(features) != 7:
#         return jsonify({'error': 'Please provide 7 numeric features.'}), 400

#     try:
#          # cutting to 6 feature for prediction without discount
#         features_six = np.array(features, dtype=float)[:-1].reshape(1, -1)
#         features = np.array(features, dtype=float).reshape(1, -1)
#     except Exception:
#         return jsonify({'error': 'Invalid feature values. Make sure all are numbers.'}), 400
        
  
   
#     prediction = model.predict(features_six)[0].astype(np.float64)
#     prediction1 = model1.predict(features)[0].astype(np.float64)
#     prediction2 = model2.predict(features_six)[0].astype(np.float64)
#     prediction3 = model3.predict(features)[0].astype(np.float64)
   

#     return jsonify({
#             "prediction": prediction,
#             "prediction1": prediction1,
#             "prediction2": prediction2,
#             "prediction3": prediction3
#             # "probability": round(float(proba), 4)
#         })


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 3000))  # Use the port Render provides
#     app.run(host="0.0.0.0", port=port)

#     # app.run(debug=True)


    
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    # Load all models once at startup
    try:
        model = joblib.load("sales_Quantity_without discount_prediction_model.pkl")
        model1 = joblib.load("sales_Revenue_with discount_prediction_model.pkl")
        model2 = joblib.load("sales_Quantity_without discount_prediction_model.pkl")
        model3 = joblib.load("sales_Quantity_with discount_prediction_model.pkl")
        print("✅ All models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = model1 = model2 = model3 = None
        
    if not all([model, model1, model2, model3]):
        return jsonify({"error": "Models not loaded"}), 500

    try:
        data = request.json
        features = data.get('features')
    except Exception as e:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400

    if not features or len(features) != 7:
        return jsonify({'error': 'Please provide 7 numeric features.'}), 400

    try:
        features = np.array(features, dtype=float).reshape(1, -1)
        features_six = features[:, :-1]  # Drop last column
    except Exception:
        return jsonify({'error': 'Invalid feature values'}), 400

    try:
        prediction = model.predict(features_six)[0].item()
        prediction1 = model1.predict(features)[0].item()
        prediction2 = model2.predict(features_six)[0].item()
        prediction3 = model3.predict(features)[0].item()

        return jsonify({
            "prediction": prediction,
            "prediction1": prediction1,
            "prediction2": prediction2,
            "prediction3": prediction3
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
