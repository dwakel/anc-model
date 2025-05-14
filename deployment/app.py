# save this as app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('../xgboost_pipeline.pkl')
numerical_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    


# Create input DataFrame (with BP category)
    input_df = pd.DataFrame([[data.get('Age', 0),
                             data.get('SystolicBP', 0),
                             data.get('DiastolicBP', 0),
                             data.get('BS', 0),
                             data.get('BodyTemp', 0),
                             data.get('HeartRate', 0)]],
                          columns=numerical_features)
    print("Making predictions with input data:")
    print(data.get('SystolicBP', 0))
    print(data.get('DiastolicBP', 0))
    print(data.get('BS', 0))
    print(data.get('BodyTemp', 0))
    print(data.get('HeartRate', 0))
    
    # Add BP category (same logic as training)
    bp_val = data.get('SystolicBP', 0)
    if bp_val < 90: bp_cat = 'Low'
    elif bp_val < 120: bp_cat = 'Normal'
    elif bp_val < 140: bp_cat = 'Pre-High'
    else: bp_cat = 'High'
    input_df['BP_Category'] = bp_cat
    
    # Predict and return
    prediction = model.predict(input_df)[0]




    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False)
