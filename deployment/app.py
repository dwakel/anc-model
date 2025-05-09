# save this as app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('../random_forest_pipeline.pkl')
numerical_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    


# Create input DataFrame (with BP category)
    input_df = pd.DataFrame([data],
                          columns=numerical_features)
    
    # Add BP category (same logic as training)
    bp_val = data[1]
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
