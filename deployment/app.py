# save this as app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__, template_folder='templates')
model = joblib.load('xgboost_pipeline.pkl')
numerical_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    


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
    bp_val = float(data.get('SystolicBP', 0))
    if bp_val < 90: bp_cat = 'Low'
    elif bp_val < 120: bp_cat = 'Normal'
    elif bp_val < 140: bp_cat = 'Pre-High'
    else: bp_cat = 'High'
    input_df['BP_Category'] = bp_cat
    
    # Predict and return
    prediction = model.predict(input_df)[0]
    print(prediction)
    predictionResult = (
        'Low Risk' if prediction == 0 else
        'Medium Risk' if prediction == 1 else
        'High Risk'
    )

    # return jsonify({'prediction': prediction.tolist()})
    return render_template('index.html', prediction=f'Prediction: {predictionResult}')


@app.route('/predict-api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    
    input_df = pd.DataFrame([[data.get('Age', 0),
                              data.get('SystolicBP', 0),
                              data.get('DiastolicBP', 0),
                              data.get('BS', 0),
                              data.get('BodyTemp', 0),
                              data.get('HeartRate', 0)]],
                            columns=numerical_features)

    print("Making predictions with input data:")
    print(data.get('Age', 0))
    print(data.get('SystolicBP', 0))
    print(data.get('DiastolicBP', 0))
    print(data.get('BS', 0))
    print(data.get('BodyTemp', 0))
    print(data.get('HeartRate', 0))

    # Add BP category (same logic as training)
    bp_val = float(data.get('SystolicBP', 0))
    if bp_val < 90:
        bp_cat = 'Low'
    elif bp_val < 120:
        bp_cat = 'Normal'
    elif bp_val < 140:
        bp_cat = 'Pre-High'
    else:
        bp_cat = 'High'
    
    input_df['BP_Category'] = bp_cat

    # Predict and return
    prediction = model.predict(input_df)[0]
    print(prediction)
    
    predictionResult = (
        'Low Risk' if prediction == 0 else
        'Medium Risk' if prediction == 1 else
        'High Risk'
    )

    return jsonify({'prediction': predictionResult})


if __name__ == '__main__':
    # app.run(debug=False)
    app.run(host='0.0.0.0', port=8080, debug=False)
