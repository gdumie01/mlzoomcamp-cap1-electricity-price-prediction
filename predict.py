import numpy as np
from flask import Flask, request, jsonify
from tensorflow import lite
import joblib
import pandas as pd

app = Flask('predict_price') # give an identity to your web service

interpreter = lite.Interpreter(model_path='electricity-price-pred-model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

scaler = joblib.load('model/train.fitted_scaler')

def predict_next_hour(hourly_status):
    hourly_status_df = pd.DataFrame([hourly_status], dtype=np.float32)
    X = scaler.transform(hourly_status_df)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return preds[0]

@app.route('/predict_price', methods=['POST']) # use decorator to add Flask's functionality to our function
def predict():
    try:
        raw_hourly_status = request.get_json()
        prediction = predict_next_hour(raw_hourly_status)
        result = {
            'next_hour_price': float(prediction)
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696