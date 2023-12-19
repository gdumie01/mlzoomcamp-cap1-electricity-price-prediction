import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

## We need to load our model and convert it into a TFLite format

model = keras.models.load_model('checkpoints/best-model_v1_36_8.344853.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open('model/electricity-price-pred-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

## And we also need to fit the scaler with the training data so that we can then scale model inputs accordingly

energy_data = pd.read_csv(r"data/energy_dataset.csv")

generation_columns = [
    'generation biomass', 'generation fossil brown coal/lignite',
    'generation fossil gas', 'generation fossil hard coal', 
    'generation fossil oil',
    'generation hydro pumped storage consumption',
    'generation hydro run-of-river and poundage',
    'generation hydro water reservoir',
    'generation nuclear', 'generation other', 'generation other renewable',
    'generation solar', 'generation waste',
    'generation wind onshore']

forecast_columns = ['forecast solar day ahead','forecast wind onshore day ahead']

demand_columns = ['total load actual', 'total load forecast']

price_columns = ['price day ahead', 'price actual']

energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True, errors='coerce')

numerical = generation_columns + forecast_columns + demand_columns + price_columns

model_data = energy_data[numerical + ['time']].copy()

# Create contextual features from the date
model_data['year'] = model_data['time'].dt.year
model_data['month'] = model_data['time'].dt.month
model_data['day'] = model_data['time'].dt.day
model_data['hour'] = model_data['time'].dt.hour
model_data['dayofweek'] = model_data['time'].dt.dayofweek

# Create lag features
model_data['price_actual_lag24h'] = model_data['price actual'].shift(24)
model_data['price_actual_lag12h'] = model_data['price actual'].shift(12)
model_data['price_actual_lag6h'] = model_data['price actual'].shift(6)
model_data['price_actual_lag3h'] = model_data['price actual'].shift(3)
model_data['price_actual_lag1h'] = model_data['price actual'].shift(1)

# We need to drop the nulls that were introducted with the shifts
model_data = model_data.dropna(how='any', axis=0)
train_data = model_data[(model_data['time'] >= '2016-01-01') & (model_data['time'] < '2017-06-01')]
X_train = train_data.drop(columns=['time'])[:-1]

scaler = MinMaxScaler()

# Fit and transform the scaler on the sample data
scaler.fit_transform(X_train)

# Save the fitted scaler to a file
scaler_filename = 'model/train.fitted_scaler'
joblib.dump(scaler, scaler_filename)