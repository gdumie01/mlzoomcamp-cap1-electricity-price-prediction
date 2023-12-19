import requests
import json

hourly_status = {
    "generation biomass":352.0,
    "generation fossil brown coal/lignite":605.0,
    "generation fossil gas":4949.0,
    "generation fossil hard coal":4900.0,
    "generation fossil oil":277.0,
    "generation hydro pumped storage consumption":2340.0,
    "generation hydro run-of-river and poundage":453.0,
    "generation hydro water reservoir":340.0,
    "generation nuclear":7102.0,
    "generation other":58.0,
    "generation other renewable":103.0,
    "generation solar":31.0,
    "generation waste":338.0,
    "generation wind onshore":4375.0,
    "forecast solar day ahead":8.0,
    "forecast wind onshore day ahead":4318.0,
    "total load actual":21168.0,
    "total load forecast":21436.0,
    "price day ahead":46.74,
    "price actual":55.55,
    "year":2017,
    "month":6,
    "day":1,
    "hour":2,
    "dayofweek":3,
    "price_actual_lag24h":52.37,
    "price_actual_lag12h":60.04,
    "price_actual_lag6h":61.01,
    "price_actual_lag3h":58.66,
    "price_actual_lag1h":55.06
}

url = 'http://localhost:9698/predict_price'
response = requests.post(url, json=hourly_status)
result = response.json()

print(json.dumps(result, indent=2))