# Electricity Price Prediction - Cap Stone Project #1 for ML Zoomcamp 2023

This repository contains the code and materials for the Cap Stone #1 project of the Zoomcamp Machine Learning course. The project focuses on predicting electricity prices for the spanish market based on several datapoints covering generation mix, demand load and historical price movements.

## Overview

In this project, I have worked on a [Kaggle hourly energy price dataset](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) that contains information related to generation mix, demand load, and historical price performance of the spanish electricity market.

I have choosen this particular problem because I was curious to understand if using deep learning techniques would enable me to build an effective model on predicting the price of electricity, thus paving the ground to develop proprietary trading mechanisms in the electricity market. In addition, I also wanted to work with a timeseries as it was something we did not had during this cohort.

After testing with several neural network configurations a stacked Recurrent Neural Network was choosen based on the RMSE vs. a naive technique such as using the previous hour price as a predictor to the next hour price. With this model I could beat the performance of the naive approach by 12% proving some predictive power in the deep learning approach. You can check my exploration journey on this [notebook](https://github.com/gdumie01/mlzoomcamp-cap1-electricity-price-prediction/blob/main/notebook.ipynb).

![Model Visualization](https://github.com/gdumie01/mlzoomcamp-cap1-electricity-price-prediction/blob/main/images/model-visualization.png)


## Problem description and usage
### Problem description
The electricity market in Spain, notably represented by [OMIE (Operador del Mercado Ibérico de Energía)](https://www.omie.es/), is dynamic and influenced by various factors such as the generation mix, demand load, and past price performance. This project aims to assess the effectiveness of deep learning models in accurately predicting electricity prices. By leveraging information about the generation mix, demand load, and historical price performance, we seek to build a robust predictive model.

For simplicity, the focus of this project is on predicting the price at which electricity will trade in the next hour based on the current hour's context (i.e.: which sources are generating electricity, the demand level and how the price has evolved up to this hour). While this might have limited practical applications due to the short time window, the primary goal is to demonstrate the predictive capabilities of Deep Learning models in this context.

### Usage
Having the ability to predict the electricity price in the Spanish market holds significant potential, as it opens doors to strategic trading opportunities. A well-constructed predictive model can provide a trading edge, potentially leading to financial gains in the electricity market.

As mentioned earlier, maybe a 1h time window prediction is not ideal for trading, but if successful, this project paves the way for building models with larger prediction windows. This expansion would provide more time for strategic decision-making and better trading outcomes.

## Dataset

### Where to find

Data is available in the [data folder](https://github.com/gdumie01/mlzoomcamp-cap1-electricity-price-prediction/tree/main/data) of this project, but you can also download it from [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather). Although the data pack from Kaggle is comprised by two files (weather and energy data), in this project I've just used the energy dataset.

### Data explanation

The dataset includes hourly reads for 4 sets of key variables:
- the production (in MW) for each power generation source available in Spain (e.g.: coal, gas, hydro, solar, etc.)
- the production forecast for renewable energies such as solar and wind for the next day
- the power demand - both for the current hour and for the forecast for the next day
- price - actual price for the current hour and the market stipulated price for the day ahead based on future contracts

Data covers a period comprised between end of 2014 and end of 2018.

For training I've used the period comprised between 2016-01-01 and 2017-06-01, leaving the rest of the data for testing.

## How to Run the Code

### Dependencies
For the notebook you´ll need:
- Python 3.9+
- NumPy
- Pandas
- Scikit-Learn
- TensforFlow
- Keras
- Seaborn
- Matplotlib
- statsmodels

To run the python scripts you´ll need:
- Flask
- gunicorn
- TensforFlow
- Joblib
- Requests (optional for the tester script)

Project contains a `Pipfile` and a `Pipfile.lock` to setup the virtual environment with the proper versions for train and predict scripts.

Finally, you´ll need to have Docker installed on your machine to build and execute the docker image that will serve the prediction webservice.

### Setup instructions

#### To train the model
##### 1. Clone this repository.
```
git clone https://github.com/gdumie01/mlzoomcamp-cap1-electricity-price-prediction.git
```
##### 2. Install the pip environment inside the repository directory
```
cd mlzoomcamp-cap1-electricity-price-prediction
pipenv install
```
Wait for pipenv to be done with installing the packages and dependencies.

NOTE: if you get any error during `pipenv install` try updating the lock file with `pipenv lock --pre --clear`

##### 3. Launch the pipenv shell
```
pipenv shell
```
##### 4. Train the model inside the pipenv shell
```
python train.py
```
should generate/update the files `electricity-price-pred-model.tflite` and `train.fitted_scaler` in the model folder
##### 5. Run the predict service
```
python predict.py
```
##### 6. On another terminal you can use the test file 
```
pipenv shell
python predict-tester.py
```
Change the variables in the script to test with different values.

#### To run the Docker container version of the service
##### 1. Clone this repository.
```
git clone https://github.com/gdumie01/mlzoomcamp-cap1-electricity-price-prediction.git
```
##### 2. Build the docker image
```
docker build -t {build-tag} .
```
`build-tag`: Specifies any user-defined tag for docker image. eg. `power-prediction`

##### 3. Run the docker image

```
docker run -it -p 9698:9698 {build-tag}:latest
```
in case you used the suggested tag above code would be:
```
docker run -it -p 9698:9698 power-prediction:latest
```

##### 4. Now use the predict-tester script to test the service
In another terminal run the predict-tester.py.
```
python predict-tester.py
```
##### 5. If you want to use other data you can just change it in the scritp and run again
## Sample output
![Sample output](https://github.com/gdumie01/mlzoomcamp-cap1-electricity-price-prediction/blob/main/images/sample-output.png)
