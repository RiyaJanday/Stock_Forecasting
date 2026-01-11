Project 6: Stock Price Prediction using LSTM

Internship: AI/ML Internship – Tamizhan Skills RISE

Objective:

 The objective of this project is to build a time-series forecasting model using Long Short-Term Memory (LSTM) neural networks to predict stock prices. The model uses historical daily closing prices of Apple Inc. (AAPL) to learn patterns and forecast both short-term prices and future trends.

Tools and Technologies Used:
- Python
- NumPy
- Pandas
- Matplotlib
- yfinance
- scikit-learn
- TensorFlow / Keras

Dataset:
- Source: yfinance library
- Stock Symbol: AAPL (Apple Inc.)
- Date Range: From 2016-01-01 to present
- Feature Used: Closing price (`Close`)
- Preprocessing: 
  - Normalization using MinMaxScaler
  - Sequence generation with 60 time steps

Project Files:
- lstm_stock_forecast.py: Complete script for downloading data, preprocessing, training the LSTM model, evaluating, and visualizing predictions.
- No additional data files required as the script dynamically downloads historical data using yfinance.

Model Architecture:
The model is a stacked LSTM network consisting of:
- LSTM Layer (50 units, return_sequences=True)
- LSTM Layer (50 units)
- Dense Output Layer (1 unit)

Compiled with:
- Optimizer: Adam
- Loss Function: Mean Squared Error

Training Details:
- Sequence Length: 60 days
- Epochs: 10
- Batch Size: 32
- Test Data: Last 1 year of stock prices
- Evaluation Metrics:
- RMSE: 10.7824
- MAE : 8.8433
 
How to Run:

 Run the forecasting script:

 python lstm_stock_forecast.py

Requirements:

 Install required libraries using pip:

 numpy 
 pandas 
 matplotlib 
 yfinance 
 scikit-learn 
 tensorflow

Output Samples:

 price_prediction_plot.png: Plot comparing actual and predicted prices for the last year.
 future_forecast_plot.png: Forecasted stock prices for the next 30 days.

Submitted By:

Name: Riya Janday
College: Parul University
Course: B.Tech CSE with specialization in AI
Project Number: 6
Project Title: Stock Price Prediction using LSTM