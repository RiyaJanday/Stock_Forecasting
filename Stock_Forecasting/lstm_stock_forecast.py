import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

ticker = 'AAPL'

start_date = '2016-01-01'
end_date = pd.Timestamp.today() - pd.Timedelta(days=1)

data = yf.download(ticker, start=start_date, end=end_date.strftime('%Y-%m-%d'))
data = data[['Close']]
data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

sequence_length = 60
x_train, y_train = [], []

for i in range(sequence_length, len(scaled_data) - 365):  # Leave last year for testing
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=10, batch_size=32)

test_data = scaled_data[-(365 + sequence_length):]
x_test, y_test = [], []

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
    y_test.append(test_data[i, 0])

x_test = np.array(x_test).reshape(-1, sequence_length, 1)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"\nEvaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")

plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='black', label='Actual Prices (Last Year)')
plt.plot(predicted_prices, color='green', label='Predicted Prices')
plt.title(f'{ticker} Stock Price Prediction (2024 to {end_date.date()})')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()

n_days = 30
last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
future_predictions = []

for _ in range(n_days):
    next_pred = model.predict(last_sequence)[0][0]
    future_predictions.append(next_pred)
    next_input = np.append(last_sequence[0, 1:], [[next_pred]], axis=0)
    last_sequence = next_input.reshape(1, sequence_length, 1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(range(len(actual_prices)), actual_prices, label='Actual Prices (2024)')
plt.plot(range(len(actual_prices), len(actual_prices) + n_days), future_predictions, color='orange', label='Future Predictions (Next 30 Days)')
plt.title(f'{ticker} Stock Forecast (Next 30 Days)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()