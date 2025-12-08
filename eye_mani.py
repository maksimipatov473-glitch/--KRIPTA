import matplotlib.pyplot as plt  # noqa: I001
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore


# --------------------------
# 1. Завантаження даних (BTC)
# --------------------------
def load_btc_data(days=7):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    if "prices" not in data or not data["prices"]:
        raise ValueError("API response does not contain price data")

    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Погодинна агрегація
    df.set_index("timestamp", inplace=True)
    df = df.resample("H").mean()

    return df


df = load_btc_data(7)

# масштабування даних
scaler = MinMaxScaler(feature_range=(0, 1))
prices = df["price"].astype(float).to_numpy().reshape(-1, 1)
scaled = scaler.fit_transform(prices)


# --------------------------
# 2. Підготовка даних для LSTM (вікно = 24 години)
# --------------------------
window = 24  # LSTM дивиться у минулі 24 години
X_list, y_list = [], []

for i in range(window, len(scaled)):
    X_list.append(scaled[i - window : i, 0])
    y_list.append(scaled[i, 0])

X = np.array(X_list)
y = np.array(y_list)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# --------------------------
# 3. LSTM модель
# --------------------------
model = Sequential(
    [LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)), LSTM(32), Dense(1)]
)

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# --------------------------
# 4. Прогноз на 10 годин уперед
# --------------------------
future_steps = 10
last_window = scaled[-window:]
predictions = []

current_input = last_window.reshape(1, window, 1)

for _ in range(future_steps):
    pred = model.predict(current_input)[0][0]
    predictions.append(pred)

    # додаємо прогноз до вікна
    current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

# розмасштабуємо назад у USD
forecast_values = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
).flatten()

# створюємо часовий індекс
last_time = df.index[-1]
forecast_index = pd.date_range(
    start=last_time + pd.Timedelta(hours=1), periods=future_steps, freq="H"
)

forecast_series = pd.Series(forecast_values, index=forecast_index)

print("📈 Погодинний LSTM прогноз на 10 годин:")
print(forecast_series)

# --------------------------
# 5. Графік
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["price"], label="Historical (hourly)")
plt.plot(forecast_series, label="LSTM Forecast (next 10 hours)", linestyle="--")
plt.title("BTC Hour-to-Hour Forecast (LSTM Neural Network)")
plt.xlabel("Time (hourly)")
plt.ylabel("USD")
plt.legend()
plt.grid()

output_file = "btc_lstm_hour_forecast.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Графік збережено у {output_file}")

plt.show()
plt.show()
