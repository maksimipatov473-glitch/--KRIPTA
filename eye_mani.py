import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.models import Sequential


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

# Масштабування
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df["price"].values.reshape(-1, 1))

# --------------------------
# 2. Підготовка даних (вікно 24 години)
# --------------------------
# ...existing code...
window = 24
X_list: list[np.ndarray] = []
y_list: list[float] = []

for i in range(window, len(scaled)):
    X_list.append(scaled[i - window : i, 0])
    y_list.append(scaled[i, 0])

X = np.array(X_list)  # Теперь X имеет тип ndarray с самого начала
y = np.array(y_list)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# ...existing code...


# --------------------------
# 3. Вибір моделі користувачем
# --------------------------
model_type = "LSTM"  # <<< ЗМІНИ НА "LSTM" або "GRU"

print(f"📌 Обрана модель: {model_type}")

model = Sequential()

if model_type.upper() == "LSTM":
    model.add(LSTM(64, return_sequences=True, input_shape=(window, 1)))
    model.add(LSTM(32))
elif model_type.upper() == "GRU":
    model.add(GRU(64, return_sequences=True, input_shape=(window, 1)))
    model.add(GRU(32))
else:
    raise ValueError("Невідомий тип моделі. Використай 'LSTM' або 'GRU'.")

model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=15, batch_size=32, verbose=1)


# --------------------------
# 4. Прогноз на 10 годин вперед
# --------------------------
future_steps = 10
last_window = scaled[-window:]
predictions = []

current_input = last_window.reshape(1, window, 1)

for _ in range(future_steps):
    pred = model.predict(current_input)[0][0]
    predictions.append(pred)

    current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

forecast_values = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
).flatten()

# Часовий індекс
last_time = df.index[-1]
forecast_index = pd.date_range(
    start=last_time + pd.Timedelta(hours=1), periods=future_steps, freq="H"
)

forecast_series = pd.Series(forecast_values, index=forecast_index)

print("📈 Погодинний прогноз на 10 годин:")
print(forecast_series)


# --------------------------
# 5. Графік
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["price"], label="Historical (hourly)")
plt.plot(
    forecast_series, label=f"{model_type} Forecast (next 10 hours)", linestyle="--"
)
plt.title(f"BTC Hour-to-Hour Forecast ({model_type} Neural Network)")
plt.xlabel("Time (hourly)")
plt.ylabel("USD")
plt.legend()
plt.grid()

output_file = f"btc_{model_type.lower()}_forecast.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Графік збережено у {output_file}")

plt.show()
