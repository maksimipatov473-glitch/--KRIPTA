import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.models import Sequential

# -------------------------------------------
# ✦ Налаштування Binance API
# -------------------------------------------
API_KEY = ""
API_SECRET = ""
client = Client(API_KEY, API_SECRET)

# -------------------------------------------
# ✦ Вибір користувача
# -------------------------------------------
symbol = input("Введіть криптовалюту (наприклад BTCUSDT): ")

print("Виберіть таймфрейм:\n1) 1m\n2) 5m\n3) 1H\n4) 1D")
tf_choice = input("Ваш вибір: ")

timeframes = {"1": "1m", "2": "5m", "3": "1h", "4": "1d"}
interval = timeframes.get(tf_choice, "1h")

print("Виберіть модель:\n1) LSTM\n2) GRU")
model_choice = input("Ваш вибір: ")

predict_steps = int(input("На скільки кроків зробити прогноз (напр. 60 = 60 хвилин): "))

# -------------------------------------------
# ✦ Завантаження даних з Binance
# -------------------------------------------
print("Завантаження даних...")

klines = client.get_klines(symbol=symbol, interval=interval, limit=2000)
df = pd.DataFrame(
    klines,
    columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ],
)

df["close"] = df["close"].astype(float)
data = df["close"].to_numpy().reshape(-1, 1)  # Використовуйте to_numpy()
# ...existing code...

# Нормалізація
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# -------------------------------------------
# ✦ Формування навчальних даних
# -------------------------------------------
window = 50
X = np.array([scaled[i - window : i, 0] for i in range(window, len(scaled))])
y = np.array([scaled[i, 0] for i in range(window, len(scaled))])
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# -------------------------------------------
# ✦ Побудова моделі
# -------------------------------------------
model = Sequential()

if model_choice == "1":
    print("\nВикористовується модель: LSTM\n")
    model.add(LSTM(64, return_sequences=False, input_shape=(window, 1)))
else:
    print("\nВикористовується модель: GRU\n")
    model.add(GRU(64, return_sequences=False, input_shape=(window, 1)))

model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Навчання
print("Навчання моделі...")
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# -------------------------------------------
# ✦ Прогноз "від часу до часу"
# -------------------------------------------
last_sequence = scaled[-window:]
future_predictions = []

current_input = last_sequence

for _ in range(predict_steps):
    pred = model.predict(current_input.reshape(1, window, 1), verbose=0)
    future_predictions.append(pred[0][0])
    current_input = np.vstack((current_input[1:], pred))

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

# -------------------------------------------
# ✦ Побудова графіка
# -------------------------------------------
close_data = df["close"].to_numpy()[-300:]

plt.figure(figsize=(12, 6))
plt.plot(close_data, label="Історія")
plt.plot(
    range(len(close_data), len(close_data) + predict_steps),
    future_predictions,
    label="Прогноз",
)
plt.title(f"{symbol} Прогноз ({interval})")
plt.legend()
plt.grid()
plt.show()
