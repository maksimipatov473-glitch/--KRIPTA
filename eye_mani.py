import matplotlib.pyplot as plt
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA


# --------------------------
# 1. Завантаження даних (BTC)
# --------------------------
def load_btc_data(days=180):
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
    df.set_index("timestamp", inplace=True)
    return df


df = load_btc_data(180)

# --------------------------
# 2. ARIMA модель
# --------------------------
model = ARIMA(df["price"], order=(5, 1, 2))
model_fit = model.fit()

# --------------------------
# 3. Прогноз на 10 годин вперед
# --------------------------
steps = 10  # кількість годин наперед
forecast = model_fit.forecast(steps=steps)

# Новий часовий індекс: кожна година
last_time = df.index[-1]
forecast_index = pd.date_range(
    start=last_time + pd.Timedelta(hours=1), periods=steps, freq="H"
)

forecast = pd.Series(forecast.values, index=forecast_index, name="predicted_mean")

print("📈 Прогноз на 10 годин:")
print(forecast)

# --------------------------
# 4. Побудова графіка
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["price"], label="Historical price")
plt.plot(forecast, label="Forecast (10 hours)", linestyle="--")
plt.title("BTC Price Forecast (Next 10 Hours)")
plt.xlabel("Time")
plt.ylabel("USD")
plt.legend()
plt.grid()

output_file = "btc_forecast_10_hours.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Графік збережено у {output_file}")

plt.show()
