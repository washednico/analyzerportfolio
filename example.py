import yfinance as yf

exchange_rate_data = yf.download("EURUSD=X", period='1d', start ="2024-01-01", end=None)
last_value = exchange_rate_data['Close'].iloc[-1]
print(last_value)

