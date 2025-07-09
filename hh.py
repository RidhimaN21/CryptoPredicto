import ccxt
import pandas as pd

# Initialize the exchange
exchange = ccxt.binance()

# Fetch historical data
symbol = 'ETH/USDT'
timeframe = '1h'
since = exchange.parse8601('2021-01-01T00:00:00Z')
limit = 1000
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

# Convert to DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
data = pd.DataFrame(ohlcv, columns=columns)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

# Save to CSV
data.to_csv('Ethereum.csv', index=False)