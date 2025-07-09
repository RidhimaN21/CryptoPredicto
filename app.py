import numpy as np
import pandas as pd
import ccxt
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

model = load_model(r"C:\Users\chira\crypto_prediction_model.keras")

def fetch_crypto_data(symbol, timeframe='1d', limit=1000):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET'])
def predict():
    crypto_pair = request.args.get('crypto_pair') 

    if not crypto_pair:
        return render_template('index.html', error="Cryptocurrency pair is required")

    data = fetch_crypto_data(crypto_pair)
    if data.empty or len(data) < 101:
        return render_template('index.html', error="Insufficient data for prediction")

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['close']])

   
    x_test, y_actual = [], []
    for i in range(100, len(data_scaled) - 1):
        x_test.append(data_scaled[i - 100:i])
        y_actual.append(data_scaled[i + 1, 0])

    x_test = np.array(x_test)
    y_actual = np.array(y_actual)

    if x_test.size == 0:
        return render_template('index.html', error="Insufficient data for prediction")

    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_actual = scaler.inverse_transform(y_actual.reshape(-1, 1))

 
    future_input = data_scaled[-100:].reshape(1, 100, 1)  
    future_predictions = []

    for _ in range(5): 
        future_pred = model.predict(future_input)
        future_predictions.append(scaler.inverse_transform(future_pred)[0][0])

        
        future_input = np.append(future_input[:, 1:, :], future_pred.reshape(1, 1, 1), axis=1)

    
    future_dates = pd.date_range(data['timestamp'].iloc[-1] + pd.Timedelta(days=1), periods=5).strftime('%Y-%m-%d').tolist()

  
    plt.figure(figsize=(10, 6))
    plt.plot(data['close'][100:].values, label='Original Price', color='cyan', linewidth=1)
    plt.plot(predictions, label='Predicted Price', color='red', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

   
    table_data = pd.DataFrame({
        'Time': data['timestamp'].iloc[-10:].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'Actual Price': y_actual[-10:].flatten().tolist(),
        'Predicted Price': predictions[-10:].flatten().tolist()
    })


    future_table_data = pd.DataFrame({
        'Time': future_dates,
        'Predicted Price': future_predictions
    })

    return render_template(
        'index.html',
        crypto_pair=crypto_pair,
        mse=float(mean_squared_error(y_actual, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_actual, predictions))),
        accuracy=100 - float(mean_absolute_percentage_error(y_actual, predictions) * 100),
        predicted_price=float(predictions[-1][0]),
        graph_url=f"data:image/png;base64,{graph_url}",
        table_data=table_data.to_dict(orient='records'),
        future_table_data=future_table_data.to_dict(orient='records')
    )

if __name__ == '__main__':
    app.run(debug=True)
