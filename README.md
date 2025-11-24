Crypto Predicto — Cryptocurrency Price Forecasting (AI-based)

Crypto Predicto is a small project where I experiment with using deep learning to forecast short-term cryptocurrency price movements.
It uses real historical OHLCV data and a simple LSTM model to learn market patterns, with results displayed through an interactive Streamlit dashboard.
The goal of the project is to explore time-series modeling, data pipelines, and basic deployment—not to provide financial advice.

Features:

Builds and trains an LSTM model for crypto price prediction
Interactive Streamlit dashboard to view charts and run model inference
Automated script for fetching and updating OHLCV data
Real-time visualization of predictions vs. actual market data
Clean, modular code structure for easier experimentation and tuning

Tech Stack:

| Category      | Tools                                  |
| ------------- | -------------------------------------- |
| Language      | Python                                 |
| ML / DL       | TensorFlow, Keras, NumPy, Scikit-learn |
| Data          | Pandas, yfinance / Binance API         |
| Deployment    | Streamlit                              |
| Visualization | Matplotlib, Seaborn                    |
