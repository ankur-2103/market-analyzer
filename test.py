import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import ta.momentum
import ta.trend
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

os.makedirs("models_online", exist_ok=True)

top_10_nse = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "HDFCBANK.NS",
              "LT.NS", "SBIN.NS", "ITC.NS", "KOTAKBANK.NS", "BHARTIARTL.NS"]

def train_online_model(ticker):
    print(f"\n🔁 Training online model for {ticker}...")

    df = yf.download(ticker, start="2020-01-01", progress=False)
    if df is None or df.empty:
        print("❌ No data available.")
        return

    df['return'] = df['Close'].pct_change()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()

    close_series = df['Close'].squeeze()
    df['rsi'] = ta.momentum.RSIIndicator(close_series).rsi()
    macd = ta.trend.MACD(close_series)
    df['macd'] = macd.macd().squeeze()
    df['macd_signal'] = macd.macd_signal().squeeze()

    df.dropna(inplace=True)

    features = ['return', 'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal']
    target = 'Close'
    X = df[features]
    y = df[target]

    model_path = f"models_online/online_{ticker}.pkl"
    if os.path.exists(model_path):
        loaded = joblib.load(model_path)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, scaler = loaded
            # Check if feature count matches
            try:
                model.n_features_in_
                if model.n_features_in_ != X.shape[1]:
                    print("⚠️ Feature count changed. Reinitializing model.")
                    raise ValueError
            except:
                model = SGDRegressor(max_iter=1, tol=1e-3, learning_rate="constant", eta0=0.001, warm_start=True)
                scaler = StandardScaler()
        else:
            print("⚠️ Old model format invalid. Reinitializing.")
            model = SGDRegressor(max_iter=1, tol=1e-3, learning_rate="constant", eta0=0.001, warm_start=True)
            scaler = StandardScaler()
        print("📦 Loaded existing model.")
    else:
        model = SGDRegressor(max_iter=1, tol=1e-3, learning_rate="constant", eta0=0.001, warm_start=True)
        scaler = StandardScaler()
        print("🆕 Initialized new model & scaler.")


    y_preds, y_actuals = [], []

    for i in range(60, len(X)):  # start from 60 for stable rolling features
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_now = X.iloc[i:i+1]

        model.fit(X_train, y_train)  # Full fit each time (SGD will still be fast)
        y_pred = model.predict(X_now)[0]

        y_preds.append(y_pred)
        y_actuals.append(y.iloc[i])

    # Plot
    plt.figure(figsize=(50, 5))
    plt.plot(y_actuals, label="Actual", linewidth=2)
    plt.plot(y_preds, label="Predicted", color="orange")
    plt.title(f"{ticker} Price Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    joblib.dump((model, scaler), model_path)
    print(f"✅ Model saved. Plot saved to models_online/{ticker}_plot.png")

    rmse = mean_squared_error(y_actuals, y_preds)
    print(f"📊 RMSE: {rmse:.2f}")

# Train all
for ticker in top_10_nse:
    train_online_model(ticker)
