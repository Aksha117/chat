import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

app = Flask(__name__)

# ✅ Function to compute technical indicators
def compute_indicators(df):
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df.fillna(method="bfill", inplace=True)
    return df

# ✅ Fetch real-time sentiment score
def fetch_sentiment(symbol):
    try:
        url = f"https://www.bing.com/news/search?q={symbol}+stock"
        soup = BeautifulSoup(requests.get(url, headers={'User-Agent':'Mozilla'}).text, 'html.parser')
        headlines = [h.text for h in soup.find_all("a") if h.text.strip()]
        return np.mean([TextBlob(h).sentiment.polarity for h in headlines]) if headlines else 0
    except Exception as e:
        print(f"Sentiment fetch error: {e}")
        return 0  # Default sentiment score

# ✅ Fetch stock data from Yahoo Finance
def fetch_stock_data(symbol):
    df = yf.download(symbol, period='70d')
    if df.empty:
        raise ValueError("Stock data not available.")

    df = compute_indicators(df)
    df['Sentiment'] = fetch_sentiment(symbol)

    return df

# ✅ Train XGBoost model if missing
def train_xgboost(symbol, df):
    print(f"⚙️ Training XGBoost model for {symbol}...")
    features = ['Close', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Sentiment']
    df = df[features].dropna()

    X, y = df.iloc[:-1], df.iloc[1:]['Close']
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200)
    model.fit(X, y)

    joblib.dump(model, f"models/{symbol}_xgb.pkl")
    print(f"✅ XGBoost model trained for {symbol}.")

# ✅ Prediction API
@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "Stock symbol is required."})

    try:
        df = fetch_stock_data(symbol)
        features = ['Close', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Sentiment']
        df = df[features].dropna()

        model_path_lstm = f"models/{symbol}_lstm.h5"
        model_path_xgb = f"models/{symbol}_xgb.pkl"

        if os.path.exists(model_path_lstm):
            print(f"🔮 Using LSTM model for {symbol}...")
            lstm_model = load_model(model_path_lstm, compile=False)

            # ✅ Properly fit scalers before transforming data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            scaler_X.fit(df)
            scaler_y.fit(df[['Close']])

            scaled_features = scaler_X.transform(df)
            sequence = scaled_features[-60:].reshape(1, 60, len(features))

            lstm_pred_scaled = lstm_model.predict(sequence)
            lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1))[0, 0]
            predicted_price = float(lstm_pred)  # Convert to Python float
            model_used = "LSTM Model"

        elif os.path.exists(model_path_xgb):
            print(f"⚠️ LSTM not found. Using XGBoost...")
            xgb_model = joblib.load(model_path_xgb)
            predicted_price = float(xgb_model.predict(df.iloc[-1:].values.reshape(1, -1))[0])  # Convert to Python float
            model_used = "XGBoost Model"

        else:
            print(f"⚠️ No trained models found. Training XGBoost on the spot...")
            train_xgboost(symbol, df)
            xgb_model = joblib.load(model_path_xgb)
            predicted_price = float(xgb_model.predict(df.iloc[-1:].values.reshape(1, -1))[0])  # Convert to Python float
            model_used = "Newly Trained XGBoost Model"

        return jsonify({
            "symbol": symbol,
            "current_price": float(df['Close'].iloc[-1]),  # Convert to Python float
            "predicted_price": round(predicted_price, 2),
            "model_used": model_used,
            "technical_indicators": {
                "SMA_50": round(float(df['SMA_50'].iloc[-1]), 2),
                "EMA_50": round(float(df['EMA_50'].iloc[-1]), 2),
                "RSI": round(float(df['RSI'].iloc[-1]), 2),
                "MACD": round(float(df['MACD'].iloc[-1]), 2),
                "Signal": round(float(df['Signal'].iloc[-1]), 2),
                "Sentiment": round(float(df['Sentiment'].iloc[-1]), 4),
            }
        })

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return jsonify({"error": "Prediction failed. Try again later."})

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
