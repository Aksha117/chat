import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import joblib
import os
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# ✅ Compute Technical Indicators
def compute_indicators(df):
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    return df.ffill().bfill()

# ✅ Fetch News Sentiment
def fetch_sentiment(symbol):
    try:
        url = f"https://www.bing.com/news/search?q={symbol}+stock"
        soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla'}).text, 'html.parser')
        headlines = [h.text for h in soup.find_all("a") if h.text.strip()]
        return np.mean([TextBlob(h).sentiment.polarity for h in headlines]) if headlines else 0
    except:
        return 0  # Neutral if sentiment can't be fetched

# ✅ Load or Train XGBoost Model
def get_or_train_xgboost(symbol, df, features):
    xgb_path = f"models/{symbol}_xgb.pkl"

    # If XGBoost model exists, load it
    if os.path.exists(xgb_path):
        return joblib.load(xgb_path)

    print(f"⚠️ XGBoost model not found for {symbol}. Training now...")

    # Prepare training data
    df['Sentiment'] = fetch_sentiment(symbol)
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaled_features = scaler_X.fit_transform(df[features])
    scaled_target = scaler_y.fit_transform(df[['Close']])

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target, test_size=0.2, shuffle=False)

    # Train XGBoost model
    model_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model_xgb.fit(X_train, y_train.ravel())

    # Save model & scalers
    joblib.dump(model_xgb, xgb_path)
    joblib.dump(scaler_X, f"models/{symbol}_scaler_X.pkl")
    joblib.dump(scaler_y, f"models/{symbol}_scaler_y.pkl")

    print(f"✅ XGBoost Model trained and saved for {symbol}")
    return model_xgb

# ✅ Load LSTM Model
def load_lstm_model(symbol):
    model_path = f"models/{symbol}_lstm.h5"
    return load_model(model_path, compile=False) if os.path.exists(model_path) else None

# ✅ Predict Stock Price
def predict_stock(symbol):
    print(f"📈 Fetching data for {symbol}...")
    df = yf.download(symbol, period="70d")[['Close', 'Volume']]
    
    if df.empty:
        return {"error": f"No data found for {symbol}"}

    df = compute_indicators(df)
    df['Sentiment'] = fetch_sentiment(symbol)

    features = ['Close', 'Volume', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal', 'Sentiment']

    # Load LSTM & XGBoost
    model_lstm = load_lstm_model(symbol)
    model_xgb = get_or_train_xgboost(symbol, df, features)

    scaler_X = joblib.load(f"models/{symbol}_scaler_X.pkl")
    scaler_y = joblib.load(f"models/{symbol}_scaler_y.pkl")

    # Scale features
    scaled_features = scaler_X.transform(df[features])
    sequence = scaled_features[-60:].reshape(1, 60, len(features))

    # Use LSTM if available, else fallback to XGBoost
    if model_lstm:
        lstm_pred_scaled = model_lstm.predict(sequence)
        predicted_price = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1))[0, 0]
        model_used = "LSTM"
    else:
        predicted_price = scaler_y.inverse_transform(model_xgb.predict(scaled_features[-1].reshape(1, -1)).reshape(-1, 1))[0, 0]
        model_used = "XGBoost"

    return {
        "symbol": symbol,
        "predicted_price": round(predicted_price, 2),
        "model_used": model_used
    }

# ✅ API Route
@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "Stock symbol is required!"})

    prediction = predict_stock(symbol)
    return jsonify(prediction)

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
