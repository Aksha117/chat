from flask import Flask, request, jsonify
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# ✅ Load Models & Scalers (Pretrained LSTM & XGBoost)
try:
    lstm_model = load_model("ITC.NS_model.h5")  # Update if using a different model
    scaler_X = joblib.load("ITC.NS_scaler_X.pkl")
    scaler_y = joblib.load("ITC.NS_scaler_y.pkl")
    xgb_model = joblib.load("ITC.NS_xgb.pkl")
except Exception as e:
    print(f"⚠️ Model Loading Error: {e}")
    lstm_model, scaler_X, scaler_y, xgb_model = None, None, None, None

# ✅ Fetch stock data
def get_stock_data(symbol):
    try:
        df = yf.download(symbol, period="60d", interval="1d")
        df = df[['Close', 'Volume']].dropna()
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

# ✅ Compute technical indicators
def compute_indicators(df):
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df.fillna(method="bfill").fillna(method="ffill")

# ✅ Fetch news sentiment
def get_sentiment(symbol):
    url = f"https://www.bing.com/news/search?q={symbol}+stock"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [h.text for h in soup.find_all("a") if h.text.strip()]
        sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in headlines]
        return np.mean(sentiment_scores) if sentiment_scores else 0
    except:
        return 0

# ✅ Predict stock price
@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    df = get_stock_data(symbol)
    if df is None or df.empty:
        return jsonify({"error": "Stock data not available"}), 404

    df = compute_indicators(df)
    df["Sentiment"] = get_sentiment(symbol)

    features = ["Close", "Volume", "SMA_50", "EMA_50", "RSI", "MACD", "Signal", "Sentiment"]
    
    try:
        scaled_features = scaler_X.transform(df[features].values[-60:])
        sequence = scaled_features.reshape(1, 60, len(features))
        lstm_pred_scaled = lstm_model.predict(sequence)
        lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1))[0, 0]
    except:
        lstm_pred = None  # LSTM fallback

    try:
        xgb_pred_scaled = xgb_model.predict(scaled_features[-1].reshape(1, -1))
        xgb_pred = scaler_y.inverse_transform(xgb_pred_scaled.reshape(-1, 1))[0, 0]
    except:
        xgb_pred = None  # XGBoost fallback

    return jsonify({
        "symbol": symbol,
        "last_close": df["Close"].iloc[-1],
        "lstm_pred": lstm_pred,
        "xgb_pred": xgb_pred,
        "rsi": df["RSI"].iloc[-1],
        "macd": df["MACD"].iloc[-1],
        "sma_50": df["SMA_50"].iloc[-1],
        "ema_50": df["EMA_50"].iloc[-1],
        "sentiment": df["Sentiment"].iloc[-1],
    })

# ✅ Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Stock Prediction API is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
