import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# ✅ Fetch Sentiment Score
def sentiment(symbol):
    url = f"https://www.bing.com/news/search?q={symbol}+stock"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
        headlines = [h.text for h in soup.find_all("a") if h.text.strip()]
        return np.mean([TextBlob(h).sentiment.polarity for h in headlines]) if headlines else 0
    except:
        return 0  # If request fails, return neutral sentiment

# ✅ Compute Technical Indicators
def indicators(df):
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() /
                            -df['Close'].diff().clip(upper=0).rolling(14).mean()))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df.fillna(method="bfill", inplace=True)
    return df

# ✅ Load or Train XGBoost if missing
def train_xgboost(symbol, df, features):
    print(f"⚠️ XGBoost not found for {symbol}. Training now...")
    
    X, y = df[features], df['Close']
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
    xgb_model.fit(X_scaled, y_scaled)
    
    joblib.dump(xgb_model, f"models/{symbol}_xgb.pkl")
    joblib.dump(scaler_X, f"models/{symbol}_scaler_X.pkl")
    joblib.dump(scaler_y, f"models/{symbol}_scaler_y.pkl")
    
    print(f"✅ XGBoost trained & saved for {symbol}.")
    return xgb_model, scaler_X, scaler_y

# ✅ Main Prediction Function
def predict(symbol):
    print(f"📈 Fetching data for {symbol}...")
    df = yf.download(symbol, period="180d")[['Close', 'Volume']]
    
    if df.empty:
        print(f"❌ No data found for {symbol}. Check the ticker.")
        return

    df = indicators(df)
    df['Sentiment'] = sentiment(symbol)

    features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'Signal', 'Sentiment']
    model_path = f"models/{symbol}_lstm.h5"
    xgb_path = f"models/{symbol}_xgb.pkl"

    # 🚀 Check if enough data is available
    if len(df) < 60:
        print("⚠️ Not enough historical data. Using XGBoost.")
        if not os.path.exists(xgb_path):
            xgb_model, scaler_X, scaler_y = train_xgboost(symbol, df, features)
        else:
            xgb_model = joblib.load(xgb_path)
            scaler_X = joblib.load(f"models/{symbol}_scaler_X.pkl")
            scaler_y = joblib.load(f"models/{symbol}_scaler_y.pkl")

        valid_features = [f for f in features if f in df.columns]
        scaled_features = scaler_X.transform(df[valid_features][-1:].values)

        pred_scaled = xgb_model.predict(scaled_features)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        print(f"🔮 **Predicted Closing Price for {symbol}:** ₹{pred:.2f} (XGBoost Model)")
        return

    # 🚀 Fallback Mechanism: Use LSTM if available, else try XGBoost
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            scaler_X = joblib.load(f"models/{symbol}_scaler_X.pkl")
            scaler_y = joblib.load(f"models/{symbol}_scaler_y.pkl")

            valid_features = [f for f in features if f in df.columns]
            scaled_features = scaler_X.transform(df[valid_features][-60:].values)
            sequence = scaled_features.reshape(1, 60, len(valid_features))

            pred_scaled = model.predict(sequence)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]
            print(f"🔮 **Predicted Closing Price for {symbol}:** ₹{pred:.2f} (LSTM Model)")
            return
        except Exception as e:
            print(f"⚠️ LSTM error: {e}. Falling back to XGBoost...")

    # 🚀 Use XGBoost if LSTM fails
    if os.path.exists(xgb_path):
        try:
            xgb_model = joblib.load(xgb_path)
            scaler_X = joblib.load(f"models/{symbol}_scaler_X.pkl")
            scaler_y = joblib.load(f"models/{symbol}_scaler_y.pkl")

            valid_features = [f for f in features if f in df.columns]
            scaled_features = scaler_X.transform(df[valid_features][-1:].values)

            pred_scaled = xgb_model.predict(scaled_features)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            print(f"🔮 **Predicted Closing Price for {symbol}:** ₹{pred:.2f} (XGBoost Model)")
            return
        except Exception as e:
            print(f"⚠️ XGBoost error: {e}. Training XGBoost...")

    # 🚀 If XGBoost is also missing, train it on the spot
    xgb_model, scaler_X, scaler_y = train_xgboost(symbol, df, features)
    valid_features = [f for f in features if f in df.columns]
    scaled_features = scaler_X.transform(df[valid_features][-1:].values)

    pred_scaled = xgb_model.predict(scaled_features)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    print(f"🔮 **Predicted Closing Price for {symbol}:** ₹{pred:.2f} (Newly Trained XGBoost)")

# ✅ Run Prediction
if __name__ == "__main__":
    symbol = input("Enter Stock Symbol: ").strip().upper()
    predict(symbol)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Stock Prediction API is Running!"

@app.route('/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get('symbol', 'ITC.NS')  # Default to ITC.NS if no symbol provided
    # Run your prediction logic here
    predicted_price = 412.03  # Replace with your actual model's prediction

    return jsonify({
        "stock": stock_symbol,
        "predicted_price": predicted_price
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
