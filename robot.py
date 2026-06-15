import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings("ignore")

def add_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["Trend"] = (df["MA10"] > df["MA50"]).astype(int)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_change"] = df["RSI"].diff()
    df["RSI_ob"] = (df["RSI"] > 70).astype(int)
    df["RSI_os"] = (df["RSI"] < 30).astype(int)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_cross"] = (
        (df["MACD"] > df["MACD_Signal"]) &
        (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    ).astype(int)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_position"] = (close - bb_lower) / bb_range
    df["BB_width"] = bb_range / bb_mid
    df["BB_squeeze"] = (
        df["BB_width"] < df["BB_width"].rolling(20).mean()
    ).astype(int)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_ratio"] = df["ATR"] / close
    df["Momentum3"] = close.pct_change(3)
    df["Momentum5"] = close.pct_change(5)
    df["Momentum10"] = close.pct_change(10)
    body = (close - df["Open"].squeeze()).abs()
    rng = (high - low).replace(0, np.nan)
    df["Body_ratio"] = body / rng
    df["Upper_wick"] = (high - close) / rng
    df["Lower_wick"] = (close - low) / rng
    df["Volume_ratio"] = volume / volume.rolling(10).mean().replace(0, np.nan)
    df["Volume_surge"] = (df["Volume_ratio"] > 1.5).astype(int)
    df["HH"] = (high > high.rolling(5).max().shift(1)).astype(int)
    df["LL"] = (low < low.rolling(5).min().shift(1)).astype(int)
    df["Price_change"] = close.pct_change()
    df["Volatility"] = close.rolling(10).std()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def prepare_data(df, lookahead=5, threshold=0.0003):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].squeeze()
    future = close.shift(-lookahead)
    pct = (future - close) / close
    df["Target"] = np.where(pct > threshold, 1,
                   np.where(pct < -threshold, 0, np.nan))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    features = [
        "MA10","MA20","MA50","Trend",
        "RSI","RSI_change","RSI_ob","RSI_os",
        "MACD","MACD_Signal","MACD_Hist","MACD_cross",
        "BB_position","BB_width","BB_squeeze",
        "ATR_ratio","Momentum3","Momentum5","Momentum10",
        "Body_ratio","Upper_wick","Lower_wick",
        "Volume_ratio","Volume_surge",
        "HH","LL","Price_change","Volatility"
    ]
    X = df[features].values
    y = df["Target"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, features, df

def train_model():
    print("📥 Downloading US30 1min data...")
    us30 = yf.download("YM=F", period="7d", interval="1m")
    print(f"✅ Total candles: {len(us30)}")
    us30 = add_indicators(us30)
    X, y, scaler, features, df = prepare_data(us30.copy())
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print("🧠 Training AI model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc*100:.2f}%")
    return model, scaler, features, df, X

def get_signal(model, X):
    prob = model.predict_proba(X[-1].reshape(1, -1))[0][1]
    signal = "BUY" if prob > 0.5 else "SELL"
    confidence = prob if prob > 0.5 else 1 - prob
    return signal, confidence

def run_robot():
    print("\n🤖 US30 AI ROBOT STARTING...")
    print("="*40)
    model, scaler, features, df, X = train_model()
    last_train = time.time()

    while True:
        try:
            if time.time() - last_train > 86400:
                print("\n🔄 Retraining with fresh data...")
                model, scaler, features, df, X = train_model()
                last_train = time.time()

            signal, confidence = get_signal(model, X)
            close = df["Close"].squeeze()
            current_price = close.iloc[-1]
            atr = df["ATR"].iloc[-1]
            sl = round(atr * 1.5, 2)
            tp = round(atr * 3.0, 2)

            if signal == "BUY":
                sl_price = round(current_price - sl, 2)
                tp_price = round(current_price + tp, 2)
                emoji = "🟢"
            else:
                sl_price = round(current_price + sl, 2)
                tp_price = round(current_price - tp, 2)
                emoji = "🔴"

            print(f"\n⏰ {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📍 Price      : {current_price:.2f}")
            print(f"📊 Signal     : {emoji} {signal}")
            print(f"🎯 Confidence : {confidence*100:.1f}%")
            print(f"🛑 Stop Loss  : {sl_price:.2f}")
            print(f"✅ Take Profit: {tp_price:.2f}")
            print(f"⚖️  RR Ratio   : 1:2")
            print("-"*40)
            time.sleep(60)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_robot()
