import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
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

def prepare_lstm_data(df, lookahead=5, threshold=0.0003, window=30):
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
    X_raw = df[features].values
    y_raw = df["Target"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_seq, y_seq = [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i-window:i])
        y_seq.append(y_raw[i])
    return np.array(X_seq), np.array(y_seq), scaler, features, df

def build_lstm(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = LSTM(256, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model():
    print("📥 Downloading US30 1min data...")
    us30 = yf.download("YM=F", period="7d", interval="1m")
    print(f"✅ Total candles: {len(us30)}")
    us30 = add_indicators(us30)
    X, y, scaler, features, df = prepare_lstm_data(us30.copy())
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print("🧠 Training LSTM...")
    model = build_lstm((X.shape[1], X.shape[2]))
    es = EarlyStopping(monitor="val_loss", patience=15,
                       restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100,
              batch_size=64, validation_split=0.1,
              callbacks=[es], verbose=0)
    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc*100:.2f}%")
    return model, scaler, features, df, X

def get_signal(model, X):
    latest = X[-1].reshape(1, X.shape[1], X.shape[2])
    prob = model.predict(latest, verbose=0)[0][0]
    signal = "BUY" if prob > 0.5 else "SELL"
    confidence = prob if prob > 0.5 else 1 - prob
    return signal, confidence

def run_robot():
    print("\n🤖 US30 AI ROBOT STARTING...")
    print("="*40)

    # Train model
    model, scaler, features, df, X = train_model()

    # Retrain every 24 hours
    last_train = time.time()

    while True:
        try:
            # Retrain every 24 hours
            if time.time() - last_train > 86400:
                print("\n🔄 Retraining model with fresh data...")
                model, scaler, features, df, X = train_model()
                last_train = time.time()

            # Get signal
            signal, confidence = get_signal(model, X)

            # Get price info
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

            # Wait 1 minute before next signal
            time.sleep(60)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_robot()
