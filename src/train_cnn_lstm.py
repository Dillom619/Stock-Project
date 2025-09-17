# src/train_cnn_lstm.py
import argparse, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--tickers", nargs="*", default=None)
args = parser.parse_args()

DATA_PATH = "data/stocks_clean.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 60

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
tickers = args.tickers if args.tickers else sorted(df["Ticker"].unique())

summary = []
for ticker in tickers:
    stock = df[df["Ticker"]==ticker].sort_values("Date")
    if len(stock) < SEQ_LEN + 10:
        print(f"Skip {ticker} for CNN-LSTM (too short)")
        continue

    features = stock[["Close"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i, 0])
        y.append(scaled[i,0])
    X = np.array(X).reshape(-1, SEQ_LEN, 1)
    y = np.array(y)

    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(SEQ_LEN,1)),
        MaxPooling1D(2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)

    preds = model.predict(X_test).flatten().reshape(-1,1)
    preds_inv = scaler.inverse_transform(preds).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    mse = mean_squared_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, preds_inv)
    summary.append([ticker, "CNN-LSTM", mse, rmse, mae])

    out = pd.DataFrame({"Date": stock["Date"].iloc[-len(y_test):].values, "y_test": y_test_inv, "preds": preds_inv})
    out.to_csv(os.path.join(OUT_DIR, f"CNNLSTM_{ticker}_preds.csv"), index=False)
    print(f"CNN-LSTM {ticker} done. RMSE={rmse:.3f}")

pd.DataFrame(summary, columns=["Ticker","Model","MSE","RMSE","MAE"]).to_csv(os.path.join(OUT_DIR, "summary_cnn.csv"), index=False)
