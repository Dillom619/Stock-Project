# src/train_xgboost.py
import argparse, os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)  # ignored
parser.add_argument("--batch_size", type=int, default=32)  # ignored
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--tickers", nargs="*", default=None)
args = parser.parse_args()

DATA_PATH = "data/stocks_clean.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
tickers = args.tickers if args.tickers else sorted(df["Ticker"].unique())

summary = []
for ticker in tickers:
    stock = df[df["Ticker"]==ticker].sort_values("Date")
    if len(stock) < 20:
        print(f"Skip {ticker} for XGBoost (too short)")
        continue

    X = stock[["Open","High","Low","Volume"]].fillna(method="ffill").values
    y = stock["Close"].values

    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=args.learning_rate, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    summary.append([ticker, "XGBoost", mse, rmse, mae])

    out = pd.DataFrame({"Date": stock["Date"].iloc[-len(y_test):].values, "y_test": y_test, "preds": preds})
    out.to_csv(os.path.join(OUT_DIR, f"XGB_{ticker}_preds.csv"), index=False)
    print(f"XGBoost {ticker} done. RMSE={rmse:.3f}")

pd.DataFrame(summary, columns=["Ticker","Model","MSE","RMSE","MAE"]).to_csv(os.path.join(OUT_DIR, "summary_xgb.csv"), index=False)
