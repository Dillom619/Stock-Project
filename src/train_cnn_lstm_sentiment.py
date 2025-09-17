# src/train_cnn_lstm_sentiment.py
import argparse, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
NEWS_PATH = "data/financial_news_events.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 60

# map company names to tickers - extend as needed
company_to_ticker = {
    "Apple Inc.":"AAPL","Microsoft":"MSFT","Tesla":"TSLA","JP Morgan Chase":"JPM",
    "Goldman Sachs":"GS","Meta":"META","Netflix":"NFLX","Amazon":"AMZN",
    "Intel":"INTC","Nvidia":"NVDA","Cisco":"CSCO","Disney":"DIS","Oracle":"ORCL",
    "Pepsi":"PEP","Visa":"V","Mastercard":"MA","IBM":"IBM","AMD":"AMD","Paypal":"PYPL",
    "Bank of America":"BAC","Boeing":"BA","ExxonMobil":"XOM","Samsung Electronics":"SSNLF"
}

# load data
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
tickers = args.tickers if args.tickers else sorted(df["Ticker"].unique())

# load news if available
if os.path.exists(NEWS_PATH):
    news = pd.read_csv(NEWS_PATH, parse_dates=["Date"])
    news["Related_Company"] = news["Related_Company"].astype(str).str.strip()
    news["Ticker"] = news["Related_Company"].map(company_to_ticker)
    # if Sentiment column missing or empty, fill with Neutral
    if "Sentiment" not in news.columns:
        news["Sentiment"] = "Neutral"
    news["Sentiment"] = news["Sentiment"].fillna("Neutral")
    # numeric mapping
    sent_map = {"Positive":1, "Neutral":0, "Negative":-1}
    news["Sentiment_Num"] = news["Sentiment"].map(sent_map).fillna(0)
    # impact numeric
    impact_map = {"Low":1,"Medium":2,"High":3}
    if "Impact_Level" not in news.columns:
        news["Impact_Level"] = "Low"
    news["Impact_Num"] = news["Impact_Level"].map(impact_map).fillna(1)
else:
    news = None

summary = []
for ticker in tickers:
    stock = df[df["Ticker"]==ticker].sort_values("Date").reset_index(drop=True)
    if news is None:
        print(f"No news file found - skipping sentiment model for {ticker}")
        continue

    # map ticker in news
    tnews = news[news["Ticker"]==ticker].sort_values("Date").reset_index(drop=True)
    if tnews.empty:
        print(f"No news for {ticker} - skipping sentiment model")
        continue

    # do a forward-fill merge_asof to align most recent news before each stock date
    merged = pd.merge_asof(stock.sort_values("Date"), tnews[["Date","Sentiment_Num","Impact_Num"]].sort_values("Date"),
                           on="Date", direction="backward")
    # fill missing sentiment/impact with neutral/low
    merged["Sentiment_Num"] = merged["Sentiment_Num"].fillna(0)
    merged["Impact_Num"] = merged["Impact_Num"].fillna(1)

    # build feature matrix: Close + Sentiment + Impact
    feat_cols = ["Close","Sentiment_Num","Impact_Num"]
    features = merged[feat_cols].values
    if len(merged) < SEQ_LEN + 10:
        print(f"Too little merged data for {ticker} - skipping")
        continue

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # create sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i, :])    # shape (seq_len, n_features)
        y.append(scaled[i, 0])  # scaled close
    X = np.array(X)
    y = np.array(y)

    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(SEQ_LEN, X.shape[2])),
        MaxPooling1D(2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)

    preds = model.predict(X_test).flatten()
    # preds and y_test are scaled close values; inverse transform only close dimension
    # build dummy arrays to inverse_transform
    dummy_pred = np.zeros((len(preds), X.shape[2]))
    dummy_pred[:,0] = preds
    preds_inv = scaler.inverse_transform(dummy_pred)[:,0]

    dummy_y = np.zeros((len(y_test), X.shape[2]))
    dummy_y[:,0] = y_test
    y_inv = scaler.inverse_transform(dummy_y)[:,0]

    mse = mean_squared_error(y_inv, preds_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_inv, preds_inv)
    summary.append([ticker, "CNN-LSTM + Sentiment", mse, rmse, mae])

    out = pd.DataFrame({"Date": merged["Date"].iloc[-len(y_test):].values, "y_test": y_inv, "preds": preds_inv})
    out.to_csv(os.path.join(OUT_DIR, f"SentCNNLSTM_{ticker}_preds.csv"), index=False)
    print(f"Sentiment CNN-LSTM {ticker} done. RMSE={rmse:.3f}")

pd.DataFrame(summary, columns=["Ticker","Model","MSE","RMSE","MAE"]).to_csv(os.path.join(OUT_DIR, "summary_sentcnn.csv"), index=False)
