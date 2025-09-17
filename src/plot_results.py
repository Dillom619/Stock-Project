# src/plot_results.py
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "models"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Collect all _preds.csv files
files = glob.glob(os.path.join(RESULTS_DIR, "*_preds.csv"))
metrics = []
for f in files:
    basename = os.path.basename(f)
    parts = basename.split("_")
    model = parts[0]
    ticker = parts[1]
    df = pd.read_csv(f, parse_dates=["Date"])
    if df.empty:
        continue
    mse = ((df["y_test"] - df["preds"])**2).mean()
    rmse = mse**0.5
    mae = (df["y_test"] - df["preds"]).abs().mean()
    metrics.append({"Ticker": ticker, "Model": model, "MSE": mse, "RMSE": rmse, "MAE": mae})

    # save per-model plot
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], df["y_test"], label="Actual")
    plt.plot(df["Date"], df["preds"], label="Predicted")
    plt.title(f"{ticker} - {model}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{ticker}_{model}_pred.png"))
    plt.close()

if metrics:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(PLOT_DIR, "metrics_summary.csv"), index=False)

    plt.figure(figsize=(12,6))
    sns.barplot(data=metrics_df, x="Ticker", y="RMSE", hue="Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "RMSE_by_ticker_model.png"))
    plt.close()
    print("Plots saved to", PLOT_DIR)
else:
    print("No prediction files found.")
