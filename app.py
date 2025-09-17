import os
import subprocess
import pandas as pd
import streamlit as st

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="📊 Stock Forecast Dashboard", layout="wide")
st.title("📊 Stock Forecast Dashboard")

DATA_DIR = "models/"
PLOT_DIR = os.path.join(DATA_DIR, "plots")

# Assign colors for model types
MODEL_COLORS = {
    "LSTM": "#1f77b4",
    "XGBoost": "#ff7f0e",
    "CNN-LSTM": "#2ca02c",
    "CNN-LSTM + Sentiment": "#9467bd"
}

# ---------------------
# MODEL NAME DETECTION
# ---------------------
def get_model_name(filename: str) -> str:
    fname = filename.lower()
    if "sent" in fname:
        return "CNN-LSTM + Sentiment"
    elif "cnn" in fname:
        return "CNN-LSTM"
    elif "xgb" in fname:
        return "XGBoost"
    elif "lstm" in fname:
        return "LSTM"
    else:
        return "Unknown"

# ---------------------
# SIDEBAR (Hyperparams + Retrain)
# ---------------------
st.sidebar.header("⚙️ Training Controls")

epochs = st.sidebar.number_input("Epochs", 1, 200, 10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128, 256], index=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.0005, 0.001, 0.005, 0.01], index=1)

st.sidebar.markdown(f"""
**Current Settings**
- Epochs: `{epochs}`
- Batch Size: `{batch_size}`
- Learning Rate: `{learning_rate}`
""")

retrain = st.sidebar.button("🚀 Retrain All Models")

if retrain:
    st.sidebar.success("Training started... check terminal logs.")
    cmd = [
        "python", "src/train_all.py",
        f"--epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--learning_rate={learning_rate}"
    ]
    subprocess.Popen(cmd)
    st.stop()

# ---------------------
# LOAD PREDICTIONS
# ---------------------
files = [f for f in os.listdir(DATA_DIR) if f.endswith("_preds.csv")]

if not files:
    st.warning("⚠️ No predictions found. Run training first.")
else:
    tickers = sorted(set(f.split("_")[1] for f in files))
    selected_ticker = st.selectbox("Choose Ticker", tickers)

    models = [f for f in files if f"_{selected_ticker}_" in f]

    for m in models:
        model_name = get_model_name(m)
        color = MODEL_COLORS.get(model_name, "gray")

        st.markdown(
            f"<h3 style='color:{color};'>📌 {model_name} – {selected_ticker}</h3>",
            unsafe_allow_html=True
        )

        df = pd.read_csv(os.path.join(DATA_DIR, m), parse_dates=["Date"]).sort_values("Date")

        # --- metrics ---
        mse = ((df["y_test"] - df["preds"]) ** 2).mean()
        mae = (df["y_test"] - df["preds"]).abs().mean()
        rmse = mse ** 0.5

        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAE", f"{mae:.2f}")

        # --- interactive chart ---
        st.line_chart(df.set_index("Date")[["y_test", "preds"]])

        # --- optional saved plots ---
        plot_path = os.path.join(PLOT_DIR, f"{selected_ticker}_{model_name}_pred.png")
        if os.path.exists(plot_path):
            st.image(plot_path, caption=f"{model_name} Predictions", use_container_width=True)

    # --- summary RMSE comparison ---
    rmse_path = os.path.join(PLOT_DIR, "RMSE_comparison.png")
    if os.path.exists(rmse_path):
        st.subheader("📉 RMSE Comparison Across Models")
        st.image(rmse_path, use_container_width=True)
