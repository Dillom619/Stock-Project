import argparse
import subprocess
import shlex

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--tickers", nargs="*", default=None, help="Optional list of tickers to train (default: all in data)")
args = parser.parse_args()

scripts = [
    "train_lstm.py",
    "train_xgboost.py",
    "train_cnn_lstm.py",
    "train_cnn_lstm_sentiment.py"
]

common_args = f"--epochs {args.epochs} --batch_size {args.batch_size} --learning_rate {args.learning_rate}"
if args.tickers:
    tickers_arg = " --tickers " + " ".join(args.tickers)
else:
    tickers_arg = ""

for s in scripts:
    cmd = f"python src/{s} {common_args}{tickers_arg}"
    print("Running:", cmd)
    subprocess.run(shlex.split(cmd), check=True)
print("All training scripts finished.")
