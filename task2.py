
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import yfinance as yf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# 0) Config & Utils
# -------------------------
TICKER = "TSLA"
START  = "2015-07-01"
END    = "2025-07-31"
TRAIN_END = "2023-12-31"
OUT_DIR = "gmf_task2_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # avoid division by zero
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    print(f"{name} — Test Metrics")
    print(f"  MAE : {mae:,.4f}")
    print(f"  RMSE: {rmse:,.4f}")
    print(f"  MAPE: {mp:,.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mp}

# -------------------------
# 1) Load data (Adj Close)
# -------------------------
print("Downloading TSLA data...")
df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=False)
if df.empty:
    raise RuntimeError("No data downloaded for TSLA; check internet or ticker.")

df = df[['Adj Close']].copy()
df.index = pd.to_datetime(df.index)
df.dropna(inplace=True)

# chronological split
train = df.loc[:TRAIN_END].copy()
test = df.loc[datetime.fromisoformat(TRAIN_END) + pd.Timedelta(days=1):].copy()

print(f"Train span: {train.index.min().date()} -> {train.index.max().date()}  (n={len(train)})")
print(f"Test  span: {test.index.min().date()} -> {test.index.max().date()}  (n={len(test)})")

# -------------------------
# 2) ARIMA (auto_arima) — rolling one-step-ahead
# -------------------------
print("\nFitting ARIMA using auto_arima on training data...")
# Allow auto_arima to difference as needed (d); equities rarely need seasonal terms, but we test seasonality lightly
arima_model = auto_arima(
    train['Adj Close'],
    start_p=0, start_q=0, max_p=5, max_q=5,
    d=None,            # let it infer differencing for stationarity
    seasonal=False,    # TSLA daily closes: weak daily seasonality
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=False
)

print("Selected ARIMA order:", arima_model.order)  # (p,d,q)

# Walk-forward rolling forecast on the test set:
arima_predictions = []
history = train['Adj Close'].copy()

for t in test.index:
    # one-step ahead prediction
    pred = arima_model.predict(n_periods=1)[0]
    arima_predictions.append(pred)
    # update model with the true observed value at time t
    arima_model.update(test.loc[t, 'Adj Close'])

arima_pred_series = pd.Series(arima_predictions, index=test.index, name="ARIMA_Pred")

# -------------------------
# 3) LSTM — supervised framing with rolling one-step ahead
# -------------------------
# Strategy:
# - Scale prices using MinMaxScaler fit on training data only
# - Build windows of length 'lookback' from training to train LSTM
# - For test predictions: one-step ahead, using latest 'lookback' true values (train + test-so-far)
#   to avoid multi-step compounding error and to be comparable to ARIMA updating

lookback = 60  # days in the input window

# scale using train only
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train[['Adj Close']])
test_scaled = scaler.transform(test[['Adj Close']])

def build_sequences(arr, lb=60):
    X, y = [], []
    for i in range(lb, len(arr)):
        X.append(arr[i-lb:i, 0])
        y.append(arr[i, 0])
    X = np.array(X)
    y = np.array(y)
    # reshape to [samples, timesteps, features]
    return X.reshape((X.shape[0], X.shape[1], 1)), y

# Prepare supervised train sequences
X_train, y_train = build_sequences(train_scaled, lb=lookback)

# LSTM model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(lookback, 1)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Validation split from tail of training sequences (time-safe)
val_frac = 0.1
val_size = int(len(X_train) * val_frac)
X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

print("\nTraining LSTM...")
hist = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=0,
    callbacks=[es]
)
print(f"LSTM training complete. Best val_loss: {min(hist.history['val_loss']):.6f}")

# Rolling one-step-ahead prediction for the test horizon
# We'll maintain a rolling buffer of the last `lookback` scaled true values
combined_scaled = np.vstack([train_scaled, test_scaled])
# Start index where test begins in the combined array
test_start_idx = len(train_scaled)

lstm_predictions_scaled = []
for i in range(test_start_idx, len(combined_scaled)):
    # form the window using the latest lookback true observed values (not predicted),
    # which ensures one-step-ahead evaluation comparable to ARIMA
    window = combined_scaled[i - lookback:i, 0].reshape(1, lookback, 1)
    pred_scaled = model.predict(window, verbose=0)[0,0]
    lstm_predictions_scaled.append(pred_scaled)

# Inverse scale LSTM predictions to price space
lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions_scaled).reshape(-1,1)).ravel()
lstm_pred_series = pd.Series(lstm_predictions, index=test.index, name="LSTM_Pred")

# -------------------------
# 4) Evaluation
# -------------------------
y_true = test['Adj Close']

arima_metrics = print_metrics("ARIMA", y_true, arima_pred_series)
lstm_metrics  = print_metrics("LSTM ", y_true, lstm_pred_series)

# Save metrics
metrics_df = pd.DataFrame([
    {"Model": "ARIMA", **arima_metrics},
    {"Model": "LSTM",  **lstm_metrics},
])
metrics_path = os.path.join(OUT_DIR, "task2_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

# -------------------------
# 5) Plots
# -------------------------
plt.figure(figsize=(14,6))
plt.plot(train.index, train['Adj Close'], label="Train (Adj Close)")
plt.plot(test.index, y_true, label="Test (Adj Close)")
plt.title("TSLA Adj Close — Train/Test Split")
plt.xlabel("Date"); plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsla_train_test_split.png"), dpi=150)
plt.show()

plt.figure(figsize=(14,6))
plt.plot(y_true.index, y_true, label="Actual")
plt.plot(arima_pred_series.index, arima_pred_series, label="ARIMA Forecast")
plt.plot(lstm_pred_series.index, lstm_pred_series, label="LSTM Forecast")
plt.title("TSLA Test Period — One-step-ahead Forecasts")
plt.xlabel("Date"); plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsla_test_forecasts.png"), dpi=150)
plt.show()

# Error curves over time
arima_err = (arima_pred_series - y_true).abs()
lstm_err  = (lstm_pred_series - y_true).abs()

plt.figure(figsize=(14,4))
plt.plot(arima_err.index, arima_err, label="|Error| ARIMA")
plt.plot(lstm_err.index, lstm_err, label="|Error| LSTM")
plt.title("Absolute Forecast Error over Time (Test)")
plt.xlabel("Date"); plt.ylabel("Absolute Error")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsla_abs_error_over_time.png"), dpi=150)
plt.show()

# -------------------------
# 6) Brief comparison summary printed to console
# -------------------------
better = "ARIMA" if arima_metrics["RMSE"] < lstm_metrics["RMSE"] else "LSTM"
print("\n================== Summary ==================")
print("Test Period:", test.index.min().date(), "to", test.index.max().date())
print(metrics_df)
print(f"\nBased on RMSE, the better model on this split is: {better}")
print("Note:")
print("- ARIMA offers interpretability and fast updating; it handled one-step-ahead updates via .update().")
print("- LSTM can capture nonlinearities but is sensitive to scaling, lookback window, and hyperparameters.")
print("- Results can change with different splits, tuning, and alternative targets (e.g., returns instead of prices).")

# Save forecasts
out_preds = pd.concat([y_true, arima_pred_series, lstm_pred_series], axis=1)
out_preds.to_csv(os.path.join(OUT_DIR, "tsla_test_actual_and_forecasts.csv"))

print(f"\nAll outputs saved in: {OUT_DIR}")
print("- task2_metrics.csv")
print("- tsla_train_test_split.png")
print("- tsla_test_forecasts.png")
print("- tsla_abs_error_over_time.png")
print("- tsla_test_actual_and_forecasts.csv")
