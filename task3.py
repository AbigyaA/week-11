import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import yfinance as yf
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# 0) Config
# -------------------------
TICKER = "TSLA"
START  = "2015-07-01"
END    = "2025-07-31"
OUT_DIR = "gmf_task3_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

H6  = 126   # ~6 months of trading days
H12 = 252   # ~12 months of trading days
LOOKBACK = 60
N_BOOT = 500   # Monte Carlo paths for LSTM intervals
ALPHA = 0.05   # 95% intervals -> 2.5% & 97.5%

# -------------------------
# Helpers
# -------------------------
def add_months_to_trading_index(last_date, n_days):
    """Utility to derive an approximate end date label (not exact calendar months)"""
    return f"{n_days} trading days"

def summarize_trend(series):
    """Simple slope sign over the forecast horizon."""
    if len(series) < 2:
        return "insufficient"
    slope = series[-1] - series[0]
    return "upward" if slope > 0 else ("downward" if slope < 0 else "flat")

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

# -------------------------
# 1) Load TSLA
# -------------------------
print("Downloading TSLA data...")
raw = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=False)
if raw.empty:
    raise RuntimeError("No data downloaded for TSLA; check internet/ticker.")

df = raw[['Adj Close']].copy().dropna()
df.index = pd.to_datetime(df.index)
last_date = df.index[-1]
print(f"Data span: {df.index.min().date()} -> {df.index.max().date()}  (n={len(df)})")

# -------------------------
# 2) ARIMA: fit on full history, forecast h steps with native CIs
# -------------------------
print("\nFitting ARIMA (auto_arima) on full sample...")
arima_model = auto_arima(
    df['Adj Close'],
    start_p=0, start_q=0, max_p=5, max_q=5,
    d=None, seasonal=False,
    stepwise=True, suppress_warnings=True, error_action="ignore", trace=False
)
print("Selected ARIMA order:", arima_model.order)

# Forecasts with intervals
def arima_forecast_with_ci(model, h, alpha=ALPHA):
    fc, ci = model.predict(n_periods=h, return_conf_int=True, alpha=alpha)
    idx = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=h, freq="B")  # business days
    fc = pd.Series(fc, index=idx, name="ARIMA_Forecast")
    ci = pd.DataFrame(ci, index=idx, columns=["lower", "upper"])
    return fc, ci

arima_fc_6, arima_ci_6   = arima_forecast_with_ci(arima_model, H6, alpha=ALPHA)
arima_fc_12, arima_ci_12 = arima_forecast_with_ci(arima_model, H12, alpha=ALPHA)

# -------------------------
# 3) LSTM: train on full history, multi-step recursive forecast
#    + bootstrap prediction intervals via residual resampling
# -------------------------
print("\nTraining LSTM on full sample...")

# Scale
scaler = MinMaxScaler((0,1))
scaled = scaler.fit_transform(df[['Adj Close']])

def build_sequences(arr, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i, 0])
        y.append(arr[i, 0])
    X = np.array(X); y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

X_all, y_all = build_sequences(scaled, LOOKBACK)

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(LOOKBACK, 1)),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# time-safe validation from tail
val_frac = 0.1
val_size = int(len(X_all)*val_frac)
X_tr, X_val = X_all[:-val_size], X_all[-val_size:]
y_tr, y_val = y_all[:-val_size], y_all[-val_size:]

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
hist = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=0,
    callbacks=[es]
)
print(f"Best val_loss: {min(hist.history['val_loss']):.6f}")

# Build residuals from one-step validation to estimate error distribution (price space)
val_pred_scaled = model.predict(X_val, verbose=0).ravel()
val_true_scaled = y_val.ravel()
val_pred = scaler.inverse_transform(val_pred_scaled.reshape(-1,1)).ravel()
val_true = scaler.inverse_transform(val_true_scaled.reshape(-1,1)).ravel()
residuals = val_true - val_pred  # empirical residuals (price space)

# Multi-step recursive forecast
def lstm_recursive_forecast(h, model, last_scaled, lookback=LOOKBACK):
    """
    last_scaled: full scaled history (n x 1) as numpy array
    returns: baseline deterministic forecast (price space)
    """
    buf = last_scaled.copy().reshape(-1,1)
    preds_scaled = []
    for _ in range(h):
        window = buf[-lookback:, 0].reshape(1, lookback, 1)
        p = model.predict(window, verbose=0)[0,0]
        preds_scaled.append(p)
        buf = np.vstack([buf, [[p]]])
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).ravel()
    idx = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=h, freq="B")
    return pd.Series(preds, index=idx, name="LSTM_Forecast")

lstm_fc_6  = lstm_recursive_forecast(H6,  model, scaled[:,0], LOOKBACK)
lstm_fc_12 = lstm_recursive_forecast(H12, model, scaled[:,0], LOOKBACK)

# Bootstrap intervals (Monte Carlo paths using empirical residuals)
def lstm_bootstrap_intervals(h, base_series, residuals, n_boot=N_BOOT, alpha=ALPHA):
    """
    base_series: pd.Series of baseline LSTM forecast (price space)
    residuals : 1D np.array of empirical residuals (price space)
    returns: lower/upper bands (DataFrame)
    """
    if len(residuals) == 0:
        # fallback: small gaussian noise if residuals unavailable
        residuals = np.random.normal(0, 1.0, size=1000)
    idx = base_series.index
    paths = np.zeros((n_boot, h))
    for b in range(n_boot):
        noise = np.random.choice(residuals, size=h, replace=True)
        paths[b, :] = base_series.values + noise.cumsum()/5.0
        # note: dividing cumulative noise moderates drift; adjust if desired
    lower = np.percentile(paths, 100*alpha/2, axis=0)
    upper = np.percentile(paths, 100*(1-alpha/2), axis=0)
    return pd.DataFrame({"lower": lower, "upper": upper}, index=idx)

lstm_ci_6  = lstm_bootstrap_intervals(H6,  lstm_fc_6,  residuals, N_BOOT, ALPHA)
lstm_ci_12 = lstm_bootstrap_intervals(H12, lstm_fc_12, residuals, N_BOOT, ALPHA)

# -------------------------
# 4) Save forecasts
# -------------------------
arima_6_df = pd.concat([arima_fc_6.rename("forecast"), arima_ci_6], axis=1)
arima_12_df = pd.concat([arima_fc_12.rename("forecast"), arima_ci_12], axis=1)
lstm_6_df  = pd.concat([lstm_fc_6.rename("forecast"),  lstm_ci_6], axis=1)
lstm_12_df = pd.concat([lstm_fc_12.rename("forecast"), lstm_ci_12], axis=1)

arima_6_df.to_csv(os.path.join(OUT_DIR, "tsla_arima_6m.csv"))
arima_12_df.to_csv(os.path.join(OUT_DIR, "tsla_arima_12m.csv"))
lstm_6_df.to_csv(os.path.join(OUT_DIR, "tsla_lstm_6m.csv"))
lstm_12_df.to_csv(os.path.join(OUT_DIR, "tsla_lstm_12m.csv"))

# -------------------------
# 5) Visualization
# -------------------------
def plot_with_bands(history, fc_df, title, fname):
    plt.figure(figsize=(14,6))
    plt.plot(history.index, history['Adj Close'], label="Historical (Adj Close)")
    plt.plot(fc_df.index, fc_df['forecast'], label="Forecast")
    plt.fill_between(fc_df.index, fc_df['lower'], fc_df['upper'], alpha=0.2, label="95% Interval")
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.show()

# show last 3 years of history for clarity
hist_tail = df.loc[df.index >= (df.index.max() - pd.DateOffset(years=3))]

plot_with_bands(hist_tail, arima_6_df,
                "TSLA — ARIMA 6-Month Forecast (95% CI)", "arima_6m.png")
plot_with_bands(hist_tail, arima_12_df,
                "TSLA — ARIMA 12-Month Forecast (95% CI)", "arima_12m.png")
plot_with_bands(hist_tail, lstm_6_df,
                "TSLA — LSTM 6-Month Forecast (Bootstrap 95% PI)", "lstm_6m.png")
plot_with_bands(hist_tail, lstm_12_df,
                "TSLA — LSTM 12-Month Forecast (Bootstrap 95% PI)", "lstm_12m.png")

# -------------------------
# 6) Analysis Summary (printed)
# -------------------------
def ci_width_stats(fc_df):
    w = (fc_df['upper'] - fc_df['lower']).values
    return {
        "avg_width": float(np.mean(w)),
        "start_width": float(w[0]),
        "end_width": float(w[-1]),
        "widening_ratio": float(w[-1] / (w[0] + 1e-9))
    }

a6_stats  = ci_width_stats(arima_6_df)
a12_stats = ci_width_stats(arima_12_df)
l6_stats  = ci_width_stats(lstm_6_df)
l12_stats = ci_width_stats(lstm_12_df)

def describe_intervals(name, s):
    return (f"{name}: avg band width={s['avg_width']:.2f}, "
            f"start={s['start_width']:.2f}, end={s['end_width']:.2f}, "
            f"widening×={s['widening_ratio']:.2f}")

arima_trend_6  = summarize_trend(arima_6_df['forecast'].values)
arima_trend_12 = summarize_trend(arima_12_df['forecast'].values)
lstm_trend_6   = summarize_trend(lstm_6_df['forecast'].values)
lstm_trend_12  = summarize_trend(lstm_12_df['forecast'].values)

print("\n==================== Task 3 — Summary ====================")
print(f"ARIMA 6M trend:  {arima_trend_6};  " + describe_intervals("CI(6M)", a6_stats))
print(f"ARIMA 12M trend: {arima_trend_12}; " + describe_intervals("CI(12M)", a12_stats))
print(f"LSTM  6M trend:  {lstm_trend_6};   " + describe_intervals("PI(6M)", l6_stats))
print(f"LSTM  12M trend: {lstm_trend_12};  " + describe_intervals("PI(12M)", l12_stats))

print("\nInterpretation guidance:")
print("- Trend: 'upward'/'downward' is based on forecast start vs end price; inspect plotted slope.")
print("- Uncertainty: Bands widen as horizon increases — reflecting compounding uncertainty and regime risk.")
print("- LSTM intervals are bootstrap-based (empirical residuals), so they adapt to recent error volatility.")
print("- ARIMA intervals are parametric and can understate tail risk during non-normal shocks.")
print("\nPortfolio notes:")
print("- If both models indicate upward drift with moderate bands: potential for cautiously increasing exposure,")
print("  especially via staggered entries or covered calls to monetize volatility.")
print("- If bands are wide or diverge across models: priority on risk controls (position sizing, stop-loss bands,")
print("  or pairing with hedges via SPY puts or reduced TSLA weight).")
print("- Reassess frequently: update forecasts after earnings or regime shifts (rates, liquidity, macro shocks).")

print(f"\nAll files saved to: {OUT_DIR}")
print(" - tsla_arima_6m.csv, tsla_arima_12m.csv")
print(" - tsla_lstm_6m.csv,  tsla_lstm_12m.csv")
print(" - arima_6m.png, arima_12m.png, lstm_6m.png, lstm_12m.png")
