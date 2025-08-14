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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from scipy.optimize import minimize

# -------------------------
# Config
# -------------------------
ASSETS = ["TSLA","BND","SPY"]
START  = "2015-07-01"
END    = "2025-07-31"
TRAIN_END = "2023-12-31"

OUT_DIR = "gmf_task4_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

TRADING_DAYS = 252
RISK_FREE = 0.02  # 2% annual (adjust as needed)
LOOKBACK_LSTM = 60
FORECAST_H = 126  # ~6 months ≈ 126 trading days

# -------------------------
# Utils
# -------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred)/denom)) * 100

def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    print(f"{name} — Test Metrics | MAE: {mae:,.4f} | RMSE: {rmse:,.4f} | MAPE: {mp:,.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mp}

def annualize_return(daily_ret, periods=TRADING_DAYS):
    return (1 + daily_ret) ** periods - 1

def annualize_cov(cov_daily, periods=TRADING_DAYS):
    return cov_daily * periods

def portfolio_return(w, mu_ann):
    return float(np.dot(w, mu_ann))

def portfolio_vol(w, cov_ann):
    return float(np.sqrt(np.dot(w, np.dot(cov_ann, w))))

def sharpe_ratio(w, mu_ann, cov_ann, rf=RISK_FREE):
    vol = portfolio_vol(w, cov_ann)
    if vol == 0:
        return -np.inf
    return (portfolio_return(w, mu_ann) - rf) / vol

def bounds_long_only(n):
    return tuple((0.0, 1.0) for _ in range(n))

def constraint_sum_to_one(n):
    return {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

def min_vol_port(mu_ann, cov_ann):
    n = len(mu_ann)
    x0 = np.repeat(1.0/n, n)
    res = minimize(lambda w: portfolio_vol(w, cov_ann),
                   x0,
                   method='SLSQP',
                   bounds=bounds_long_only(n),
                   constraints=[constraint_sum_to_one(n)])
    return res.x, portfolio_return(res.x, mu_ann), portfolio_vol(res.x, cov_ann)

def max_sharpe_port(mu_ann, cov_ann, rf=RISK_FREE):
    n = len(mu_ann)
    x0 = np.repeat(1.0/n, n)
    # maximize Sharpe -> minimize negative Sharpe
    res = minimize(lambda w: -(portfolio_return(w, mu_ann) - rf) / (portfolio_vol(w, cov_ann) + 1e-12),
                   x0,
                   method='SLSQP',
                   bounds=bounds_long_only(n),
                   constraints=[constraint_sum_to_one(n)])
    return res.x, portfolio_return(res.x, mu_ann), portfolio_vol(res.x, cov_ann)

def efficient_frontier(mu_ann, cov_ann, n_points=50):
    """
    Compute EF via volatility minimization subject to target return and full investment.
    Long-only.
    """
    n = len(mu_ann)
    mu_min, mu_max = mu_ann.min(), mu_ann.max()
    targets = np.linspace(mu_min, mu_max, n_points)
    vols, rets, weights = [], [], []
    for target in targets:
        cons = [
            constraint_sum_to_one(n),
            {'type': 'eq', 'fun': lambda w, t=target, mu=mu_ann: float(np.dot(w, mu) - t)}
        ]
        x0 = np.repeat(1.0/n, n)
        res = minimize(lambda w: np.dot(w, np.dot(cov_ann, w)),
                       x0,
                       method='SLSQP',
                       bounds=bounds_long_only(n),
                       constraints=cons)
        if res.success:
            w = res.x
            weights.append(w)
            rets.append(portfolio_return(w, mu_ann))
            vols.append(portfolio_vol(w, cov_ann))
    ef = pd.DataFrame({"vol": vols, "ret": rets})
    return ef.sort_values("vol").reset_index(drop=True), np.array(weights)

# -------------------------
# 1) Download data
# -------------------------
print("Downloading data...")
panel = {}
for t in ASSETS:
    df = yf.download(t, start=START, end=END, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {t}")
    panel[t] = df[['Adj Close']].rename(columns={'Adj Close': t}).dropna()

# Align on common dates
prices = panel[ASSETS[0]].join([panel[a] for a in ASSETS[1:]], how='inner')
prices = prices.sort_index()
assert set(prices.columns) == set(ASSETS)

# Daily simple returns
rets = prices.pct_change().dropna()

# -------------------------
# 2) Pick best TSLA model (ARIMA vs LSTM) as in Task 2, then forecast 6M
# -------------------------
tsla = prices[['TSLA']].copy()
tsla_train = tsla.loc[:TRAIN_END].copy()
tsla_test  = tsla.loc[datetime.fromisoformat(TRAIN_END) + pd.Timedelta(days=1):].copy()

# --- ARIMA (auto_arima) with walk-forward on test
print("\nFitting ARIMA on TSLA train...")
arima_model = auto_arima(
    tsla_train['TSLA'],
    start_p=0, start_q=0, max_p=5, max_q=5,
    d=None, seasonal=False,
    stepwise=True, suppress_warnings=True, error_action="ignore", trace=False
)

arima_preds = []
for t in tsla_test.index:
    arima_preds.append(arima_model.predict(n_periods=1)[0])
    # update with true
    arima_model.update(tsla_test.loc[t, 'TSLA'])
arima_pred_series = pd.Series(arima_preds, index=tsla_test.index, name="ARIMA")

# --- LSTM with one-step-ahead on test
print("Training LSTM on TSLA train...")
scaler = MinMaxScaler((0,1))
train_scaled = scaler.fit_transform(tsla_train.values)
test_scaled  = scaler.transform(tsla_test.values)

def build_sequences(arr, lookback=LOOKBACK_LSTM):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i, 0])
        y.append(arr[i, 0])
    X = np.array(X); y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

X_tr, y_tr = build_sequences(train_scaled, LOOKBACK_LSTM)
val_frac = 0.1
val_size = max(1, int(len(X_tr)*val_frac))
X_tr_, X_val = X_tr[:-val_size], X_tr[-val_size:]
y_tr_, y_val = y_tr[:-val_size], y_tr[-val_size:]

lstm = Sequential([
    LSTM(64, input_shape=(LOOKBACK_LSTM,1)),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
lstm.fit(X_tr_, y_tr_, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0, callbacks=[es])

# one-step-ahead on test using rolling true buffer
combined_scaled = np.vstack([train_scaled, test_scaled])
test_start_idx = len(train_scaled)
lstm_preds_scaled = []
for i in range(test_start_idx, len(combined_scaled)):
    window = combined_scaled[i-LOOKBACK_LSTM:i, 0].reshape(1, LOOKBACK_LSTM, 1)
    pred_s = lstm.predict(window, verbose=0)[0,0]
    lstm_preds_scaled.append(pred_s)
lstm_pred = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1,1)).ravel()
lstm_pred_series = pd.Series(lstm_pred, index=tsla_test.index, name="LSTM")

# Compare on test
y_true = tsla_test['TSLA']
arima_metrics = print_metrics("ARIMA", y_true, arima_pred_series)
lstm_metrics  = print_metrics("LSTM ", y_true, lstm_pred_series)

best_model = "ARIMA" if arima_metrics["RMSE"] < lstm_metrics["RMSE"] else "LSTM"
print(f"\nBest model by RMSE on (2024-01-01..2025-07-31): {best_model}")

# Refit best model on FULL TSLA history and produce 6M forecast
def arima_forecast_prices(series, h):
    model = auto_arima(series, start_p=0, start_q=0, max_p=5, max_q=5, d=None,
                       seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
    fc = model.predict(n_periods=h)
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=h, freq="B")
    return pd.Series(fc, index=idx)

def lstm_forecast_prices(series, h, lookback=LOOKBACK_LSTM):
    values = series.values.reshape(-1,1)
    sc = MinMaxScaler((0,1))
    scaled = sc.fit_transform(values)
    # build sequences for fit
    X_all, y_all = build_sequences(scaled, lookback)
    model = Sequential([
        LSTM(64, input_shape=(lookback,1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    # time-safe val
    val_size = max(1, int(len(X_all)*0.1))
    X_tr, X_val = X_all[:-val_size], X_all[-val_size:]
    y_tr, y_val = y_all[:-val_size], y_all[-val_size:]
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0, callbacks=[es])
    # recursive multi-step
    buf = scaled.copy()
    preds_s = []
    for _ in range(h):
        win = buf[-lookback:, 0].reshape(1, lookback, 1)
        p = model.predict(win, verbose=0)[0,0]
        preds_s.append(p)
        buf = np.vstack([buf, [[p]]])
    preds = sc.inverse_transform(np.array(preds_s).reshape(-1,1)).ravel()
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=h, freq="B")
    return pd.Series(preds, index=idx)

print("\nProducing 6-month TSLA forecast from best model...")
if best_model == "ARIMA":
    tsla_fc = arima_forecast_prices(tsla['TSLA'], FORECAST_H)
else:
    tsla_fc = lstm_forecast_prices(tsla['TSLA'], FORECAST_H)

# -------------------------
# 3) Build expected returns vector (annualized)
# -------------------------
last_price = tsla['TSLA'].iloc[-1]
# derive expected **daily** return from forecast: use log-return slope across horizon
# daily mean return ≈ geometric average over forecast horizon
end_price = tsla_fc.iloc[-1]
n_days = len(tsla_fc)
tsla_expected_daily = (end_price / last_price) ** (1.0 / n_days) - 1.0
tsla_expected_ann = annualize_return(tsla_expected_daily)

# Historical daily returns (full period) for BND/SPY -> annualized mean
hist_daily_mean = rets.mean()
bnd_expected_ann = annualize_return(hist_daily_mean['BND'])
spy_expected_ann = annualize_return(hist_daily_mean['SPY'])

mu_ann = pd.Series({
    'TSLA': tsla_expected_ann,
    'BND' : bnd_expected_ann,
    'SPY' : spy_expected_ann
})

print("\nExpected Annual Returns (inputs):")
print(mu_ann.apply(lambda x: f"{x*100:.2f}%"))

# -------------------------
# 4) Covariance matrix (historical, annualized)
# -------------------------
cov_daily = rets[['TSLA','BND','SPY']].cov()
cov_ann   = annualize_cov(cov_daily)

# -------------------------
# 5) Efficient Frontier + Key Portfolios
# -------------------------
ef_df, ef_weights = efficient_frontier(mu_ann.values, cov_ann.values, n_points=60)

w_minvol, r_minvol, v_minvol = min_vol_port(mu_ann.values, cov_ann.values)
w_tan,    r_tan,    v_tan    = max_sharpe_port(mu_ann.values, cov_ann.values, rf=RISK_FREE)

sr_minvol = (r_minvol - RISK_FREE) / (v_minvol + 1e-12)
sr_tan    = (r_tan    - RISK_FREE) / (v_tan    + 1e-12)

# -------------------------
# 6) Plot Efficient Frontier
# -------------------------
plt.figure(figsize=(10,7))
plt.scatter(ef_df['vol'], ef_df['ret'], s=18, alpha=0.7, label='Efficient Frontier')
plt.scatter(v_minvol, r_minvol, marker='o', s=120, label='Min Vol', edgecolor='k')
plt.scatter(v_tan,    r_tan,    marker='*', s=200, label='Max Sharpe (Tangency)', edgecolor='k')

# Capital Market Line from risk-free to tangency
cml_x = np.linspace(0, max(ef_df['vol'].max(), v_tan)*1.1, 50)
cml_y = RISK_FREE + ( (r_tan - RISK_FREE) / (v_tan + 1e-12) ) * cml_x
plt.plot(cml_x, cml_y, linestyle='--', label='Capital Market Line')

plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Expected Return')
plt.title('Efficient Frontier — TSLA / BND / SPY (Long-only)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "efficient_frontier.png"), dpi=150)
plt.show()

# -------------------------
# 7) Recommended Portfolio (explainable choice)
# -------------------------
# Default: prioritize **maximum risk-adjusted return** (Tangency portfolio).
# If user prefers lower total risk, they can choose Min-Vol instead.
recommended_name = "Tangency (Max Sharpe)"
recommended_w = w_tan
recommended_r = r_tan
recommended_v = v_tan
recommended_sr = sr_tan

# Prepare outputs
weights_df = pd.DataFrame({
    'Asset': ['TSLA','BND','SPY'],
    'Weight_Tangency': w_tan,
    'Weight_MinVol'  : w_minvol
})
metrics_df = pd.DataFrame([
    {'Portfolio':'Tangency', 'Exp_Return': r_tan, 'Vol': v_tan, 'Sharpe': sr_tan},
    {'Portfolio':'MinVol'  , 'Exp_Return': r_minvol, 'Vol': v_minvol, 'Sharpe': sr_minvol},
])

weights_df.to_csv(os.path.join(OUT_DIR, "portfolio_weights.csv"), index=False)
metrics_df.to_csv(os.path.join(OUT_DIR, "portfolio_metrics.csv"), index=False)

# -------------------------
# 8) Console Summary
# -------------------------
def pct(x): return f"{x*100:.2f}%"

print("\n==================== Inputs ====================")
print("Annualized Expected Returns used:")
for a, mu in mu_ann.items():
    print(f"  {a}: {pct(mu)}")

print("\nAnnualized Covariance Matrix (excerpt):")
print(pd.DataFrame(cov_ann, index=['TSLA','BND','SPY'], columns=['TSLA','BND','SPY']).round(6))

print("\n==================== Frontier Key Points ====================")
print("Minimum Volatility Portfolio")
print("  Weights:", dict(zip(['TSLA','BND','SPY'], w_minvol.round(4))))
print(f"  Exp Return: {pct(r_minvol)} | Vol: {pct(v_minvol)} | Sharpe: {sr_minvol:.3f}")

print("\nTangency (Max Sharpe) Portfolio")
print("  Weights:", dict(zip(['TSLA','BND','SPY'], w_tan.round(4))))
print(f"  Exp Return: {pct(r_tan)} | Vol: {pct(v_tan)} | Sharpe: {sr_tan:.3f}")

print("\n==================== Recommendation ====================")
print(f"Recommended: {recommended_name}")
print("  Rationale: Maximizes risk-adjusted return (Sharpe). Suitable if the mandate prioritizes growth")
print("  with disciplined risk budgeting. For lower absolute risk mandates, choose Min-Vol alternative.")
print("\nFiles saved in:", OUT_DIR)
print(" - efficient_frontier.png")
print(" - portfolio_weights.csv")
print(" - portfolio_metrics.csv")
