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
from scipy.optimize import minimize

# -------------------------
# Config
# -------------------------
ASSETS = ["TSLA","BND","SPY"]
START = "2015-07-01"
END   = "2025-07-31"

BACKTEST_START = "2024-08-01"
BACKTEST_END   = "2025-07-31"

OUT_DIR = "gmf_task5_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE = 0.02  # annual risk-free for Sharpe

LOOKBACK_LSTM = 60
FORECAST_H = 126  # ~6 months used in Task 4 to derive expected TSLA return (keeps consistency)

np.random.seed(42)

# -------------------------
# Helper functions
# -------------------------
def annualized_return_from_daily_mean(daily_mean, periods=TRADING_DAYS):
    return (1 + daily_mean) ** periods - 1

def annualize_vol(daily_std, periods=TRADING_DAYS):
    return daily_std * np.sqrt(periods)

def portfolio_return(w, mu_ann):
    return float(np.dot(w, mu_ann))

def portfolio_vol(w, cov_ann):
    return float(np.sqrt(np.dot(w, np.dot(cov_ann, w))))

def sharpe_from_series(ret_series, rf=RISK_FREE, periods=TRADING_DAYS):
    # ret_series: daily returns (simple)
    excess = ret_series - ( (1+rf)**(1/periods) - 1 )
    ann_mean = excess.mean() * periods
    ann_std  = ret_series.std() * np.sqrt(periods)
    return ann_mean / ann_std if ann_std != 0 else np.nan

def cumulative_returns_from_weights(prices_df, weights, rebalance_dates=None):
    """
    prices_df: DataFrame of asset prices (aligned, daily)
    weights: initial weights (len = n_assets)
    rebalance_dates: list-like of dates (index labels in prices_df) where weights are reset to `weights`.
                     If None -> no rebalancing (hold weights)
    Returns: daily portfolio simple returns series and cumulative returns series (starting at 1.0)
    """
    # compute daily simple returns per asset
    rets = prices_df.pct_change().fillna(0)
    dates = rets.index
    n = len(weights)

    if rebalance_dates is None:
        # hold: compute returns based on initial weights and changing asset returns (weights drift)
        # To simulate no rebalancing, compute portfolio return each day as dot(current asset weights, daily returns)
        # but current asset weights drift: easier approach compute portfolio value dynamic with initial allocation
        # initialize portfolio value 1.0 with allocation by initial weights
        alloc = weights
        pv = 1.0
        pv_series = []
        # track asset holdings in units: units = (w * pv) / price_at_start
        # use first available price as start
        start_prices = prices_df.iloc[0].values
        units = (alloc * pv) / start_prices
        for i, d in enumerate(dates):
            # portfolio value = sum(units * price_t)
            pv = np.dot(units, prices_df.iloc[i].values)
            pv_series.append(pv)
        pv_series = pd.Series(pv_series, index=dates)
        # convert to daily returns
        daily_rets = pv_series.pct_change().fillna(0)
        cum = (1 + daily_rets).cumprod()
        return daily_rets, cum
    else:
        # monthly rebalancing to fixed target weights at rebalance_dates
        pv = 1.0
        pv_list = []
        # initialize units at first day
        curr_date = dates[0]
        price0 = prices_df.iloc[0].values
        units = (weights * pv) / price0
        for i, d in enumerate(dates):
            # at rebalance date (if d in rebalance_dates) -> reset holdings to target weights
            if pd.to_datetime(d).normalize() in set(pd.to_datetime(rebalance_dates).normalize()):
                # compute pv at that day first
                pv = np.dot(units, prices_df.iloc[i].values)
                # reallocate to target weights
                units = (weights * pv) / prices_df.iloc[i].values
            pv = np.dot(units, prices_df.iloc[i].values)
            pv_list.append(pv)
        pv_series = pd.Series(pv_list, index=dates)
        daily_rets = pv_series.pct_change().fillna(0)
        cum = (1 + daily_rets).cumprod()
        return daily_rets, cum

def max_sharpe_port(mu_ann, cov_ann, rf=RISK_FREE):
    n = len(mu_ann)
    x0 = np.repeat(1.0/n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    def neg_sharpe(w):
        port_ret = float(np.dot(w, mu_ann))
        port_vol = float(np.sqrt(np.dot(w, np.dot(cov_ann, w))))
        return -(port_ret - rf) / (port_vol + 1e-12)
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def min_vol_port(mu_ann, cov_ann):
    n = len(mu_ann)
    x0 = np.repeat(1.0/n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    res = minimize(lambda w: np.dot(w, np.dot(cov_ann, w)), x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def max_drawdown(cum_series):
    # cum_series: cumulative value series starting at 1
    roll_max = cum_series.cummax()
    drawdown = (cum_series - roll_max) / roll_max
    return drawdown.min()

# -------------------------
# 1) Download & prepare price series
# -------------------------
print("Downloading data...")
panel = {}
for t in ASSETS:
    df = yf.download(t, start=START, end=END, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {t}")
    panel[t] = df[['Adj Close']].rename(columns={'Adj Close': t}).dropna()

# Align on common dates (inner join)
prices = panel[ASSETS[0]].join([panel[a] for a in ASSETS[1:]], how='inner')
prices = prices.sort_index()

# Subset backtest window
back_prices = prices.loc[BACKTEST_START:BACKTEST_END].copy()
if back_prices.empty:
    raise RuntimeError("No data in backtest window. Check dates and available data.")

# Daily returns full history for cov matrix and historical means
full_rets = prices.pct_change().dropna()

# -------------------------
# 2) Recompute expected returns vector & cov matrix consistent with Task 4
#    We'll reuse the same approach: best model (ARIMA or LSTM) for TSLA 6-month forecast
# -------------------------
# First split for model selection as in Task 4
train_end = "2023-12-31"
tsla = prices[['TSLA']].copy()
tsla_train = tsla.loc[:train_end].copy()
tsla_test  = tsla.loc[datetime.fromisoformat(train_end) + pd.Timedelta(days=1):].copy()

# --- ARIMA candidate
print("Fitting ARIMA for model selection...")
arima_model = auto_arima(tsla_train['TSLA'], start_p=0, start_q=0, max_p=5, max_q=5,
                         d=None, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
arima_preds = []
for t in tsla_test.index:
    arima_preds.append(arima_model.predict(n_periods=1)[0])
    arima_model.update(tsla_test.loc[t,'TSLA'])
arima_rmse = np.sqrt(((tsla_test['TSLA'].values - np.array(arima_preds))**2).mean())

# --- LSTM candidate
print("Training LSTM for model selection...")
# scale train/test
scaler = MinMaxScaler((0,1))
train_scaled = scaler.fit_transform(tsla_train.values)
test_scaled  = scaler.transform(tsla_test.values)

def build_sequences_local(arr, lookback=LOOKBACK_LSTM):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i,0])
        y.append(arr[i,0])
    X = np.array(X); y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

# Check there's enough data to train LSTM
if len(train_scaled) > LOOKBACK_LSTM + 5:
    X_tr, y_tr = build_sequences_local(train_scaled, LOOKBACK_LSTM)
    # small time-safe split
    val_size = max(1, int(0.1*len(X_tr)))
    X_train_l, X_val_l = X_tr[:-val_size], X_tr[-val_size:]
    y_train_l, y_val_l = y_tr[:-val_size], y_tr[-val_size:]

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential([
        LSTM(64, input_shape=(LOOKBACK_LSTM,1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    model.fit(X_train_l, y_train_l, validation_data=(X_val_l, y_val_l), epochs=100, batch_size=32, verbose=0, callbacks=[es])

    # rolling one-step on test using true past
    combined = np.vstack([train_scaled, test_scaled])
    test_start_idx = len(train_scaled)
    lstm_preds_s = []
    for i in range(test_start_idx, len(combined)):
        window = combined[i-LOOKBACK_LSTM:i,0].reshape(1,LOOKBACK_LSTM,1)
        p = model.predict(window, verbose=0)[0,0]
        lstm_preds_s.append(p)
    lstm_preds = scaler.inverse_transform(np.array(lstm_preds_s).reshape(-1,1)).ravel()
    lstm_rmse = np.sqrt(((tsla_test['TSLA'].values - lstm_preds)**2).mean())
else:
    # not enough data — fallback to ARIMA
    lstm_rmse = 1e9

best_model = "ARIMA" if arima_rmse <= lstm_rmse else "LSTM"
print(f"Model selection result: best_model = {best_model} (ARIMA_RMSE={arima_rmse:.3f}, LSTM_RMSE={lstm_rmse:.3f})")

# Produce TSLA 6-month forecast from the best model as in Task 4
def arima_forecast(series, h):
    model = auto_arima(series, start_p=0, start_q=0, max_p=5, max_q=5, d=None,
                       seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
    fc = model.predict(n_periods=h)
    return fc  # numpy array

def lstm_forecast(series, h, lookback=LOOKBACK_LSTM):
    values = series.values.reshape(-1,1)
    sc = MinMaxScaler((0,1))
    scaled = sc.fit_transform(values)
    if len(scaled) <= lookback + 5:
        raise RuntimeError("Not enough data to train LSTM for forecasting.")
    X_all, y_all = build_sequences_local(scaled, lookback)
    # train
    model = Sequential([
        LSTM(64, input_shape=(lookback,1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    val_size = max(1, int(0.1*len(X_all)))
    model.fit(X_all[:-val_size], y_all[:-val_size], validation_data=(X_all[-val_size:], y_all[-val_size:]),
              epochs=100, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    # recursive forecast
    buf = scaled.copy()
    preds_s = []
    for _ in range(h):
        window = buf[-lookback:,0].reshape(1,lookback,1)
        p = model.predict(window, verbose=0)[0,0]
        preds_s.append(p)
        buf = np.vstack([buf, [[p]]])
    preds = sc.inverse_transform(np.array(preds_s).reshape(-1,1)).ravel()
    return preds

print("Generating 6-month TSLA price forecast...")
if best_model == "ARIMA":
    tsla_fc_vals = arima_forecast(tsla['TSLA'], FORECAST_H)  # numpy
else:
    tsla_fc_vals = lstm_forecast(tsla['TSLA'], FORECAST_H, LOOKBACK_LSTM)

# derive expected daily TSLA return from forecast (geometric mean across horizon)
last_price = tsla['TSLA'].iloc[-1]
end_price = tsla_fc_vals[-1]
n_days = len(tsla_fc_vals)
tsla_expected_daily = (end_price / last_price) ** (1.0 / n_days) - 1.0
tsla_expected_ann = (1 + tsla_expected_daily) ** TRADING_DAYS - 1

# For BND and SPY: use historical mean daily returns over full history, annualize
hist_daily_mean = full_rets.mean()
bnd_expected_ann = (1 + hist_daily_mean['BND']) ** TRADING_DAYS - 1
spy_expected_ann = (1 + hist_daily_mean['SPY']) ** TRADING_DAYS - 1

mu_ann = np.array([tsla_expected_ann, bnd_expected_ann, spy_expected_ann])
cov_ann = full_rets[['TSLA','BND','SPY']].cov().values * TRADING_DAYS

print("\nExpected annual returns used (input):")
print(f" TSLA: {tsla_expected_ann*100:.2f}%")
print(f" BND : {bnd_expected_ann*100:.2f}%")
print(f" SPY : {spy_expected_ann*100:.2f}%")

# -------------------------
# 3) Compute Tangency weights (initial) — long-only
# -------------------------
w_tan = max_sharpe_port(mu_ann, cov_ann, rf=RISK_FREE)
w_minv = min_vol_port(mu_ann, cov_ann)

print("\nComputed portfolios (long-only):")
print(" Tangency weights:", dict(zip(ASSETS, w_tan.round(4))))
print(" Min-Vol weights :", dict(zip(ASSETS, w_minv.round(4))))

# For backtest, choose Tangency as strategy (you can change to min-vol alternative)
strategy_weights = w_tan

# -------------------------
# 4) Build backtest scenarios
#    - Strategy A: Hold initial weights for the full year (no rebalancing)
#    - Strategy B: Monthly rebalance to initial weights (start of month)
#    - Benchmark: 60% SPY / 40% BND, monthly rebalance
# -------------------------
# Prepare price subset for backtest (use Adj Close)
bt_prices = back_prices.copy()
# Ensure business-day index matching (we assume daily prices available)
bt_prices = bt_prices.asfreq('B').fillna(method='ffill')  # forward fill missing non-trading days to keep index consistent

# monthly rebalance dates: first business day of each month in backtest window
months = pd.date_range(start=bt_prices.index.min(), end=bt_prices.index.max(), freq='MS')
rebalance_dates = [d for d in months if d in bt_prices.index]
if len(rebalance_dates) == 0:
    # fallback: pick the first trading day of each month present
    months = pd.date_range(start=bt_prices.index.min(), end=bt_prices.index.max(), freq='MS')
    rebalance_dates = [bt_prices.index[bt_prices.index.get_loc(m, method='nearest')] for m in months]

# build strategy portfolios
strat_daily_hold, strat_cum_hold = cumulative_returns_from_weights(bt_prices[ASSETS], strategy_weights, rebalance_dates=None)
strat_daily_monthly, strat_cum_monthly = cumulative_returns_from_weights(bt_prices[ASSETS], strategy_weights, rebalance_dates=rebalance_dates)

# build benchmark: 60% SPY / 40% BND (note: no TSLA)
bench_weights = np.array([0.0, 0.4, 0.6])  # order TSLA, BND, SPY (we set 0 TSLA)
bench_daily, bench_cum = cumulative_returns_from_weights(bt_prices[ASSETS], bench_weights, rebalance_dates=rebalance_dates)

# -------------------------
# 5) Performance metrics
# -------------------------
def summarize_performance(daily_returns, cum_series, name):
    total_return = cum_series.iloc[-1] - 1.0
    ann_return = (1 + daily_returns.mean()) ** TRADING_DAYS - 1
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = sharpe_from_series(daily_returns, rf=RISK_FREE)
    mdd = max_drawdown(cum_series)
    return {
        'Name': name,
        'Total Return': total_return,
        'Annual Return': ann_return,
        'Annual Vol': ann_vol,
        'Sharpe': sharpe,
        'Max Drawdown': mdd
    }

perf_hold = summarize_performance(strat_daily_hold, strat_cum_hold, "Strategy (Hold)")
perf_monthly = summarize_performance(strat_daily_monthly, strat_cum_monthly, "Strategy (Monthly Rebalance)")
perf_bench = summarize_performance(bench_daily, bench_cum, "Benchmark (60%SPY/40%BND)")

perf_df = pd.DataFrame([perf_hold, perf_monthly, perf_bench]).set_index('Name')
print("\nPerformance summary (backtest period: {} to {}):".format(BACKTEST_START, BACKTEST_END))
print(perf_df[['Total Return','Annual Return','Annual Vol','Sharpe','Max Drawdown']].applymap(lambda x: f"{x:.4f}" if isinstance(x, float) else x))

# Save metrics
perf_df.to_csv(os.path.join(OUT_DIR, "backtest_performance_summary.csv"))

# -------------------------
# 6) Plot cumulative returns
# -------------------------
plt.figure(figsize=(12,7))
plt.plot(strat_cum_hold.index, strat_cum_hold, label="Strategy (Hold)", linewidth=2)
plt.plot(strat_cum_monthly.index, strat_cum_monthly, label="Strategy (Monthly Rebalance)", linewidth=2)
plt.plot(bench_cum.index, bench_cum, label="Benchmark (60%SPY/40%BND)", linewidth=2)
plt.title("Backtest: Cumulative Returns ({} to {})".format(BACKTEST_START, BACKTEST_END))
plt.xlabel("Date"); plt.ylabel("Cumulative Return (1 = start value)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "backtest_cumulative_returns.png"), dpi=150)
plt.show()

# -------------------------
# 7) Concluding summary printed
# -------------------------
print("\nConcluding summary:")
print("- Strategy used initial Tangency weights (recomputed from forecast inputs).")
print("- Two variants: Hold (no rebal) and Monthly Rebalance (rebalanced to the same target weights each month).")
print("- Benchmark: monthly-rebalanced 60% SPY / 40% BND.")
print()
print("Key Results (Total Return, Annual Return, Sharpe, Max Drawdown):")
for name, row in perf_df.iterrows():
    print(f" {name}: Total={row['Total Return']:.2%}, Annual={row['Annual Return']:.2%}, Vol={row['Annual Vol']:.2%}, Sharpe={row['Sharpe']:.3f}, MDD={row['Max Drawdown']:.2%}")

print("\nInterpretation guidance:")
print("- If the strategy outperforms the benchmark (higher annual return and Sharpe), that supports the viability of the model-driven portfolio allocation over the backtest window.")
print("- Watch drawdown: higher returns with much larger drawdowns may be unacceptable depending on mandate.")
print("- This single-year backtest is indicative but not definitive — extend to multiple rolling windows, add transaction costs, slippage, realistic reforecast cadence, and position limits for production readiness.")

print(f"\nAll outputs saved to folder: {OUT_DIR}")
print("- backtest_performance_summary.csv")
print("- backtest_cumulative_returns.png")
