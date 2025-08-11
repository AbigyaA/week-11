
import os
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config / Parameters
# -------------------------
ASSETS = ["TSLA", "BND", "SPY"]
START = "2015-07-01"
END = "2025-07-31"
ROLL_WINDOW = 21  # approx. 1 month trading days
OUTPUT_DIR = "gmf_task1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def download_assets(tickers, start, end):
    """
    Download historical OHLCV data using yfinance for a list of tickers.
    Returns a dict of DataFrames keyed by ticker.
    """
    data = {}
    for t in tickers:
        print(f"Downloading {t}...")
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)  # keep Adj Close column
        if df.empty:
            print(f"Warning: No data for {t}.")
        df.index = pd.to_datetime(df.index)
        data[t] = df
    return data

def basic_checks_and_clean(df):
    """
    Ensure columns, dtypes, missing values handling.
    Strategy:
      - Keep ['Open','High','Low','Close','Adj Close','Volume']
      - If missing days exist (non-trading), do not forward-fill beyond business days.
      - For small gaps: forward-fill then backfill for safety for prices, fill 0 for volume.
    """
    expected_cols = ['Open','High','Low','Close','Adj Close','Volume']
    df = df.copy()
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Convert types
    df[expected_cols] = df[expected_cols].astype(float)
    # Missing values summary
    missing_before = df[expected_cols].isna().sum()
    print("Missing values before cleaning:\n", missing_before)

    # Strategy: forward fill small gaps then backfill
    df[['Open','High','Low','Close','Adj Close']] = df[['Open','High','Low','Close','Adj Close']].ffill().bfill()
    # Volume: fill small gaps with 0 (if any)
    df['Volume'] = df['Volume'].fillna(0)

    missing_after = df[expected_cols].isna().sum()
    print("Missing values after cleaning:\n", missing_after)

    return df

def compute_returns(df, price_col='Adj Close'):
    """
    Compute daily simple returns and log returns.
    """
    df = df.copy()
    df['return'] = df[price_col].pct_change()  # simple returns
    df['log_return'] = np.log(df[price_col]).diff()
    return df

def rolling_stats(df, window=ROLL_WINDOW, price_col='Adj Close'):
    df = df.copy()
    df['rolling_mean_price'] = df[price_col].rolling(window=window).mean()
    df['rolling_std_price'] = df[price_col].rolling(window=window).std()
    df['rolling_mean_return'] = df['return'].rolling(window=window).mean()
    df['rolling_std_return'] = df['return'].rolling(window=window).std()
    return df

def adf_test(series, title="Series"):
    """
    Run Augmented Dickey-Fuller test and return results as dict.
    """
    series = series.dropna()
    adf_res = adfuller(series, autolag='AIC')
    result = {
        'adf_stat': adf_res[0],
        'p_value': adf_res[1],
        'used_lag': adf_res[2],
        'n_obs': adf_res[3],
        'critical_values': adf_res[4],
        'icbest': adf_res[5]
    }
    print(f"ADF test for {title}: adf_stat = {result['adf_stat']:.4f}, p-value = {result['p_value']:.4f}")
    for k,v in result['critical_values'].items():
        print(f"   critical value ({k}): {v:.4f}")
    return result

def detect_outliers_zscore(df, col='return', thresh=3.0):
    """
    Identify outliers using z-score on specified column.
    """
    s = df[col].dropna()
    z = np.abs(stats.zscore(s))
    outlier_idx = s.index[z > thresh]
    return outlier_idx

def detect_outliers_iqr(df, col='return', k=1.5):
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outlier_idx = s[(s < lower) | (s > upper)].index
    return outlier_idx

def historical_var(returns, alpha=0.05):
    """
    Historical (non-parametric) VaR for given alpha (e.g., 0.05 for 95% VaR).
    Returns the negative VaR (positive number representing loss).
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    var = -np.quantile(returns, alpha)  # e.g., 5th percentile negative-> positive loss
    return var

def parametric_var(returns, alpha=0.05):
    """
    Parametric VaR assuming normal distribution: mu + sigma * z_alpha
    We return positive loss magnitude.
    """
    returns = returns.dropna()
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(alpha)
    var = -(mu + sigma * z)
    return var

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Annualized Sharpe Ratio (excess returns / volatility)
    risk_free_rate is annualized (e.g., 0.03 for 3%)
    """
    r = returns.dropna()
    if r.empty:
        return np.nan
    rf_daily = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    excess = r - rf_daily
    ann_mean = excess.mean() * periods_per_year
    ann_std = r.std() * np.sqrt(periods_per_year)
    return ann_mean / ann_std if ann_std != 0 else np.nan

# -------------------------
# Step 1: Download Data
# -------------------------
data_raw = download_assets(ASSETS, START, END)

# -------------------------
# Step 2: Clean & Preprocess
# -------------------------
data = {}
for t, df in data_raw.items():
    print(f"\nProcessing {t}")
    if df is None or df.empty:
        print(f"No data for {t} -- skipping.")
        continue
    df_clean = basic_checks_and_clean(df)
    df_clean = compute_returns(df_clean, price_col='Adj Close')
    df_clean = rolling_stats(df_clean, window=ROLL_WINDOW, price_col='Adj Close')
    data[t] = df_clean
    # Save cleaned CSV
    df_clean.to_csv(os.path.join(OUTPUT_DIR, f"{t}_cleaned.csv"))

# -------------------------
# Step 3: Basic Statistics & Distribution Checks
# -------------------------
summary_stats = {}
for t, df in data.items():
    print(f"\nSummary stats for {t}:")
    # Price stats (Adj Close)
    price_desc = df['Adj Close'].describe()
    print("Price (Adj Close):")
    print(price_desc)

    # Returns stats
    ret_desc = df['return'].describe()
    print("\nDaily returns (simple):")
    print(ret_desc)

    summary_stats[t] = {
        "price_desc": price_desc,
        "return_desc": ret_desc,
        "skew_return": df['return'].skew(),
        "kurt_return": df['return'].kurtosis()
    }

# Save summary to CSV
ss_df = []
for t, s in summary_stats.items():
    row = {
        'ticker': t,
        'price_mean': s['price_desc']['mean'],
        'price_std': s['price_desc']['std'],
        'return_mean': s['return_desc']['mean'],
        'return_std': s['return_desc']['std'],
        'return_skew': s['skew_return'],
        'return_kurtosis': s['kurt_return']
    }
    ss_df.append(row)
pd.DataFrame(ss_df).to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)

# -------------------------
# Step 4: EDA Plots (prices, returns, rolling vol)
# -------------------------
sns.set(style="whitegrid", context="talk")
for t, df in data.items():
    fig, axes = plt.subplots(3,1, figsize=(14,12), sharex=True)
    fig.suptitle(f"{t} — Price, Returns, Rolling Volatility (window={ROLL_WINDOW})", fontsize=16)

    # price
    axes[0].plot(df.index, df['Adj Close'], label='Adj Close')
    axes[0].plot(df.index, df['rolling_mean_price'], label=f'{ROLL_WINDOW}-day MA', alpha=0.9)
    axes[0].legend()
    axes[0].set_ylabel("Price")

    # returns
    axes[1].plot(df.index, df['return'], label='Daily Return', linewidth=0.5)
    axes[1].axhline(0, color='k', linewidth=0.5, linestyle='--')
    axes[1].set_ylabel("Daily Return")

    # rolling std (volatility)
    axes[2].plot(df.index, df['rolling_std_return'] * np.sqrt(252), label='Annualized rolling vol')  # approx annualized
    axes[2].set_ylabel("Annualized Volatility")
    axes[2].set_xlabel("Date")
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(OUTPUT_DIR, f"{t}_eda_plots.png"))
    plt.close(fig)

print(f"EDA plots saved to {OUTPUT_DIR}/")

# -------------------------
# Step 5: Stationarity Tests (ADF)
# -------------------------
adf_results = {}
for t, df in data.items():
    print("\n--- ADF Tests for", t, "---")
    # ADF on prices
    r_price = adf_test(df['Adj Close'], title=f"{t} Adj Close")
    # ADF on returns
    r_return = adf_test(df['return'].dropna(), title=f"{t} Daily Returns")
    adf_results[t] = {'price_adf': r_price, 'return_adf': r_return}

# Interpretation guidance:
# - If price series ADF p-value > 0.05, fail to reject unit root => non-stationary -> differencing needed for ARIMA.
# - Returns series are usually stationary (ADF p-value < 0.05).

# -------------------------
# Step 6: Volatility analysis and extreme days
# -------------------------
extreme_days_summary = []
for t, df in data.items():
    # top moves
    top_up = df['return'].nlargest(10)
    top_down = df['return'].nsmallest(10)
    vol = df['rolling_std_return'] * np.sqrt(252)  # approx annualized
    
    # Outliers
    out_z = detect_outliers_zscore(df, col='return', thresh=3.5)
    out_iqr = detect_outliers_iqr(df, col='return', k=1.5)
    outlier_union = out_z.union(out_iqr)

    # VaR & Sharpe
    var_hist_95 = historical_var(df['return'], alpha=0.05)
    var_param_95 = parametric_var(df['return'], alpha=0.05)
    sharpe = sharpe_ratio(df['return'], risk_free_rate=0.02, periods_per_year=252)  # assuming 2% annual rf

    extreme_days_summary.append({
        'ticker': t,
        'num_outliers_zscore': len(out_z),
        'num_outliers_iqr': len(out_iqr),
        'num_outliers_union': len(outlier_union),
        'var_hist_95': var_hist_95,
        'var_param_95': var_param_95,
        'sharpe_annual': sharpe,
        'top_10_up_dates': top_up.index.strftime("%Y-%m-%d").tolist(),
        'top_10_down_dates': top_down.index.strftime("%Y-%m-%d").tolist()
    })

    # Save lists of extreme events
    pd.DataFrame({
        'top_10_up': top_up,
        'top_10_down': top_down
    }).to_csv(os.path.join(OUTPUT_DIR, f"{t}_extreme_days.csv"))

pd.DataFrame(extreme_days_summary).to_csv(os.path.join(OUTPUT_DIR, "extreme_days_summary.csv"), index=False)

# -------------------------
# Step 7: Outlier Visualizations (returns histogram + top anomalies)
# -------------------------
for t, df in data.items():
    plt.figure(figsize=(12,6))
    sns.histplot(df['return'].dropna(), bins=200, kde=True)
    plt.title(f"{t} Return Distribution")
    plt.xlabel("Daily Return")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{t}_return_hist.png"))
    plt.close()

    # Plot returns with anomalies highlighted
    outliers = detect_outliers_iqr(df, 'return')
    plt.figure(figsize=(14,4))
    plt.plot(df.index, df['return'], linewidth=0.5, label='Daily Return')
    plt.scatter(outliers, df.loc[outliers, 'return'], color='red', s=20, label='IQR Outliers')
    plt.legend()
    plt.title(f"{t} Returns with IQR Outliers Highlighted")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{t}_returns_outliers.png"))
    plt.close()

# -------------------------
# Step 8: Print/Save Key Insights
# -------------------------
print("\nKey risk metrics and a short summary (saved to CSV):")
insights = pd.DataFrame(extreme_days_summary)
insights.to_csv(os.path.join(OUTPUT_DIR, "risk_insights_summary.csv"), index=False)
print(insights)

print(f"\nAll outputs saved in folder: {OUTPUT_DIR}")
print("Files include cleaned CSVs, EDA plots, histograms, extreme days CSVs, and summaries.")

# -------------------------
# End of Task 1
# -------------------------
