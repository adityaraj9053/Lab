import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from datetime import datetime

sns.set_style("darkgrid")

# -----------------------
# Parameters (edit here)
# -----------------------
ticker = "SPY"                   
start_date = "2014-01-01"  
end_date = None              
n_states = 2                     
use_log_returns = False         
scale_data = True                
random_seed = 42

# -----------------------
# 1. Download data
# -----------------------
if end_date is None:
    end_date = datetime.today().strftime("%Y-%m-%d")

df = yf.download(ticker, start=start_date, end=end_date, progress=False)
if df.empty:
    raise ValueError("No data downloaded. Check ticker or dates.")

# FIX: Handle MultiIndex columns from yfinance (auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Use Adjusted Close or Close (if auto_adjust=True removed Adj Close)
price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
df = df[[price_col]].rename(columns={price_col: 'adj_close'})
df.dropna(inplace=True)

# -----------------------
# 2. Preprocessing: returns
# -----------------------
if use_log_returns:
    df['returns'] = np.log(df['adj_close']).diff()
else:
    df['returns'] = df['adj_close'].pct_change()

df.dropna(inplace=True)  # remove the first NaN

# optional: add volatility (rolling std) as a feature (multi-dimensional HMM)
# df['vol_10'] = df['returns'].rolling(10).std().fillna(method='bfill')

# Use only the returns (1D) or stack features for multivariate HMM
features = df[['returns']].values  # shape (n_samples, n_features)

# scaling
if scale_data:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
else:
    features_scaled = features

# -----------------------
# 3. Fit Gaussian HMM
# -----------------------
np.random.seed(random_seed)
model = GaussianHMM(n_components=n_states,
                    covariance_type="diag",
                    n_iter=2000,
                    random_state=random_seed,
                    verbose=False)
model.fit(features_scaled)

# decode hidden states (Viterbi)
hidden_states = model.predict(features_scaled)
df['state'] = hidden_states

# posterior probabilities (optional)
posteriors = model.predict_proba(features_scaled)

# -----------------------
# 4. Parameter analysis
# -----------------------
# For each hidden state: mean and variance (in original return scale)
means_scaled = model.means_.flatten()  # in scaled space
covars_scaled = model.covars_.flatten()

if scale_data:
    # convert scaled mean/cov back to original return units
    means = means_scaled * scaler.scale_[0] + scaler.mean_[0]
    vars_ = covars_scaled * (scaler.scale_[0] ** 2)
else:
    means = means_scaled
    vars_ = covars_scaled

state_info = pd.DataFrame({
    'state': np.arange(n_states),
    'mean_return': means,
    'std_return': np.sqrt(vars_)
}).sort_values('mean_return').reset_index(drop=True)

# Sort mapping: optional make state 0 = low volatility etc by ordering
# but here we keep original numeric labels and present sorted view separately.

# -----------------------
# 5. Transition matrix
# -----------------------
transmat = model.transmat_

# stationary distribution (eigenvector)
evals, evecs = np.linalg.eig(transmat.T)
stat = np.real(evecs[:, np.argmax(np.real(evals))])
stationary = stat / stat.sum()

# -----------------------
# 6. Visualization
# -----------------------
import matplotlib.dates as mdates

dates = df.index

# plot returns with colored regimes
plt.figure(figsize=(14, 6))
unique_states = np.unique(hidden_states)
colors = sns.color_palette("tab10", n_states)
for s in unique_states:
    mask = (hidden_states == s)
    plt.scatter(dates[mask], df['returns'].values[mask], s=10, label=f"State {s}")
plt.title(f"{ticker} daily returns colored by HMM state (n_states={n_states})")
plt.ylabel("Daily return")
plt.legend()
plt.tight_layout()
plt.show()

# plot price with background shading by regime
plt.figure(figsize=(14, 6))
plt.plot(dates, df['adj_close'], label=f"{ticker} Price")
# fill background by state segments
current_state = hidden_states[0]
start_idx = 0
for i in range(1, len(hidden_states)):
    if hidden_states[i] != current_state:
        # shade region
        plt.axvspan(dates[start_idx], dates[i-1], alpha=0.12, color=colors[current_state])
        start_idx = i
        current_state = hidden_states[i]
# last region
plt.axvspan(dates[start_idx], dates[-1], alpha=0.12, color=colors[current_state])
plt.title(f"{ticker} price with regime shading (states={n_states})")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# Plot transition matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(transmat, annot=True, fmt=".3f", cmap="Blues", xticklabels=[f"S{i}" for i in range(n_states)], yticklabels=[f"S{i}" for i in range(n_states)])
plt.title("HMM transition matrix")
plt.tight_layout()
plt.show()

# parameter table
print("\nState summary (sorted by mean return):")
print(state_info)

print("\nTransition matrix:")
print(transmat)

print("\nStationary distribution (long-run prob of each state):")
print(np.round(stationary, 4))

# -----------------------
# 7. Model evaluation: log-likelihood, AIC, BIC
# -----------------------
log_likelihood = model.score(features_scaled)
n_params = (n_states - 1) + n_states * features_scaled.shape[1] + n_states * features_scaled.shape[1]
n_obs = features_scaled.shape[0]
aic = 2 * n_params - 2 * log_likelihood
bic = np.log(n_obs) * n_params - 2 * log_likelihood

print(f"\nLog-likelihood: {log_likelihood:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}")

# -----------------------
# 8. Predict next-state probabilities (short-term forecast)
# -----------------------
# method: use last inferred state (most probable) and apply transition matrix
last_state = hidden_states[-1]
next_state_prob = transmat[last_state, :]

print(f"\nLast inferred state: {last_state}")
print("One-step ahead next-state probabilities given last inferred state:")
for s, p in enumerate(next_state_prob):
    print(f"  P(next state = {s}) = {p:.3f}")

# Alternatively compute unconditional next-state (using posterior at last time)
last_post = posteriors[-1]
next_state_prob_from_post = last_post.dot(transmat)
print("\nOne-step ahead next-state probabilities using last posterior distribution:")
for s, p in enumerate(next_state_prob_from_post):
    print(f"  P(next state = {s}) = {p:.3f}")

# Expected mean return next day
expected_next_return = np.dot(next_state_prob_from_post, means)
print(f"\nExpected next-day mean return (based on HMM) = {expected_next_return:.6f}")

# Save results to df
df['state'] = hidden_states
df['state_mean'] = df['state'].map({i: means[i] for i in range(n_states)})