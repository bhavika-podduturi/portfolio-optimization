import yfinance as yf # type: ignore
import sqlite3
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Part 1: Import stock price data 

# 1a. Import stock price data for 15 top companies/ETFs over 5 years
# Choose companies/ETFs across different industries for a diversified list
tickers = ["VTI", "VOO", "QQQ", "BRK-B", "GOOGL", "MSFT", "NET", "AVGO", "UBER", "LMT", "PG", "ABNB", "V", "MRK", "JPM"]
raw = yf.download(tickers, start="2021-01-01", end="2025-12-31", auto_adjust=True)
prices = raw["Close"]

# 1b. Convert to monthly data to reduce noise/volatility and focus on long-term trends 
monthly_prices = prices.resample("ME").last() 

# 1c. Calculate month-over-month returns
monthly_rets = monthly_prices.pct_change().dropna()*100

# 1d. Convert to long format to be loaded into SQL
monthly_rets_long = monthly_rets.stack().reset_index()
monthly_rets_long.columns = ["date", "ticker", "return_pct"]

conn = sqlite3.connect("portfolio.db")
monthly_rets_long.to_sql("monthly_rets", conn, if_exists="replace", index=False)

# 1e. Filter out assets without at least 4 years of data or with negative returns
query = """
SELECT ticker
FROM monthly_rets
GROUP BY ticker
HAVING COUNT(*) >= 48 AND AVG(return_pct) > 0
"""

valid_tickers = pd.read_sql(query, conn)

# 1f. Load filtered dataset back into Python
ticker_list = valid_tickers["ticker"].tolist()
placeholders = ",".join(["?" for _ in ticker_list])

returns_query = f"""
SELECT date, ticker, return_pct
FROM monthly_rets
WHERE ticker IN ({placeholders})
ORDER BY ticker, date
"""

filtered_rets = pd.read_sql(returns_query,conn,params=ticker_list)
returns_wide = filtered_rets.pivot(index="date", columns="ticker", values="return_pct")

# Part 2: Develop framework to determine various optimal portfolios

# 2a. Calculate parameter estimates and store in numpy arrays
mu = returns_wide.mean().values
sigma = returns_wide.cov().values
N = len(mu)

# 2b. Create an equal weights vector for the number of tickers in the portfolio (N)
ewgts = np.repeat(1/N, N)

# 2c. Create a function that takes weights allocated to assets in a portfolio
# and calculates the mean, volatility, and Sharpe ratio of the portfolio
def mvs(weights):
    pmean = weights.T @ mu
    pvol = np.sqrt(weights.T @ sigma @ weights)
    return pmean, pvol, pmean/pvol

# 2d. Determine the mean, volatility, and Sharpe ratio for an equal weighted portfolio
print(mvs(ewgts))

# Part 3: Utilize minimization algorithms to find a portfolio that achieves a target return with minimum variance
import scipy.optimize as sco

# 3a. Set target return and constraints 
target_ret = 1.5

cons = [
    # expected return equals target return
    {'type': 'eq', 'fun': lambda w: mvs(w)[0] - target_ret},

    # weights sum to one
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
]
# REMOVE LATER: 'type': 'eq' indicates it is an equality constraint 

# 3b. Create a function that calculates the variance of a portfolio given its weights
# by squaring the volatility of the portfolio
def portfoliovar(weights):
    return mvs(weights)[1]**2

# 3c. Call the optimization algorithm
result1 = sco.minimize(portfoliovar,    # function to minimize
                       ewgts,           # starting point for weights
                       constraints = cons,
                       method = 'SLSQP' # minimization algorithm
                        )
print(result1)

# 3d. Pass weights from the minimized function to calculate 
# the expected return, variance, and Sharpe ratio of this portfolio.
# Expected return should match the target of 1.5%
print(mvs(result1['x']))

# Part 4: Determine the tangency portfolio, which is the portfolio with the maximum possible Sharpe ratio

# 4a. The optimization algorithm using scipy only offers a minimizer
# So, to maximize the Sharpe ratio, we have to minimize the negative Sharpe 
def neg_sharpe(weights):
    return -1*mvs(weights)[2]

cons_tangency = {'type': 'eq',
                 'fun': lambda w: np.sum(w) - 1}

tangency = sco.minimize(neg_sharpe, ewgts, constraints=cons_tangency, method='SLSQP')

print(tangency)

# 4b. Run weights from tangency portfolio through mvs function to calculate 
# expected return, variance, and Sharpe ratio for the tangency portfolio
print(mvs(tangency['x']))

# Part 5: Determine the efficient frontier by sweeping across multiple target return values
# and, for each one, finding the portfolio with minimum variance

# 5a. Create range of target returns
target_rets = np.linspace(-1, 3, 50)

# 5b. Sweep through range of target returns, running the optimization algorithm and calculating variance
target_vols = []
for t in target_rets:
    cons = ({'type': 'eq', 'fun': lambda w: mvs(w)[0] - t},
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    result = sco.minimize(portfoliovar, ewgts, method='SLSQP', constraints=cons)
    target_vol = np.sqrt(result['fun'])
    target_vols.append(target_vol)

# 5c. Display the efficient frontier with the tangency portfolio and equal-weighted portfolio
fig, ax = plt.subplots(figsize=(10,6))

ax.plot(target_vols, target_rets, 'g-', linewidth=2, label='Efficient Frontier')

for i in range(N):
    ax.scatter(np.sqrt(sigma[i,i]), mu[i], 
               color='steelblue', alpha=0.6, s=60, zorder=3)
    ax.annotate(returns_wide.columns[i],
                (np.sqrt(sigma[i,i]), mu[i]),
                textcoords="offset points",
                xytext=(5, 5), fontsize=8, color='steelblue')

tang_ret, tang_vol, _ = mvs(tangency['x'])
ew_ret, ew_vol, _     = mvs(ewgts)

ax.scatter(tang_vol, tang_ret, color='red', 
           s=80, zorder=5, label='Tangency Portfolio')
ax.scatter(ew_vol, ew_ret, color='blue', 
           s=80, zorder=5, label='Equal Weight Portfolio')

ax.set_xlabel("Volatility (Monthly %)", fontsize=12)
ax.set_ylabel("Expected Return (Monthly %)", fontsize=12)
ax.set_title("Efficient Frontier — Multi-Asset Portfolio", fontsize=14)
ax.set_xlim(0, None)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()