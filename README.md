# Portfolio Optimization 

## Project Intro/Objective
The objective of this project is to identify optimal portfolio allocations that either minimize risk for a given return target or maximize risk-adjusted returns. The project takes stock price data over five years for a list of assets and constructs three portfolios:
* The Minimum Variance Portfolio: the least risky allocation to meet a target return
* The Efficient Frontier: the full set of optimal risk/return tradeoffs across multiple target returns
* The Tangency Portfolio: the portfolio with the highest Sharpe ratio (best return per unit of risk)

## Methodology
Part 1 - Data Collection & Storage: Historical daily prices are pulled from Yahoo Finance. Prices are resampled and converted to month-over-month percentage returns. Assets are filtered via SQL to retain only those with sufficient history and positive average returns.

Part 2 - Parameter Estimation: An equal-weighted portfolio serves as the baseline to calculate the mean return vector and the covariance matrix for the filtered asset list.

Part 3 - Single Portfolio Optimization: The project finds the minimum variance portfolio that achieves a target return of 1.5% per month. Two equality constraints are imposed: the portfolio must hit the target return, and weights must sum to one.

Part 4 - Tangency Portfolio: The portfolio with the maximum possible Sharpe ratio is found by minimizing the negative Sharpe ratio since scipy only offers minimizing functions. 

Part 5 - Efficient Frontier: The optimizer is run repeatedly across a range of target returns. Each point on the frontier represents the lowest possible volatility achievable for that target return. Individual asset risk/return profiles are plotted alongside the frontier as well as the tangency portfolio and equal-weights portfolio.

## Technologies Used
* Python
* SQl
* yfinance
* Pandas
* NumPy
* SciPy
* Matplotlib

## How to Use This Code
1. Clone or download the project
2. Customize the timeframe, the tickers, and/or the target return
    * To change the data history window, edit the start and end parameters in part 1a
    * To change the assets in the portfolio, edit the tickers list in part 1a
    * To change the target return for the minimum variance portfolio, edit target_ret in part 3a
3. Run the script, which will
    * Download the price data for your list of tickers
    * Compute monthly returns and store them in a local SQL file
    * Filter your list of assets
    * Print the equal-weighted portfolio's return, volatility, and Sharpe ratio
    * Run the minimum variance optimizer for your target return
    * Run the maximize Sharpe ratio optimizer and display the tangency portfolio's return, volatility, and Sharpe Ratio
    * Sweep through target returns, run the minimum variance optimizer and display the efficient frontier chart with the tangency portfolio and the equal-weights portfolio

## Needs of this Project
* Retail investors looking to build a data-driven, diversified portfolio that balances risk tolerance against expected returns
* Asset managers and analysts who need a code-based framework for testing various asset allocations and presenting risk/return tradeoffs to clients 