import pandas as pd
import numpy as np
import yfinance as yf
from utils import plot_rolling_mean


# Downloading S&P 500 futures data
ticker = "ES=F"  # S&P 500 front-month futures ticker symbol
data = yf.download(ticker)["Adj Close"]

# Creating a pandas dataframe
sp500_data = pd.DataFrame(data)

# Computing log returns
log_returns = np.log(sp500_data).diff()

# Calculating cumulative returns
cumulative_returns = np.exp(np.cumsum(log_returns))

# Calculating rolling annualized mean return
rolling_mean_1yr = log_returns.rolling(252).mean() * 252
rolling_mean_3yrs = log_returns.rolling(756).mean() * 252
rolling_mean_10yrs = log_returns.rolling(2520).mean() * 252

plot_rolling_mean(cumulative_returns, rolling_mean_1yr)
plot_rolling_mean(cumulative_returns, rolling_mean_3yrs)
plot_rolling_mean(cumulative_returns, rolling_mean_10yrs)
