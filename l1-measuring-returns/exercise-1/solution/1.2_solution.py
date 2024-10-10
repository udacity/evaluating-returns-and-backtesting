# Import libraries
import numpy as np
import yfinance as yf

# Use the `yfinance` library to download the front month S&P500 futures price data.
sp500_prices = yf.download('ES=F')['Adj Close']

# Calculate the daily logarithmic returns of the futures prices.
log_returns = np.log(sp500_prices).diff()

# Annualize the mean of the logarithmic returns.
annualized_mean_return = log_returns.mean() * 252
print()
print(f"annualized_mean_return: {np.round(annualized_mean_return * 100, 2)}%")
print()

