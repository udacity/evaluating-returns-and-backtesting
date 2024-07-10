import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# download front-month futures data of S&P500, 10-year Treasuries, gold and US dollar
symbols = ['ES=F', 'ZN=F', 'GC=F', 'DX=F']
data = yf.download(symbols)
# resample data so that we deal with monthly data instead of daily to reduce noise
data = data.resample("M").last()
data.index = pd.to_datetime(data.index)
# subset adjusted close prices and fill NaNs with value know at time t
# drop rows with unknow prices in the beginning of the dataset
prices = data["Adj Close"].ffill().dropna()
prices.index = pd.to_datetime(prices.index)
# compute logarithmic returns
log_returns = np.log(prices).diff()

def compute_risk_parity_weights(returns, window_size=36):
    # compute volatility known at time t
    rolling_vol = returns.rolling(window_size).std()
    rolling_inverse_vol = 1 / rolling_vol
    # divide inverse volatility by the sum of inverse volatilities
    risk_parity_weights = rolling_inverse_vol.apply(
        lambda column: column / rolling_inverse_vol.sum(1)
        )
    # shift weights by one period to use only information available at time t
    return risk_parity_weights.shift(1)

def evaluate_performance(returns, freq=12):
    annualized_mean_return = (returns.mean() * freq)
    print()
    print(f"annualized_mean_return: {annualized_mean_return}")
    annualized_volatility = (returns.std() * np.sqrt(freq))#.round(3)
    print(f"annualized_volatility: {annualized_volatility}")
    skewness = (returns.skew())#.round(2)
    print(f"skewness: {skewness}")
    kurtosis = (returns.kurtosis())#.round(2)
    print(f"kurtosis: {kurtosis}")
    cum_returns = np.exp(returns.cumsum())
    drawdowns = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
    max_drawdown = np.round(drawdowns.max(), 2)
    print(f"max_drawdown: {max_drawdown}")
    sharpe_ratio = (annualized_mean_return / annualized_volatility).round(2)
    print(f"sharpe_ratio: {sharpe_ratio}")
    downside_volatility = returns[returns < 0].std() * np.sqrt(freq)
    sortino_ratio = (annualized_mean_return / downside_volatility).round(2)
    print(f"sortino_ratio: {sortino_ratio}")
    calmar_ratio = (annualized_mean_return / max_drawdown)#.round(2)
    print(f"calmar_ratio: {calmar_ratio}")
    print()
    plt.plot(cum_returns-1, label='Cumulative Returns')
    plt.plot(cum_returns.cummax()-1, label='Cumulative Max', linewidth=.5)
    plt.fill_between(
        drawdowns.index, -drawdowns, color='red', alpha=0.5, label="Drawdowns"
        )
    # Setting x-axis major locator to each year and formatter
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Adding grid with vertical lines for each year
    plt.grid(True, which='major', linestyle='--', color='grey')
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    # Adjusting the legend to include all plots
    plt.legend(loc='best')
    # plt.yscale("log")
    plt.title("Cumulative Returns of Risk-Parity Portfolio")
    plt.savefig("plots/risk_parity_returns.png")
    return None

risk_parity_weights = compute_risk_parity_weights(log_returns, 36)
weighted_returns = (log_returns * risk_parity_weights).dropna()
risk_parity_portfolio_returns = weighted_returns.sum(axis=1)
evaluate_performance(risk_parity_portfolio_returns)
