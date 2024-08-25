import pandas as pd
import numpy as np
import yfinance as yf

def beta(tickers: list, weights: list, start_date: str, end_date: str, market_index: str = "^GSPC") -> float:
    """
    Calculate the beta of a portfolio relative to the market using yfinance data.
    
    Parameters:
    tickers (list): List of stock tickers in the portfolio.
    weights (list): List of corresponding weights for each stock in the portfolio.
    start_date (str): The start date for fetching historical data (e.g., '2020-01-01').
    end_date (str): The end date for fetching historical data (e.g., '2021-01-01').
    market_index (str): The market index to compare against (default is S&P 500, '^GSPC').
    
    Returns:
    float: The beta of the portfolio.
    """
    if len(tickers) != len(weights):
        raise ValueError("The number of tickers must match the number of weights.")
    
    # Fetch data for each stock and the market index
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    market_data = yf.download(market_index, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()
    
    # Calculate portfolio returns as a weighted sum of individual stock returns
    portfolio_returns = (stock_returns * weights).sum(axis=1)
    
    # Calculate covariance between portfolio returns and market returns
    covariance_matrix = np.cov(portfolio_returns, market_returns)
    covariance = covariance_matrix[0, 1]
    
    # Calculate variance of market returns
    market_variance = np.var(market_returns)
    
    # Calculate beta
    beta = covariance / market_variance
    
    return beta