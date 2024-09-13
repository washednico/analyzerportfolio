import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import logging
import statsmodels.api as sm

from analyzerportfolio.utils import (
    get_stock_info, 
    get_current_rate, 
    get_currency, 
    get_exchange_rate, 
    convert_to_base_currency
)

def download_data(tickers: list[str], market_ticker: str, start_date: str, end_date: str, base_currency: str) -> pd.DataFrame:
    """
    Download stock and market data, convert to base currency, and return the processed data.
    
    Parameters:
    tickers (list): List of stock tickers.
    market_ticker (str): Market index ticker.
    start_date (str): Start date for historical data.
    end_date (str): End date for historical data.
    base_currency (str): The base currency for the portfolio (e.g., 'USD').

    Returns:
    pd.DataFrame: DataFrame containing the adjusted and converted prices for all tickers and the market index.
    """
    exchange_rate_cache = {}
    stock_data = pd.DataFrame()
    
    # Fetch and process each stock's data
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        currency = get_currency(ticker)
        if currency != base_currency:
            exchange_rate = get_exchange_rate(base_currency, currency, start_date, end_date, exchange_rate_cache)
            data = convert_to_base_currency(data, exchange_rate)
        stock_data[ticker] = data
    
    # Fetch and process market index data
    market_data = yf.download(market_ticker, start=start_date, end=end_date)['Adj Close']
    market_currency = get_currency(market_ticker)
    if market_currency != base_currency:
        exchange_rate = get_exchange_rate(base_currency, market_currency, start_date, end_date, exchange_rate_cache)
        market_data = convert_to_base_currency(market_data, exchange_rate)
    
    # Add market data to stock data
    stock_data[market_ticker] = market_data
    
    # Drop rows with missing data to ensure alignment
    stock_data = stock_data.dropna()
    
    return stock_data


def calculate_beta_and_alpha(portfolio_returns: pd.DataFrame, market_returns: pd.DataFrame, risk_free_rate: float = 0.01) -> tuple:
    """
    Calculate the beta and alpha of a portfolio using monetary investments.

    Parameters:
    portfolio (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers and the market index.
    market_ticker (str): The market index to compare against (e.g., S&P 500).
    risk_free_rate (float): The risk-free rate to use in the alpha calculation (default is 1%).

    Returns:
    tuple: A tuple containing the beta and alpha of the portfolio.
    """

    # Convert the annual risk-free rate to a daily rate
    daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1

    # Excess returns
    excess_portfolio_returns = portfolio_returns.tail(-1) - daily_risk_free_rate
    excess_market_returns = market_returns.tail(-1) - daily_risk_free_rate

    # Add a constant for the intercept in the regression model
    X = sm.add_constant(excess_market_returns)

    # Perform linear regression to find alpha and beta
    model = sm.OLS(excess_portfolio_returns, X).fit()
    alpha = model.params['const']
    beta = model.params[market_returns.name]
    annualized_alpha = (1 + alpha) ** 252 - 1

    return beta, annualized_alpha

