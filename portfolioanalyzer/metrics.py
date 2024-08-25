import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

def calculate_beta_and_alpha(tickers: list, investments: list, start_date: str, end_date: str, market_index: str = "^GSPC", risk_free_rate: float = 0.01) -> tuple:
    """
    Calculate the beta and alpha of a portfolio using monetary investments.

    Parameters:
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).
    start_date (str): The start date for fetching historical data (e.g., '2020-01-01').
    end_date (str): The end date for fetching historical data (e.g., '2021-01-01').
    market_index (str): The market index to compare against (default is S&P 500, '^GSPC').
    risk_free_rate (float): The risk-free rate to use in the alpha calculation (default is 1%).

    Returns:
    tuple: A tuple containing the beta and alpha of the portfolio.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = [investment / total_investment for investment in investments]
    
    # Fetch adjusted closing prices for the tickers and the market index
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    market_data = yf.download(market_index, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()
    
    # Handle the case where there is only one stock in the portfolio
    if isinstance(stock_returns, pd.Series):
        portfolio_returns = stock_returns * weights[0]
    else:
        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = (stock_returns * weights).sum(axis=1)
    
    # Calculate the cumulative returns over the period
    cumulative_portfolio_return = (1 + portfolio_returns).prod() - 1
    cumulative_market_return = (1 + market_returns).prod() - 1
    
    # Calculate the portfolio beta
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    
    # Calculate alpha
    alpha = (cumulative_portfolio_return - risk_free_rate) - beta * (cumulative_market_return - risk_free_rate)
    
    return beta, alpha


def calculate_sharpe_ratio(tickers: list, investments: list, start_date: str, end_date: str, risk_free_rate: float = 0.01) -> float:
    """
    Calculate the Sharpe ratio of a portfolio using monetary investments.

    Parameters:
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).
    start_date (str): The start date for fetching historical data (e.g., '2020-01-01').
    end_date (str): The end date for fetching historical data (e.g., '2021-01-01').
    risk_free_rate (float): The risk-free rate to use in the Sharpe ratio calculation (default is 1%).

    Returns:
    float: The Sharpe ratio of the portfolio.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = [investment / total_investment for investment in investments]
    
    # Fetch adjusted closing prices for the tickers
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    
    # Handle the case where there is only one stock in the portfolio
    if isinstance(stock_returns, pd.Series):
        portfolio_returns = stock_returns * weights[0]
    else:
        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = (stock_returns * weights).sum(axis=1)
    
    # Calculate average portfolio return and standard deviation
    average_portfolio_return = portfolio_returns.mean()
    portfolio_std_dev = portfolio_returns.std()

    # Calculate Sharpe ratio
    sharpe_ratio = (average_portfolio_return - risk_free_rate) / portfolio_std_dev
    
    return sharpe_ratio


def calculate_sortino_ratio(tickers: list, investments: list, start_date: str, end_date: str, target_return: float = 0.0, risk_free_rate: float = 0.01) -> float:
    """
    Calculate the Sortino ratio of a portfolio using monetary investments.

    Parameters:
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).
    start_date (str): The start date for fetching historical data (e.g., '2020-01-01').
    end_date (str): The end date for fetching historical data (e.g., '2021-01-01').
    target_return (float): The minimum acceptable return (MAR), often set to 0 or the risk-free rate.
    risk_free_rate (float): The risk-free rate to use in the Sortino ratio calculation (default is 1%).

    Returns:
    float: The Sortino ratio of the portfolio.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = [investment / total_investment for investment in investments]
    
    # Fetch adjusted closing prices for the tickers
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    
    # Handle the case where there is only one stock in the portfolio
    if isinstance(stock_returns, pd.Series):
        portfolio_returns = stock_returns * weights[0]
    else:
        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = (stock_returns * weights).sum(axis=1)
    
    # Calculate the average portfolio return
    average_portfolio_return = portfolio_returns.mean()
    
    # Calculate downside deviation
    downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns - target_return) ** 2))
    
    # Calculate Sortino ratio
    sortino_ratio = (average_portfolio_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio


def calculate_var(tickers: list, investments: list, start_date: str, end_date: str, confidence_level: float = 0.95, time_horizon: int = 1, method:str ='parametric') -> float:
    """
    Calculate the Value at Risk (VaR) of a portfolio using either the Parametric or Historical method.

    Parameters:
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock.
    start_date (str): Start date for historical data.
    end_date (str): End date for historical data.
    confidence_level (float): Confidence level for VaR (default is 95%).
    time_horizon (int): Time horizon in days for VaR calculation (default is 1 day).
    method (str): Method to calculate VaR, either 'parametric' or 'historical' (default is 'parametric').

    Returns:
    float: The Value at Risk (VaR) of the portfolio in monetary terms.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = np.array(investments) / total_investment
    
    # Fetch adjusted closing prices for the tickers
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()
    
    # Calculate portfolio daily returns as a weighted sum of individual stock returns
    portfolio_returns = daily_returns.dot(weights)
    
    if method == 'parametric':
        # Calculate the mean and standard deviation of portfolio returns
        mean_return = portfolio_returns.mean()
        std_dev = portfolio_returns.std()
        
        # Z-score for the given confidence level
        z_score = norm.ppf(1 - confidence_level)
        
        # Calculate Parametric VaR scaled by sqrt(T)
        var = (z_score * std_dev * np.sqrt(time_horizon)) * total_investment
    
    elif method == 'historical':
        # Calculate rolling T-day returns
        t_day_returns = portfolio_returns.rolling(window=time_horizon).apply(lambda x: np.prod(1 + x) - 1).dropna()
        
        # Calculate Historical VaR
        var = np.percentile(t_day_returns, (1 - confidence_level) * 100) * total_investment
    
    else:
        raise ValueError("Method must be either 'parametric' or 'historical'.")
    
    return abs(var)

