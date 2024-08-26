import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

def get_currency(ticker):
    """Fetch the currency of the given ticker using yfinance."""
    ticker_info = yf.Ticker(ticker).info
    return ticker_info['currency']

def get_exchange_rate(base_currency, quote_currency, start_date, end_date, exchange_rate_cache):
    """Fetch the historical exchange rates from quote_currency to base_currency, using a cache to avoid redundant API calls."""
    if base_currency == quote_currency:
        return None  # No conversion needed

    # Check if the exchange rate is already in the cache
    cache_key = (quote_currency, base_currency)
    if cache_key in exchange_rate_cache:
        return exchange_rate_cache[cache_key]
    
    # Fetch the exchange rate if not cached
    exchange_rate_ticker = f'{quote_currency}{base_currency}=X'
    exchange_rate_data = yf.download(exchange_rate_ticker, start=start_date, end=end_date)['Adj Close']
    
    # Store the fetched exchange rate in the cache
    exchange_rate_cache[cache_key] = exchange_rate_data
    return exchange_rate_data

def convert_to_base_currency(prices, exchange_rate):
    """Convert prices to the base currency using the exchange rate."""
    if exchange_rate is None:
        return prices  # Already in base currency
    return prices * exchange_rate


def download_data(tickers, market_index, start_date, end_date, base_currency):
    """
    Download stock and market data, convert to base currency, and return the processed data.
    
    Parameters:
    tickers (list): List of stock tickers.
    market_index (str): Market index ticker.
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
    market_data = yf.download(market_index, start=start_date, end=end_date)['Adj Close']
    market_currency = get_currency(market_index)
    if market_currency != base_currency:
        exchange_rate = get_exchange_rate(base_currency, market_currency, start_date, end_date, exchange_rate_cache)
        market_data = convert_to_base_currency(market_data, exchange_rate)
    
    # Add market data to stock data
    stock_data[market_index] = market_data
    
    # Drop rows with missing data to ensure alignment
    stock_data = stock_data.dropna()
    
    return stock_data


def calculate_beta_and_alpha(data: pd.DataFrame, tickers: list, investments: list, market_index: str, risk_free_rate: float = 0.01) -> tuple:
    """
    Calculate the beta and alpha of a portfolio using monetary investments.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers and the market index.
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).
    market_index (str): The market index to compare against (e.g., S&P 500).
    risk_free_rate (float): The risk-free rate to use in the alpha calculation (default is 1%).

    Returns:
    tuple: A tuple containing the beta and alpha of the portfolio.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")

    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = np.array(investments) / total_investment
    
    # Ensure the market index is in the DataFrame
    if market_index not in data.columns:
        raise ValueError(f"Market index '{market_index}' not found in the provided data.")

    # Calculate daily returns
    stock_returns = data[tickers].pct_change().dropna()
    market_returns = data[market_index].pct_change().dropna()
    
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



def calculate_sharpe_ratio(data: pd.DataFrame, tickers: list, investments: list, risk_free_rate: float = 0.01) -> float:
    """
    Calculate the Sharpe ratio of a portfolio using monetary investments.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers.
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).
    risk_free_rate (float): The risk-free rate to use in the Sharpe ratio calculation (default is 1%).

    Returns:
    float: The Sharpe ratio of the portfolio.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = np.array(investments) / total_investment
    
    # Ensure all tickers are in the DataFrame
    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]
    if missing_tickers:
        raise ValueError(f"Tickers {missing_tickers} not found in the provided data.")
    
    # Calculate daily returns
    stock_returns = data[tickers].pct_change().dropna()
    
    # Calculate portfolio returns as a weighted sum of individual stock returns
    portfolio_returns = (stock_returns * weights).sum(axis=1)
    
    # Calculate average portfolio return and standard deviation
    average_portfolio_return = portfolio_returns.mean()
    portfolio_std_dev = portfolio_returns.std()

    # Calculate Sharpe ratio
    sharpe_ratio = (average_portfolio_return - risk_free_rate) / portfolio_std_dev
    
    return sharpe_ratio



def calculate_sortino_ratio(data: pd.DataFrame, tickers: list, investments: list, target_return: float = 0.0, risk_free_rate: float = 0.01) -> float:
    """
    Calculate the Sortino ratio of a portfolio using monetary investments.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers.
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).
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
    weights = np.array(investments) / total_investment
    
    # Ensure all tickers are in the DataFrame
    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]
    if missing_tickers:
        raise ValueError(f"Tickers {missing_tickers} not found in the provided data.")
    
    # Calculate daily returns
    stock_returns = data[tickers].pct_change().dropna()
    
    # Calculate portfolio returns as a weighted sum of individual stock returns
    portfolio_returns = (stock_returns * weights).sum(axis=1)
    
    # Calculate the average portfolio return
    average_portfolio_return = portfolio_returns.mean()
    
    # Calculate downside deviation
    downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns - target_return) ** 2))
    
    # Calculate Sortino ratio
    sortino_ratio = (average_portfolio_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio


def calculate_var(data: pd.DataFrame, tickers: list, investments: list, confidence_level: float = 0.95, time_horizon: int = 1, method: str = 'parametric') -> float:
    """
    Calculate the Value at Risk (VaR) of a portfolio using either the Parametric or Historical method.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers.
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock.
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
    
    # Ensure all tickers are in the DataFrame
    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]
    if missing_tickers:
        raise ValueError(f"Tickers {missing_tickers} not found in the provided data.")
    
    # Calculate daily returns
    daily_returns = data[tickers].pct_change().dropna()
    
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




#would make sense to do a module to download the data only once and then pass the df to the functions.
