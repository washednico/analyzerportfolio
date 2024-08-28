import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf

from portfolioanalyzer.utils import (
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

def calculate_daily_returns(stock_df:pd.DataFrame) -> pd.DataFrame:
    """Calculate the daily returns from adjusted prices."""
    if not pd.api.types.is_datetime64_any_dtype(stock_df.index):
        raise ValueError("Index must be of datetime type")
    return stock_df.pct_change().dropna()

def calculate_portfolio_returns(investments:list[float], stock_returns:pd.DataFrame) -> pd.Series:
    """
    Calculate portfolio returns as a weighted sum of individual stock returns.

    Parameters:
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).

    Returns:
    pd.Series: A pandas series containing portofolio returns. 
    """
    total_investment = sum(investments)
    weights = np.array(investments) / total_investment
    portfolio_returns = (stock_returns * weights).sum(axis=1)

    return portfolio_returns.dropna()

def check_dataframe(data: pd.DataFrame, tickers: list[str], investments:list[float] = False, market_ticker:str = False) -> bool:
    """
    Check if necessary variables exist in the provided data
    """
    #Ensure there is a position in all tickers
    if investments:
        if len(tickers) != len(investments):
            raise ValueError("The number of tickers must match the number of investments.")
    
    # Ensure the market index is in the DataFrame (OPTIONAL)
    if market_ticker:
        if market_ticker not in data.columns:
            raise ValueError(f"Market index '{market_ticker}' not found in the provided data.")
    
    # Ensure all tickers are in the DataFrame
    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]
    if [ticker for ticker in tickers if ticker not in data.columns]:
        raise ValueError(f"Tickers {missing_tickers} not found in the provided data.")
    
    return True

def calculate_beta_and_alpha(data: pd.DataFrame, tickers: list[str], investments: list[float], market_ticker: str, risk_free_rate: float = 0.01) -> tuple:
    """
    Calculate the beta and alpha of a portfolio using monetary investments.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers and the market index.
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock (e.g., $1000 in AAPL, $2000 in MSFT).
    market_ticker (str): The market index to compare against (e.g., S&P 500).
    risk_free_rate (float): The risk-free rate to use in the alpha calculation (default is 1%).

    Returns:
    tuple: A tuple containing the beta and alpha of the portfolio.
    """

    if check_dataframe(data, tickers, investments, market_ticker):

        #Calculate market and stocks daily returns
        market_returns = calculate_daily_returns(data[market_ticker])
        stock_returns = calculate_daily_returns(data[tickers])

        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)

        # Calculate the cumulative returns over the period
        cumulative_portfolio_return = (1 + portfolio_returns).prod() - 1
        cumulative_market_return = (1 + market_returns).prod() - 1
        
        # Calculate the portfolio Alpha and Beta
        beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
        alpha = (cumulative_portfolio_return - risk_free_rate) - beta * (cumulative_market_return - risk_free_rate)
        
        return beta, alpha

def calculate_sharpe_ratio(data: pd.DataFrame, tickers: list[str], investments: list[float], risk_free_rate: float = 0.01) -> float:
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
    if check_dataframe(data, tickers, investments, market_ticker=False):

        # Calculate daily returns
        stock_returns = calculate_daily_returns(data[tickers]) 

        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)
        
        # Calculate average portfolio return and standard deviation
        average_portfolio_return = portfolio_returns.mean()
        portfolio_std_dev = portfolio_returns.std()

        # Calculate Sharpe ratio
        sharpe_ratio = (average_portfolio_return - risk_free_rate) / portfolio_std_dev
        
        return sharpe_ratio

def calculate_sortino_ratio(data: pd.DataFrame, tickers: list[str], investments: list[float], target_return: float = 0.0, risk_free_rate: float = 0.01) -> float:
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
    if check_dataframe(data, tickers, investments):
    
        # Calculate portfolio returns as a weighted sum of individual stock returns
        stock_returns = calculate_daily_returns(data[tickers]) 
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)
    
        # Calculate the average portfolio return
        average_portfolio_return = portfolio_returns.mean()
        
        # Calculate downside deviation
        downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns - target_return) ** 2))
        
        # Calculate Sortino ratio
        sortino_ratio = (average_portfolio_return - risk_free_rate) / downside_deviation
        
        return sortino_ratio

def calculate_var(data: pd.DataFrame, tickers: list[str], investments: list[float], confidence_level: float = 0.95, time_horizon: int = 1, method: str = 'parametric') -> float:
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
    if check_dataframe(data, tickers, investments):

        # Calculate portfolio returns as a weighted sum of individual stock returns
        stock_returns = calculate_daily_returns(data[tickers]) 
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)

        #Total monetary amount invested
        total_investment = sum(investments)

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

def calculate_portfolio_scenarios(tickers: list[str], investments: list[float], base_currency: str ='USD') -> dict:
    """
    Calculate the portfolio value in different scenarios based on analyst target prices.

    Parameters:
    tickers (list): List of stock tickers.
    investments (list): Corresponding investment amounts for each ticker.
    base_currency (str): The base currency for calculating portfolio value (default is 'USD').

    Returns:
    dict: Portfolio values in low, mean, median, and high scenarios.
    """
    portfolio_value_low = 0
    portfolio_value_mean = 0
    portfolio_value_median = 0
    portfolio_value_high = 0

    #Ensure there is a position in all tickers
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    for ticker, investment in zip(tickers, investments):
        stock_info = get_stock_info(ticker)
        current_price = stock_info['currentPrice']
        target_low = stock_info['targetLowPrice']
        target_mean = stock_info['targetMeanPrice']
        target_median = stock_info['targetMedianPrice']
        target_high = stock_info['targetHighPrice']
        stock_currency = stock_info['currency']

        # Convert prices to the base currency if necessary
        exchange_rate = get_current_rate(base_currency, stock_currency)
        target_low *= exchange_rate
        target_mean *= exchange_rate
        target_median *= exchange_rate
        target_high *= exchange_rate

        # Calculate the number of shares bought with the investment
        shares = investment / (current_price * exchange_rate)

        # Calculate portfolio value in each scenario
        portfolio_value_low += shares * target_low
        portfolio_value_mean += shares * target_mean
        portfolio_value_median += shares * target_median
        portfolio_value_high += shares * target_high

    return {
        'Low Scenario': portfolio_value_low,
        'Mean Scenario': portfolio_value_mean,
        'Median Scenario': portfolio_value_median,
        'High Scenario': portfolio_value_high
    }

def calculate_dividend_yield(tickers: list[str], investments: list[float]) -> float:
    """
    Calculate the overall dividend yield of the portfolio.

    Parameters:
    tickers (list): List of stock tickers.
    investments (list): Corresponding investment amounts for each ticker.

    Returns:
    float: The overall dividend yield of the portfolio as a percentage.
    """

    #Ensure there is a position in all tickers
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    total_investment = sum(investments)
    weighted_dividend_yield = 0

    for ticker, investment in zip(tickers, investments):
        stock_info = get_stock_info(ticker)
        dividend_yield = stock_info.get('dividendYield', 0)

        if dividend_yield is None:
            continue  # Skip this stock if dividend yield is not available

        # Calculate the weight of this stock in the portfolio
        weight = investment / total_investment

        # Calculate the contribution to the overall dividend yield
        weighted_dividend_yield += weight * dividend_yield

    return weighted_dividend_yield

def calculate_max_drawdown(data: pd.DataFrame, tickers: list[str], investments: list[float]) -> float:
    """
    Calcualte maxdrawdawn of the portfolio

    Parameters: 
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers.
    tickers (list): List of stock tickers.
    investments (list): Corresponding investment amounts for each ticker.

    Returns:
    float: The overall maxdrawdown of the portfolio as a percentage.
    """ 

    if check_dataframe(data, tickers, investments, market_ticker=False):

        # Calculate portfolio returns as a weighted sum of individual stock returns
        stock_returns = calculate_daily_returns(data[tickers]) 
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)
    
        cumulative_portfolio_return = (1 + portfolio_returns).cumprod()
        cumulative_portfolio_returns_max = cumulative_portfolio_return.cummax()

        drawdown = (cumulative_portfolio_return - cumulative_portfolio_returns_max) / cumulative_portfolio_returns_max
        max_drawdown = min(drawdown[1:])

        return max_drawdown
