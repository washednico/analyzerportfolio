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
    convert_to_base_currency,
    check_dataframe
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

        # Calculate daily market and stock returns
        market_returns = calculate_daily_returns(data[market_ticker])
        stock_returns = calculate_daily_returns(data[tickers])

        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)

        # Convert the annual risk-free rate to a daily rate
        daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1

        # Excess returns
        excess_portfolio_returns = portfolio_returns - daily_risk_free_rate
        excess_market_returns = market_returns - daily_risk_free_rate

        # Add a constant for the intercept in the regression model
        X = sm.add_constant(excess_market_returns)

        # Perform linear regression to find alpha and beta
        model = sm.OLS(excess_portfolio_returns, X).fit()
        alpha = model.params['const']
        beta = model.params[market_returns.name]
        annualized_alpha = (1 + alpha) ** 252 - 1

        return beta, annualized_alpha

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
    if check_dataframe(data, tickers, investments):

        # Calculate daily returns
        stock_returns = calculate_daily_returns(data[tickers]) 

        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)
        
        # Calculate  portfolio standard deviation
        portfolio_std_dev = portfolio_returns.std()

        trading_days = 252
        annualized_std_dev = portfolio_std_dev * (trading_days ** 0.5)

        # Calculate compounded return over the entire period
        cumulative_return = (1 + portfolio_returns).prod()  # Product of (1 + daily returns)

        # Annualize the cumulative return
        num_days = len(portfolio_returns)
        annualized_return = cumulative_return ** (252 / num_days) - 1

        # Calculate Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std_dev        
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
    
       # Calculate the compounded return over the period
        cumulative_return = (1 + portfolio_returns).prod()  # Calculate cumulative return
        num_days = len(portfolio_returns)
        
        # Annualize the portfolio return
        annualized_return = cumulative_return ** (252 / num_days) - 1
        
        # Calculate downside deviation
        downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns - target_return) ** 2))
        
        trading_days = 252
        annualized_downside_deviation = downside_deviation * np.sqrt(trading_days)
        
        
        # Calculate Sortino ratio
        sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_deviation
        
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

    if check_dataframe(data, tickers, investments):

        # Calculate portfolio returns as a weighted sum of individual stock returns
        stock_returns = calculate_daily_returns(data[tickers]) 
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)
    
        cumulative_portfolio_return = (1 + portfolio_returns).cumprod()
        cumulative_portfolio_returns_max = cumulative_portfolio_return.cummax()

        drawdown = (cumulative_portfolio_return - cumulative_portfolio_returns_max) / cumulative_portfolio_returns_max
        max_drawdown = min(drawdown[1:])

        return max_drawdown
    
def calculate_analyst_suggestion(tickers: list[str], investments: list[float]) -> dict:
    """
    Calculate the weighted average analyst suggestion for a portfolio based on Yahoo Finance data. 1 is a strong buy and 5 is a strong sell.

    Parameters:
    tickers (list[str]): A list of stock tickers in the portfolio.
    investments (list[float]): A list of investment amounts corresponding to each stock in the portfolio.

    Returns:
    dict: A dictionary containing individual ticker suggestions and the weighted average suggestion for the portfolio.
    """
    suggestions = []
    weighted_suggestions = []
    adjusted_investments = []
    # Set up basic configuration for logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    for ticker, investment in zip(tickers, investments):
        try:
            stock_info = yf.Ticker(ticker).info
            analyst_suggestion = stock_info.get('recommendationMean', None)
            
            if analyst_suggestion:
                suggestions.append({
                    "ticker": ticker,
                    "suggestion": analyst_suggestion
                })
                weighted_suggestions.append(analyst_suggestion * investment)
                adjusted_investments.append(investment)
            else:
                logging.warning(f"No analyst suggestion available for {ticker}. Skipping...")
        
        except Exception as e:
            logging.error(f"Error retrieving data for {ticker}: {e}")
            continue

    if not suggestions:
        logging.warning("No valid analyst suggestions were retrieved for the portfolio.")
        return {
            "individual_suggestions": [],
            "weighted_average_suggestion": None
        }

    # Calculate the weighted average suggestion
    total_adjusted_investment = sum(adjusted_investments)
    weighted_average_suggestion = sum(weighted_suggestions) / total_adjusted_investment if total_adjusted_investment > 0 else None

    return {
        "individual_suggestions": suggestions,
        "weighted_average_suggestion": weighted_average_suggestion
    }

def calculate_portfolio_metrics(
    price_df: pd.DataFrame,
    tickers: list[str],
    investments: list[float], 
    start_date_report: str = None,  # Optional parameter for calculating returns from a start date
    investment_at_final_date: bool = True,  # Indicates if investments are based on the final date
    market_ticker: str = '^GSPC',
    risk_free_rate: float = 0.01
) -> dict:
    """
    Calculate portfolio metrics and returns.

    Parameters:
    price_df (pd.DataFrame): A DataFrame containing the historical price data of the stocks in the portfolio.
    tickers (list[str]): A list of stock tickers in the portfolio.
    investments (list[float]): A list of investment amounts corresponding to each stock in the portfolio.
    start_date_report (str, optional): The start date of the report to calculate returns in the format 'YYYY-MM-DD'.
    investment_at_final_date (bool): Whether investments are based on the final date. If False, uses the start date.
    market_ticker (str, optional): The ticker symbol of the market index to compare the portfolio against. Default is '^GSPC' (S&P 500).
    risk_free_rate (float, optional): The risk-free rate used in calculating the Sharpe and Sortino ratios. Default is 0.01 (1%).

    Returns:
    dict: A dictionary containing the calculated metrics and other data.
    """
    # Determine the last available date in the price data
    last_day = price_df.index[-1].strftime('%Y-%m-%d')
    first_day = price_df.index[0].strftime('%Y-%m-%d')

    # Use the start_date_report if provided; otherwise, use the first available date
    start_date = start_date_report if start_date_report else first_day

    # Calculate the number of shares for each stock
    if investment_at_final_date:
        shares = [investment / price_df[ticker].loc[last_day] for ticker, investment in zip(tickers, investments)]
    else:
        shares = [investment / price_df[ticker].loc[start_date] for ticker, investment in zip(tickers, investments)]

    # Calculate portfolio values
    portfolio_initial_value = sum(price_df[ticker].loc[start_date] * share for ticker, share in zip(tickers, shares))
    portfolio_final_value = sum(price_df[ticker].loc[last_day] * share for ticker, share in zip(tickers, shares))

    # Calculate market and portfolio returns from the start_date_report to the last available day
    market_return = (price_df[market_ticker].loc[last_day] / price_df[market_ticker].loc[start_date] - 1) * 100
    portfolio_return = (portfolio_final_value / portfolio_initial_value - 1) * 100

    # Calculate portfolio metrics
    beta, alpha = calculate_beta_and_alpha(price_df, tickers, investments, market_ticker)
    sharpe_ratio = calculate_sharpe_ratio(price_df, tickers, investments, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(price_df, tickers, investments, risk_free_rate)
    var = calculate_var(price_df, tickers, investments)
    max_drawdown = calculate_max_drawdown(price_df, tickers, investments)
    dividend_yield = calculate_dividend_yield(tickers, investments)

    # Calculate individual stock returns and monetary surplus/deficit
    stock_details = []
    for ticker, investment, share in zip(tickers, investments, shares):
        initial_value = share * price_df[ticker].loc[start_date]
        final_value = share * price_df[ticker].loc[last_day]
        stock_return = (final_value / initial_value - 1) * 100
        surplus_or_deficit = final_value - initial_value
        stock_details.append({
            "ticker": ticker,
            "initial_value": initial_value,
            "final_value": final_value,
            "return": stock_return,
            "surplus_or_deficit": surplus_or_deficit
        })

    return {
        "last_day": last_day,
        "first_metric_day": first_day,
        "portfolio_initial_value": portfolio_initial_value,
        "portfolio_final_value": portfolio_final_value,
        "portfolio_return": portfolio_return,
        "market_return": market_return,
        "beta": beta,
        "alpha": alpha,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "var": var,
        "max_drawdown": max_drawdown,
        "dividend_yield": dividend_yield,
        "stock_details": stock_details
    }