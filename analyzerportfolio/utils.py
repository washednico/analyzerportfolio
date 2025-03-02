import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import timedelta
import re 
from .logger import logger 
import os
from typing import Union, List, Dict, Tuple
import logging


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
    exchange_rate_data = yf.download(exchange_rate_ticker, start=start_date, end=end_date)['Close']
    
    # Store the fetched exchange rate in the cache
    exchange_rate_cache[cache_key] = exchange_rate_data
    return exchange_rate_data

def convert_to_base_currency(prices, exchange_rate):
    """Convert prices to the base currency using the exchange rate."""
    if exchange_rate is None:
        return prices  # Already in base currency

    # Align exchange_rate index with prices index
    exchange_rate_aligned = exchange_rate.reindex(prices.index).ffill().bfill()

    # Ensure exchange_rate_aligned is a Series
    if isinstance(exchange_rate_aligned, pd.DataFrame):
        exchange_rate_aligned = exchange_rate_aligned.squeeze()

    return prices.multiply(exchange_rate_aligned, axis=0)


def get_stock_info(ticker):
    """Fetch stock info including target prices from yfinance."""
    stock_info = yf.Ticker(ticker).info
    stock_data = yf.Ticker(ticker).info
    try:
        stock_info['currentPrice'] = stock_data['currentPrice']
    except KeyError:
        logger.warning(f"Current price not found for {ticker}.")
        stock_info['currentPrice'] = None

    try:
        stock_info['targetLowPrice'] = stock_data.get('targetLowPrice', stock_info['currentPrice'])
    except KeyError:
        logger.warning(f"Target low price not found for {ticker}.")
        stock_info['targetLowPrice'] = stock_info['currentPrice']

    try:
        stock_info['targetMeanPrice'] = stock_data.get('targetMeanPrice', stock_info['currentPrice'])
    except KeyError:
        logger.warning(f"Target mean price not found for {ticker}.")
        stock_info['targetMeanPrice'] = stock_info['currentPrice']

    try:
        stock_info['targetHighPrice'] = stock_data.get('targetHighPrice', stock_info['currentPrice'])
    except KeyError:
        logger.warning(f"Target high price not found for {ticker}.")
        stock_info['targetHighPrice'] = stock_info['currentPrice']

    try:
        stock_info['targetMedianPrice'] = stock_data.get('targetMedianPrice', stock_info['currentPrice'])
    except KeyError:
        logger.warning(f"Target median price not found for {ticker}.")
        stock_info['targetMedianPrice'] = stock_info['currentPrice']

    try:
        stock_info['dividendYield'] = stock_data.get('dividendYield', 0)  # Default to 0 if not available
    except KeyError:
        logger.warning(f"Dividend yield not found for {ticker}.")
        stock_info['dividendYield'] = 0

    try:
        stock_info['currency'] = stock_data['currency']
    except KeyError:
        logger.warning(f"Currency not found for {ticker}. Defaulting to USD.")
        stock_info['currency'] = 'USD'

    return stock_info

def get_current_rate(base_currency, quote_currency):
    """Fetch the exchange rate from quote_currency to base_currency."""
    if base_currency == quote_currency:
        return 1.0
    exchange_rate_ticker = f'{quote_currency}{base_currency}=X'
    exchange_rate_data = yf.download(exchange_rate_ticker, period='1d', start ="2024-01-01", end=None)['Close'].iloc[-1]
    return exchange_rate_data


def download_data(tickers: list[str], market_ticker: str, start_date: str, end_date: str, base_currency: str,risk_free: str = "DTB3", use_cache: bool = False, folder_path: str = None) -> pd.DataFrame:
    """
    Download stock and market data, convert to base currency, and return the processed data.
    
    Parameters:
    tickers (list): List of stock tickers.
    market_ticker (str): Market index ticker.
    start_date (str): Start date for historical data.
    end_date (str): End date for historical data.
    base_currency (str): The base currency for the portfolio (e.g., 'USD').
    risk_free (str): The risk free rate to use in the calculations written as ticker on fred (e.g., 'DTB3' for USD).
    use_cache (bool): Whether to use cache to retrieve data, if data is not cached it will be stored for future computations. Default is False. 
    folder_path (str): Path to the folder where the cache will be stored. Default is None. 
    

    Returns:
    pd.DataFrame: DataFrame containing the adjusted and converted prices for all tickers and the market index.
    """

    def validate_tickers(tickers):
        """
        Validates and corrects a ticker list to ensure there are no formatting issues:
        1. Ensures the list contains valid tickers as strings.
        2. Detects and fixes concatenated tickers (e.g., missing commas).
        3. Checks if each ticker follows the expected format (valid stock ticker).

        Returns:
        list: A corrected ticker list with issues fixed where possible.

        Raises:
        ValueError: If validation conditions are violated.
        """
        # Ensure the input is a list
        if not isinstance(tickers, list):
            raise ValueError("Input must be a list of tickers.")
        # Ensure all elements are strings
        if not all(isinstance(ticker, str) for ticker in tickers):
            raise ValueError("All tickers in the list must be strings.")

        # Pattern for a valid stock ticker
        symbol_pattern = re.compile(r'^[A-Za-z0-9-]+(\.[A-Za-z]{1,3})?$')

        corrected_list = []

        for idx, ticker in enumerate(tickers):
            # Validate if the ticker matches the expected pattern
            if symbol_pattern.fullmatch(ticker):
                corrected_list.append(ticker)  # Valid ticker
            else:
                # Try to detect and fix concatenated tickers
                fixed = False
                for i in range(1, len(ticker)):
                    part1 = ticker[:i]
                    part2 = ticker[i:]
                    if symbol_pattern.fullmatch(part1) and symbol_pattern.fullmatch(part2):
                        print(f"Missing comma detected at position {idx}: '{ticker}' -> '{part1}', '{part2}'")
                        corrected_list.extend([part1, part2])
                        fixed = True
                        break
                if not fixed:
                    # If unable to fix, log an error and raise an exception
                    raise ValueError(f"Invalid ticker at position {idx}: '{ticker}'")

        return corrected_list
            
    exchange_rate_cache = {}
    stock_data = pd.DataFrame()
    tickers = validate_tickers(tickers)

    def cached_exchange_rates(base_currency, currency, start_date, end_date, exchange_rate_cache, folder_path):
        csv_exchange_rate = "/"+base_currency+"_"+currency+".csv"
        try:
            exchange_rate_df = pd.read_csv(folder_path + csv_exchange_rate, index_col=0, parse_dates=True)
            exchange_rate = exchange_rate_df.squeeze()
            exchange_rate.index = pd.to_datetime(exchange_rate.index, errors='coerce')
            # Get the first and last date in the cached data
            first_date_cached = exchange_rate.index[0]
            last_date_cached = exchange_rate.index[-1]
            

            # Convert start_date and end_date to datetime for comparison
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)


            if first_date_cached > start_date_dt or last_date_cached < end_date_dt - timedelta(days=1):
                missing_data = get_exchange_rate(base_currency, currency, start_date, end_date, exchange_rate_cache)
                exchange_rate.index = pd.to_datetime(exchange_rate.index, errors='coerce')
                missing_data.index = pd.to_datetime(missing_data.index, errors='coerce')
                full_exchange_rate_data = pd.concat([exchange_rate, missing_data]).sort_index()
                full_exchange_rate_data.to_csv(folder_path + csv_exchange_rate)
                return missing_data
            
            else:
                filtered_data = exchange_rate[(exchange_rate.index >= start_date) & (exchange_rate.index <= end_date)]
                return filtered_data

        
        except FileNotFoundError:
            exchange_rate = get_exchange_rate(base_currency, currency, start_date, end_date, exchange_rate_cache)
            exchange_rate.to_csv(folder_path + csv_exchange_rate)

        return exchange_rate
    
    def get_interest_rates(risk_free, start_date, end_date):
        # Make a request to the specified URL
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="+risk_free+"&cosd="+start_date+"&coed="+end_date+"&fq=Daily%2C%207-Day&fam=avg"
        
        response = requests.get(url)
        if response.status_code == 200:
            # Get the CSV data from the response
            csv_data = response.text
            
            interest_data = pd.read_csv(
                StringIO(csv_data)
            ).rename(columns={"DATE": "observation_date", risk_free: "Interest_Rates"})
            
            # Convert 'Date' to datetime and set as index
            interest_data['observation_date'] = pd.to_datetime(interest_data['observation_date'])
            interest_data.set_index('observation_date', inplace=True)
            

            # Ensure 'stock_data' index is datetime and aligned
            stock_data.index = pd.to_datetime(stock_data.index)

            stock_data.index = stock_data.index.tz_localize(None)
            interest_data.index = interest_data.index.tz_localize(None)
            # Reindex 'interest_data' to match 'stock_data' dates
            interest_data = interest_data.reindex(stock_data.index)
            
            # Convert the column to float, forcing invalid strings to NaN (if any)
            interest_data['Interest_Rates'] = interest_data['Interest_Rates'].replace('.', np.nan)

            interest_data['Interest_Rates'] = interest_data['Interest_Rates'].astype(float)

            # Forward-fill missing interest rates
            interest_data['Interest_Rates'] = interest_data['Interest_Rates'].ffill()
            interest_data['Interest_Rates'] = interest_data['Interest_Rates'].bfill()
            return interest_data

        else:
            print("Risk Free request failed with status code:", response.status_code)
            return None
        
    def cached_interest_rates(risk_free, start_date, end_date, folder_path):
        try:
            interest_rate_df = pd.read_csv(folder_path + "/"+risk_free+".csv", index_col=0, parse_dates=True)
            interest_rate = interest_rate_df.squeeze()
            interest_rate.index = pd.to_datetime(interest_rate.index, errors='coerce')
            # Get the first and last date in the cached data
            first_date_cached = interest_rate.index[0]
            last_date_cached = interest_rate.index[-1]

            # Convert start_date and end_date to datetime for comparison
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            if first_date_cached > start_date_dt or last_date_cached < end_date_dt - timedelta(days=1):
                missing_data = get_interest_rates(risk_free, start_date, end_date)
                interest_rate.index = pd.to_datetime(interest_rate.index, errors='coerce')
                missing_data.index = pd.to_datetime(missing_data.index, errors='coerce')
                full_interest_rate_data = pd.concat([interest_rate, missing_data]).sort_index()
                full_interest_rate_data.to_csv(folder_path + "/"+risk_free+".csv")
                return missing_data
            else:
                filtered_data = interest_rate[(interest_rate.index >= start_date) & (interest_rate.index <= end_date)]
                return filtered_data
        except FileNotFoundError:
            interest_rate = get_interest_rates(risk_free, start_date, end_date)
            interest_rate.to_csv(folder_path + "/"+risk_free+".csv")
            return interest_rate

    # Fetch and process each stock's data
    if use_cache:
        for ticker in tickers + [market_ticker]:
            try:
                ticker_data_df = pd.read_csv(folder_path + "/"+ticker+".csv", index_col=0, parse_dates=True)
                ticker_data = ticker_data_df.squeeze()
                # Ensure the index is datetime
                ticker_data.index = pd.to_datetime(ticker_data.index, errors='coerce')
                
                column_split = ticker_data.name.split(" ")
                currency = column_split[-1]
                ticker_data.name = "Close"

                first_date_cached = ticker_data.index[0]
                last_date_cached = ticker_data.index[-1]
                

                # Convert start_date and end_date to datetime for comparison
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)
                

                if first_date_cached > start_date_dt or last_date_cached < end_date_dt - timedelta(days=1):
                    missing_data = yf.download(ticker, start=start_date, end=end_date)['Close']
                    missing_data.index = missing_data.index.tz_localize(None)
                    # Create a complete date range from start to end date
                    date_range = pd.date_range(start=start_date_dt, end=end_date_dt - timedelta(days=1))

                    # Reindex the data to the complete date range and fill missing entries with NaN
                    missing_data = missing_data.reindex(date_range)
                    
                    ticker_data.index = pd.to_datetime(ticker_data.index, errors='coerce')
                    missing_data.index = pd.to_datetime(missing_data.index, errors='coerce')
                    full_stock_data = pd.concat([ticker_data, missing_data]).sort_index()
                    full_stock_data = full_stock_data[~full_stock_data.index.duplicated(keep='last')]
                    
                    data_to_save = full_stock_data.copy()
                    data_to_save.name = f"Close {currency}"
                    data_to_save.to_csv(folder_path + "/"+ticker+".csv")

                    if currency != base_currency:
                        exchange_rate = cached_exchange_rates(base_currency, currency, start_date, end_date, exchange_rate_cache, folder_path)
                        data = convert_to_base_currency(missing_data, exchange_rate)
                    else:
                        data = missing_data
                
                else:
                    data = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= end_date)]
                    
                
                stock_data[ticker] = ticker_data
                        
                    
            except FileNotFoundError:
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)

                data = yf.download(ticker, start=start_date, end=end_date)['Close']
                data.index = data.index.tz_localize(None)

                date_range = pd.date_range(start=start_date_dt, end=end_date_dt - timedelta(days=1))
                data = data.reindex(date_range)
                # Reindex the data to the complete date range and fill missing entries with NaN
                

                currency = get_currency(ticker)
                data_to_save = data.copy()
                data_to_save.name = f"Close {currency}"
                data_to_save.to_csv(folder_path + "/"+ticker+".csv")

                
                
                if currency != base_currency:
                    
                    exchange_rate = cached_exchange_rates(base_currency, currency, start_date, end_date, exchange_rate_cache, folder_path)
                    data = convert_to_base_currency(data, exchange_rate)
        
                stock_data[ticker] = data

                
    else:
        for ticker in tickers + [market_ticker]:
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            currency = get_currency(ticker)
            if currency != base_currency:
                exchange_rate = get_exchange_rate(base_currency, currency, start_date, end_date, exchange_rate_cache)
                data = convert_to_base_currency(data, exchange_rate)
            stock_data[ticker] = data  


    
    if use_cache:
        interest_data = cached_interest_rates(risk_free, start_date, end_date, folder_path)
    else:
        interest_data = get_interest_rates(risk_free, start_date, end_date)
        
        
            
             
            
    
    if interest_data is not None:
        # Merge on index
        dataframe = stock_data.join(interest_data, how='inner')
        return dataframe
    else:
        return None

        
        
def remove_small_weights(portfolio_returns:dict, threshold:float=0.0005) -> dict:
    """
    Removes assets with weights below a specified threshold while maintaining the original structure of the portfolio data.

    Parameters:
    - portfolio_returns (dict): Dictionary containing portfolio data, including tickers, investments, and weights.
    - threshold (float): Optional float representing the minimum acceptable weight for assets to remain in the portfolio. Default is 0.001.

    Returns:
    - portfolio_returns(dict): Updated portfolio dictionary with assets that meet the weight threshold. 
                        The 'tickers', 'investments', and 'weights' fields are filtered accordingly.
    """
    # Extract the relevant data
    tickers = portfolio_returns["tickers"]
    investments = portfolio_returns["investments"]
    weights = portfolio_returns["weights"]
    
    # Filter assets based on the weight threshold
    filtered_data = [(ticker, investment, weight) for ticker, investment, weight in zip(tickers, investments, weights) if weight >= threshold]
    
    # Unzip the filtered data back into separate lists
    if filtered_data:
        tickers, investments, weights = zip(*filtered_data)
    else:
        tickers, investments, weights = [], [], []

    # Update the portfolio dictionary
    portfolio_returns["tickers"] = list(tickers)
    portfolio_returns["investments"] = list(investments)
    portfolio_returns["weights"] = list(weights)

    new_portfolio = update_portfolio(portfolio_returns)

    return new_portfolio

def create_portfolio(
    data: pd.DataFrame,
    tickers : list[str],
    investments : list[float],
    market_ticker: str,
    name_portfolio: str,
    base_currency: str,
    return_period_days : int = 1,
    rebalancing_period_days: int = None,
    target_weights: list[float] = None,
    exclude_ticker_time: int = 7,
    exclude_ticker: bool = False
) -> pd.DataFrame:
    """
    Calculates returns and value amounts for specified stocks over a return period,
    the portfolio without rebalancing, optionally the portfolio with auto-rebalancing,
    and includes market index calculations.

    Parameters:
    - data: DataFrame with adjusted closing prices (index as dates, columns as tickers).
    - tickers: List of stock tickers in the portfolio.
    - investments: List or array of initial investments for each stock.
    - market_ticker: String representing the market index ticker.
    - name_portfolio: String representing the name of the portfolio
    - base_currency: String representing the base currency for the portfolio.
    - return_period_days: Integer representing the return period in days. Default is 1.
    - rebalancing_period_days: Optional integer representing the rebalancing period in days.
                               If None, no rebalancing is performed.
    - market_ticker: Optional string representing the market index ticker.
                     If provided, market returns and values will be calculated.
    - target_weights: Optional list or array of target weights (should sum to 1).
                      If not provided, it will be calculated from the initial investments.
    - exclude_ticker_time (int): if ticker is not available withing +- x days from start date, exclude it. Default is 7.
    - exclude_ticker (bool): Apply the exclusion of tickers based on the exclude_ticker_time parameter. Default is False.

    Returns:
    - returns_df: DataFrame containing:
        - Stock returns and values for each ticker.
        - 'Portfolio_Returns' and 'Portfolio_Value' columns for the portfolio without rebalancing.
        - 'Rebalanced_Portfolio_Returns' and 'Rebalanced_Portfolio_Value' columns (if rebalancing is performed).
        - 'Market_Returns' and 'Market_Value' columns (if market_ticker is provided).
    """
    

    # Configure the logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def count_initial_nans(series):
        count = 0
        for value in series:
            if pd.isna(value):
                count += 1
            else:
                break
        return count

    if exclude_ticker:
        # Create a list of tickers to exclude based on consecutive NaNs at the start
        to_exclude = [ticker for ticker in tickers if count_initial_nans(data[ticker]) > exclude_ticker_time]
        
        # Loop through the tickers to be excluded
        for ticker in to_exclude:
            # Find the index before modifying the tickers list
            index = tickers.index(ticker)
            
            # Remove the ticker from tickers and investments based on the index
            tickers.pop(index)
            investments.pop(index)
            
            # Log the exclusion message
            print(f"{ticker} has been excluded from the portfolio due to missing data.")
            data.drop(columns=ticker, inplace=True)
        
        # Now drop NaN values from the remaining data
        data.dropna(inplace=True)
    else:
        data.dropna(inplace=True)

    # Ensure investments and target_weights are numpy arrays
    investments = np.array(investments)
    total_investment = investments.sum()
    initial_target = target_weights

    if target_weights is None:
        # Calculate target weights from initial investments
        target_weights = investments / total_investment
    else:
        target_weights = np.array(target_weights)

    # Validate inputs
    if len(tickers) != len(investments):
        raise ValueError("The length of 'tickers' and 'investments' must be the same.")
    if not np.isclose(target_weights.sum(), 1):
        raise ValueError("Target weights must sum to 1.")
    if not set(tickers).issubset(data.columns):
        missing_tickers = set(tickers) - set(data.columns)
        raise ValueError(f"The following tickers are missing in the data: {missing_tickers}")
    if market_ticker and market_ticker not in data.columns:
        raise ValueError(f"Market ticker '{market_ticker}' is not present in the data.")
    
    # Select data for the specified tickers
    stock_data = data[tickers]
    
    # Calculate returns for each stock over the specified return period
    stock_returns = stock_data.pct_change(return_period_days)

    # ----- Stock Values Over Time -----
    # Calculate initial shares for each stock
    initial_prices = stock_data.iloc[0]
    shares = investments / initial_prices

    # Calculate stock values over time
    stock_values = stock_data.multiply(shares, axis=1)

    # ----- Portfolio Without Rebalancing -----

    # Calculate portfolio value over time
    portfolio_values = stock_values.sum(axis=1)

    # Calculate returns of the portfolio over the specified return period
    portfolio_returns = portfolio_values.pct_change(return_period_days)

    # Create a DataFrame to store returns and values
    returns_df = pd.DataFrame(index=stock_data.index)
    


    # ----- Portfolio With Auto-Rebalancing (if rebalancing_period_days is provided) -----
    if rebalancing_period_days is not None:
        # Initialize Series to store portfolio values with rebalancing
        rebalanced_portfolio_values = pd.Series(index=stock_data.index, dtype=float)
        rebalanced_portfolio_values.iloc[0] = total_investment

        # Initialize shares based on initial target weights
        shares_rebalanced = (total_investment * target_weights) / initial_prices
        last_rebalance_date = stock_data.index[0]

        # Initialize DataFrame to store rebalanced stock values
        rebalanced_stock_values = pd.DataFrame(index=stock_data.index, columns=tickers, dtype=float)

        for date in stock_data.index:
            current_prices = stock_data.loc[date]
            # Update stock values
            stock_values_rebalanced = shares_rebalanced * current_prices
            rebalanced_stock_values.loc[date] = stock_values_rebalanced

            # Update portfolio value
            portfolio_value = stock_values_rebalanced.sum()
            rebalanced_portfolio_values.loc[date] = portfolio_value

            # Check if it's time to rebalance
            days_since_last_rebalance = (date - last_rebalance_date).days
            if days_since_last_rebalance >= rebalancing_period_days or date == stock_data.index[-1]:
                # Rebalance the portfolio
                shares_rebalanced = (portfolio_value * target_weights) / current_prices
                last_rebalance_date = date

        # Calculate returns of the rebalanced portfolio over the specified return period
        rebalanced_portfolio_returns = rebalanced_portfolio_values.pct_change(return_period_days)
        
        # Add rebalanced portfolio returns and values to the DataFrame
        returns_df['Portfolio_Returns'] = rebalanced_portfolio_returns
        returns_df['Portfolio_Value'] = rebalanced_portfolio_values

        # Add rebalanced stock values to the DataFrame
        for ticker in tickers:
            returns_df[f'Rebalanced_{ticker}_Value'] = rebalanced_stock_values[ticker]

    # ----- Market Index Calculations (if market_ticker is provided) -----

    if market_ticker is not None:
        # Extract market data
        market_data = data[market_ticker]

        # Calculate market returns over the specified return period
        market_returns = market_data.pct_change(return_period_days)

        # Calculate market investment value over time
        initial_market_price = market_data.iloc[0]
        market_shares = total_investment / initial_market_price
        market_values = market_data * market_shares


    # ----- Interest Rate Adjustments -----

    # Convert annual interest rates to decimal form if necessary
    
    if  data['Interest_Rates'] .max() > 1:
         data['Interest_Rates']  =  data['Interest_Rates']  / 100
    

    # Number of days in a year
    N = 252

    # Convert annual rates to daily rates using compounded formula
    data['daily_risk_free_rate'] = (1 + data['Interest_Rates']) ** (1 / N) - 1

    # Calculate cumulative risk-free returns over the return period
    data['risk_free_return'] = data['daily_risk_free_rate'].rolling(window=return_period_days).apply(
        lambda x: np.prod(1 + x) - 1, raw=True
    )

    # Add risk-free return to returns_df
    

        # Add stock returns
    columns_to_add = {
                        **{f'{ticker}_Return': stock_returns[ticker] for ticker in tickers},
                        **{f'{ticker}_Value': stock_values[ticker] for ticker in tickers},
                        'Portfolio_Returns': portfolio_returns,
                        'Portfolio_Value': portfolio_values,
                        'Risk_Free_Return' : data['risk_free_return'],
                        # Add market returns and values to the DataFrame
                        'Market_Returns' : market_returns,
                        'Market_Value' : market_values
                    }

# Create a new DataFrame with these columns
    returns_df = pd.DataFrame(columns_to_add, index=stock_data.index)
    

    # Drop rows with all NaN values (if any)
    returns_df.notna()

    portfolio_returns = {
        "name": name_portfolio,
        "auto_rebalance" : rebalancing_period_days,
        "tickers": tickers,
        "investments": investments,
        "weights": investments / total_investment,
        "base_currency": base_currency,
        "returns": returns_df,
        "market_ticker": market_ticker,
        "return_period_days": return_period_days,
        "market_returns" : returns_df["Market_Returns"],
        "market_value" : returns_df["Market_Value"],
        "portfolio_returns" : returns_df["Portfolio_Returns"],
        "portfolio_value" : returns_df["Portfolio_Value"],
        "risk_free_returns" : returns_df["Risk_Free_Return"],
        "untouched_data" : data,
        "target_weights" : initial_target,
        "exclude_ticker_time" : exclude_ticker_time,
        "exclude_ticker" : exclude_ticker

    }
    
    return portfolio_returns


def read_portfolio_composition(portfolio, min_value=0.01):
    """Read the composition of the portfolio from the portfolio dictionary and filter by minimum weight."""
    composition = {}
    tickers = portfolio["tickers"]
    weights = portfolio["weights"]
    
    for ticker, weight in zip(tickers, weights):
        if weight > min_value:
            composition[ticker] = weight
    
    return composition


def update_portfolio(portfolio_dict):
    result = create_portfolio(
        data=portfolio_dict["untouched_data"],
        tickers=portfolio_dict["tickers"],
        investments=portfolio_dict["investments"],
        market_ticker=portfolio_dict["market_ticker"],
        name_portfolio=portfolio_dict["name"],
        base_currency=portfolio_dict["base_currency"],
        return_period_days=portfolio_dict["return_period_days"],
        rebalancing_period_days=portfolio_dict["auto_rebalance"],
        target_weights=portfolio_dict["target_weights"],
        exclude_ticker_time=portfolio_dict["exclude_ticker_time"],
        exclude_ticker=portfolio_dict["exclude_ticker"]
    )
    return result



## - Graphics utility functions -- ##

def prepare_portfolios(portfolios: Union[dict, List[dict]]) -> List[dict]:
    """
    Ensures portfolios are in a consistent list format and validates their existence.

    Parameters
    ----------
    portfolios : Union[dict, List[dict], None]
        A dictionary, a list of dictionaries, or None representing portfolios.

    Returns
    -------
    List[dict]
        A list of prepared portfolios.

    Raises
    ------
    ValueError
        If no portfolios are provided or if portfolios is None.
    """
    # Check if portfolios is None
    if portfolios is None:
        logger.error("Portfolios cannot be None.")
        raise ValueError("Portfolios cannot be None.")

    # Ensure portfolios is a list
    if isinstance(portfolios, dict):
        portfolios = [portfolios]
        logger.debug("Converted a single portfolio dictionary into a list.")

    # Check if portfolios list is empty
    if len(portfolios) == 0:
        logger.error("At least one portfolio must be provided.")
        raise ValueError("At least one portfolio must be provided.")

    return portfolios

def prepare_portfolios_colors(
    portfolios: Union[dict, List[dict]],
    colors: Union[None, str, List[str]] = None
) -> Tuple[List[dict], List[Union[str, None]]]:
    """
    Prepares portfolios and colors for consistent processing.

    Parameters
    ----------
    portfolios : Union[dict, List[dict]]
        A dictionary or list of dictionaries representing portfolios.
    colors : Union[None, str, List[str]], optional
        Colors associated with each portfolio. Can be None, a single string, or a list of strings.

    Returns
    -------
    Tuple[List[dict], List[Union[str, None]]]
        A tuple of prepared portfolios (as a list) and their corresponding colors (as a list).

    Raises
    ------
    ValueError
        If the length of 'colors' does not match the number of portfolios.
    TypeError
        If 'colors' is not of type None, str, or List[str].
    """
    # Use prepare_portfolios to validate and prepare portfolios
    portfolios = prepare_portfolios(portfolios)

    # Prepare colors
    if colors is None:
        colors = [None] * len(portfolios)
        logger.debug("No colors provided; using default colors.")
    elif isinstance(colors, str):
        colors = [colors]
        logger.debug("Single color provided; converted to list.")
    elif isinstance(colors, list):
        if len(colors) != len(portfolios):
            logger.error("Length of 'colors' does not match the number of portfolios.")
            raise ValueError("The length of 'colors' must match the number of portfolios.")
    else:
        logger.error("Invalid type for 'colors' parameter.")
        raise TypeError("Invalid type for 'colors' parameter.")

    return portfolios, colors

def process_market(
    portfolios: List[dict],
    type: str = "value"
) -> pd.DataFrame:
    """
    Processes market values and retrieves the market name.

    Parameters
    ----------
    portfolios : List[dict]
        A list of portfolio dictionaries. The first portfolio containing market data is expected.

    Returns
    -------
    Tuple[pd.Series, str]
        The market values (pd.Series) and the market name (str).

    Raises
    ------
    ValueError
        If market values are not provided or invalid.
    """

    # Iterate portofolios to find a market_name
    for portfolio in portfolios:
        market_type = portfolio.get(f'market_{type}', None)           # Get makret values or returns
        market_name = portfolio.get('market_ticker', None)            # Get market_ticker (str)

        if market_name and market_type is not None:
            # Convert in case market data are in pd.Dataframe
            if isinstance(market_type, pd.DataFrame):
                market_type = market_type.squeeze()
                logger.debug(f"Converted Market_{type} from DataFrame to Series.")

            try:
                market_type.index = pd.to_datetime(market_type.index)
                logger.debug(f"Successfully loaded Market_{type} for '{market_name}'.")
                return market_type, market_name
            except Exception as e:
                logger.error(f"Error processing  Market_{type} index for '{market_name}': {e}")
                raise ValueError("Invalid market returns index format.") from e
        
    logger.warning("Market returns not found in any portfolio.")
    raise ValueError("Market returns not provided or invalid.")

def align_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns all time series in the DataFrame by dropping rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing multiple time series to be aligned.

    Returns
    -------
    pd.DataFrame
        The aligned DataFrame with no missing values.

    Logs
    ----
    - Logs the shape of the DataFrame before and after alignment.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    original_shape = df.shape
    aligned_df = df.dropna()

    logger.debug(f"Aligned data by dropping rows with missing values. Original shape: {original_shape}, Final shape: {aligned_df.shape}")

    return aligned_df

def save_data(
    df: pd.DataFrame,
    save_format: str = "csv",
    save_path: str = None,
    default_filename: str = "default_data",
) -> None:
    """
    Saves a DataFrame based on the provided parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    save_format : str, optional
        The file format to save in ('csv' or 'xlsx'). Default is 'csv'.
    should_save : bool, optional
        Whether to save the DataFrame. Default is False.
    save_path : str, optional
        The path or directory where the file will be saved. If None, defaults to the working directory.
    default_filename : str, optional
        The default filename (without extension) to use if no specific file is provided.

    Returns
    -------
    None

    Logs
    ----
    - Logs success or failure during the save operation.

    Raises
    ------
    ValueError
        If the specified file format is unsupported.
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if save_format not in ["csv", "xlsx"]:
        raise ValueError("Unsupported file format. Please choose 'csv' or 'xlsx'.")

    # Determine the full file path
    if save_path is None:
        save_path = os.path.join(os.getcwd(), f"{default_filename}.{save_format}")
    elif os.path.isdir(save_path):
        save_path = os.path.join(save_path, f"{default_filename}.{save_format}")
    elif not save_path.endswith(f".{save_format}"):
        save_path = f"{save_path}.{save_format}"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if save_format == "csv":
            df.to_csv(save_path, index=True)
            logger.info(f"DataFrame successfully saved as CSV at '{save_path}'.")
        elif save_format == "xlsx":
            df.to_excel(save_path, index=True, engine="openpyxl")
            logger.info(f"DataFrame successfully saved as XLSX at '{save_path}'.")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {save_format.upper()} at '{save_path}': {e}")
        raise

