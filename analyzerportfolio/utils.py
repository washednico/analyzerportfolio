import yfinance as yf
import pandas as pd
import logging
import numpy as np

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

def get_stock_info(ticker):
    """Fetch stock info including target prices from yfinance."""
    stock_info = yf.Ticker(ticker).info
    stock_data = yf.Ticker(ticker).info
    try:
        stock_info['currentPrice'] = stock_data['currentPrice']
    except KeyError:
        logging.warning(f"Current price not found for {ticker}.")
        stock_info['currentPrice'] = None

    try:
        stock_info['targetLowPrice'] = stock_data.get('targetLowPrice', stock_info['currentPrice'])
    except KeyError:
        logging.warning(f"Target low price not found for {ticker}.")
        stock_info['targetLowPrice'] = stock_info['currentPrice']

    try:
        stock_info['targetMeanPrice'] = stock_data.get('targetMeanPrice', stock_info['currentPrice'])
    except KeyError:
        logging.warning(f"Target mean price not found for {ticker}.")
        stock_info['targetMeanPrice'] = stock_info['currentPrice']

    try:
        stock_info['targetHighPrice'] = stock_data.get('targetHighPrice', stock_info['currentPrice'])
    except KeyError:
        logging.warning(f"Target high price not found for {ticker}.")
        stock_info['targetHighPrice'] = stock_info['currentPrice']

    try:
        stock_info['targetMedianPrice'] = stock_data.get('targetMedianPrice', stock_info['currentPrice'])
    except KeyError:
        logging.warning(f"Target median price not found for {ticker}.")
        stock_info['targetMedianPrice'] = stock_info['currentPrice']

    try:
        stock_info['dividendYield'] = stock_data.get('dividendYield', 0)  # Default to 0 if not available
    except KeyError:
        logging.warning(f"Dividend yield not found for {ticker}.")
        stock_info['dividendYield'] = 0

    try:
        stock_info['currency'] = stock_data['currency']
    except KeyError:
        logging.warning(f"Currency not found for {ticker}. Defaulting to USD.")
        stock_info['currency'] = 'USD'

    return stock_info

def get_current_rate(base_currency, quote_currency):
    """Fetch the exchange rate from quote_currency to base_currency."""
    if base_currency == quote_currency:
        return 1.0
    exchange_rate_ticker = f'{quote_currency}{base_currency}=X'
    exchange_rate_data = yf.download(exchange_rate_ticker, period='1d', start ="2024-01-01", end=None)['Adj Close'].iloc[-1]
    return exchange_rate_data


def forming_portfolio(
    data: pd.DataFrame,
    tickers : list[str],
    investments : list[float],
    market_ticker: str,
    return_period_days : int = 1,
    rebalancing_period_days: int = None,
    target_weights: list[float] = None
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
    - return_period_days: Integer representing the return period in days. Default is 1.
    - rebalancing_period_days: Optional integer representing the rebalancing period in days.
                               If None, no rebalancing is performed.
    - market_ticker: Optional string representing the market index ticker.
                     If provided, market returns and values will be calculated.
    - target_weights: Optional list or array of target weights (should sum to 1).
                      If not provided, it will be calculated from the initial investments.

    Returns:
    - returns_df: DataFrame containing:
        - Stock returns and values for each ticker.
        - 'Portfolio_Returns' and 'Portfolio_Value' columns for the portfolio without rebalancing.
        - 'Rebalanced_Portfolio_Returns' and 'Rebalanced_Portfolio_Value' columns (if rebalancing is performed).
        - 'Market_Returns' and 'Market_Value' columns (if market_ticker is provided).
    """
    # Ensure investments and target_weights are numpy arrays
    investments = np.array(investments)

    if target_weights is None:
        # Calculate target weights from initial investments
        total_investment = investments.sum()
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
    # Add stock returns
    for ticker in tickers:
        returns_df[f'{ticker}_Return'] = stock_returns[ticker]
        returns_df[f'{ticker}_Value'] = stock_values[ticker]

    # Add portfolio returns and values
    returns_df['Portfolio_Returns'] = portfolio_returns
    returns_df['Portfolio_Value'] = portfolio_values

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
        returns_df['Rebalanced_Portfolio_Returns'] = rebalanced_portfolio_returns
        returns_df['Rebalanced_Portfolio_Value'] = rebalanced_portfolio_values

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

        # Add market returns and values to the DataFrame
        returns_df['Market_Returns'] = market_returns
        returns_df['Market_Value'] = market_values

    # Drop rows with all NaN values (if any)
    returns_df.notna()

    return returns_df