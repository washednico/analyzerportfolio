import yfinance as yf
import pandas as pd
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

def check_dataframe(data: pd.DataFrame, tickers: list[str], investments:list[float] = None, market_ticker:str = None) -> bool:
    """
    Check if necessary variables exist in the provided data
    """
    #Ensure there is a position in all tickers
    if investments is not None:
        if len(tickers) != len(investments):
            raise ValueError("The number of tickers must match the number of investments.")

    # Ensure the market index is in the DataFrame (OPTIONAL)
    if market_ticker is not None:
        if market_ticker not in data.columns:
            raise ValueError(f"Market index '{market_ticker}' not found in the provided data.")
    
    # Ensure all tickers are in the DataFrame
    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]
    if [ticker for ticker in tickers if ticker not in data.columns]:
        raise ValueError(f"Tickers {missing_tickers} not found in the provided data.")
    
    return True