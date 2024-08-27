import yfinance as yf
import pandas as pd

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

def get_analyst_info(ticker):
    """Fetch stock info including target prices from yfinance."""
    stock_info = yf.Ticker(ticker).info
    try:
        return {
            'currentPrice': stock_info['currentPrice'],
            'targetLowPrice': stock_info['targetLowPrice'],
            'targetMeanPrice': stock_info['targetMeanPrice'],
            'targetHighPrice': stock_info['targetHighPrice'],
            'targetMedianPrice': stock_info['targetMedianPrice'],
            'currency': stock_info['currency']
        }
    except KeyError:
        # If target prices are not available, assume the current price remains the same
        return {
            'currentPrice': stock_info['currentPrice'],
            'targetLowPrice': stock_info['currentPrice'],
            'targetMeanPrice': stock_info['currentPrice'],
            'targetHighPrice': stock_info['currentPrice'],
            'targetMedianPrice': stock_info['currentPrice'],
            'currency': stock_info['currency']
        }

def get_current_rate(base_currency, quote_currency):
    """Fetch the exchange rate from quote_currency to base_currency."""
    if base_currency == quote_currency:
        return 1.0
    exchange_rate_ticker = f'{quote_currency}{base_currency}=X'
    exchange_rate_data = yf.download(exchange_rate_ticker, period='1d')['Adj Close'].iloc[-1]
    return exchange_rate_data

