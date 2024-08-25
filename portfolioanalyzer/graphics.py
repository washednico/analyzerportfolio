import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_currency(ticker):
    """Fetch the currency of the given ticker using yfinance."""
    ticker_info = yf.Ticker(ticker).info
    return ticker_info['currency']

def get_exchange_rate(base_currency, quote_currency, start_date, end_date):
    """Fetch the historical exchange rates from quote_currency to base_currency."""
    if base_currency == quote_currency:
        return None  # No conversion needed

    exchange_rate_ticker = f'{quote_currency}{base_currency}=X'
    exchange_rate_data = yf.download(exchange_rate_ticker, start=start_date, end=end_date)['Adj Close']
    return exchange_rate_data

def convert_to_base_currency(prices, exchange_rate):
    """Convert prices to the base currency using the exchange rate."""
    if exchange_rate is None:
        return prices  # Already in base currency
    return prices * exchange_rate

def compare_portfolio_to_market(tickers: list, investments: list, start_date: str, end_date: str, market_index: str = '^GSPC', base_currency: str = 'USD'):
    """
    Compare the portfolio's return with the market's return and plot the comparison with currency conversion.

    Parameters:
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock.
    start_date (str): Start date for historical data.
    end_date (str): End date for historical data.
    market_index (str): The market index to compare against (default is S&P 500, '^GSPC').
    base_currency (str): The base currency for the portfolio (e.g., 'USD').

    Returns:
    None: The function plots the results and shows them.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = np.array(investments) / total_investment
    
    # Fetch adjusted closing prices for the tickers
    stock_data = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        currency = get_currency(ticker)
        if currency != base_currency:
            exchange_rate = get_exchange_rate(base_currency, currency, start_date, end_date)
            data = convert_to_base_currency(data, exchange_rate)
        stock_data[ticker] = data
    
    # Fetch and convert market index data
    market_data = yf.download(market_index, start=start_date, end=end_date)['Adj Close']
    market_currency = get_currency(market_index)
    if market_currency != base_currency:
        exchange_rate = get_exchange_rate(base_currency, market_currency, start_date, end_date)
        market_data = convert_to_base_currency(market_data, exchange_rate)
    
    # Combine stock data and market data into one DataFrame and drop rows with missing data
    stock_data['Market'] = market_data
    combined_data = stock_data.dropna()
    
    # Calculate daily returns
    stock_returns = combined_data[tickers].pct_change().dropna()
    market_returns = combined_data['Market'].pct_change().dropna()
    
    # Calculate portfolio daily returns as a weighted sum of individual stock returns
    portfolio_returns = stock_returns.dot(weights)
    
    # Calculate cumulative returns
    portfolio_cumulative_return = (1 + portfolio_returns).cumprod() * total_investment
    market_cumulative_return = (1 + market_returns).cumprod() * total_investment
    
    # Plotting the results with a black background
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cumulative_return, label='Portfolio', color='orange', linewidth=2)
    plt.plot(market_cumulative_return, label=f'{market_index}', color='green', linewidth=2)
    
    # Customizing the plot with a black background
    plt.title('Portfolio vs Market Performance', color='white', fontsize=16)
    plt.xlabel('Date', color='white')
    plt.ylabel('Value ($)', color='white')
    plt.legend(facecolor='black', edgecolor='white', fontsize=12, loc='best', labelcolor='white')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    
    # Set background color
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    # Customize the tick colors
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    
    plt.show()