import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_portfolio_to_market(data: pd.DataFrame, tickers: list, investments: list, market_index: str):
    """
    Compare the portfolio's return with the market's return and plot the comparison.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers and the market index.
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of monetary investments for each stock.
    market_index (str): The market index to compare against.

    Returns:
    None: The function plots the results and shows them.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Calculate the total portfolio value
    total_investment = sum(investments)
    
    # Convert monetary investments to weights (percentages of total portfolio value)
    weights = np.array(investments) / total_investment
    
    # Ensure all tickers and market index are in the DataFrame
    missing_columns = [ticker for ticker in tickers + [market_index] if ticker not in data.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the provided data.")
    
    # Calculate daily returns
    stock_returns = data[tickers].pct_change().dropna()
    market_returns = data[market_index].pct_change().dropna()
    
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
