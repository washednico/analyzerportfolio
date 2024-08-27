import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

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


def simulate_pac(data: pd.DataFrame, tickers: list, initial_investment: float, periodic_investment: float, investment_interval: int, investment_weights: list):
    """
    Simulate a PAC (Piano di Accumulo) investment strategy and plot the portfolio's growth over time.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers.
    tickers (list): List of stock tickers in the portfolio.
    initial_investment (float): Initial amount to invest at the start.
    investment_amount (float): Amount to invest at each interval.
    investment_interval (int): Interval (in days) between each additional investment.
    investment_weights (list): List of weights representing the percentage of each stock in the portfolio.

    Returns:
    None: The function plots the portfolio growth over time.
    """
    if len(tickers) != len(investment_weights):
        raise ValueError("The number of tickers must match the number of weights.")
    
    # Normalize weights to ensure they sum to 1
    weights = np.array(investment_weights)
    weights = weights / weights.sum()
    
    # Initialize variables
    dates = data.index
    portfolio_values = []
    total_invested = []
    
    # Initialize holdings (number of shares) for each stock
    holdings = {ticker: 0 for ticker in tickers}
    cumulative_investment = initial_investment
    next_investment_date = dates[0]
    
    # Initial investment
    current_prices = data.loc[next_investment_date]
    for ticker, weight in zip(tickers, weights):
        shares = (initial_investment * weight) / current_prices[ticker]
        holdings[ticker] += shares
    
    # Record initial portfolio value and investment
    total_value = sum(holdings[ticker] * current_prices[ticker] for ticker in tickers)
    portfolio_values.append(total_value)
    total_invested.append(cumulative_investment)
    
    # Iterate through each date
    for date in dates[1:]:
        # Make periodic investment on specified intervals
        if (date - next_investment_date).days >= investment_interval:
            current_prices = data.loc[date]
            for ticker, weight in zip(tickers, weights):
                shares = (periodic_investment * weight) / current_prices[ticker]
                holdings[ticker] += shares
            cumulative_investment += periodic_investment
            next_investment_date = date
        
        # Calculate portfolio value for the current date
        current_prices = data.loc[date]
        total_value = sum(holdings[ticker] * current_prices[ticker] for ticker in tickers)
        portfolio_values.append(total_value)
        total_invested.append(cumulative_investment)
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Portfolio Value': portfolio_values,
        'Total Invested': total_invested
    }, index=dates)
    
    # Plotting the results with a black background
    plt.figure(figsize=(14, 8))
    plt.plot(plot_data.index, plot_data['Portfolio Value'], label='Portfolio Value', color='orange', linewidth=2)
    plt.plot(plot_data.index, plot_data['Total Invested'], label='Total Invested', color='green', linewidth=2, linestyle='--')
    
    # Customizing the plot with a black background
    plt.title('Portfolio Growth with PAC Strategy', color='white', fontsize=18, pad=20)
    plt.xlabel('Date', color='white', fontsize=14)
    plt.ylabel('Value ($)', color='white', fontsize=14)
    plt.legend(facecolor='black', edgecolor='white', fontsize=12, loc='upper left', labelcolor='white')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set background color
    plt.gca().set_facecolor('#121212')
    plt.gcf().set_facecolor('#121212')
    
    # Customize the tick colors and sizes
    plt.tick_params(axis='x', colors='white', labelsize=12)
    plt.tick_params(axis='y', colors='white', labelsize=12)
    
    # Add annotations for final values
    final_date = plot_data.index[-1]
    final_portfolio_value = plot_data['Portfolio Value'][-1]
    final_total_invested = plot_data['Total Invested'][-1]
    gain = final_portfolio_value - final_total_invested
    gain_percentage = (gain / final_total_invested) * 100
    
    plt.text(
        x=final_date,
        y=final_portfolio_value,
        s=f' ${final_portfolio_value:,.2f}',
        color='orange',
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='#121212', edgecolor='orange', boxstyle='round,pad=0.5')
    )
    
    plt.text(
        x=final_date,
        y=final_total_invested,
        s=f' ${final_total_invested:,.2f}',
        color='green',
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='#121212', edgecolor='green', boxstyle='round,pad=0.5')
    )
    
    plt.text(
        x=plot_data.index[int(len(plot_data)/2)],
        y=final_portfolio_value,
        s=f'Total Gain: ${gain:,.2f} ({gain_percentage:.2f}%)',
        color='white',
        fontsize=14,
        verticalalignment='center',
        horizontalalignment='right',
        bbox=dict(facecolor='#121212', edgecolor='white')
    )
    
    plt.tight_layout()
    plt.show()



def plot_garch_volatility(data: pd.DataFrame, tickers: list, investments: list):
    """
    Apply a GARCH model to the portfolio returns based on USD investments and plot the resulting volatility.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted prices for all tickers.
    tickers (list): List of stock tickers in the portfolio.
    investments (list): List of USD amounts invested in each stock.

    Returns:
    None: The function plots the volatility of the portfolio over time.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Normalize investments to convert to an equivalent weight in portfolio
    investments = np.array(investments)
    
    # Calculate the number of shares bought for each stock
    shares = investments / data[tickers].iloc[0]

    # Calculate the portfolio value at each time step
    portfolio_values = (shares * data[tickers]).sum(axis=1)
    
    # Calculate daily returns of the portfolio
    portfolio_returns = portfolio_values.pct_change().dropna()
    
    # Fit a GARCH(1,1) model to the portfolio returns
    model = arch_model(portfolio_returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    
    # Forecast the conditional volatility
    forecasted_volatility = model_fit.conditional_volatility
    
    # Plotting the conditional volatility
    

    plt.figure(figsize=(12, 6))
    plt.plot(forecasted_volatility, color='orange', linewidth=2, label='GARCH Volatility')
    
    # Customizing the plot with a black background
    plt.title('GARCH Volatility of Portfolio', color='white', fontsize=16)
    plt.xlabel('Date', color='white')
    plt.ylabel('Volatility', color='white')
    plt.legend(facecolor='black', edgecolor='white', fontsize=12, loc='best', labelcolor='white')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    
    # Set background color
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    # Customize the tick colors
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    
    plt.show()

def plot_montecarlo(
    price_df: pd.DataFrame,
    tickers: list[str],
    investments: list[float],
    simulation_length: int,
    num_portfolio_simulations: int = 100,
    num_market_simulations: int = 0,
    market_ticker: str = '^GSPC'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Perform a Monte Carlo simulation for a portfolio and optionally for a market index.

    Parameters:
    price_df (pd.DataFrame): DataFrame containing adjusted daily prices for the portfolio assets and market.
    tickers (list[str]): List of stock tickers in the portfolio.
    investments (list[float]): Corresponding investment amounts for each ticker.
    simulation_length (int): The length of each simulation (in days).
    num_portfolio_simulations (int): Number of simulations to run for the portfolio (default is 100).
    num_market_simulations (int): Number of simulations to run for the market index (default is 0, meaning no market simulations).
    market_ticker (str): Ticker symbol for the market index (default is '^GSPC').

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames: one for the portfolio simulation and one for the market simulation.
    """
    
    def calculate_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the daily returns from adjusted prices."""
        return price_df.pct_change().dropna()

    def run_monte_carlo(returns_df: pd.DataFrame, weights: list[float], num_simulations: int, simulation_length: int, initial_value: float) -> pd.DataFrame:
        """Run Monte Carlo simulations."""
        simulation_results = np.zeros((simulation_length, num_simulations))
        
        for ticker, weight in zip(returns_df.columns, weights):
            mean_return = returns_df[ticker].mean()
            std_dev_return = returns_df[ticker].std()
            
            for sim in range(num_simulations):
                simulated_prices = [initial_value]  # Start with the initial investment value
                for _ in range(simulation_length):
                    simulated_price = simulated_prices[-1] * (1 + np.random.normal(mean_return, std_dev_return))
                    simulated_prices.append(simulated_price)
                
                simulation_results[:, sim] += weight * np.array(simulated_prices[1:])
        
        return pd.DataFrame(simulation_results)
    
    # Calculate daily returns for the portfolio
    returns_df = calculate_daily_returns(price_df[tickers])
    total_investment = sum(investments)
    portfolio_weights = np.array(investments) / total_investment

    # Run simulations for the portfolio
    portfolio_sim_df = run_monte_carlo(returns_df, portfolio_weights, num_portfolio_simulations, simulation_length, initial_value=total_investment)
    
    # Initialize market simulation DataFrame as None
    market_sim_df = pd.DataFrame()
    
    # If market simulations are requested, calculate and run those as well
    if num_market_simulations > 0 and market_ticker in price_df.columns:
        market_returns_df = calculate_daily_returns(price_df[[market_ticker]])
        market_sim_df = run_monte_carlo(market_returns_df, [1], num_market_simulations, simulation_length, initial_value=total_investment)
    
    # Create subplots for the portfolio and market simulations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot portfolio simulations
    ax1.plot(portfolio_sim_df, color='orange', alpha=0.5)  # Increase alpha for better visibility
    ax1.set_title('Monte Carlo Simulation of Portfolio', color='white', fontsize=18)
    ax1.set_ylabel('Portfolio Value', color='white', fontsize=14)
    ax1.set_facecolor('black')
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    
    # Plot market simulations
    if not market_sim_df.empty:
        ax2.plot(market_sim_df, color='green', alpha=0.5)  # Increase alpha for better visibility
        ax2.set_title('Monte Carlo Simulation of Market', color='white', fontsize=18)
        ax2.set_xlabel('Day', color='white', fontsize=14)
        ax2.set_ylabel('Market Value', color='white', fontsize=14)
        ax2.set_facecolor('black')
        ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
    
    # Calculate and display statistics on the plot for the portfolio
    portfolio_end_values = portfolio_sim_df.iloc[-1]
    portfolio_stats = {
        'Max': portfolio_end_values.max(),
        'Min': portfolio_end_values.min(),
        'Median': portfolio_end_values.median(),
        'Mean': portfolio_end_values.mean()
    }
    stats_text = '\n'.join([f'{key}: {value:.2f}' for key, value in portfolio_stats.items()])
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='black', edgecolor='orange', boxstyle='round,pad=0.5'),
             color='orange')
    
    # Calculate and display statistics on the plot for the market
    if not market_sim_df.empty:
        market_end_values = market_sim_df.iloc[-1]
        market_stats = {
            'Max': market_end_values.max(),
            'Min': market_end_values.min(),
            'Median': market_end_values.median(),
            'Mean': market_end_values.mean()
        }
        stats_text = '\n'.join([f'{key}: {value:.2f}' for key, value in market_stats.items()])
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(facecolor='black', edgecolor='green', boxstyle='round,pad=0.5'),
                 color='green')
    
    plt.show()

    return portfolio_sim_df, market_sim_df


#DA AGGIUNGERE:
#CONO VOLATILITA' PORTFOLIO
#RIVEDERE ARCH/GARCH
