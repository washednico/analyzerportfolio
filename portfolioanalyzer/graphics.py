import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from arch import arch_model
from portfolioanalyzer.metrics import (
    calculate_daily_returns,
    calculate_portfolio_returns,
    check_dataframe
)

# Set plotly template
pio.templates.default = "plotly_dark"

def montecarlo(
    price_df: pd.DataFrame,
    tickers: list[str],
    investments: list[float],
    simulation_length: int,
    num_portfolio_simulations: int = 100,
    num_market_simulations: int = 0,
    market_ticker: str = '^GSPC',
    plot: bool = True
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
    plot (bool): Whether to plot the results (default is True).

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames: one for the portfolio simulation and one for the market simulation.
    """
    
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
    
    if check_dataframe(price_df, tickers, investments, market_ticker):
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
        
        if plot:
            # Create subplots for the portfolio and market simulations
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Portfolio Simulations', 'Market Simulations'))

            # Plot portfolio simulations
            for i in range(num_portfolio_simulations):
                fig.add_trace(go.Scatter(x=list(range(simulation_length)), y=portfolio_sim_df[i], mode='lines', line=dict(color='orange', width=1), showlegend=False), row=1, col=1)

            # Plot market simulations
            if not market_sim_df.empty:
                for i in range(num_market_simulations):
                    fig.add_trace(go.Scatter(x=list(range(simulation_length)), y=market_sim_df[i], mode='lines', line=dict(color='green', width=1), showlegend=False), row=2, col=1)

            fig.update_layout(height=800, width=1000, title_text="Monte Carlo Simulation of Portfolio and Market")
            fig.show()

        # Calculate and return statistics for portfolio
        portfolio_end_values = portfolio_sim_df.iloc[-1]
        print("Portfolio Simulation Statistics:")
        print(f"Max Value: {portfolio_end_values.max():.2f}")
        print(f"Min Value: {portfolio_end_values.min():.2f}")
        print(f"Median Value: {portfolio_end_values.median():.2f}")
        print(f"Average Value: {portfolio_end_values.mean():.2f}")
        
        # Calculate and return statistics for market
        if not market_sim_df.empty:
            market_end_values = market_sim_df.iloc[-1]
            print("\nMarket Simulation Statistics:")
            print(f"Max Value: {market_end_values.max():.2f}")
            print(f"Min Value: {market_end_values.min():.2f}")
            print(f"Median Value: {market_end_values.median():.2f}")
            print(f"Average Value: {market_end_values.mean():.2f}")

        return portfolio_sim_df, market_sim_df

def compare_portfolio_to_market(
    data: pd.DataFrame,
    tickers: list[str],
    investments: list[float],
    market_index: str,
    plot: bool = True
) -> pd.DataFrame:
    """
    Compare the portfolio's return with the market's return and optionally plot the comparison.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers and the market index.
    tickers (list[str]): List of stock tickers in the portfolio.
    investments (list[float]): Corresponding investment amounts for each ticker.
    market_index (str): The market index to compare against.
    plot (bool): Whether to plot the results (default is True).

    Returns:
    pd.DataFrame: A DataFrame with the cumulative returns of both the portfolio and the market index.
    """
    
    if check_dataframe(data, tickers, investments, market_index):
        # Calculate the total portfolio value
        total_investment = sum(investments)
    
        #Calculate market and stocks daily returns
        market_returns = calculate_daily_returns(data[market_index])
        stock_returns = calculate_daily_returns(data[tickers])

        # Calculate portfolio returns as a weighted sum of individual stock returns
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)
    
        # Calculate cumulative returns
        portfolio_cumulative_return = (1 + portfolio_returns).cumprod() * total_investment
        market_cumulative_return = (1 + market_returns).cumprod() * total_investment
        
        # Combine results into a DataFrame for easier analysis
        comparison_df = pd.DataFrame({
            'Portfolio': portfolio_cumulative_return,
            f'{market_index}': market_cumulative_return
        })

        if plot:
            # Plot the cumulative returns
            fig = go.Figure()

            # Plot portfolio cumulative returns
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative_return.index,
                y=portfolio_cumulative_return,
                mode='lines',
                name='Portfolio',
                line=dict(color='orange')
            ))

            # Plot market cumulative returns
            fig.add_trace(go.Scatter(
                x=market_cumulative_return.index,
                y=market_cumulative_return,
                mode='lines',
                name=f'{market_index}',
                line=dict(color='green')
            ))

            # Update layout
            fig.update_layout(
                title="Portfolio vs Market Performance",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_dark",
                height=600
            )

            fig.show()

        return comparison_df

def simulate_pac(
    data: pd.DataFrame,
    tickers: list[str],
    initial_investment: float,
    periodic_investment: float,
    investment_interval: int,
    investment_weights: list[float],
    plot: bool = True
) -> pd.DataFrame:
    """
    Simulate a PAC (Piano di Accumulo) investment strategy and optionally plot the portfolio's growth over time.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers.
    tickers (list[str]): List of stock tickers in the portfolio.
    initial_investment (float): Initial amount to invest at the start.
    periodic_investment (float): Amount to invest at each interval.
    investment_interval (int): Interval (in days) between each additional investment.
    investment_weights (list[float]): List of weights representing the percentage of each stock in the portfolio.
    plot (bool): Whether to plot the results (default is True).

    Returns:
    pd.DataFrame: A DataFrame with the portfolio value and total invested amount over time.
    """
    if len(tickers) != len(investment_weights):
        raise ValueError("The number of tickers must match the number of weights.")
    
    # Normalize weights to ensure they sum to 1
    weights = np.array(investment_weights)
    #weights are normalized so they sum to 1.
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
    
    # Calculate final values for summary
    final_portfolio_value = round(portfolio_values[-1],2)
    final_total_invested = total_invested[-1]
    profit = round((final_portfolio_value - final_total_invested),2)
    profit_percentage = round(((profit / final_total_invested) * 100),2)
    
    # Summary string for annotation
    summary = "Total Invested: "+str(final_total_invested)+"    Final portfolio value: "+str(final_portfolio_value)+"     Percentage Increase: "+str(profit_percentage)+"%"
    
    
    if plot:
        # Plotting the results with Plotly
        fig = go.Figure()

        # Portfolio Value trace
        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=plot_data['Portfolio Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='orange', width=2)
        ))

        # Total Invested trace
        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=plot_data['Total Invested'],
            mode='lines',
            name='Total Invested',
            line=dict(color='green', width=2, dash='dash')
        ))

        # Update layout and add annotation
        fig.update_layout(
            title="Portfolio Growth with PAC Strategy",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_dark",
            height=600,
            annotations=[
                go.layout.Annotation(
                    text=summary,
                    xref="paper", yref="paper",
                    x=1.05, y=1.05, showarrow=False,
                    font=dict(color="orange", size=12),
                    align="left",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(0, 0, 0, 0.75)"
                )
            ]
        )

        fig.show()

    return plot_data

def garch(
    data: pd.DataFrame,
    tickers: list[str],
    investments: list[float],
    plot: bool = True
) -> pd.Series:
    """
    Apply a GARCH model to the portfolio returns based on USD investments and optionally plot the resulting volatility.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted prices for all tickers.
    tickers (list[str]): List of stock tickers in the portfolio.
    investments (list[float]): List of USD amounts invested in each stock.
    plot (bool): Whether to plot the volatility (default is True).

    Returns:
    pd.Series: The conditional volatility series from the GARCH model.
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
    
    if plot:
        # Plotting the conditional volatility with Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecasted_volatility.index,
            y=forecasted_volatility,
            mode='lines',
            name='GARCH Volatility',
            line=dict(color='orange', width=2)
        ))

        # Update layout
        fig.update_layout(
            title="GARCH Volatility of Portfolio",
            xaxis_title="Date",
            yaxis_title="Volatility",
            template="plotly_dark",
            height=600
        )

        fig.show()

    return forecasted_volatility

def heatmap(
        data: pd.DataFrame,
        tickers: list[str],
        market_ticker: str = '^GSPC',
        plot: bool = True
        ):
    """
    Plot a heatmap for correaltion analysis

    Parameters: 
    price_df (pd.DataFrame): DataFrame containing adjusted daily prices for the portfolio assets and market.
    tickers (list[str]): List of stock tickers in the portfolio.
    market_ticker (str): Ticker symbol for the market index (default is '^GSPC').
    plot (bool): Whether to plot the results (default is True).

    Returns:
    pd.DataFrame: A DataFrame with the correlation coefficients. 
    """
    if check_dataframe(data, tickers, market_ticker = market_ticker):
    
        # Calculate daily returns
        stock_returns = calculate_daily_returns(data[tickers])
        market_returns = calculate_daily_returns(data[market_ticker])
        
        # Combine stock and market returns into a single DataFrame
        combined_returns = pd.concat([stock_returns, market_returns], axis=1)

        # Calculate the correlation matrix
        corr_matrix = combined_returns.corr()

        if plot:
            # Plot the clustermap using px 
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
            fig.update_layout(height=800, width=1000, title_text="Heatmap of Stocks and Market")
            fig.show()

        return corr_matrix

def drawdown_plot(
        data: pd.DataFrame, 
        tickers: list[str], 
        investments: list[float], 
        plot: bool = True) -> pd.DataFrame:
    
    """
    Plot drawdown of the portfolio

    Parameters: 
    data (pd.DataFrame): DataFrame containing adjusted and converted prices for all tickers.
    tickers (list): List of stock tickers.
    investments (list): Corresponding investment amounts for each ticker.

    Returns:
    pd.DataFrame: The drawdown of the portfolio as a percentage.
    """ 


    if check_dataframe(data, tickers, investments):

        # Calculate portfolio returns as a weighted sum of individual stock returns
        stock_returns = calculate_daily_returns(data[tickers])
        portfolio_returns = calculate_portfolio_returns(investments, stock_returns)
    
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()
        cumulative_portfolio_returns_max = cumulative_portfolio_returns.cummax()

        drawdown = (cumulative_portfolio_returns - cumulative_portfolio_returns_max) / cumulative_portfolio_returns_max

        if plot:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='orange', width=2)
            ))

            fig.update_layout(
                title="Drawdown",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                template="plotly_dark",
                height=600
            )

            fig.show()

        return drawdown

def volatility_cone(
    data: pd.DataFrame,
    tickers: list[str],
    investments: list[float],
    time_horizon: int,
    confidence_intervals: list[float] = [0.90, 0.95, 0.99],
    plot: bool = True
) -> pd.DataFrame:
    """
    Calculate and plot the probability cone for a portfolio over a time horizon.

    Parameters:
    data (pd.DataFrame): DataFrame containing adjusted prices for all tickers.
    tickers (list[str]): List of stock tickers in the portfolio.
    investments (list[float]): List of USD amounts invested in each stock.
    time_horizon (int): The number of days over which to calculate the probability cone.
    confidence_intervals (list[float]): List of confidence intervals to use for the cone (default is [0.90, 0.95, 0.99]).
    plot (bool): Whether to plot the probability cone (default is True).

    Returns:
    pd.DataFrame: A DataFrame containing the expected value and the upper and lower bounds for each confidence interval.
    """
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    # Normalize investments to convert to an equivalent weight in the portfolio
    investments = np.array(investments)
    
    # Calculate the number of shares bought for each stock
    shares = investments / data[tickers].iloc[0]

    # Calculate the portfolio value at each time step
    portfolio_values = (shares * data[tickers]).sum(axis=1)
    
    # Calculate daily returns of the portfolio
    portfolio_returns = portfolio_values.pct_change().dropna()

    # Calculate the mean return and volatility of the portfolio
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    # Generate a range of time steps (days)
    time_steps = np.arange(1, time_horizon + 1)
    
    # Calculate the expected value over time
    initial_value = portfolio_values.iloc[0]
    expected_value = initial_value * np.exp(mean_return * time_steps)
    
    # Prepare data for the confidence intervals
    probability_cone_data = {'Days': time_steps, 'Expected Value': expected_value}
    
    for ci in confidence_intervals:
        z_score = np.abs(np.log(1 - (1 - ci) / 2))  # Z-score for the given confidence interval
        lower_bound = initial_value * np.exp((mean_return - z_score * volatility) * time_steps)
        upper_bound = initial_value * np.exp((mean_return + z_score * volatility) * time_steps)
        
        probability_cone_data[f'Lower Bound {int(ci*100)}%'] = lower_bound
        probability_cone_data[f'Upper Bound {int(ci*100)}%'] = upper_bound
    
    # Convert to DataFrame
    probability_cone_df = pd.DataFrame(probability_cone_data)

    if plot:
        # Plotting the probability cone with Plotly
        fig = go.Figure()

        # Plot expected value
        fig.add_trace(go.Scatter(
            x=probability_cone_df['Days'],
            y=probability_cone_df['Expected Value'],
            mode='lines',
            name='Expected Value',
            line=dict(color='orange')
        ))

        # Plot confidence intervals
        for ci in confidence_intervals:
            fig.add_trace(go.Scatter(
                x=probability_cone_df['Days'],
                y=probability_cone_df[f'Lower Bound {int(ci*100)}%'],
                mode='lines',
                name=f'Lower Bound {int(ci*100)}%',
                line=dict(color='red', dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=probability_cone_df['Days'],
                y=probability_cone_df[f'Upper Bound {int(ci*100)}%'],
                mode='lines',
                name=f'Upper Bound {int(ci*100)}%',
                line=dict(color='green', dash='dash')
            ))

        # Update layout
        fig.update_layout(
            title="Probability Cone",
            xaxis_title="Days",
            yaxis_title="Portfolio Value",
            template="plotly_dark",
            height=600
        )

        fig.show()

    return probability_cone_df