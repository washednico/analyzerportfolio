import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from arch import arch_model
from scipy.stats import norm
from typing import Union, List, Dict


# Set plotly template
pio.templates.default = "plotly_dark"


def portfolio_value(
    portfolios: Union[dict, List[dict]],
    colors: Union[str, List[str]] = None,
    market_color: str = 'green'
) -> pd.DataFrame:
    """
    Compare the portfolio(s) return with the market's return and plot the comparison.
    If multiple portfolios are passed, it will use market values from the first portfolio.

    Parameters:
    - portfolios (dict or list of dict): Portfolio dictionary or list of portfolio dictionaries
      created from the create_portfolio function.
    - colors (str or list[str], optional): Color or list of colors for each portfolio plot line.
    - market_color (str, optional): Color for the market plot line (default is 'green').

    Returns:
    - pd.DataFrame: A DataFrame with the cumulative values of the portfolio(s) and the market index.
    """
    # Ensure portfolios is a list
    if isinstance(portfolios, dict):
        portfolios = [portfolios]

    # Ensure colors is a list
    if colors is None:
        colors = [None] * len(portfolios)
    elif isinstance(colors, str):
        colors = [colors]
    elif isinstance(colors, list):
        if len(colors) != len(portfolios):
            raise ValueError("The length of 'colors' must match the number of portfolios.")
    else:
        raise ValueError("Invalid type for 'colors' parameter.")

    # Initialize an empty DataFrame to hold cumulative values
    cumulative_values = pd.DataFrame()

    # Create a Plotly figure
    fig = go.Figure()

    # Check if portfolios are passed
    if len(portfolios) > 0:
        # Extract market values from the first portfolio if available
        market_values = portfolios[0].get('market_value', None)
        market_name = portfolios[0].get('market_ticker', 'Market')

        if market_values is not None:
            # Ensure the index is datetime
            market_values.index = pd.to_datetime(market_values.index)

            # Add market values to cumulative_values DataFrame
            cumulative_values[market_name] = market_values

            # Add market trace to the figure
            fig.add_trace(go.Scatter(
                x=market_values.index,
                y=market_values,
                mode='lines',
                name=market_name,
                line=dict(color=market_color, width=2)
            ))

    # Iterate over each portfolio and corresponding color
    for portfolio, color in zip(portfolios, colors):
        # Extract portfolio values and name
        portfolio_values = portfolio['portfolio_value']
        name = portfolio['name']

        # Ensure the index is datetime
        portfolio_values.index = pd.to_datetime(portfolio_values.index)

        # Add the portfolio values to the cumulative_values DataFrame
        cumulative_values[name] = portfolio_values

        # Add a trace to the figure with the specified color
        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values,
            mode='lines',
            name=name,
            line=dict(color=color) if color else {}
        ))

    # Align all series by their dates
    cumulative_values = cumulative_values.dropna()

    # Update figure layout
    fig.update_layout(
        title="Portfolio(s) vs Market Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_dark"
    )

    # Display the plot
    fig.show()

    return cumulative_values

def garch(
    portfolios: Union[dict, List[dict]],
    colors: Union[str, List[str]] = None,
    market_color: str = 'green',
    plot: bool = True
) -> pd.DataFrame:
    """
    Compare the GARCH volatilities of one or multiple portfolios and plot the comparison.
    If multiple portfolios are passed, it will use market values from the first portfolio.

    Parameters:
    - portfolios (dict or list of dict): Portfolio dictionary or list of portfolio dictionaries
      created from the create_portfolio function.
    - colors (str or list[str], optional): Color or list of colors for each portfolio plot line.
    - market_color (str, optional): Color for the market volatility plot line (default is 'green').
    - plot (bool, optional): Whether to plot the results (default is True).

    Returns:
    - pd.DataFrame: A DataFrame with the GARCH volatilities of all portfolios (divided by 100).
    """
    # Ensure portfolios is a list
    if isinstance(portfolios, dict):
        portfolios = [portfolios]

    # Ensure colors is a list
    if colors is None:
        colors = [None] * len(portfolios)
    elif isinstance(colors, str):
        colors = [colors]
    elif isinstance(colors, list):
        if len(colors) != len(portfolios):
            raise ValueError("The length of 'colors' must match the number of portfolios.")
    else:
        raise ValueError("Invalid type for 'colors' parameter.")

    # Initialize an empty DataFrame to hold volatilities
    volatility_df = pd.DataFrame()

    # Create a Plotly figure
    fig = go.Figure()

    # Check if portfolios are passed
    if len(portfolios) > 0:
        # Extract market returns from the first portfolio if available
        market_returns = portfolios[0].get('market_returns', None)
        market_name = portfolios[0].get('market_ticker', 'Market')

        if market_returns is not None:
            # Clean market_returns
            market_returns = market_returns.replace([np.inf, -np.inf], np.nan).dropna()

            # Ensure market_returns is a Pandas Series
            if isinstance(market_returns, pd.DataFrame):
                market_returns = market_returns.squeeze()

            # Sort the index to ensure proper alignment
            market_returns = market_returns.sort_index()

            # Multiply returns by 100 to convert to percentage
            market_returns_pct = market_returns * 100

            # Fit a GARCH(1,1) model to the market returns
            m_model = arch_model(market_returns_pct, vol='Garch', p=1, q=1, rescale=False)
            m_model_fit = m_model.fit(disp='off')

            # Get the conditional volatility and divide by 100
            m_forecasted_volatility = m_model_fit.conditional_volatility / 100

            # Add the market volatility series to the volatility_df DataFrame
            volatility_df[market_name] = m_forecasted_volatility

            # Add market volatility trace
            fig.add_trace(go.Scatter(
                x=m_forecasted_volatility.index,
                y=m_forecasted_volatility,
                mode='lines',
                name=market_name,
                line=dict(color=market_color, width=2)
            ))

    # Iterate over each portfolio and corresponding color
    for portfolio, color in zip(portfolios, colors):
        # Extract portfolio returns and name
        portfolio_returns = portfolio['portfolio_returns']
        name = portfolio['name']

        # Clean portfolio_returns
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()

        # Ensure portfolio_returns is a Pandas Series
        if isinstance(portfolio_returns, pd.DataFrame):
            portfolio_returns = portfolio_returns.squeeze()

        # Sort the index to ensure proper alignment
        portfolio_returns = portfolio_returns.sort_index()

        # Multiply returns by 100 to convert to percentage
        portfolio_returns_pct = portfolio_returns * 100

        # Fit a GARCH(1,1) model to the portfolio returns
        model = arch_model(portfolio_returns_pct, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp='off')

        # Get the conditional volatility and divide by 100
        forecasted_volatility = model_fit.conditional_volatility / 100

        # Add the volatility series to the volatility_df DataFrame
        volatility_df[name] = forecasted_volatility

        # Add a trace to the figure with the specified color
        fig.add_trace(go.Scatter(
            x=forecasted_volatility.index,
            y=forecasted_volatility,
            mode='lines',
            name=name,
            line=dict(color=color) if color else {}
        ))

    # Align all series by their dates
    volatility_df = volatility_df.dropna()

    # Update figure layout
    fig.update_layout(
        title="Comparison of GARCH Volatilities",
        xaxis_title="Date",
        yaxis_title="Volatility",
        template="plotly_dark"
    )

    # Display the plot
    if plot:
        fig.show()

    return volatility_df

def montecarlo(
    portfolios: Union[dict, List[dict]],
    simulation_length: int,
    num_simulations: int = 100,
    plot: bool = True
) -> Dict[str, pd.DataFrame]:
    if isinstance(portfolios, dict):
        portfolios = [portfolios]

    simulation_results = {}
    if len(portfolios) == 0:
        raise ValueError("At least one portfolio must be provided.")
    
    days_per_step = portfolios[0]["return_period_days"]
    market_returns = portfolios[0].get('market_returns', None)
    market_name = portfolios[0].get('market_ticker', 'Market')
    total_investment = sum(portfolios[0]['investments'])

    def run_monte_carlo(returns_series: pd.Series, num_simulations: int, simulation_length: int, initial_value: float) -> pd.DataFrame:
        mean_return = returns_series.mean()
        std_dev_return = returns_series.std()
        simulation_results = np.zeros((simulation_length + 1, num_simulations))
        simulation_results[0, :] = initial_value
        for sim in range(num_simulations):
            random_returns = np.random.normal(mean_return, std_dev_return, simulation_length)
            cumulative_returns = np.cumprod(1 + random_returns)
            simulation_results[1:, sim] = initial_value * cumulative_returns
        return pd.DataFrame(simulation_results)

    for portfolio in portfolios:
        name = portfolio['name']
        investments = portfolio['investments']
        portfolio_returns = portfolio['portfolio_returns'].dropna()
        initial_value = sum(investments)
        sim_df = run_monte_carlo(portfolio_returns, num_simulations, simulation_length, initial_value)
        simulation_results[name] = sim_df

    if market_returns is not None:
        market_returns = market_returns.dropna()
        market_sim_df = run_monte_carlo(market_returns, num_simulations, simulation_length, total_investment)
        simulation_results[market_name] = market_sim_df

    if plot:
        num_plots = len(simulation_results)
        num_cols = 2
        num_rows = -(-num_plots // num_cols)  
        fig = make_subplots(rows=num_rows, cols=num_cols)

        plot_index = 0
        for name, sim_df in simulation_results.items():
            plot_index += 1
            row = (plot_index - 1) // num_cols + 1
            col = (plot_index - 1) % num_cols + 1
            for i in range(num_simulations):
                fig.add_trace(
                    go.Scatter(x=np.arange(simulation_length + 1), y=sim_df.iloc[:, i], mode='lines', line=dict(width=1), showlegend=False),
                    row=row, col=col
                ) 

            xref = f'x{plot_index}' if plot_index != 1 else 'x'
            yref = f'y{plot_index}' if plot_index != 1 else 'y'
            fig.add_annotation(text=f"{name} - Max: {sim_df.iloc[-1].max():.2f} Avg: {sim_df.iloc[-1].mean():.2f} Med: {sim_df.iloc[-1].median():.2f} Min: {sim_df.iloc[-1].min():.2f}",
                               xref=f"{xref} domain", yref=f"{yref} domain", x=0.1, y=1.05, showarrow=False, row=row, col=col)
            

        fig.update_layout(title_text="Monte Carlo Simulations - Simulation length: "+str(simulation_length)+" (Days x Step: "+str(days_per_step)+") - Simuluations per portfolio: "+str(num_simulations), template="plotly_dark")
        fig.show()

    return simulation_results


def drawdown(portfolios: Union[str, List[str]],
             plot: bool = True, 
             colors: Union[str, List[str]] = None,
            market_color: str = 'green',) -> pd.DataFrame:
    """
    Plot drawdown of multiple portfolios using their portfolio values and optionally plot market drawdown.

    Parameters:
    - portfolios (list[dict]): List of portfolio dictionaries containing 'name' and 'portfolio_value'.
      Optionally, the first portfolio can contain 'market_value' for market drawdown comparison.
    - plot (bool): Whether to plot the results (default is True).

    Returns:
    - pd.DataFrame: DataFrame containing the drawdown series for each portfolio and optionally for the market.
    """
    if isinstance(portfolios, dict):
        portfolios = [portfolios]

    if len(portfolios) == 0:
        raise ValueError("At least one portfolio must be provided.")
    
     # Ensure colors is a list
    if colors is None:
        colors = [None] * len(portfolios)
    elif isinstance(colors, str):
        colors = [colors]
    elif isinstance(colors, list):
        if len(colors) != len(portfolios):
            raise ValueError("The length of 'colors' must match the number of portfolios.")
    else:
        raise ValueError("Invalid type for 'colors' parameter.")
    
    def calculate_drawdown(values):
        """Calculate drawdowns given a series of portfolio values."""
        peak = values.cummax()
        drawdown = (values - peak) / peak
        return drawdown * 100  # convert to percentage
    
    fig = go.Figure()
    drawdown_data = {}

    for portfolio, color in zip(portfolios,colors):
        portfolio_name = portfolio['name']
        portfolio_value = portfolio['portfolio_value']
        drawdown = calculate_drawdown(portfolio_value)
        drawdown_data[portfolio_name] = drawdown

        # Add portfolio drawdown trace to the plot
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            name=f"{portfolio_name} (Max DD: {drawdown.min():.2f}%)",
            line=dict(color=color) if color else {}
        ))

        
    market_drawdown = calculate_drawdown(portfolios[0]['market_value'])
    drawdown_data['Market'] = market_drawdown
    fig.add_trace(go.Scatter(
        x=market_drawdown.index,
        y=market_drawdown,
        mode='lines',
        name=f"Market (Max DD: {market_drawdown.min():.2f}%)",
        line=dict(color=market_color, width=2)
    ))

    if plot:
        fig.update_layout(
            title="Portfolio Drawdown Comparison",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark"
        )
        fig.show()

    return pd.DataFrame(drawdown_data)

def heatmap(
        portfolio: dict,
        plot: bool = True
        ):
    """
    Plot a heatmap for correaltion analysis

    Parameters: 
    portfolio (dict): Portfolio dictionary created from the create_portfolio function.
    plot (bool): Whether to plot the results (default is True).

    Returns:
    pd.DataFrame: A DataFrame with the correlation coefficients. 
    """
    
    tickers = portfolio['tickers']
    
    revised_tickers = []
    for i in tickers:
        revised_tickers.append(i+"_Return")
    
    returns_df = portfolio['returns']
    market_ticker = portfolio['market_ticker']

    # Retrieving returns
    stock_returns = returns_df[revised_tickers]
    market_returns = portfolio['market_returns']
    stock_returns = stock_returns.copy()
    stock_returns.rename(columns=lambda x: x.replace("_Return",""), inplace=True)
    
    # Combine stock and market returns into a single DataFrame
    combined_returns = pd.concat([stock_returns, market_returns], axis=1)

    # Calculate the correlation matrix
    corr_matrix = combined_returns.corr()

    if plot:
        # Plot the clustermap using px 
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        fig.update_layout(title_text="Heatmap of Stocks and Market")
        fig.show()

    return corr_matrix


def distribution_return(
    portfolios: Union[str, List[str]],
    bins: int = 100,
    colors: Union[str, List[str]] = None,
    market_color: str = 'green'
) -> pd.DataFrame:
    """
    Plot the distribution of portfolio returns over a specified time interval.

    Parameters: 
    portfolio (dict): Portfolio dictionary created from the create_portfolio function.
    bins (int): Number of bins for the histogram (default is 100).
    plot (bool): Whether to plot the distribution returns (default is True).

    Returns:
    
    """

    if isinstance(portfolios, dict):
        portfolios = [portfolios]

    if len(portfolios) == 0:
        raise ValueError("At least one portfolio must be provided.")
    
     # Ensure colors is a list
    if colors is None:
        colors = [None] * len(portfolios)
    elif isinstance(colors, str):
        colors = [colors]
    elif isinstance(colors, list):
        if len(colors) != len(portfolios):
            raise ValueError("The length of 'colors' must match the number of portfolios.")
    else:
        raise ValueError("Invalid type for 'colors' parameter.")

    fig = go.Figure()
    

    days_per_step = portfolios[0]["return_period_days"]
    market_returns = portfolios[0].get('market_returns', None)
    market_name = portfolios[0].get('market_ticker', 'Market')

    for portfolio, color in zip(portfolios,colors):
        portfolio_name = portfolio['name']
        portfolio_returns = portfolio['portfolio_returns']
        fig.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=bins,
        histnorm='probability density',  # Normalizes histogram so area under histogram equals 1
        marker=dict(
            color=color,
            line=dict(
                color='black',
                width=1
            )
        ),
        opacity=1.0,
        name= portfolio_name
    ))
    
    fig.add_trace(go.Histogram(
        x=market_returns,
        nbinsx=bins,
        histnorm='probability density',  # Normalizes histogram so area under histogram equals 1
        marker=dict(
            color=market_color,
            line=dict(
                color='black',
                width=1
            )
        ),
        opacity=1.0,
        name= market_name
    ))

    # Update layout to make the plot visually appealing
    fig.update_layout(
        title="Distribution of Portfolio Returns",
        template="plotly_dark",
        xaxis=dict(
            title=f"{days_per_step}-Day Returns",
            tickformat='.2%',  # Formats the x-axis ticks as percentages
            showgrid=True,
            zeroline=True
        ),
        yaxis_title="Probability Density",
        bargap=0.02  # Adjusts the gap between bars
    )

    fig.show()


def simulate_dca(
    portfolios: Union[dict, List[dict]],
    initial_investment: float,
    periodic_investment: float,
    investment_interval: int,
    rebalance_interval: int = None,  
    colors: Union[str, List[str]] = None,
    plot: bool = True
) -> pd.DataFrame:
    """
    Simulate a Dollar Cost Averaging investment strategy with optional periodic rebalancing.

    Parameters:
    portfolios (Union[dict, List[dict]]): A portfolio dictionary or a list of portfolio dictionaries.
    initial_investment (float): Initial amount to invest at the start.
    periodic_investment (float): Amount to invest at each interval.
    investment_interval (int): Interval (in days) between each additional investment.
    rebalance_interval (int): Interval (in days) between rebalancing the portfolio to its original weights.
    plot (bool): Whether to plot the results (default is True).

    Returns:
    pd.DataFrame: A DataFrame with the portfolio values and total invested amount over time for each portfolio.
    """

    if isinstance(portfolios, dict):
        portfolios = [portfolios]

    if colors is None:
        colors = [None] * len(portfolios)
    elif isinstance(colors, str):
        colors = [colors]
    elif isinstance(colors, list):
        if len(colors) != len(portfolios):
            raise ValueError("The length of 'colors' must match the number of portfolios.")
    else:
        raise ValueError("Invalid type for 'colors' parameter.")

    results = {}
    fig = go.Figure()
    summary_data = []

    for portfolio, color in zip(portfolios,colors):
        tickers = portfolio['tickers']
        data = portfolio['untouched_data']  # Assuming 'prices' contains the historical data for each ticker
        investment_weights = portfolio['investments']/sum(portfolio['investments'])
        
        # Ensure that weights are normalized
        if len(tickers) != len(investment_weights):
            raise ValueError("The number of tickers must match the number of weights.")
        
        weights = np.array(investment_weights)
        weights = weights / weights.sum()  # Normalize the weights

        # Initialize variables
        dates = data.index
        portfolio_values = []
        total_invested = []

        # Initialize holdings for each stock
        holdings = {ticker: 0 for ticker in tickers}
        cumulative_investment = initial_investment
        next_investment_date = dates[0]
        next_rebalance_date = dates[0] if rebalance_interval else None

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

            # Perform rebalancing on specified intervals
            if rebalance_interval and (date - next_rebalance_date).days >= rebalance_interval:
                # Calculate the total value of the portfolio at current prices
                current_value = sum(holdings[ticker] * current_prices[ticker] for ticker in tickers)

                # Rebalance the portfolio to match the target weights
                for ticker, weight in zip(tickers, weights):
                    target_value = current_value * weight
                    holdings[ticker] = target_value / current_prices[ticker]

                next_rebalance_date = date

            # Calculate portfolio value for the current date
            current_prices = data.loc[date]
            total_value = sum(holdings[ticker] * current_prices[ticker] for ticker in tickers)
            portfolio_values.append(total_value)
            total_invested.append(cumulative_investment)

        # Create a DataFrame for each portfolio's simulation results
        results[portfolio['name']] = pd.DataFrame({
            'Portfolio Value': portfolio_values,
            'Total Invested': total_invested
        }, index=dates)

        # Calculate final values for summary
        final_portfolio_value = round(portfolio_values[-1], 2)
        final_total_invested = total_invested[-1]
        profit = round((final_portfolio_value - final_total_invested), 2)
        profit_percentage = round(((profit / final_total_invested) * 100), 2)

        # Append data to summary table
        summary_data.append({
            'Portfolio': portfolio['name'],
            'Total Invested': final_total_invested,
            'Final Value': final_portfolio_value,
            'Profit': profit,
            'Profit (%)': profit_percentage
        })

        

        if plot:
            # Plotting the results with Plotly
            

            # Portfolio Value trace
            fig.add_trace(go.Scatter(
                x=results[portfolio['name']].index,
                y=results[portfolio['name']]['Portfolio Value'],
                mode='lines',
                name=f'Portfolio Value ({portfolio["name"]})',
                line=dict(color=color, width=2)
            ))

            # Total Invested trace
            fig.add_trace(go.Scatter(
                x=results[portfolio['name']].index,
                y=results[portfolio['name']]['Total Invested'],
                mode='lines',
                name='Total Invested',
                line=dict(color='green', width=2, dash='dash')
            ))

    # Convert summary data to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Generate the text to display as a table on the graph
    summary_text = ""
    for index, row in summary_df.iterrows():
        summary_text += (
            f"{row['Portfolio']} - Total Invested: &#36;{row['Total Invested']:,.2f}, "
            f"Final Value: &#36;{row['Final Value']:,.2f}, "
            f"Profit: &#36;{row['Profit']:,.2f} ({row['Profit (%)']}%)<br>"
        )
    

    # Update layout and add annotation
    fig.update_layout(
        title=f"Portfolio Growth with DCA and Rebalancing",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_dark",
        annotations=[
            go.layout.Annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0, y=1, showarrow=False,
                font=dict(color="white", size=12),
                align="left",
                bordercolor="white",
                borderwidth=1,
                borderpad=4,
                bgcolor="rgba(0, 0, 0, 0.75)"
            )
        ]
    )

    fig.show()

    return results


def probability_cone(
    portfolio: dict,
    time_horizon: int,
    confidence_intervals: list[float] = [0.90, 0.95, 0.99],
    plot: bool = True
) -> pd.DataFrame:
    """
    Calculate and plot the probability cone for a portfolio over a time horizon.

    Parameters:
    portfolio (dict): Portfolio dictionary created from the create_portfolio function.
    time_horizon (int): The number of days over which to calculate the probability cone.
    confidence_intervals (list[float]): List of confidence intervals to use for the cone (default is [0.90, 0.95, 0.99]).
    plot (bool): Whether to plot the probability cone (default is True).

    Returns:
    pd.DataFrame: A DataFrame containing the expected value and the upper and lower bounds for each confidence interval.
    """
    
    portfolio_returns = portfolio['portfolio_returns']
    days_per_step = portfolio['return_period_days']
    initial_value = sum(portfolio['investments'])

    # Calculate the mean return and volatility of the portfolio
    mean_return = (1 + portfolio_returns.mean()) ** (1 / days_per_step) - 1
    volatility = portfolio_returns.std() / np.sqrt(days_per_step)
    
    # Initialize arrays to hold the expected value and confidence intervals
    time_steps = np.arange(1, time_horizon + 1)
    expected_value = np.zeros(time_horizon)
    lower_bounds = {ci: np.zeros(time_horizon) for ci in confidence_intervals}
    upper_bounds = {ci: np.zeros(time_horizon) for ci in confidence_intervals}

    # Calculate the expected value and confidence intervals for each time step
    for t in time_steps:
        cumulative_return = mean_return * t
        expected_value[t-1] = initial_value * np.exp(cumulative_return)
        
        for ci in confidence_intervals:
            z_score = norm.ppf((1 + ci) / 2)  # Z-score for the given confidence interval
            std_dev = z_score * volatility * np.sqrt(t)  # Scaling with sqrt(t)
            lower_bounds[ci][t-1] = initial_value * np.exp(cumulative_return - std_dev)
            upper_bounds[ci][t-1] = initial_value * np.exp(cumulative_return + std_dev)

    # Combine everything into a DataFrame
    probability_cone_data = {
        'Days': time_steps,
        'Expected Value': expected_value
    }
    
    for ci in confidence_intervals:
        probability_cone_data[f'Lower Bound {int(ci*100)}%'] = lower_bounds[ci]
        probability_cone_data[f'Upper Bound {int(ci*100)}%'] = upper_bounds[ci]

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