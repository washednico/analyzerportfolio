import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from arch import arch_model
from scipy.stats import norm


# Set plotly template
pio.templates.default = "plotly_dark"


def compare_portfolio_mkt(
    portfolio: dict
) -> pd.DataFrame:
    """
    Compare the portfolio's return with the market's return and optionally plot the comparison.

    Parameters:
    portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    pd.DataFrame: A DataFrame with the cumulative returns of both the portfolio and the market index.
    """
    
    portfolio_returns = portfolio['portfolio_value']
    market_returns = portfolio['market_value']
    name = portfolio['name']
    
    # Plot the cumulative returns
    fig = go.Figure()

    # Plot portfolio cumulative returns
    fig.add_trace(go.Scatter(
        x=portfolio_returns.index,
        y=portfolio_returns,
        mode='lines',
        name=name,
        line=dict(color='orange')
    ))

    # Plot market cumulative returns
    fig.add_trace(go.Scatter(
        x=market_returns.index,
        y=market_returns,
        mode='lines',
        name=f'Market',
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

def compare_multiple_portfolios(
    portfolios: list[dict],
    colors: list[str] = None
) -> pd.DataFrame:
    """
    Compare the returns of multiple portfolios and optionally plot the comparison.

    Parameters:
    - portfolios (list[dict]): List of portfolio dictionaries created from the create_portfolio function.
    - colors (list[str], optional): List of colors for each portfolio plot line.

    Returns:
    - pd.DataFrame: A DataFrame with the cumulative values of all portfolios (and market index if provided).
    """
    # Initialize an empty DataFrame to hold cumulative values
    cumulative_values = pd.DataFrame()

    # Create a Plotly figure
    fig = go.Figure()

    # If colors are not provided, use default Plotly colors
    if colors is None:
        colors = [None] * len(portfolios)

    # Check if the length of colors matches the number of portfolios
    if len(colors) != len(portfolios):
        raise ValueError("The length of 'colors' must match the number of portfolios.")

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
        title="Comparison of Multiple Portfolios",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_dark",
        height=600
    )

    # Display the plot
    fig.show()

    return cumulative_values