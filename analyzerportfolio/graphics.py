import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from arch import arch_model
from scipy.stats import norm
from typing import Union, List


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
        template="plotly_dark",
        height=600
    )

    # Display the plot
    fig.show()

    return cumulative_values

def garch(
    portfolios: Union[dict, List[dict]],
    colors: Union[str, List[str]] = None,
    market_color: str = 'green'
) -> pd.DataFrame:
    """
    Compare the GARCH volatilities of one or multiple portfolios and plot the comparison.
    If multiple portfolios are passed, it will use market values from the first portfolio.

    Parameters:
    - portfolios (dict or list of dict): Portfolio dictionary or list of portfolio dictionaries
      created from the create_portfolio function.
    - colors (str or list[str], optional): Color or list of colors for each portfolio plot line.
    - market_color (str, optional): Color for the market volatility plot line (default is 'green').

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
        template="plotly_dark",
        height=600
    )

    # Display the plot
    fig.show()

    return volatility_df