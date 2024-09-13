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



def plot_distribution_returns(
    portfolio_returns: pd.DataFrame,
    bins: int = 100,
    plot: bool = True
) -> pd.DataFrame:
    """
    Plot the distribution of portfolio returns over a specified time interval.

    Parameters:
    portfolio_returns (pd.DataFrame): Portfolio returns dataframe.
    bins (int): Number of bins for the histogram (default is 100).

    Returns:
    pd.DataFrame: A DataFrame containing the portfolio returns.
    """
        
    # Calculate the mean and standard deviation
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()


    # Create the histogram with probability density normalization
    fig = go.Figure()

    # Add the histogram trace with solid orange bars and solid white border
    fig.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=bins,
        histnorm='probability density',  # Normalizes histogram so area under histogram equals 1
        marker=dict(
            color='orange',
            line=dict(
                color='black',
                width=1
            )
        ),
        opacity=1.0,
        name='Portfolio Returns'
    ))

    # Generate data for the normal distribution curve
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
    y = norm.pdf(x, mu, sigma)

    # Add the normal distribution curve in green
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='green', width=2)
    ))

    # Update layout to make the plot visually appealing
    fig.update_layout(
        title="Distribution of Portfolio Returns",
        template="plotly_dark",
        height=600,
        xaxis=dict(
            title=f"Returns",
            tickformat='.2%',  # Formats the x-axis ticks as percentages
            showgrid=True,
            zeroline=True
        ),
        yaxis_title="Probability Density",
        bargap=0.02  # Adjusts the gap between bars
    )

    fig.show()
    
    return portfolio_returns
    

def compare_portfolio_to_market(
    portfolio_value: pd.DataFrame,
    market_value: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare the portfolio's return with the market's return and optionally plot the comparison.

    Parameters:
    portfolio_value (pd.DataFrame): DataFrame containing adjusted values of portfolio price at close.
    market_value (str): The market index to compare against.

    Returns:
    pd.DataFrame: A DataFrame with the cumulative returns of both the portfolio and the market index.
    """
    
    comparison_df = pd.DataFrame({
        'Portfolio': portfolio_value,
        f'Market': market_value
    })

    
    # Plot the cumulative returns
    fig = go.Figure()

    # Plot portfolio cumulative returns
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value,
        mode='lines',
        name='Portfolio',
        line=dict(color='orange')
    ))

    # Plot market cumulative returns
    fig.add_trace(go.Scatter(
        x=market_value.index,
        y=market_value,
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

    return comparison_df