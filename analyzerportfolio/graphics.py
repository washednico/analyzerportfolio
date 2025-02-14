from analyzerportfolio.logger import logger
from analyzerportfolio.config import get_plotly_template

from analyzerportfolio.utils import(
    align_series,
    process_market,
    prepare_portfolios_colors,
    prepare_portfolios

)

from typing import Union, List, Dict
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from arch import arch_model
from scipy.stats import norm
import yfinance as yf

def portfolio_value(
    portfolios: Union[dict, List[dict]],
    colors: Union[str, List[str]] = None,
    market_color: str = 'green',
    plot: bool = True,
    should_align: bool = True,
) -> pd.DataFrame:
    """
    Compare the portfolio(s) return with the market's return and plot the comparison.
    If multiple portfolios are passed, it will use market values from the first portfolio.

    Parameters
    ----------
    portfolios : dict or list of dict
        Portfolio dictionary or list of portfolio dictionaries.
    colors : str or list of str, optional
        Color or list of colors for each portfolio plot line.
        If not provided, Plotly's default colors will be used.
    market_color : str, optional
        Color for the market plot line. Defaults to 'green'.
    plot : bool, optional
        Whether to display the plot. Defaults to True.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the aligned portfolio and market values for comparison.

    Notes
    -----
    - All time series (portfolio and market values) are aligned by their dates. Rows with missing
      values are dropped to ensure consistency in comparison and plotting.
    - If plotting is enabled (`plot=True`), an interactive Plotly chart is displayed showing the
      performance of each portfolio and the market.
    - If `colors` are not specified, Plotly's default color palette will be used for the portfolios.

    Examples
    --------
    >>> portfolio1 = {
    ...     'name': 'Tech Portfolio',
    ...     'portfolio_value': pd.Series([100, 110, 120], index=['2021-01-01', '2021-01-02', '2021-01-03']),
    ...     'market_value': pd.Series([95, 100, 110], index=['2021-01-01', '2021-01-02', '2021-01-03']),
    ...     'market_ticker': 'S&P 500'
    ... }
    >>> portfolio_value(portfolio1)
                S&P 500  Tech Portfolio
    2021-01-01     95.0           100.0
    2021-01-02    100.0           110.0
    2021-01-03    110.0           120.0

    """

    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for portfolio comparison.")

    # Initialize the comparison DataFrame and plotly figure
    cumulative_values = pd.DataFrame()
    fig = go.Figure() if plot is True else None

    # Map portfolios to colors
    portfolio_colors = {portfolio['name']: color for portfolio, color in zip(portfolios, colors)}

    # Process market values 
    market_values, market_name = process_market(portfolios)
    cumulative_values[market_name] = market_values
    logger.debug(f"Added market data '{market_name}' to cumulative values.")


    # Iterate over portfolios
    for idx, portfolio in enumerate(portfolios):
        name = portfolio.get('name', 'Unnamed Portfolio')
        portfolio_values = portfolio.get('portfolio_value')

        if portfolio_values is None:
            logger.warning(f"Portfolio '{name}' has no 'portfolio_value' key.")
            continue

        # Add portfolio values to DataFrame
        portfolio_values.index = pd.to_datetime(portfolio_values.index)
        cumulative_values[name] = portfolio_values
        logger.debug(f"Added portfolio '{name}' values to cumulative DataFrame.")

    # Align all series if needed
    if should_align:
        cumulative_values = align_series(cumulative_values)

    # Plot if enabled
    if plot:
        for name in cumulative_values.columns:
            aligned_values = cumulative_values[name]

            # Retrieve the color for the current portfolio or use default
            current_color = portfolio_colors.get(name) if name != market_name else market_color

            # Define line properties
            line_props = dict(color=current_color, width=2) if name == market_name else dict(color=current_color)

            # Add trace to the plot
            fig.add_trace(go.Scatter(
                x=aligned_values.index,
                y=aligned_values,
                mode='lines',
                name=name,
                line=line_props
            ))
            logger.debug(f"Added aligned trace for '{name}' to the plot with color '{current_color}'.")

        # Update the layout of the plot
        fig.update_layout(
            title="Portfolio(s) vs Market Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            template=get_plotly_template()
        )

        # Show the plot
        fig.show()
        logger.debug("Displayed portfolio vs market performance plot.")

    # Always return the cumulative_values DataFrame
    return cumulative_values

def garch(
    portfolios: Union[dict, List[dict]],
    colors: Union[str, List[str]] = None,
    market_color: str = 'green',
    plot: bool = True,
    should_align: bool = True
) -> pd.DataFrame:
    """
    Compare the GARCH volatilities of one or multiple portfolios and optionally plot the comparison.
    If multiple portfolios are passed, it will use market returns from the first portfolio.

    Parameters
    ----------
    portfolios : dict or list of dict
        Portfolio dictionary or list of portfolio dictionaries. Each dictionary must include:
        - 'name': str, Name of the portfolio.
        - 'portfolio_returns': pandas.Series, Time series of portfolio returns indexed by date.
        - 'market_returns' (optional): pandas.Series, Time series of market returns for comparison (used from the first portfolio).
        - 'market_ticker' (optional): str, Name of the market, used for labeling in the plot.
    colors : str or list of str, optional
        Color or list of colors for each portfolio plot line. If not provided, default colors will be used.
    market_color : str, optional
        Color for the market volatility plot line. Defaults to 'green'.
    plot : bool, optional
        Whether to display the plot. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the GARCH volatilities of all portfolios (divided by 100), indexed by date.

    Raises
    ------
    ValueError
        If the length of `colors` does not match the number of portfolios.
    TypeError
        If `colors` is not a string or a list of strings.
    ValueError
        If market returns are not provided in the first portfolio.
    Exception
        If GARCH model fitting fails for a portfolio or the market.

    Notes
    -----
    - This function applies a GARCH(1,1) model to the time series of portfolio and market returns.
    - The first portfolio is used as the reference for market returns.
    - Infinite and NaN values are removed from the data before processing.
    - If `plot` is True, a Plotly figure comparing GARCH volatilities is displayed.
    - The returned DataFrame (`garch_df`) contains one column for each portfolio and the market, labeled by their names.

    Example
    -------
    >>> portfolios = [
    ...     {'name': 'Portfolio A', 'portfolio_returns': pd.Series([...]), 'market_returns': pd.Series([...])},
    ...     {'name': 'Portfolio B', 'portfolio_returns': pd.Series([...])}
    ... ]
    >>> garch_df = garch(portfolios, colors=['blue', 'red'], market_color='green', plot=True)
    >>> print(garch_df)

    """

    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for GARCH simulation.")

    # Initialize the comparison DataFrame and plotly figure
    garch_df = pd.DataFrame()
    fig = go.Figure() if plot is True else None

    # Map portfolios to colors
    portfolio_colors = {portfolio['name']: color for portfolio, color in zip(portfolios, colors)}

    # Process market values 
    market_returns, market_name = process_market(portfolios, type="returns")
    garch_df[market_name] = market_returns
    # Clean market_returns
    market_returns = market_returns.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    logger.debug(f"Processed 'market_returns' for '{market_name}'.")

    # Fit a GARCH(1,1) model to the market returns
    if not market_returns.empty:
        try:
            m_model = arch_model(market_returns * 100, vol='Garch', p=1, q=1, rescale=False)
            m_model_fit = m_model.fit(disp='off')
            market_volatility = m_model_fit.conditional_volatility / 100
            garch_df[market_name] = market_volatility
            logger.info(f"Fitted GARCH model for market '{market_name}'.")
        except Exception as e:
            logger.error(f"GARCH model failed for market '{market_name}': {e}")

    else:
        logger.warning(f"Market returns for '{market_name}' are empty after cleaning.")
        raise ValueError(f"Market returns for '{market_name}' are empty after cleaning.")

    # Iterate over each portfolio
    for idx, portfolio in enumerate(portfolios):
        name = portfolio.get('name', f'Portfolio {idx + 1}')
        portfolio_returns = portfolio.get('portfolio_returns', None)

        if portfolio_returns is None:
            logger.warning(f"Portfolio '{name}' has no 'portfolio_returns' key.")
            continue

        # Clean portfolio_returns
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
        logger.debug(f"Cleaned 'portfolio_returns' for '{name}'.")

        # Sort the index to ensure proper alignment
        portfolio_returns = portfolio_returns.sort_index()
        logger.debug(f"Sorted 'portfolio_returns' for '{name}'.")

        # Multiply returns by 100 to convert to percentage
        portfolio_returns_pct = portfolio_returns * 100
        logger.debug(f"Converted 'portfolio_returns' to percentage for '{name}'.")

        # Fit a GARCH(1,1) model to the portfolio returns
        try:
            model = arch_model(portfolio_returns_pct, vol='Garch', p=1, q=1, rescale=False)
            model_fit = model.fit(disp='off')
            logger.debug(f"Fitted GARCH(1,1) model for '{name}'.")
        except Exception as e:
            logger.error(f"Failed to fit GARCH model for '{name}': {e}")
            continue

        # Get the conditional volatility and divide by 100 to revert to original scale
        forecasted_volatility = model_fit.conditional_volatility / 100
        logger.debug(f"Computed forecasted volatility for '{name}'.")

        # Add the volatility series to the garch_df DataFrame
        garch_df[name] = forecasted_volatility
        logger.debug(f"Added forecasted volatility for '{name}' to 'garch_df'.")

        # Store color if provided
        if portfolio_colors:
            portfolio_colors[name] = colors[idx]
            logger.debug(f"Assigned color '{colors[idx]}' to portfolio '{name}'.")
        

    # Align all series if needed
    if should_align:
        garch_df = align_series(garch_df)

    # Process the figure
    if fig:
        for name in garch_df.columns:
            # Retrieve aligned values for the series
            aligned_values = garch_df[name]

            # Determine the color for the series
            line_color = portfolio_colors.get(name, market_color if name == 'Market' else None)

            # Add trace to the figure for each series
            fig.add_trace(go.Scatter(
                x=aligned_values.index,
                y=aligned_values,
                mode='lines',
                name=name,
                line=dict(
                    color=line_color,
                    width=2 if name == 'Market' else 1
                )
            ))
            logger.debug(f"Added trace for '{name}' with color '{line_color}' to the plot.")

        # Update the layout for the figure
        fig.update_layout(
            title="Comparison of GARCH Volatilities",
            xaxis_title="Date",
            yaxis_title="Volatility",
            template=get_plotly_template()
        )

        # Show the plot
        fig.show()
        logger.info("Displayed GARCH volatility comparison plot.")

    # Always return the DataFrame
    return garch_df

def montecarlo(
    portfolios: Union[dict, List[dict]],
    simulation_length: int,
    num_simulations: int = 100,
    plot: list[bool, str] = [True, None],
) -> Dict[str, pd.DataFrame]:
    """
    Perform Monte Carlo simulations on one or multiple portfolios to simulate future portfolio values and optionally simulate market values.

    Parameters
    ----------
    portfolios : Union[dict, List[dict]]
        A dictionary or list of dictionaries, each representing a portfolio.
    simulation_length : int
        The number of periods to simulate (e.g., number of days or months).
    num_simulations : int, optional
        Number of Monte Carlo simulations to run for each portfolio (default is 100).
    plot : bool, optional
        Whether to plot the simulation results (default is True).


    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary where the keys are portfolio names (or 'market') and the values are DataFrames
        with the simulated portfolio values for each simulation.

    Raises
    ------
    ValueError
        If no portfolios are provided or inputs are invalid.
    TypeError
        If `market_returns` is of an invalid type.
    """
    
    logger.debug("Starting Monte Carlo simulation...")

    # Prepare portfolios and colors
    portfolios = prepare_portfolios(portfolios)
    logger.info(f"Prepared {len(portfolios)} portfolios for simulation.")

    # Initiate new dictionary for simulation paths 
    simulation_results = {}

    # Process market return 
    market_returns, market_name = process_market(portfolios, type="returns")
    
    # Other params for montecarlo
    days_per_step = portfolios[0]["return_period_days"]
    total_investment = sum(portfolios[0]['investments'])

    def run_monte_carlo(returns_series: pd.Series, num_simulations: int, simulation_length: int, initial_value: float) -> pd.DataFrame:
        """
        Function to launch montecarlo simularion 
        """
        mean_return = returns_series.mean()
        std_dev_return = returns_series.std()
        simulation_results = np.zeros((simulation_length + 1, num_simulations))
        simulation_results[0, :] = initial_value
        for sim in range(num_simulations):
            random_returns = np.random.normal(mean_return, std_dev_return, simulation_length)
            cumulative_returns = np.cumprod(1 + random_returns)
            simulation_results[1:, sim] = initial_value * cumulative_returns
        return pd.DataFrame(simulation_results)

    # Iterate portfolios, clean data and launch montecarlo
    for portfolio in portfolios:
        name = portfolio['name']
        investments = portfolio['investments']
        initial_value = sum(investments)
        portfolio_returns = portfolio['portfolio_returns'].dropna()

        try:
            sim_df = run_monte_carlo(portfolio_returns, num_simulations, simulation_length, initial_value)
            simulation_results[name] = sim_df
            logger.info(f'Successfully completed montecarlo stimulation for {name}')
        except:
            simulation_results[name] = None
            logger.warning(f'Montecarlo simulation failed for {name}')
            continue

    # Clean and launch montecarlo
    market_returns = market_returns.dropna()
    market_sim_df = run_monte_carlo(market_returns, num_simulations, simulation_length, total_investment)
    simulation_results[market_name] = market_sim_df
    logger.info(f'Successfully completed montecarlo stimulation for {market_name}')


    # If should be plot
    if plot:
        fig = go.Figure()                       
         # Set the template (default: plotly_dark)
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
            

        fig.update_layout(
            title_text="Monte Carlo Simulations - Simulation length: "+str(simulation_length)+" (Days x Step: "+str(days_per_step)+") - Simuluations per portfolio: "+str(num_simulations), 
            template=get_plotly_template()
            )
        fig.show()
        logger.info("Displayed Montecarlo comparison plot.")

    return simulation_results

def drawdown(
            portfolios: Union[str, List[str]],
            plot: bool = True, 
            colors: Union[str, List[str]] = None,
            market_color: str = 'green',
            should_align: bool = True
) -> pd.DataFrame:
    """
    Calculate and optionally plot the drawdowns of multiple portfolios, with optional market comparison.

    Parameters:
    ----------
    portfolios : Union[str, List[str]]
        A list of portfolio dictionaries where each dictionary contains:
        - 'name' (str): The name of the portfolio.
        - 'portfolio_value' (pd.Series): The series of portfolio values over time.
        Optionally, the first portfolio can include:
        - 'market_value' (pd.Series): The market value series for drawdown comparison.

    plot : bool, optional
        Whether to plot the drawdown comparisons. Default is True.

    colors : Union[str, List[str]], optional
        The colors to use for the portfolio drawdowns. Can be:
        - A single string (same color for all portfolios).
        - A list of strings (one color per portfolio).
        If not provided, default colors will be used.

    market_color : str, optional
        The color to use for the market drawdown. Default is 'green'.

    should_align : bool, optional
        Whether to align the time indices of all portfolios and the market for consistent comparison.
        Default is True.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the drawdown series for each portfolio and the market, if applicable.

    Raises:
    ------
    ValueError
        If no portfolios are provided or if the length of the colors list does not match the number of portfolios.

    Notes:
    -----
    - Drawdown is calculated as the percentage decline from the peak value over time.
    - If 'market_value' is provided in the first portfolio, its drawdown will also be calculated and included.
    - When plotting, the market drawdown is visually distinct with the specified `market_color`.
    """

    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for simulation.")

    def calculate_drawdown(values):
        """Calculate drawdowns given a series of portfolio values."""
        peak = values.cummax()
        drawdown = (values - peak) / peak
        return drawdown * 100  # convert to percentage
    
    # Initialize the comparison DataFrame and plotly figure
    drawdown_data = pd.DataFrame()
    fig = go.Figure() if plot is True else None

    # Map portfolios to colors
    portfolio_colors = {portfolio['name']: color for portfolio, color in zip(portfolios, colors)}

    # Process portfolio values
    for portfolio, color in zip(portfolios,colors):
        portfolio_name = portfolio['name']
        portfolio_value = portfolio['portfolio_value']

        logger.debug(f"Processing portfolio '{portfolio_name}' with color '{color}'.")
        drawdown_data[portfolio_name] = calculate_drawdown(portfolio_value)

    # Process market values 
    market_values, market_name = process_market(portfolios)
    market_drawdown = calculate_drawdown(market_values)
    drawdown_data[market_name] = market_drawdown

    # Align all series if needed
    if should_align:
        drawdown_data = align_series(drawdown_data)

    # Plot the data
    if plot:
        fig = go.Figure()
        for name, color in portfolio_colors.items():
            fig.add_trace(go.Scatter(
                x=drawdown_data.index,
                y=drawdown_data[name],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))

        # Add market drawdown if available
        if market_name and market_name in drawdown_data:
            fig.add_trace(go.Scatter(
                x=drawdown_data.index,
                y=drawdown_data[market_name],
                mode='lines',
                name=market_name,
                line=dict(color=market_color, width=3)
            ))

        # Update layout
        fig.update_layout(
            title="Portfolio Drawdown Comparison",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=get_plotly_template()
        )

        fig.show()
        logger.info("Displayed Drawdown comparison plot.")

    return drawdown_data

def heatmap(
        portfolios: Union[dict, List[dict]],
        colors: Union[str, List[str]] = None,
        plot: bool = True,
        disassemble: bool = False
 )-> None:
    """
    Generate and optionally plot a heatmap for correlation analysis between portfolios or their individual assets.

    Parameters:
    ----------
    portfolios : Union[dict, List[dict]]
        A dictionary or a list of portfolio dictionaries, each representing a portfolio. Each dictionary should contain:
        - 'name' (str): The name of the portfolio.
        - If `disassemble` is True:
            - 'tickers' (list[str]): A list of asset tickers included in the portfolio.
            - 'returns' (pd.DataFrame): A DataFrame with asset-level returns.
        - If `disassemble` is False:
            - 'portfolio_returns' (pd.Series): A series of overall portfolio returns over time.
        Optionally, the first portfolio may include:
        - 'market_returns' (pd.Series): Market return series for correlation analysis.
        - 'market_ticker' (str): Name of the market index (default is 'Market').

    colors : Union[str, List[str]], optional
        A string representing a single color or a list of colors corresponding to the portfolios.
        Default colors are used if not provided.

    plot : bool, optional
        Whether to plot the heatmap using Plotly. Default is True.

    disassemble : bool, optional
        If True, break down portfolios into individual asset returns for the correlation analysis.
        If False (default), use overall portfolio returns.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the correlation coefficients between the returns of the portfolios and, optionally, the market.

    Notes:
    -----
- If `disassemble` is True, the correlation analysis is performed at the asset level.
    - If `disassemble` is False, the correlation analysis is based on overall portfolio returns.
    - The market return series, if provided, is included in the correlation analysis.
    - The heatmap is visualized using Plotly if `plot=True`.
    """
    
    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for portfolio comparison.")

    # Initialize the all_returns dict and plotly figure
    all_returns = {}
    fig = go.Figure() if plot is True else None

    # Process market values 
    market_returns, market_name = process_market(portfolios, type = "returns")

    # Clean, sort and add market_returns to all_returns dict
    market_returns = market_returns.replace([np.inf, -np.inf], np.nan).dropna()
    market_returns = market_returns.sort_index()
    all_returns[market_name] = market_returns

    # Disassemble portfolios into individual asset returns if nedeed
    if disassemble:
        for portfolio in portfolios:
            tickers = portfolio['tickers']
            revised_tickers = [ticker + "_Return" for ticker in tickers]
            
            # Extract asset returns
            portfolio_returns = portfolio['returns'][revised_tickers].copy()
            portfolio_returns.rename(columns=lambda x: x.replace("_Return", ""), inplace=True)
            all_returns.update(portfolio_returns.to_dict(orient='series'))

    else:
        # Use overall portfolio returns
        for portfolio in portfolios:
            name = portfolio['name']
            portfolio_returns = portfolio['portfolio_returns']

            # Ensure portfolio_returns is a Pandas Series
            if isinstance(portfolio_returns, pd.DataFrame):
                portfolio_returns = portfolio_returns.squeeze()

            # Clean, sort and add portfolio_returns to all_returns dict
            portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
            portfolio_returns = portfolio_returns.sort_index()
            all_returns[name] = portfolio_returns

    # Create a DataFrame from all the returns
    combined_returns = pd.DataFrame(all_returns)

    # Calculate the correlation matrix
    corr_matrix = combined_returns.corr()

    # Set plot layout and plot the result
    if plot:
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        fig.update_layout(
            title_text="Heatmap of Portfolio Correlations" if not disassemble else "Heatmap of Asset Correlations",
            template = get_plotly_template())
        fig.show()

    return combined_returns

def pie_chart(
        portfolios: Union[dict, List[dict]], 
        colors: Union[str, List[str]] = None, 
        plot: bool = True,
        threshold: float = 0.001):
    """
    Plot a pie chart to visualize the asset allocation of an ETF-based investment strategy for one or multiple portfolios.

    Small allocations (less than the specified threshold) are grouped into an "All Others" category for simplicity.

    Parameters:
    ----------
    portfolios : Union[dict, List[dict]]
        A dictionary or a list of portfolio dictionaries, each representing a portfolio. Each dictionary should contain:
        - 'name' (str): The name of the portfolio.
        - 'tickers' (list[str]): A list of asset tickers in the portfolio.
        - 'weights' (list[float]): A list of corresponding allocation weights for the assets.

    colors : Union[str, List[str]], optional
        A string representing a single color or a list of colors for the portfolio segments.
        If not provided, default colors are used.

    plot : bool, optional
        Whether to display the pie chart using Plotly. Default is True.

    threshold : float, optional
        The allocation threshold below which allocations are grouped into an "All Others" category. Default is 0.001.


    Returns:
    -------
    None
        The function displays the pie chart if `plot` is set to True. Otherwise, no output is returned.

    Notes:
    -----
    - Small allocations below the threshold are summed up into an "All Others" category to declutter the chart.
    - The pie chart displays each segment's label and percentage inside the chart slices.
    - The chart's background can be customized with the `transparent` parameter.
    """

    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for portfolio comparison.")
    
    for portfolio, color in zip(portfolios, colors):
        name = portfolio['name']
        tickers = portfolio['tickers']
        weights = portfolio['weights']

        # Create a dictionary of tickers and their corresponding weights
        allocations = dict(zip(tickers, weights))
        logger.debug(f"Allocations for portfolio '{name}': {allocations}")

        # Separate small allocations into "All Others"
        large_allocations = {k: v for k, v in allocations.items() if v >= threshold}
        small_allocations_total = sum(v for v in allocations.values() if v < threshold)
        
        # Create "All Others" if there are small allocations
        if small_allocations_total > 0:
            large_allocations["All Others"] = round(small_allocations_total, 8)
            logger.debug(f"Grouped small allocations into 'All Others' for portfolio '{name}'.")
        
        # Extract asset names and their respective sizes
        labels = list(large_allocations.keys())
        sizes = list(large_allocations.values())
        logger.debug(f"Final labels and sizes for portfolio '{name}': {labels}, {sizes}")
        
    
        # Show the pie chart if plot is True
        if plot:
            # Create a pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=sizes, 
                hole=.3,
                textinfo='label+percent',  # Show both label and percentage on the chart
                textposition='inside',  # Position text inside the slices
            )])

            # Update the layout for the pie chart
            fig.update_layout(
                title_text=f"{name} - Portfolio Asset Allocation",
                showlegend=False,  # Hide the legend
                title_font_size=20,
                paper_bgcolor='black',  # Set background color to black
                font_color='white',     # Set text color to white
                template = get_plotly_template()
            )
            
            fig.show()
            logger.info(f"Displayed pie chart for portfolio '{name}'.")

def sector_pie(
        portfolios: Union[dict, List[dict]], 
        colors: Union[str, List[str]] = None, 
        plot: bool = True,
        threshold: float = 0.001
        ) -> pd.DataFrame :
    """
    Generate a sector distribution pie chart based on equity portfolios and return a DataFrame of sector weights.

    Parameters:
    ----------
    portfolios : Union[dict, List[dict]]
        Portfolio(s) containing 'name', 'tickers', and 'weights'.
    colors : Union[str, List[str]], optional
        Color scheme for the pie chart. Defaults to None.
    plot : bool, optional
        Whether to display the plot. Defaults to True.
    threshold : float, optional
        Minimum allocation for a sector to be displayed separately. Defaults to 0.001.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the aggregated sector weights for all portfolios.
    """
    
    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for portfolio comparison.")

    # Initialize a list of sectors data
    all_sector_data = []
    
    for portfolio, color in zip(portfolios, colors):
        portfolio_name = portfolio['name']
        logger.info(f"Processing portfolio: {portfolio_name}")

        # Create allocations
        allocations = dict(zip(portfolio['tickers'], portfolio['weights']))

        # Fetch sector data
        sector_info = []
        for ticker, weight in allocations.items():
            try:
                stock = yf.Ticker(ticker)
                sector = stock.info.get('sector', 'Unknown')
                sector_info.append({'Sector': sector, 'Weight': weight})
            except Exception as e:
                logger.warning(f"Error fetching data for {ticker}: {e}")
                sector_info.append({'Sector': 'Error', 'Weight': weight})

        # Create a DataFrame
        df = pd.DataFrame(sector_info)

        # Aggregate weights by sector
        sector_weights = df.groupby('Sector')['Weight'].sum()

        # Filter sectors below the threshold
        large_sectors = sector_weights[sector_weights >= threshold]
        small_sectors_total = sector_weights[sector_weights < threshold].sum()

        if small_sectors_total > 0:
            large_sectors["All Others"] = small_sectors_total
            logger.info(f"Grouped small sectors into 'All Others' for portfolio: {portfolio_name}")

        # Add portfolio name to the sector data for aggregation
        large_sectors_df = pd.DataFrame({
            'Portfolio': portfolio_name,
            'Sector': large_sectors.index,
            'Weight': large_sectors.values
        })
        all_sector_data.append(large_sectors_df)

        # Plot pie chart
        if plot:
            fig = go.Figure(data=[go.Pie(
                labels=large_sectors.index,
                values=large_sectors.values,
                hole=0.3,
                textinfo='label+percent',
                textposition='inside',
                marker=dict(colors=color) if color else None,
            )])

            fig.update_layout(
                title_text=f"Sector Distribution: {portfolio_name}",
                template = get_plotly_template(),
                font_color='black'
            )
            fig.show()
            logger.info(f"Displayed pie chart for portfolio: {portfolio_name}")

    # Combine all sector data into a single DataFrame
    combined_sector_data = pd.concat(all_sector_data, ignore_index=True)
    logger.info("Sector pie chart generation and aggregation completed.")
    return combined_sector_data

def country_pie(
        portfolios: Union[dict, List[dict]], 
        colors: Union[str, List[str]] = None, 
        plot: bool = True,
        threshold: float = 0.001
        ) -> pd.DataFrame:
    """
    Generate a geographical distribution pie chart based on equity portfolios and return a DataFrame of country weights.

    Parameters:
    ----------
    portfolios : Union[dict, List[dict]]
        Portfolio(s) containing 'name', 'tickers', and 'weights'.
    colors : Union[str, List[str]], optional
        Color scheme for the pie chart. Defaults to None.
    plot : bool, optional
        Whether to display the plot. Defaults to True.
    threshold : float, optional
        Minimum allocation for a country to be displayed separately. Defaults to 0.001.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the aggregated country weights for all portfolios.
    """

    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for geographical distribution analysis.")

    # Initiate all_country data list 
    all_country_data = []

    for portfolio, color in zip(portfolios, colors):
        portfolio_name = portfolio['name']
        logger.info(f"Processing portfolio: {portfolio_name}")

        # Create allocations
        allocations = dict(zip(portfolio['tickers'], portfolio['weights']))

        # Fetch geographical data
        geographical_info = []
        for ticker, weight in allocations.items():
            try:
                stock = yf.Ticker(ticker)
                country = stock.info.get('country', 'Unknown')
                geographical_info.append({'Portfolio': portfolio_name, 'Country': country, 'Weight': weight})
            except Exception as e:
                logger.warning(f"Error fetching data for {ticker}: {e}")
                geographical_info.append({'Portfolio': portfolio_name, 'Country': 'Error', 'Weight': weight})

        # Create a DataFrame for this portfolio
        df = pd.DataFrame(geographical_info)

        # Aggregate weights by country
        country_weights = df.groupby('Country')['Weight'].sum()

        # Filter countries below the threshold
        large_countries = country_weights[country_weights >= threshold]
        small_countries_total = country_weights[country_weights < threshold].sum()

        if small_countries_total > 0:
            large_countries["All Others"] = small_countries_total
            logger.info(f"Grouped small countries into 'All Others' for portfolio: {portfolio_name}")

        # Add portfolio name to the country data for aggregation
        large_countries_df = pd.DataFrame({
            'Portfolio': portfolio_name,
            'Country': large_countries.index,
            'Weight': large_countries.values
        })
        all_country_data.append(large_countries_df)

        # Plot pie chart
        if plot:
            fig = go.Figure(data=[go.Pie(
                labels=large_countries.index,
                values=large_countries.values,
                hole=0.3,
                textinfo='label+percent',
                textposition='inside',
                marker=dict(colors=color) if color else None,
            )])

            fig.update_layout(
                title_text=f"Geographical Distribution: {portfolio_name}",
                font_color='black',
                template = get_plotly_template(),
            )
            fig.show()
            logger.info(f"Displayed pie chart for portfolio: {portfolio_name}")

    # Combine all country data into a single DataFrame
    combined_country_data = pd.concat(all_country_data, ignore_index=True)
    logger.info("Geographical pie chart generation and aggregation completed.")
    return combined_country_data
    
def distribution_return(
    portfolios: Union[dict, List[dict]],
    bins: int = 100,
    colors: Union[str, List[str]] = None,
    market_color: str = 'green',
    plot: bool = True
) -> pd.DataFrame:
    """
    Plot the distribution of portfolio returns over a specified time interval and return a DataFrame of histogram data.

    Parameters:
    ----------
    portfolios : Union[dict, List[dict]]
        A dictionary or a list of dictionaries representing portfolios. Each dictionary should contain:
        - 'name' (str): The name of the portfolio.
        - 'portfolio_returns' (pd.Series): A series of portfolio returns.
        - 'return_period_days' (int): Number of days in the return period.
        Optionally, the first portfolio may include:
        - 'market_returns' (pd.Series): A series of market returns for comparison.
        - 'market_ticker' (str): The name of the market (default is 'Market').

    bins : int, optional
        Number of bins for the histogram. Default is 100.

    colors : Union[str, List[str]], optional
        A string or a list of colors for the portfolios. Defaults to None.

    market_color : str, optional
        The color to use for the market returns histogram. Default is 'green'.

    plot : bool, optional
        Whether to display the histogram plot. Default is True.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the bin edges and the normalized frequencies of returns for each portfolio and the market.
    """

    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for portfolio comparison.")

    # Initialize DataFrame for histogram data and optional plot
    combined_histogram_data = []
    fig = go.Figure() if plot else None

    # Extract market details if present in the first portfolio
    days_per_step = portfolios[0].get("return_period_days", 1)
    market_returns = portfolios[0].get('market_returns', None)
    market_name = portfolios[0].get('market_ticker', 'Market')

    # Process each portfolio
    for portfolio, color in zip(portfolios, colors):
        portfolio_name = portfolio.get('name', 'Unnamed Portfolio')
        portfolio_returns = portfolio.get('portfolio_returns')

        if portfolio_returns is None:
            logger.warning(f"Portfolio '{portfolio_name}' has no 'portfolio_returns' key.")
            continue

        # Ensure portfolio_returns is a pandas Series
        portfolio_returns = pd.Series(portfolio_returns).dropna()

        # Calculate histogram data
        hist, bin_edges = np.histogram(portfolio_returns, bins=bins, density=True)
        combined_histogram_data.append(pd.DataFrame({
            'Portfolio': portfolio_name,
            'BinEdges': bin_edges[:-1],  # Exclude the last bin edge
            'Density': hist
        }))

        # Add portfolio histogram to the plot
        if plot:
            fig.add_trace(go.Histogram(
                x=portfolio_returns,
                nbinsx=bins,
                histnorm='probability density',
                marker=dict(
                    color=color,
                    line=dict(color='black', width=1)
                ),
                opacity=0.75,
                name=portfolio_name
            ))

    # Add market histogram if market returns are provided
    if market_returns is not None:
        market_returns = pd.Series(market_returns).dropna()
        hist, bin_edges = np.histogram(market_returns, bins=bins, density=True)
        combined_histogram_data.append(pd.DataFrame({
            'Portfolio': market_name,
            'BinEdges': bin_edges[:-1],
            'Density': hist
        }))

        if plot:
            fig.add_trace(go.Histogram(
                x=market_returns,
                nbinsx=bins,
                histnorm='probability density',
                marker=dict(
                    color=market_color,
                    line=dict(color='black', width=1)
                ),
                opacity=0.75,
                name=market_name
            ))

    # Combine all histogram data into a single DataFrame
    combined_histogram_data = pd.concat(combined_histogram_data, ignore_index=True)

    # Update layout for the plot
    if plot:
        fig.update_layout(
            title="Distribution of Portfolio Returns",
            template=get_plotly_template(),
            xaxis=dict(
                title=f"{days_per_step}-Day Returns",
                tickformat='.2%',
                showgrid=True,
                zeroline=True
            ),
            yaxis_title="Probability Density",
            bargap=0.02
        )
        fig.show()

    return combined_histogram_data

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

    # Prepare portfolios and colors
    portfolios, colors = prepare_portfolios_colors(portfolios, colors)
    logger.info(f"Prepared {len(portfolios)} portfolios for portfolio comparison.")

    # Initialize the results dict, summary data list and plotly figure
    results = {}
    fig = go.Figure() if plot is True else None
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
        template=get_plotly_template(),
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
            template=get_plotly_template(),
            height=600
        )

        fig.show()

    return probability_cone_df

def garch_diff(
    portfolios: Union[dict, List[dict]],
    colors: Union[str, List[str]] = None,
    market_color: str = 'green',
    plot: bool = True
) -> pd.DataFrame:
    """
    Compare the GARCH volatilities of the difference between portfolio returns and market returns.
    Plot the comparison of volatilities.

    Parameters:
    - portfolios (dict or list of dict): Portfolio dictionary or list of portfolio dictionaries
      created from the create_portfolio function.
    - colors (str or list[str], optional): Color or list of colors for each portfolio plot line.
    - market_color (str, optional): Color for the market volatility plot line (default is 'green').
    - plot (bool, optional): Whether to plot the results (default is True).

    Returns:
    - pd.DataFrame: A DataFrame with the GARCH volatilities of the portfolio-market return differences (divided by 100).
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

        # Align portfolio returns with market returns
        aligned_portfolio, aligned_market = portfolio_returns.align(market_returns, join='inner')

        # Calculate the difference between portfolio returns and market returns
        diff_returns = (aligned_portfolio - aligned_market)

        # Multiply returns by 100 to convert to percentage
        diff_returns_pct = diff_returns * 100

        # Fit a GARCH(1,1) model to the portfolio-market return difference
        model = arch_model(diff_returns_pct, vol='Garch', p=1, q=1, rescale=False)
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
        title="Comparison of GARCH Volatilities (Portfolio Returns - Market Returns)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        template=get_plotly_template()
    )

    # Display the plot
    if plot:
        fig.show()

    return volatility_df
