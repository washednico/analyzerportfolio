import analyzerportfolio as ap
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import pandas as pd
from typing import Union, List, Dict
from analyzerportfolio.logger import logger
from analyzerportfolio.config import get_plotly_template


def optimize(portfolio: dict, 
             metric: str = 'sharpe'):
    """
    Optimize the investments in the portfolio to maximize the Sharpe ratio.
    
    Parameters:
    portfolio (dict): A portfolio dictionary created using the `create_portfolio` function.
    metric (str): The metric to optimize for. It can be 'sharpe', 'drawdown', 'volatility', or 'information_ratio'.
    
    Returns:
    dict: The optimized portfolio dictionary with updated investments to maximize Sharpe ratio.
    """

    if metric not in ['sharpe', 'drawdown', 'volatility', 'information_ratio']:
        raise ValueError("Invalid metric. Choose from 'sharpe', 'drawdown', 'volatility', or 'information_ratio'.")
    
    tickers = portfolio['tickers']
    initial_investments = portfolio['investments']
    data = portfolio['untouched_data']
    market_ticker = portfolio['market_ticker']
    base_currency = portfolio['base_currency']
    rebalancing_period_days = portfolio['auto_rebalance']
    risk_free_rate = portfolio['risk_free_returns']
    target_weights = portfolio["target_weights"]  
    return_period_days = portfolio['return_period_days']

    def negative_sharpe_ratio(weights):
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights /= np.sum(weights)

        # Update the portfolio with the new weights (investments)
        updated_investment = weights * sum(initial_investments)
        updated_portfolio = ap.create_portfolio(data, tickers, investments=updated_investment, 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)

        # Calculate the Sharpe ratio
        sharpe_ratio = ap.c_sharpe(updated_portfolio)

        return -sharpe_ratio  # We negate since we're minimizing
    
    def drawdwon(weights):
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights /= np.sum(weights)

        # Update the portfolio with the new weights (investments)
        updated_investment = weights * sum(initial_investments)
        updated_portfolio = ap.create_portfolio(data, tickers, investments=updated_investment, 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)

        # Calculate the Sharpe ratio
        drawdown = ap.c_max_drawdown(updated_portfolio)

        return drawdown
    
    def volatility(weights):
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights /= np.sum(weights)

        # Update the portfolio with the new weights (investments)
        updated_investment = weights * sum(initial_investments)
        updated_portfolio = ap.create_portfolio(data, tickers, investments=updated_investment, 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)

        # Calculate the Sharpe ratio
        volatility = ap.c_volatility(updated_portfolio)

        return volatility

    def negative_information_ratio(weights):

        # Ensure weights sum to 1
        weights = np.array(weights)
        weights /= np.sum(weights)

        # Update the portofolio with the new weights (investments)
        updated_investment = weights * sum(initial_investments)
        updated_portfolio = ap.create_portfolio(data, tickers, investments=updated_investment, 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)

        # Calculate the Information ratio
        information_ratio = ap.c_info_ratio(updated_portfolio)
        return -information_ratio

    
    # Constraints: the weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds for weights: between 0 and 1 (no short selling)
    bounds = [(0, 1) for _ in range(len(tickers))]

    # Initial guess: normalize the initial investments
    initial_weights = np.array(initial_investments) / np.sum(initial_investments)

    if metric == 'sharpe':


        # Optimization using scipy's minimize function
        result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Optimized weights (investments)
        optimal_weights = result.x
        portfolio['investments'] = optimal_weights * sum(initial_investments)  # Scale weights back to investment amounts
        
        new_portfolio = ap.create_portfolio(data, tickers, investments=portfolio['investments'], 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)
        # Return the updated portfolio with optimized investments
        return new_portfolio
    
    if metric == "drawdown":
                # Optimization using scipy's minimize function
        result = minimize(drawdwon, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Optimized weights (investments)
        optimal_weights = result.x
        portfolio['investments'] = optimal_weights * sum(initial_investments)  # Scale weights back to investment amounts
        
        new_portfolio = ap.create_portfolio(data, tickers, investments=portfolio['investments'], 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)
        # Return the updated portfolio with optimized investments
        return new_portfolio
    
    if metric == "volatility":
        # Optimization using scipy's minimize function
        result = minimize(volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Optimized weights (investments)
        optimal_weights = result.x
        portfolio['investments'] = optimal_weights * sum(initial_investments)  # Scale weights back to investment amounts
        
        new_portfolio = ap.create_portfolio(data, tickers, investments=portfolio['investments'], 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)
        # Return the updated portfolio with optimized investments
        return new_portfolio
    
    if metric == "information_ratio":
        # Optimization using scipy's minimize function
        result = minimize(negative_information_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Optimized weights (investments)
        optimal_weights = result.x
        portfolio['investments'] = optimal_weights * sum(initial_investments)  # Scale weights back to investment amounts
        
        new_portfolio = ap.create_portfolio(data, tickers, investments=portfolio['investments'], 
                                             market_ticker=market_ticker, 
                                             name_portfolio=portfolio['name'], 
                                             base_currency=base_currency, 
                                             rebalancing_period_days=rebalancing_period_days,
                                             return_period_days=return_period_days,
                                             target_weights= target_weights)
        # Return the updated portfolio with optimized investments
        return new_portfolio


# Move this function outside efficient_frontier so it's picklable
def minimize_volatility_for_target_return(args):
    (
        target_return,
        data,
        tickers,
        total_investment,
        initial_weights,
        bounds,
        constraints,
        portfolio_return,
        volatility_objective,
        method,
    ) = args

    # Constraint: portfolio expected return equals target_return
    return_constraint = {
        'type': 'eq',
        'fun': lambda weights: portfolio_return(weights) - target_return
    }

    result = minimize(
        volatility_objective,
        initial_weights,
        method=method,
        bounds=bounds,
        constraints=[constraints, return_constraint]
    )
    return result

# Top-level function for the weight sum constraint
def weight_sum_constraint(weights):
    return np.sum(weights) - 1

# Top-level function for the return constraint
def return_constraint_function(weights, target_return, data, tickers, total_investment, other_params):
    ret = portfolio_return(weights, data, tickers, total_investment, other_params)
    return ret - target_return

# Top-level function for portfolio return
def portfolio_return(weights, data, tickers, total_investment, other_params):
    weights = np.array(weights)
    investments = weights * total_investment
    updated_portfolio = ap.create_portfolio(
        data, tickers, investments=investments, **other_params
    )
    return ap.c_return(updated_portfolio)

# Top-level function for portfolio volatility
def volatility_objective(weights, data, tickers, total_investment, other_params):
    weights = np.array(weights)
    weights /= np.sum(weights)
    investments = weights * total_investment
    updated_portfolio = ap.create_portfolio(
        data, tickers, investments=investments, **other_params
    )
    return ap.c_volatility(updated_portfolio)

# Top-level function for negative portfolio return (for maximization)
def negative_portfolio_return(weights, data, tickers, total_investment, other_params):
    return -portfolio_return(weights, data, tickers, total_investment, other_params)

# Top-level function for portfolio return scalar (used in minimization)
def portfolio_return_scalar(weights, data, tickers, total_investment, other_params):
    return portfolio_return(weights, data, tickers, total_investment, other_params)

# Top-level function for minimizing volatility for a target return
def minimize_volatility_for_target_return(args):
    (
        target_return,
        data,
        tickers,
        total_investment,
        initial_weights,
        bounds,
        method,
        other_params
    ) = args

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': weight_sum_constraint},
        {'type': 'eq', 'fun': return_constraint_function,
         'args': (target_return, data, tickers, total_investment, other_params)}
    ]

    # Minimize volatility
    result = minimize(
        volatility_objective,
        initial_weights,
        args=(data, tickers, total_investment, other_params),
        method=method,
        bounds=bounds,
        constraints=constraints
    )
    return result

def efficient_frontier(
    portfolio: dict,
    num_points: int,
    multi_thread: bool = False,
    num_threads: int = 4,
    method: str = 'SLSQP',
    additional_portfolios: Union[str, List[str]] = None,
    colors: Union[str, List[str]] = None,
):
    """
    Computes and plots the upper part of the efficient frontier for a given portfolio,
    optimizing asset allocations to achieve various target returns while minimizing volatility.
    
    Parameters:
    - portfolio (dict): Dictionary containing portfolio data, including tickers, investments, and market details.
    - num_points (int): Number of points to compute along the efficient frontier.
    - multi_thread (bool, optional): Whether to use multi-threading for optimization. Defaults to False.
    - num_threads (int, optional): Number of threads to use if multi-threading is enabled. Defaults to 4.
    - method (str, optional): Optimization method for portfolio allocation. Defaults to 'SLSQP'.
    - additional_portfolios (Union[str, List[str]], optional): Additional portfolios to plot alongside the efficient frontier.
    - colors (Union[str, List[str]], optional): Colors corresponding to additional portfolios in the plot.

    Returns:
    - dict: A dictionary where keys represent portfolio points along the efficient frontier, and values are the corresponding portfolios.
    """
    # Ensure portfolios is a list
    if isinstance(additional_portfolios, dict):
        additional_portfolios = [additional_portfolios]

    # Ensure colors is a list
    if colors is None:
        colors = [None] * len(additional_portfolios)
    elif isinstance(colors, str):
        colors = [colors]
    elif isinstance(colors, list):
        if len(colors) != len(additional_portfolios):
            raise ValueError("The length of 'colors' must match the number of portfolios.")
    else:
        raise ValueError("Invalid type for 'colors' parameter.")
    
    # Extract necessary data
    tickers = portfolio['tickers']
    initial_investments = portfolio['investments']
    data = portfolio['untouched_data']
    market_ticker = portfolio['market_ticker']
    base_currency = portfolio['base_currency']
    rebalancing_period_days = portfolio['auto_rebalance']
    risk_free_rate = portfolio['risk_free_returns']
    target_weights = portfolio["target_weights"]  
    return_period_days = portfolio['return_period_days']
    portfolio_name = portfolio['name']
    exclude_ticker_time=portfolio["exclude_ticker_time"]
    exclude_ticker=portfolio["exclude_ticker"]

    total_investment = sum(initial_investments)

    # Initial guess: normalize the initial investments
    initial_weights = np.array(initial_investments) / total_investment

    num_assets = len(tickers)

    # Bounds for weights: between 0 and 1 (no short selling)
    bounds = [(0, 1) for _ in range(num_assets)]

    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': weight_sum_constraint}]

    # Prepare 'other_params' to pass to create_portfolio
    other_params = {
        'market_ticker': market_ticker,
        'name_portfolio': portfolio_name,
        'base_currency': base_currency,
        'rebalancing_period_days': rebalancing_period_days,
        'return_period_days': return_period_days,
        'target_weights': target_weights,
        "exclude_ticker_time" : exclude_ticker_time,
        "exclude_ticker" :  exclude_ticker,
         # or weights, depending on your library
    }

    # Compute the Minimum Variance Portfolio (MVP) using volatility_objective
    min_var_result = minimize(
        volatility_objective,  # Minimize volatility
        initial_weights,
        args=(data, tickers, total_investment, other_params),
        method=method,
        bounds=bounds,
        constraints=constraints
    )

    # Extract the optimal weights from the result
    min_var_weights = min_var_result.x

    # Calculate the volatility and return for the Minimum Variance Portfolio
    min_var_volatility = ap.c_volatility(ap.create_portfolio(data, tickers, investments=min_var_weights * total_investment, **other_params))
    min_var_return = portfolio_return(min_var_weights, data, tickers, total_investment, other_params)


    # Compute the Maximum Return Portfolio (MRP)
    max_return_result = minimize(
        negative_portfolio_return,
        initial_weights,
        args=(data, tickers, total_investment, other_params),
        method=method,
        bounds=bounds,
        constraints=constraints
    )
    max_return = portfolio_return(max_return_result.x, data, tickers, total_investment, other_params)
    

    # Generate target returns including those below MVP return, but filter later
    target_returns = np.linspace(min_var_return, max_return, num_points)

    # Prepare arguments for optimization
    args_list = [
        (
            tr,
            data,
            tickers,
            total_investment,
            initial_weights,
            bounds,
            method,
            other_params
        )
        for tr in target_returns
    ]

    # Compute efficient frontier only for the upper part
    if multi_thread:
        with multiprocessing.Pool(num_threads) as pool:
            results = pool.map(minimize_volatility_for_target_return, args_list)
    else:
        results = [minimize_volatility_for_target_return(args) for args in args_list]

    # Initialize lists to store portfolio metrics
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_weights = []

    for result in results:
        if result.success:
            weights = result.x
            investments = weights * total_investment
            updated_portfolio = ap.create_portfolio(
                data, tickers, investments=investments, **other_params
            )
            ret = ap.c_return(updated_portfolio)
            vol = ap.c_volatility(updated_portfolio)
            
            # Keep only portfolios that are part of the upper frontier
            if vol <= min_var_volatility or ret >= min_var_return:
                portfolio_returns.append(ret)
                portfolio_volatilities.append(vol)
                portfolio_weights.append(weights)
        else:
            # Handle optimization failure
            print(f"Optimization failed for target return {result.x}")
            continue

    # Plot the efficient frontier using Plotly (only upper part)
    fig = go.Figure()
    portfolio_volatilities.insert(0, min_var_volatility)
    portfolio_returns.insert(0, min_var_return)

    # Add efficient frontier trace
    fig.add_trace(go.Scatter(
        x=portfolio_volatilities,
        y=portfolio_returns,
        mode='lines',
        name='Efficient Frontier'
    ))

    # Plot the Minimum Variance Portfolio
    fig.add_trace(go.Scatter(
        x=[min_var_volatility],
        y=[min_var_return],
        mode='markers',
        name='Minimum Variance Portfolio',
        marker=dict(color='red', size=10)
    ))

    # Add maximum return portfolio
    max_ret_index = np.argmax(portfolio_returns)
    fig.add_trace(go.Scatter(
        x=[portfolio_volatilities[max_ret_index]],
        y=[portfolio_returns[max_ret_index]],
        mode='markers',
        name='Maximum Return Portfolio',
        marker=dict(color='green', size=10)
    ))

    fig.update_layout(
        title='Efficient Frontier (Upper Part Only)',
        xaxis_title='Volatility',
        yaxis_title='Return',
        showlegend=True,
        template = get_plotly_template()
    )

    if additional_portfolios is not None:
        for port, color in zip(additional_portfolios, colors):
            port_vol = ap.c_volatility(port)
            port_ret = ap.c_return(port)
            fig.add_trace(go.Scatter(
                x=[port_vol],
                y=[port_ret],
                mode='markers',
                name=port["name"],
                marker=dict(color=color, size=10)
            ))

    fig.show()

    # Create the dictionary of portfolios
    portfolios_dict = {}

    # TODO - Check on portfolio dict
    
    for i, weights in enumerate(portfolio_weights):
        investments = weights * total_investment
        # Create portfolio using 'create_portfolio'
        new_portfolio = ap.create_portfolio(
            data, tickers, investments=investments, **other_params
        )
        portfolios_dict[i+1] = new_portfolio

    # Return the dictionary of portfolios
    return portfolios_dict
   
