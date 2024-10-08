import analyzerportfolio as ap
import numpy as np
from scipy.optimize import minimize

def optimize(portfolio: dict, 
             metric: str = 'sharpe'):
    """
    Optimize the investments in the portfolio to maximize the Sharpe ratio.
    
    Parameters:
    portfolio (dict): A portfolio dictionary created using the `create_portfolio` function.
    
    Returns:
    dict: The optimized portfolio dictionary with updated investments to maximize Sharpe ratio.
    """
    
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

