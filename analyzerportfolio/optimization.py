import numpy as np
import pandas as pd

from scipy import optimize

from analyzerportfolio.metrics import (
    calculate_daily_returns
)

from analyzerportfolio.utils import (
    check_dataframe
)

def markowitz_optimization(data: pd.DataFrame, tickers: list[str], investments: list[float], rf_rate:float = 0.0, plot: bool = True, method = 'sharpe', target = 0.05):
    """
    Perform Markowitz optimization to find the minimum variance portfolio and plot the efficient frontier.

    Parameters:
    data (pd.DataFrame): DataFrame containing historical price data for assets and the market index.
    tickers (list[str]): List of asset tickers in the portfolio.
    investments (list[float]): List of monetary investments for each asset.
    rf_rate (float): Indicating risk free rate (defaul is 0.0).
    plot (bool): Whether to plot the results (default is True).
    method (str): Optimization method to use (default is 'sharpe'). Value accepted are 'sharpe', 'variance', 'return', 'sortino'.
    target (float): Target return for the portfolio (default is 0.05).

    Returns:
    dict: Dictionary containg portfolio metrics based on optimal weights

    -
    All weights, of course, must be between 0 and 1. Thus we set 0 and 1 as the boundaries. 
    The second boundary is the sum of weights = 1.
    
    Sequential Least Squares Programming (SLSQP) Algorithm
    - https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html

    -

    NOTE: we are minimizing the negative of sharpe ratio since maximise function is not supported by scipy
    """

    def portfolio_performance(weights, mean_returns, cov_matrix):
        """
        Calculate portfolio metrics.
        """
        returns =  np.matmul(mean_returns.T, weights) *252

        variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        vol = np.sqrt(variance) * np.sqrt(252)

        sharpe = (returns-rf_rate) / vol

        downside_deviation = np.sqrt(np.mean(np.minimum(0, returns - rf_rate) ** 2))
        sortino_ratio = (returns - rf_rate) / downside_deviation

        return {
            'return': returns, 
            'volatility': vol, 
            'sharpe': sharpe,
            'sortino':sortino_ratio
                }
    
    def minimize_sortino(weights):
        return -portfolio_performance(weights, mean_returns, covar_matrix)['sortino'] 
    
    def minimize_sharpe(weights):  
        return -portfolio_performance(weights, mean_returns, covar_matrix)['sharpe'] 
    
    def minimize_volatility(weights):
        return portfolio_performance(weights, mean_returns, covar_matrix)['volatility'] 
    
    def minimize_return(weights): 
        return -portfolio_performance(weights, mean_returns, covar_matrix)['return']

    def optimize_portfolio(minimize_func, initializer, bounds, constraints, tickers, mean_returns, covar_matrix):
        """
        SLSQP ALGORITHM
        """
        optimal = optimize.minimize(minimize_func,
                                    initializer,
                                    method='SLSQP',
                                    bounds=bounds,
                                    constraints=constraints)

        # Extract and round the optimized weights
        optimal_weights = optimal['x'].round(4)

        # Calculate portfolio performance metrics
        portfolio_metrics = portfolio_performance(optimal_weights, mean_returns, covar_matrix)
        optimal_return = portfolio_metrics['return']
        optimal_vol = portfolio_metrics['volatility']

        # Combine the results into a structured dictionary
        result = {
            'weights': list(zip(tickers, list(optimal_weights))),
            'return': optimal_return,
            'volatility': optimal_vol
        }

        return result
    
    if check_dataframe(data, tickers, investments):

        data = data.divide(data.iloc[0] / 100) # Normalize prices 
        stock_returns = calculate_daily_returns(data[tickers])
        mean_returns = np.mean(stock_returns,axis=0)
        covar_matrix = np.cov(stock_returns, rowvar=False)
        num_assets = len(tickers)  

        #  Constrains and initial guess
        constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
        bounds = tuple((0,1) for x in range(num_assets))
        initializer = num_assets * [1./num_assets,]
        
        if method == 'sharpe':
            return optimize_portfolio(minimize_sharpe, initializer, bounds, constraints, tickers, mean_returns, covar_matrix)
        elif method == 'variance':
            return optimize_portfolio(minimize_volatility, initializer, bounds, constraints, tickers, mean_returns, covar_matrix)
        elif method == 'return':
            return optimize_portfolio(minimize_return, initializer, bounds, constraints, tickers, mean_returns, covar_matrix)
        elif method == 'sortino':
            return optimize_portfolio(minimize_sortino, initializer, bounds, constraints, tickers, mean_returns, covar_matrix)


