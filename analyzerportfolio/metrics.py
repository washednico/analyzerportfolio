import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import logging
import statsmodels.api as sm

from analyzerportfolio.utils import (
    get_stock_info, 
    get_current_rate, 
    get_currency, 
    get_exchange_rate, 
    convert_to_base_currency
)


def calculate_beta_and_alpha(portfolio: dict, risk_free_rate: float = 0.01) -> tuple:
    """
    Calculate the beta and alpha of a portfolio using monetary investments.

    Parameters:
    portfolio (dict): Dictionary created from the create_portfolio function.
    market_ticker (str): The market index to compare against (e.g., S&P 500).
    risk_free_rate (float): The risk-free rate to use in the alpha calculation (default is 1%).

    Returns:
    tuple: A tuple containing the beta and alpha of the portfolio.
    """

    portfolio_returns = portfolio['portfolio_returns']
    market_returns = portfolio['market_returns']

    # Convert the annual risk-free rate to a daily rate
    daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1

    # Excess returns
    excess_portfolio_returns = portfolio_returns.tail(-1) - daily_risk_free_rate
    excess_market_returns = market_returns.tail(-1) - daily_risk_free_rate

    # Add a constant for the intercept in the regression model
    X = sm.add_constant(excess_market_returns)

    # Perform linear regression to find alpha and beta
    model = sm.OLS(excess_portfolio_returns, X).fit()
    alpha = model.params['const']
    beta = model.params[market_returns.name]
    annualized_alpha = (1 + alpha) ** 252 - 1

    return beta, annualized_alpha

