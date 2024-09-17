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


def calc_beta(portfolio: dict) -> tuple:
    """
    Calculate the beta and alpha of a portfolio using monetary investments.

    Parameters:
    portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    tuple: A tuple containing the beta and alpha of the portfolio.
    """


    # Extract return period
    days_return = int(portfolio['return_period_days'])

    # Extract returns and risk-free rates from the portfolio
    portfolio_returns = portfolio['portfolio_returns']
    market_returns = portfolio['market_returns']
    risk_free_returns = portfolio['risk_free_returns']  # Ensure the key matches your data

    # Align the data to ensure they have matching indices
    portfolio_returns, market_returns = portfolio_returns.align(market_returns, join='inner')
    portfolio_returns, risk_free_returns = portfolio_returns.align(risk_free_returns, join='inner')
    market_returns, risk_free_returns = market_returns.align(risk_free_returns, join='inner')

    # Calculate excess returns
    excess_portfolio_returns = portfolio_returns - risk_free_returns
    excess_market_returns = market_returns - risk_free_returns

    # Combine excess returns into a DataFrame and drop NaN values
    data = pd.DataFrame({
        'Excess_Portfolio_Returns': excess_portfolio_returns,
        'Excess_Market_Returns': excess_market_returns
    }).dropna()

    # Prepare the independent and dependent variables for regression
    X = sm.add_constant(data['Excess_Market_Returns'])
    y = data['Excess_Portfolio_Returns']

    # Perform linear regression to find alpha and beta
    model = sm.OLS(y, X).fit()

    # Extract alpha and beta
    alpha = model.params['const']
    beta = model.params['Excess_Market_Returns']

    # Annualize alpha
    annualization_factor = 252 / days_return  # 252 trading days in a year
    annualized_alpha = (1 + alpha) ** annualization_factor - 1

    return beta, annualized_alpha

