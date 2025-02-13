import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
import statsmodels.api as sm
import yfinance as yf
from scipy.optimize import minimize

from analyzerportfolio.utils import (
    get_stock_info, 
    get_current_rate, 
    get_currency, 
    get_exchange_rate, 
    convert_to_base_currency
)


def c_total_return(portfolio: dict) -> float:
    """
    Calculate the return of a portfolio using monetary investments.

    Parameters:
    portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    float: The return of the portfolio as a percentage.
    """

    # Extract the portfolio value at the beginning and end of the period
    portfolio_value = portfolio['portfolio_value']
    initial_value = portfolio_value.iloc[0]
    final_value = portfolio_value.iloc[-1]

    # Calculate the return as the percentage change in portfolio value
    return_percentage = (final_value - initial_value) / initial_value

    return return_percentage

def c_return(portfolio: dict) -> float:
    """
    Calculate the annual return of a portfolio using monetary investments.
    """
    # Extract the portfolio value at the beginning and end of the period
    portfolio_value = portfolio['portfolio_value']
    initial_value = portfolio_value.iloc[0]
    final_value = portfolio_value.iloc[-1]

    # Calculate the return as the percentage change in portfolio value
    return_percentage = (final_value - initial_value) / initial_value

    # Extract the return period from the portfolio
    return_period_days = len(portfolio_value)

    # Annualize the return
    annualization_factor = 252 / return_period_days  # 252 trading days in a year
    annualized_return = (1 + return_percentage) ** annualization_factor - 1

    return annualized_return

def c_volatility(portfolio: dict) -> float:
    """
    Calculate the annual volatility of a portfolio using monetary investments.

    Parameters:
    portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    float: The volatility of the portfolio as a percentage. (1Y)
    """

    # Extract the portfolio returns from the portfolio dictionary
    portfolio_returns = portfolio['portfolio_returns']

    # Calculate the standard deviation of the portfolio returns
    portfolio_std_dev = portfolio_returns.std()

    # Annualize the standard deviation
    return_period_days = portfolio['return_period_days']
    annualization_factor = np.sqrt(252 / return_period_days)  # 252 trading days in a year
    annualized_std_dev = portfolio_std_dev * annualization_factor

    return annualized_std_dev

def c_beta(portfolio: dict) -> tuple:
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

def c_info_ratio(portfolio: dict) -> float:
    """
    Calculate the information ratio of a portfolio using monetary investments 

    Parameters:
    portofolio (dict): Dictionary created from the create_portfolio function

    Returns:
    float: The Information ratio of the portfolio
    """

    # Extract portfolio returns from the portfolio dictionary
    portfolio_returns = portfolio['portfolio_returns']

    # Extract market returns form the portfolio dictionary
    market_returns = portfolio['market_returns']

    # Ensure that the returns are aligned on the same dates
    portfolio_returns, market_returns = portfolio_returns.align(market_returns, join='inner')

    # Calculate excess returns
    excess_returns = portfolio_returns - market_returns

    # Calculate excess return std dev
    excess_return_std_dev = excess_returns.std()

    # Calculate the mean of the excess returns
    mean_excess_return = excess_returns.mean()

    # Get the return period from the portfolio
    return_period_days = portfolio['return_period_days']
    
    # Annualization factor based on the return period
    annualization_factor = 252 / return_period_days  # 252 trading days in a year

    # Annualize the mean excess return
    annualized_mean_excess_return = mean_excess_return * annualization_factor

    # Annualize the excess return std dev 
    annulized_excess_return_std_dev = excess_return_std_dev * np.sqrt(annualization_factor)

    # Calculate the information raito
    information_ratio = annualized_mean_excess_return / annulized_excess_return_std_dev

    return information_ratio

def c_sharpe(portfolio: dict) -> float:
    """
    Calculate the Sharpe ratio of a portfolio using monetary investments.

    Parameters:
    portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    float: The Sharpe ratio of the portfolio.
    """

    # Extract portfolio returns from the portfolio dictionary
    portfolio_returns = portfolio['portfolio_returns']
    
    # Extract risk-free returns from the portfolio dictionary
    risk_free_returns = portfolio['risk_free_returns']

    # Ensure that the returns are aligned on the same dates
    portfolio_returns, risk_free_returns = portfolio_returns.align(risk_free_returns, join='inner')

    # Calculate excess returns
    excess_returns = portfolio_returns - risk_free_returns

    # Calculate the mean of the excess returns
    mean_excess_return = excess_returns.mean()

    # Calculate the standard deviation of the portfolio returns
    portfolio_std_dev = portfolio_returns.std()

    # Get the return period from the portfolio
    return_period_days = portfolio['return_period_days']
    
    # Annualization factor based on the return period
    annualization_factor = 252 / return_period_days  # 252 trading days in a year

    # Annualize the mean excess return
    annualized_mean_excess_return = mean_excess_return * annualization_factor

    # Annualize the standard deviation
    annualized_std_dev = portfolio_std_dev * np.sqrt(annualization_factor)

    # Calculate the Sharpe ratio
    sharpe_ratio = annualized_mean_excess_return / annualized_std_dev

    return sharpe_ratio

def c_sortino(portfolio: dict, target_return: float = 0.0) -> float:
    """
    Calculate the Sortino ratio of a portfolio using monetary investments.

    Parameters:
    - portfolio (dict): Dictionary created from the create_portfolio function.
    - target_return (float, optional): The target return (minimum acceptable return). 
      Defaults to 0.0, which considers only negative returns as downside risk.

    Returns:
    - float: The Sortino ratio of the portfolio.
    """

    # Extract portfolio returns from the portfolio dictionary
    portfolio_returns = portfolio['portfolio_returns']
    
    # Ensure that the returns are aligned on the same dates
    portfolio_returns = portfolio_returns.dropna()

    # Create a Series with the target return, aligned with portfolio_returns
    if isinstance(target_return, (float, int)):
        target_return_series = pd.Series(target_return, index=portfolio_returns.index)
    else:
        # Assume target_return is a Series, align indices
        portfolio_returns, target_return_series = portfolio_returns.align(target_return, join='inner')

    # Calculate the differences between portfolio returns and target return
    downside_diff = portfolio_returns - target_return_series

    # Keep only negative differences (returns below target), set others to zero
    downside_diff[downside_diff > 0] = 0

    # Square the negative differences
    squared_downside_diff = downside_diff ** 2

    # Calculate the mean of the squared negative differences
    mean_squared_downside_diff = squared_downside_diff.mean()

    # Calculate the downside deviation
    downside_deviation = np.sqrt(mean_squared_downside_diff)

    # Calculate the mean portfolio return
    mean_portfolio_return = portfolio_returns.mean()

    # Get the return period from the portfolio
    return_period_days = portfolio['return_period_days']
    
    # Annualization factor based on the return period
    annualization_factor = 252 / return_period_days  # Adjust if necessary

    # Annualize the mean portfolio return
    annualized_mean_return = mean_portfolio_return * annualization_factor

    # Annualize the target return
    annualized_target_return = target_return * annualization_factor

    # Annualize the downside deviation
    annualized_downside_deviation = downside_deviation * np.sqrt(annualization_factor)

    # Calculate the Sortino ratio
    sortino_ratio = (annualized_mean_return - annualized_target_return) / annualized_downside_deviation

    return sortino_ratio

def c_analyst_scenarios(portfolio) -> dict:
    """
    Calculate the portfolio value in different scenarios based on analyst target prices.

    Parameters:
    - portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    dict: Portfolio values in low, mean, median, and high scenarios.
    """
    portfolio_value_low = 0
    portfolio_value_mean = 0
    portfolio_value_median = 0
    portfolio_value_high = 0

    tickers = portfolio['tickers']
    investments = portfolio['investments']
    base_currency = portfolio['base_currency']

    #Ensure there is a position in all tickers
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    for ticker, investment in zip(tickers, investments):
        stock_info = get_stock_info(ticker)
        current_price = stock_info['currentPrice']
        target_low = stock_info['targetLowPrice']
        target_mean = stock_info['targetMeanPrice']
        target_median = stock_info['targetMedianPrice']
        target_high = stock_info['targetHighPrice']
        stock_currency = stock_info['currency']

        # Convert prices to the base currency if necessary
        exchange_rate = get_current_rate(base_currency, stock_currency)
        target_low *= exchange_rate
        target_mean *= exchange_rate
        target_median *= exchange_rate
        target_high *= exchange_rate

        # Calculate the number of shares bought with the investment
        shares = investment / (current_price * exchange_rate)

        # Calculate portfolio value in each scenario
        portfolio_value_low += shares * target_low
        portfolio_value_mean += shares * target_mean
        portfolio_value_median += shares * target_median
        portfolio_value_high += shares * target_high

    return {
        'Low Scenario': portfolio_value_low,
        'Mean Scenario': portfolio_value_mean,
        'Median Scenario': portfolio_value_median,
        'High Scenario': portfolio_value_high
    }

def c_analyst_score(portfolio) -> dict:
    """
    Calculate the weighted average analyst suggestion for a portfolio based on Yahoo Finance data. 1 is a strong buy and 5 is a strong sell.

    Parameters:
    - portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    dict: A dictionary containing individual ticker suggestions and the weighted average suggestion for the portfolio.
    """
    suggestions = []
    weighted_suggestions = []
    adjusted_investments = []

    tickers = portfolio['tickers']
    investments = portfolio['investments']

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    for ticker, investment in zip(tickers, investments):
        try:
            stock_info = yf.Ticker(ticker).info
            analyst_suggestion = stock_info.get('recommendationMean', None)
            
            if analyst_suggestion:
                suggestions.append({
                    "ticker": ticker,
                    "suggestion": analyst_suggestion
                })
                weighted_suggestions.append(analyst_suggestion * investment)
                adjusted_investments.append(investment)
            else:
                logging.warning(f"No analyst suggestion available for {ticker}. Skipping...")
        
        except Exception as e:
            logging.error(f"Error retrieving data for {ticker}: {e}")
            continue

    if not suggestions:
        logging.warning("No valid analyst suggestions were retrieved for the portfolio.")
        return {
            "individual_suggestions": [],
            "weighted_average_suggestion": None
        }

    # Calculate the weighted average suggestion
    total_adjusted_investment = sum(adjusted_investments)
    weighted_average_suggestion = sum(weighted_suggestions) / total_adjusted_investment if total_adjusted_investment > 0 else None

    return {
        "individual_suggestions": suggestions,
        "weighted_average_suggestion": weighted_average_suggestion
    }

def c_dividend_yield(portfolio : dict) -> float:
    """
    Calculate the overall dividend yield of the portfolio.

    Parameters:
    portfolio: dict created from the create_portfolio function.

    Returns:
    float: The overall dividend yield of the portfolio as a percentage.
    """

    tickers = portfolio['tickers']
    investments = portfolio['investments']

    #Ensure there is a position in all tickers
    if len(tickers) != len(investments):
        raise ValueError("The number of tickers must match the number of investments.")
    
    total_investment = sum(investments)
    weighted_dividend_yield = 0

    for ticker, investment in zip(tickers, investments):
        stock_info = get_stock_info(ticker)
        dividend_yield = stock_info.get('dividendYield', 0)

        if dividend_yield is None:
            continue  # Skip this stock if dividend yield is not available

        # Calculate the weight of this stock in the portfolio
        weight = investment / total_investment

        # Calculate the contribution to the overall dividend yield
        weighted_dividend_yield += weight * dividend_yield

    return weighted_dividend_yield

def c_VaR(portfolio: dict, confidence_level: float = 0.95, horizon_days: int = 1, method: str = "historical", portfolio_value: int = None, n_bootstrap_samples: int = 10000) -> float:
    """
    Calculate the Value at Risk (VaR) of a portfolio using a chosen method: historical, parametric, or bootstrap.

    Parameters:
    - portfolio (dict): Dictionary created from the create_portfolio function.
    - confidence_level (float, optional): The confidence level for the VaR calculation.
      Defaults to 0.95 (95% confidence).
    - horizon_days (int, optional): The number of days ahead for the VaR calculation.
      Defaults to 1 day.
    - method (str, optional): The method used to calculate VaR. Options are 'historical', 'parametric', or 'bootstrap' or 'EVT'.
      Defaults to 'historical'.
    - portfolio_value (int, optional): The value of the portfolio. If not provided, the VaR will be calculated based on the last portfolio value.
    - n_bootstrap_samples (int, optional): Number of bootstrap samples to generate for the bootstrap method.
      Defaults to 10,000.

    Returns:
    - float: The Value at Risk (VaR) of the portfolio at the specified confidence level
      and time horizon.
    """
    return_days = portfolio['return_period_days']

    if portfolio_value is None:
        # Extract the last portfolio value from the portfolio dictionary
        portfolio_value = portfolio['portfolio_value'].iloc[-1]

    # Extract portfolio returns from the portfolio dictionary
    portfolio_returns = portfolio['portfolio_returns']
    if method == "historical":
        # Calculate the daily Value at Risk (VaR) using the historical method
        VaR = portfolio_returns.quantile(1 - confidence_level)

    elif method == "parametric":
        # Calculate the daily Value at Risk (VaR) by assuming normal distribution of returns
        mean_return = portfolio_returns.mean()
        std_dev = portfolio_returns.std()
        # Calculate the z-score for the specified confidence level
        z_score = norm.ppf(confidence_level)
        # Calculate the Value at Risk (VaR) using the parametric method
        VaR = mean_return - z_score * std_dev

    elif method == "bootstrap": # Reference: Bootstrap for Value at Risk Prediction - Rjiba, Meriem et all. (2015)
        
        # Drop NaN values from portfolio returns
        portfolio_returns = portfolio['portfolio_returns'].dropna()

        # Check if there is sufficient data after dropping NaN values. The limit has been choosen for daily returns following the reference
        # Reference: Basel Committee on Banking Supervision. “Amendment to the Capital Accord to Incorporate Market Risks.” Bank for International Settlements, 1996.
        if len(portfolio_returns) < 100:
            raise ValueError("Insufficient data in portfolio returns after removing NaN values.")
        
        # Generating bootstrap samples from the returns 
        bootstrap_samples = np.random.choice(portfolio_returns, (n_bootstrap_samples, len(portfolio_returns)), replace=True) 
        # Calculate the desidered percentile for each sample
        bootstrap_quantile = np.percentile(bootstrap_samples, 100*(1-confidence_level), axis=1)

        # Check for NaN values in the quantile array before calculating the mean
        if np.isnan(bootstrap_quantile).any():
            raise ValueError("NaN values found in bootstrap quantiles.")
        
        # Calculate the aveage of the percentile
        VaR = np.mean(bootstrap_quantile)

    elif method == "EVT":
        """
        The EVT estimates are calculated using the Generalized Pareto Distribution (GPD)
        and maximum likelihood estimation (MLE).
        """

        def gpd_log_likelihood(params, exceedances, threshold):
            """
            Generalized Pareto Distribution. Reference: Gnedenko (1943)
            """
            β, ξ = params
            if β <= 0 or ξ <= 0:
                 return np.inf  # Ensure valid parameters
            term = 1 + (ξ * (exceedances - threshold) / β)
            if np.any(term <= 0): # Prevent invalid log or power operations
                return np.inf
            return -np.sum(np.log((1 / β) * (term ** (-1 / ξ - 1))))

        # Drop NaN values from portfolio returns
        portfolio_returns = portfolio['portfolio_returns'].dropna()

        # Handle the case the portfolio is empty 
        if portfolio_returns.empty:
            raise ValueError("Portfolio returns data is empty.")

        # sorting returns in ascending order
        sorted_losses = np.sort(portfolio_returns)[::-1] * -1 

        # Set the threshold as the quantile of the portfolio returns
        threshold = np.quantile(sorted_losses, confidence_level)
        n = len(sorted_losses)

        # Filter exceedances above the threshold
        exceedances = sorted_losses[sorted_losses > threshold]
        nu = len(exceedances)
        print("nu is ", nu)

        # Ensure sufficient right/left hand losses
        if nu < 10:
            raise ValueError("Insufficient exceedances for EVT modeling. Consider lowering the confidence_level.")

        # initializing guess for β and ξ
        initial_params = [40, 0.3]

        # defining bounds for β and ξ to ensure they are > 0
        bounds = [(0.001, None), (0.001, None)] 

        # optimizing using scipy's minimize function
        result = minimize(gpd_log_likelihood, initial_params,  method="L-BFGS-B", args=(exceedances, threshold), bounds=bounds)
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        ξ_hat, β_hat = result.x

        # Calculate Value at Risk (VaR)
        VaR = threshold + np.divide(β_hat, ξ_hat) * ((np.divide(n, nu) * np.power((1 - confidence_level), -ξ_hat) - 1))

    else:
        raise ValueError("Invalid method. Choose 'historical' or 'parametric' or 'bootstrap'.")

    # Annualize the VaR for the specified time horizon
    VaR_annualized = VaR * np.sqrt(horizon_days/return_days) * portfolio_value

    return abs(VaR_annualized)

def c_ES(portfolio: dict, confidence_level: float = 0.95, horizon_days: int = 1, method: str = "historical", portfolio_value: int = None, n_bootstrap_samples: int = 10000) -> float:
    """
    Calculate the Expected Shortfall (ES), also known as Conditional VaR (CVaR), of a portfolio using a chosen method: historical, parametric, or bootstrap.

    Parameters:
    - portfolio (dict): Dictionary created from the create_portfolio function.
    - confidence_level (float, optional): The confidence level for the ES calculation.
      Defaults to 0.95 (95% confidence).
    - horizon_days (int, optional): The number of days ahead for the ES calculation. 
      Defaults to 1 day.
    - method (str, optional): The method used to calculate ES. Options: 'historical', 'parametric', or 'bootstrap' or 'EVT' . 
      Defaults to 'historical'.
    - portfolio_value (int, optional): The value of the portfolio. If not provided, the ES will be calculated based on the last portfolio value.
    - n_bootstrap_samples (int, optional): Number of bootstrap samples to generate for the bootstrap method. 
      Defaults to 10,000.

    Returns:
    - float: The Expected Shortfall (ES) of the portfolio at the specified confidence level and time horizon.
    """
    return_days = portfolio['return_period_days']

    if portfolio_value is None:
        # Extract the last portfolio value from the portfolio dictionary
        portfolio_value = portfolio['portfolio_value'].iloc[-1]

    # Extract portfolio returns from the portfolio dictionary
    portfolio_returns = portfolio['portfolio_returns']

    if method == "historical":
        # Calculate VaR using the historical method
        VaR = portfolio_returns.quantile(1 - confidence_level)
        
        # Calculate Expected Shortfall (ES) as the average of losses beyond VaR
        ES = portfolio_returns[portfolio_returns <= VaR].mean()
    
    elif method == "parametric":
        # Calculate the mean and standard deviation of the portfolio returns
        mean_return = portfolio_returns.mean()
        std_dev = portfolio_returns.std()

        # Calculate the z-score for the specified confidence level
        z_score = norm.ppf(confidence_level)

        # Calculate the VaR using the parametric method
        VaR = - (mean_return - z_score * std_dev)

        # Calculate Expected Shortfall (ES) using the parametric method
        # ES = mean - std_dev * (PDF of z-score) / (1 - confidence level)
        pdf_z = norm.pdf(z_score)
        ES = - (mean_return - (std_dev * (pdf_z / (1 - confidence_level))))
    
    elif method == "bootstrap":
        # Drop NaN values from portfolio returns
        portfolio_returns = portfolio['portfolio_returns'].dropna()

        # Check if there is sufficient data after dropping NaN values
        if len(portfolio_returns) < 100:
            raise ValueError("Insufficient data in portfolio returns after removing NaN values.")
        
        # Generating bootstrap samples from the returns
        bootstrap_samples = np.random.choice(portfolio_returns, (n_bootstrap_samples, len(portfolio_returns)), replace=True)

        # Calculate ES for each bootstrap sample
        bootstrap_ES = []
        for sample in bootstrap_samples:
            # Calculate VaR for the bootstrap sample
            sample_VaR = np.percentile(sample, 100 * (1 - confidence_level))
            
            # Calculate ES as the average of returns below the sample VaR
            sample_ES = np.mean(sample[sample <= sample_VaR])
            bootstrap_ES.append(sample_ES)
        
        # Calculate the average of the ES values from all bootstrap samples
        ES = np.mean(bootstrap_ES)
    
    elif method == "EVT":
        """
        The EVT estimates are calculated using the Generalized Pareto Distribution (GPD)
        and maximum likelihood estimation (MLE).
        """

        def gpd_log_likelihood(params, exceedances, threshold):
            """
            Generalized Pareto Distribution. Reference: Gnedenko (1943)
            """
            β, ξ = params
            if β <= 0 or ξ <= 0:
                 return np.inf  # Ensure valid parameters
            term = 1 + (ξ * (exceedances - threshold) / β)
            if np.any(term <= 0): # Prevent invalid log or power operations
                return np.inf
            return -np.sum(np.log((1 / β) * (term ** (-1 / ξ - 1))))

        # Drop NaN values from portfolio returns
        portfolio_returns = portfolio['portfolio_returns'].dropna()

        # Handle the case the portfolio is empty 
        if portfolio_returns.empty:
            raise ValueError("Portfolio returns data is empty.")

        # sorting returns in ascending order
        sorted_losses = np.sort(portfolio_returns)[::-1] * -1 

        # Set the threshold as the quantile of the portfolio returns
        threshold = np.quantile(sorted_losses, confidence_level)
        n = len(sorted_losses)

        # Filter exceedances above the threshold
        exceedances = sorted_losses[sorted_losses > threshold]
        nu = len(exceedances)
        #print("nu is ", nu)

        # Ensure sufficient right/left hand losses
        if nu < 10:
            raise ValueError("Insufficient exceedances for EVT modeling. Consider lowering the confidence_level.")

        # initializing guess for β and ξ
        initial_params = [40, 0.3]

        # defining bounds for β and ξ to ensure they are > 0
        bounds = [(0.001, None), (0.001, None)] 

        # optimizing using scipy's minimize function
        result = minimize(gpd_log_likelihood, initial_params,  method="L-BFGS-B", args=(exceedances, threshold), bounds=bounds)
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        ξ_hat, β_hat = result.x

        # Calculate Value at Risk (VaR)
        VaR = threshold + np.divide(β_hat, ξ_hat) * ((np.divide(n, nu) * np.power((1 - confidence_level), -ξ_hat) - 1))

        # Calculate the ES
        ES = np.divide((VaR + β_hat - ξ_hat * threshold), 1 - ξ_hat)

    else:
        raise ValueError("Invalid method. Choose 'historical', 'parametric', or 'bootstrap'.")

    # Annualize the Expected Shortfall for the specified time horizon
    ES_annualized = ES * np.sqrt(horizon_days / return_days) * portfolio_value

    return abs(ES_annualized)

def c_max_drawdown(portfolio: dict) -> float:
    """
    Calculate the maximum drawdown of a portfolio.

    Parameters:
    - portfolio (dict): Dictionary created from the create_portfolio function.

    Returns:
    - float: The maximum drawdown of the portfolio.
    """
    portfolio_value = portfolio['portfolio_value']
    peak = portfolio_value.cummax()
    drawdown = ((portfolio_value - peak) / peak)

    # Calculate the maximum drawdown as the minimum value in the drawdown series
    max_drawdown = -drawdown.min()

    return max_drawdown
