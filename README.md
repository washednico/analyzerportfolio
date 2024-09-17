# AnalyzerPortfolio

analyzerportfolio is a Python package designed for comprehensive portfolio analysis. Built on top of the `yfinance` library, it provides a wide range of metrics for analyzing portfolio performance, including beta, alpha, Sharpe ratio, Sortino ratio, and Value at Risk (VaR). The package also offers advanced tools for generating graphical outputs, performing Monte Carlo simulations, and optimizing portfolios. Additionally, AnalyzerPortfolio includes AI-powered features for monitoring stock news and generating investment reports.

## We are currenyly working on a ncompletely new revised version, check NEW branch

## Installation

You can install analyzerportfolio using pip:

```bash
pip install analyzerportfolio
```


## Usage

Below is an example of how to use AnalyzerPortfolio to calculate various portfolio metrics, generate graphical outputs, and analyze stock news.

### Example: Variables

```python
ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
investments = [100,200,300,300,200,500]
start_date = '2019-01-01'
end_date = '2024-08-28'
market_ticker = '^GSPC'
base_currency = 'EUR'
risk_free = "PCREDIT8"
rebalancing_period_days = 250
```
### Example: Metrics Calculation
```python
import analyzerportfolio as ap

# Download historical data
data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free)
portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", rebalancing_period_days=rebalancing_period_days)

beta, alpha = ap.calc_beta(portfolio_1)
sharpe = ap.calc_sharpe(portfolio_1)

print("Beta: ", beta)
print("Alpha: ", alpha)
print("Sharpe Ratio: ", sharpe_ratio)
```
### Output Metrics Calculation

```text
Beta:  1.117957602334291
Alpha:  0.12284375076780196
Sharpe Ratio:  1.0003270932926993
```

### Example: Graphics Module
```python
ap.compare_portfolio_mkt(portfolio_1)

#Compare with other portfolios created
compare_multiple_portfolios([portfolio_1,portfolio_2])
```

### Output Portfolio vs Market  
![alt text](https://github.com/washednico/analyzerportfolio/blob/NEW/img/pfvsmkt.png?raw=true)
### Output Portfolio Comparison  
![alt text](https://github.com/washednico/analyzerportfolio/blob/NEW/img/pfvspf.png?raw=true)


## Current requirements
```text
    install_requires=[
        'openai>=1.43.0',
        'pandas>=1.5.1',
        'yfinance>=0.2.32',
        'numpy==1.26.4',
        'plotly>=5.18.0',
        'arch>=7.0.0',
        'scipy==1.14.0',
        "statsmodels==0.14.1",
        "nbformat>=4.2.0"
    ]
```
We are currently forcing version of `numpy`, `statsmodel` and `scipy` due to compatibility issues.
See more here https://github.com/statsmodels/statsmodels/issues/9333#issuecomment-2305438605

`nbformat` is required not directly for `analyzerportfolio` but indirectly for `plotly`, solving plotting issues.

## Contributions

Contributions are welcome! Please submit pull requests or report issues via the GitHub repository.
