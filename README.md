# AnalyzerPortfolio main branch

analyzerportfolio is a Python package designed for comprehensive portfolio analysis. Built on top of the `yfinance` library, it provides a wide range of metrics for analyzing portfolio performance, including beta, alpha, Sharpe ratio, Sortino ratio, and Value at Risk (VaR). The package also offers advanced tools for generating graphical outputs, performing Monte Carlo simulations, and optimizing portfolios. Additionally, AnalyzerPortfolio includes AI-powered features for monitoring stock news and generating investment reports.

## Rewriting status
~sortino~  
~beta~  
~sharpe~  
~montecarlo~  
~drawdown~  
~compare_returns~  
~portfolio_scanerios~   
~dividend_yield~   
~analyst_suggestion~  
~heatmap~  
~distribution_returns~  
~var~  
~simulate_dca~  
~monitor_news~  
~probability_cone~        
optimization 

## Working in progress
~Heatmap between portfolios~
Kelly criterion  
Fractals model  
~Expected Shortfall~  
Different types of VaR    

## To Push once everything is finished
newsletter_report   
get_suggestion  
Other AI functions


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
data = ap.download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free)
portfolio_1 = ap.create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", rebalancing_period_days=rebalancing_period_days)

beta, alpha = ap.calc_beta(portfolio_1)
sharpe = ap.calc_sharpe(portfolio_1)
sortino = ap.c_sortino(portfolio_1)
scenarios = ap.c_analyst_scenarios(portfolio_1)
score = ap.c_analyst_score(portfolio_1)
dividend_yield = ap.c_dividend_yield(portfolio_1)
var_95 = ap.c_VaR(portfolio_1, confidence_level=0.95)
var_99_30d = ap.c_VaR(portfolio_1, confidence_level=0.95, horizon_days=30)
var_95_p = ap.c_VaR(portfolio_1, confidence_level=0.95, method="parametric")
var_99_p_30d = ap.c_VaR(portfolio_1, confidence_level=0.95, horizon_days=30, method="parametric")

```
### Output Metrics Calculation

```text
Beta:  1.117957602334291
Alpha:  0.12284375076780196
Sharpe Ratio:  1.0003270932926993
Sortino: 1.524920163925947
Scenarios: {'Low Scenario': 1431.780134608739, 'Mean Scenario': 1800.2148927427404, 'Median Scenario': 1810.3485673486473, 'High Scenario': 2138.98597652531}
Score: {'individual_suggestions': [{'ticker': 'AAPL', 'suggestion': 2.0}, {'ticker': 'MSFT', 'suggestion': 1.7}, {'ticker': 'GOOGL', 'suggestion': 1.9}, {'ticker': 'AMZN', 'suggestion': 1.8}, {'ticker': 'TSLA', 'suggestion': 2.7}, {'ticker': 'E', 'suggestion': 2.5}], 'weighted_average_suggestion': 2.15}
Dividend Yield: 0.0229375
VaR_95: 189.85826340825247
VaR_99_30d: 1039.8965359745753
VaR_95_p: 197.53601012635366
VaR_99_p_30d: 1081.9492866577282

```

### Example: Graphics Module
```python
colors_1 = "orange" #OPTIONAL
colors_4 = ["orange","blue","purple","red"]  #OPTIONAL

start_date = '2013-02-01'
end_date = '2024-08-28'
market_ticker = '^GSPC'
base_currency = 'EUR'
risk_free = "DTB3"


ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
investments = [100,200,300,300,200,500]
data = ap.download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free)
portfolio_1 = ap.create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio 1", base_currency=base_currency, rebalancing_period_days=250)

ticker = ['AAPL','MSFT','GOOGL']
investments = [500,300,800]
portfolio_2 = ap.create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="3 STOCK",base_currency=base_currency, rebalancing_period_days=250)

ticker = ["VWCE.DE","IGLN.L","IUSN.DE"]
investments = [500,300,800]
data = ap.download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free)
portfolio_3 = ap.create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="3 ETF", base_currency=base_currency, rebalancing_period_days=250)

ticker = ["VWCE.DE","IGLN.L"]
investments = [1300,300]
portfolio_4 = ap.create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="2 ETF",base_currency=base_currency, rebalancing_period_days=250)

ap.portfolio_value(portfolio_1, colors=colors_1)
ap.portfolio_value([portfolio_1,portfolio_2,portfolio_3,portfolio_4], colors=colors_4)

ap.garch(portfolio_1, colors=colors_1)
ap.garch([portfolio_1,portfolio_2,portfolio_3,portfolio_4], colors=colors_4)

ap.montecarlo(portfolio_1, simulation_length=1000)
ap.montecarlo([portfolio_1,portfolio_2,portfolio_3,portfolio_4], simulation_length=1000)

ap.drawdown(portfolio_1, colors=colors_1)
ap.drawdown([portfolio_1,portfolio_2,portfolio_3,portfolio_4], colors=colors_4)

ap.heatmap(portfolio_1)

ap.distribution_return(portfolio_1, colors=colors_1)
ap.distribution_return([portfolio_1,portfolio_2,portfolio_3,portfolio_4], colors=colors_4)

ap.simulate_dca(portfolio_1, initial_investment=10000, periodic_investment=500, investment_interval=30, colors=colors_1)
ap.simulate_dca([portfolio_1,portfolio_2,portfolio_3,portfolio_4], initial_investment=10000, periodic_investment=500, investment_interval=30, colors=colors_4)

ap.probability_cone(portfolio_1, time_horizon=1000)
```

### Output Portfolio Value  
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img1.png?raw=true)
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img2.png?raw=true)
### Output Garch  
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img3.png?raw=true)
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img4.png?raw=true)
### Output Montecarlo  
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img5.png?raw=true)
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img6.png?raw=true)
### Output Drawdown  
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img7.png?raw=true)
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img8.png?raw=true)
### Output Heatmap
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img9.png?raw=true)
### Output Distribution Returns
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img10.png?raw=true)
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img11.png?raw=true)
### Output Simulate DCA
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img12.png?raw=true)
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img13.png?raw=true)
### Output Probability Cone
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/img14.png?raw=true)

### Optimization Module
```python
information_ratio1 = c_info_ratio(portfolio_1)
portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
info_optimized = c_info_ratio(portfolio_optimized)

print("Information Ratio before optimization: ", information_ratio1)
print("Information Ratio optimization: ", info_optimized)
```


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
