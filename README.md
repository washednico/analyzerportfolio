# AnalyzerPortfolio

analyzerportfolio is a Python package designed for comprehensive portfolio analysis. Built on top of the `yfinance` library, it provides a wide range of metrics for analyzing portfolio performance, including beta, alpha, Sharpe ratio, Sortino ratio, and Value at Risk (VaR). The package also offers advanced tools for generating graphical outputs, performing Monte Carlo simulations, and optimizing portfolios. Additionally, AnalyzerPortfolio includes AI-powered features for monitoring stock news and generating investment reports.

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
risk_free_rate = 0.01
base_currency = 'EUR'
openai_key = ""  # Add your OpenAI key here
```
### Example: Metrics Calculation
```python
import analyzerportfolio as ap

# Download historical data
data = ap.download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker)

# Calculate portfolio metrics - ap.calculate_portfolio_metrics() will return a dict with the major metrics.
beta, alpha = ap.calculate_beta_and_alpha(data, ticker, investments, market_ticker)
sharpe_ratio = ap.calculate_sharpe_ratio(data, ticker, investments, risk_free_rate)
sortino_ratio = ap.calculate_sortino_ratio(data, ticker, investments, risk_free_rate)
var = ap.calculate_var(data, ticker, investments)
var_5_h = ap.calculate_var(data, ticker, investments, confidence_level=0.99, time_horizon=5, method='historical')
portfolio_scenarios = ap.calculate_portfolio_scenarios(ticker, investments, base_currency)
dividend_yield = ap.calculate_dividend_yield(ticker, investments)
max_drawdown = ap.calculate_max_drawdown(data, ticker, investments)
analyst_info = ap.calculate_analyst_suggestion(ticker, investments)

print("Beta: ", beta)
print("Alpha: ", alpha)
print("Sharpe Ratio: ", sharpe_ratio)
print("Sortino Ratio: ", sortino_ratio)
print("Value at Risk: ", var)
print("Value at Risk (5 days, 99% confidence level): ", var_5_h)
print("Dividend Yield: ", dividend_yield)
print("Max Drawdown: ", max_drawdown)
print("Portfolio Scenarios: ", portfolio_scenarios)
print("Analyst Info: ", analyst_info)
```
### Output Metrics Calculation

```text
Beta:  1.1181819930813468
Alpha:  0.08161211021082404
Sharpe Ratio:  0.9268951221173983
Sortino Ratio:  0.8919636462200137
Value at Risk:  44.8207337284658
Value at Risk (5 days, 99% confidence level):  152.097583515136
Dividend Yield:  0.022281250037500002
Max Drawdown:  -0.394113981716177
Portfolio Scenarios:  {'Low Scenario': 1451.0997033591789, 'Mean Scenario': 1843.0520713925248, 'Median Scenario': 1861.6080779118356, 'High Scenario': 2168.7750095303313}
Analyst Info:  {'individual_suggestions': [{'ticker': 'AAPL', 'suggestion': 2.0}, {'ticker': 'MSFT', 'suggestion': 1.7}, {'ticker': 'GOOGL', 'suggestion': 1.9}, {'ticker': 'AMZN', 'suggestion': 1.8}, {'ticker': 'TSLA', 'suggestion': 2.8}, {'ticker': 'E', 'suggestion': 2.5}], 'weighted_average_suggestion': 2.1625}
```

### Example: Graphics Module
```python
ap.compare_portfolio_to_market(data, ticker, investments, market_ticker)
ap.garch(data, ticker, investments)
ap.simulate_pac(data, ticker, 1000, 100, 30, [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
ap.montecarlo(data, ticker, investments, 250, 50, 50, market_ticker)
ap.heatmap(data, ticker, market_ticker)
ap.probability_cone(data, ticker, investments, 750)
ap.drawdown_plot(data, ticker, investments)
```
### Output Portfolio vs Market  
![alt text](https://github.com/washednico/portfolioanalyzer/blob/main/img/pfvsmkt.png?raw=true)
### Output Simulate Pac 
![alt text](https://github.com/washednico/portfolioanalyzer/blob/main/img/pac.png?raw=true)
### Output Heatmap  
![alt text](https://github.com/washednico/portfolioanalyzer/blob/main/img/heatmap.png?raw=true)
### Output Montecarlo 
![alt text](https://github.com/washednico/portfolioanalyzer/blob/main/img/montecarlo.png?raw=true)
### Output Probability Cone 
![alt text](https://github.com/washednico/portfolioanalyzer/blob/main/img/prob_cone.png?raw=true)
### Output Drawdown 
![alt text](https://github.com/washednico/portfolioanalyzer/blob/main/img/drawdown.png?raw=true)



### Example: Portfolio Optimization
```python
markowitz_portfolio = ap.markowitz_optimization(data, ticker, investments, method='sortino')
print("Optimal Weights:", markowitz_portfolio)
```

### Output Portfolio optimization
```text
Optimal Weights: {'weights': [('AAPL', 0.1667), ('MSFT', 0.1667), ('GOOGL', 0.1667), ('AMZN', 0.1667), ('TSLA', 0.1667), ('E', 0.1667)], 'return': 0.3177235229590848, 'volatility': 0.2860601198632393}
```

### Example: News Monitor - AI
```python
ap.monitor_news(ticker, delay=60, loop_forever=True, openai_key=openai_key)
```

### Output News Monitor - AI
```text
--------------portfolioanalyzer--------------
Retrieved news for AAPL at 01-09-2024 14:13:53
Title: Warren Buffett Just Sold 389,368,450 Shares of Apple Stock. Was That a Good Idea?
Link: https://finance.yahoo.com/m/0595db4e-58b4-3e46-9397-7e454c575a48/warren-buffett-just-sold.html
Importance Score: 9

Sentiment: Negative 

Explanation: Warren Buffett is a highly influential figure in investment circles. His actions are closely watched and often copied by other investors. Therefore, his decision to sell off a large number of shares can send a negative signal about the future outlook of Apple's stock to the market at large, which might have a major impact on the company's stock price. Hence, this news is quite important. The sentiment is negative because it suggests a lack of confidence in the company's stock from one of the world's most respected investors.
```

### Example: Newsletter Report - AI
```python
report = ap.newsletter_report(data, ticker, investments, start_date_report="2024-08-01", openai_key=openai_key)
print(report)
```

### Output Newsletter Report - AI
```text
Portfolio Newsletter Report
---------------------------
Portfolio Metrics (Calculated from 2019-01-02 to 2024-08-27)
- Beta: 1.12
- Alpha: 0.08
- Sharpe Ratio: 0.93
- Sortino Ratio: 0.89
- Value at Risk: 44.82
- Max Drawdown: -39.41%
- Dividend Yield: 2.23%

---------------------------
Returns (Calculated from 2024-08-01 to 2024-08-27)
Portfolio Initial Value: 1,664.58
Portfolio Final Value: 1,600.00
Portfolio Return: -3.88%
Market Return: 0.14%

---------------------------
Individual Stock Performance:
AAPL:
  - Initial Value: 98.66
  - Final Value: 100.00
  - Return: 1.36%
  - Surplus/Deficit: 1.34
MSFT:
  - Initial Value: 207.55
  - Final Value: 200.00
  - Return: -3.64%
  - Surplus/Deficit: -7.55
GOOGL:
  - Initial Value: 320.87
  - Final Value: 300.00
  - Return: -6.50%
  - Surplus/Deficit: -20.87
AMZN:
  - Initial Value: 329.02
  - Final Value: 300.00
  - Return: -8.82%
  - Surplus/Deficit: -29.02
TSLA:
  - Initial Value: 213.84
  - Final Value: 200.00
  - Return: -6.47%
  - Surplus/Deficit: -13.84
E:
  - Initial Value: 494.64
  - Final Value: 500.00
  - Return: 1.08%
  - Surplus/Deficit: 5.36


General Comments:
Overall, the portfolio has experienced a negative return of -3.88% from August 1st, 2024, to August 27th, 2024. This outcome stands in contrast to the market's positive return of 0.14% during the same period. 

The individual stock performances within the portfolio were varied. The best performing stocks were Apple (AAPL) and stock 'E,' both delivering a positive return. AAPL showed a slight increase in its value from $98.66 to $100.00, resulting in a return of 1.36%. The 'E' stock also saw a small gain over the period, moving from $494.64 to $500.00, providing a return of 1.08%. 

In contrast, Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), and Tesla (TSLA) all suffered declines. The hardest hit was AMZN, which fell from $329.02 to $300.00, a decrease of -8.82%. GOOGL and TSLA followed closely with declines of -6.50% and -6.47% respectively. MSFT saw a less drastic dip of -3.64%.

We accumulated a net deficit in surplus/deficit for the portfolio from these investments with AAPL and E yielding surplus of 1.34 and 5.36 respectively while MSFT, GOOGL, AMZN, and TSLA presenting deficits of -7.55, -20.87, -29.02, and -13.84 respectively. 

Going forward, investors may wish to re-evaluate the portfolio composition and consider possible changes in strategy to align with market trends and their risk tolerance levels.
```

### Example: Portfolio Suggestions - AI
```python
report = ap.newsletter_report(data, ticker, investments, start_date_report="2024-08-01", openai_key=openai_key)
print(report)
```

### Output Portfolio Suggestions - AI
```text
Portfolio Improvement Suggestions
---------------------------
Portfolio Metrics (Calculated from 2019-01-02 to 2024-08-27)
- Beta: 1.12
- Alpha: 0.08
- Sharpe Ratio: 0.93
- Sortino Ratio: 0.89
- Value at Risk: 44.82
- Max Drawdown: -39.41%
- Dividend Yield: 2.23%

---------------------------
Individual Stock Performance:
AAPL:
  - Initial Value: 100.00
  - Final Value: 620.12
  - Return: 520.12%
  - Surplus/Deficit: 520.12
MSFT:
  - Initial Value: 200.00
  - Final Value: 889.72
  - Return: 344.86%
  - Surplus/Deficit: 689.72
GOOGL:
  - Initial Value: 300.00
  - Final Value: 962.88
  - Return: 220.96%
  - Surplus/Deficit: 662.88
AMZN:
  - Initial Value: 300.00
  - Final Value: 692.83
  - Return: 130.94%
  - Surplus/Deficit: 392.83
TSLA:
  - Initial Value: 200.00
  - Final Value: 2,077.67
  - Return: 938.83%
  - Surplus/Deficit: 1,877.67
E:
  - Initial Value: 500.00
  - Final Value: 763.06
  - Return: 52.61%
  - Surplus/Deficit: 263.06


Suggestions for Improvement:
Based on the given metrics and return figures, here are some suggestions to improve the portfolio:

1. Consider Redistributing Investments: The existing portfolio seems to be heavily invested in more volatile tech stocks like Apple, Microsoft, and Tesla. While these shares have brought significant returns, high volatility also typically correlates with higher risk, as indicated by the portfolio's high Beta value. To decrease risk, consider redistributing some of your investment from these volatile stocks to more stable sectors or stocks with a lower Beta.

2. Beta: The portfolio's Beta is above one, meaning it's more volatile than the market average. By diversifying the portfolio more widely across different sectors and types of securities, you can lower the portfolio Beta, potentially reducing risk.

3. Negative Sharpe and Sortino Ratios: The negative Sharpe and Sortino ratios indicate the portfolio's returns are not justifying the level of risk taken. This suggests the need for re-evaluation and optimization of the portfolio to enhance return relative to the level of risk. 

4. Dividend Yield: The portfolio's dividend yield is 2.23% which seems low considering the risk associated with it. Invest in more dividend-paying stocks to potentially generate regular income and boost overall returns.

5. Max Drawdown and Value at Risk: The portfolio metrics show a max drawdown of -39.41% and a value at risk of 44.82. This indicates a significant downside risk in the portfolio. Consider implementing a more stringent risk management strategy.

6. Focus on Low-Performers: Based on individual stock performance, 'E' has provided the least return (52.61%). Consider whether it's beneficial to keep holding this stock, or if investment capital could be better utilized elsewhere.

Please note, these suggestions are general in nature. A detailed risk profile and understanding of investment goals are crucial for making portfolio modifications. It's recommended to consult with a financial advisor prior to making any changes.
```


## Contributions

Contributions are welcome! Please submit pull requests or report issues via the GitHub repository.
