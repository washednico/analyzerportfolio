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
ap.garch(data, ticker, investments,market_ticker)
ap.simulate_dca(data, ticker, 1000, 100, 30, [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
ap.montecarlo(data, ticker, investments, 250, 50, 50, market_ticker)
ap.heatmap(data, ticker, market_ticker)
ap.probability_cone(data, ticker, investments, 750)
ap.drawdown_plot(data, ticker, investments,market_ticker)
```
### Output Heatmap  
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/heatmap.png?raw=true)
### Output Portfolio vs Market  
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/pfvsmkt.png?raw=true)
### Output Simulate DCA 
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/pac.png?raw=true)
### Output Montecarlo 
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/montecarlo.png?raw=true)
### Output Probability Cone 
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/prob_cone.png?raw=true)
### Output Drawdown 
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/drawdown.png?raw=true)
### Output Garch 
![alt text](https://github.com/washednico/analyzerportfolio/blob/main/img/garch.png?raw=true)



### Example: Portfolio Optimization
```python
markowitz_portfolio = ap.markowitz_optimization(data, ticker, investments, method='volatility')
print("Optimal Weights:", markowitz_portfolio)
```

### Output Portfolio optimization
```text
Optimal Weights: {'weights': [('AAPL', 0.1523), ('MSFT', 0.1773), ('GOOGL', 0.1333), ('AMZN', 0.168), ('TSLA', 0.0), ('E', 0.3691)], 'return': 0.23008349841981968, 'volatility': 0.2544446620192334}
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
In the span from August 1st to August 27th, 2024, our portfolio experienced a decline, with a return of -3.88%. Despite the general market gaining 0.14% during this time, it is important to note that our performance should be seen in the context of the behavior of individual stocks in the portfolio. 

For individual stocks, Apple (AAPL) and company E yielded positive results, with returns of 1.36% and 1.08% respectively. Apple's worth rose from an initial value of 98.66 to a final value of 100. Company E also saw a surplus, with the value slightly increasing from an initial 494.64 to 500. However, these positive returns were counterbalanced by losses in the rest of the portfolio, with Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN) and Tesla (TSLA) experiencing returns of -3.64%, -6.50%, -8.82%, and -6.47% respectively.

Microsoft’s stock fell from 207.55 to 200, leading to a deficit of -7.55. Alphabet underwent a similar decline, with its value dropping from 320.87 to 300, which led to a -20.87 deficit. Amazon’s stock experienced one of the steepest percentages of loss with the value falling from 329.02 to 300, a decline that resulted in a -29.02 deficit. Tesla also saw its initial value of 213.84 fall to 200, leading to a deficit of -13.84.

The performance of these stocks contributed significantly to the overall negative return of our portfolio during this period. Nevertheless, it is crucial to maintain a long-term perspective in evaluating portfolio performance, taking into account both market volatility and the strengths of individual holdings.
```

### Example: Portfolio Suggestions - AI
```python
report = ap.get_suggestion(data, ticker, investments, openai_key)
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
Based on the metrics and returns provided here are a few suggestions to improve this portfolio:

1. **Diversification**: The stocks are primarily focused on the technology sector (AAPL, MSFT, GOOGL, AMZN, TSLA). While it appears to have performed really well in this period, over-reliance on a single sector can be risky. Consider diversifying the portfolio by investing in different sectors like financials, healthcare, consumer staples, utilities, and industrials. 

2. **Portfolio Beta**: The Beta value of 1.12 suggests that the portfolio is more volatile than the market. Reducing volatility might come with lowering returns, but it could provide better down-side protection. You can reduce the portfolio's overall risk by investing in stocks with lower beta metrics or by investing in generally less volatile sectors.

3. **Max Drawdown**: The highest losses endured in this period is quite huge at 39.41%. This high drawdown suggests there could be fairly high risk within the portfolio. Stocks that have a volatile historical price action or which dropped significantly during this period may need to be balanced with low-volatility equities.

4. **Underperforming Stocks**: The stock with the ticker 'E' has a return of only 52.61%, which is much lower than the returns from the rest of the stocks in the portfolio. You may need to review this investment and possibly consider replacing it with another stock or increasing investments in stocks that are outperforming.

5. **Alpha Value**: While the value of Alpha signifies that the portfolio may be outperforming when compared to the benchmark index, there is always room for improvement. Re-evaluate the risk-adjusted returns of each stock and consider shifting investments into those with higher alpha values.

6. **Risk-Adjusted Returns Metrics**: Sharpe and Sortino ratios are below 1. This indicates the potential for improving the quality of returns relative to the risks taken. Shifting towards investments which have high Sharpe and Sortino ratios within their respective categories would help to improve these metrics.

Remember, while these suggestions could help to optimize the portfolio, it's always important to align investment decisions with specific financial goals and risk-tolerance levels.
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