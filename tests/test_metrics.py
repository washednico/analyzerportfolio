from analyzerportfolio.metrics import (
    calculate_beta_and_alpha, 
    calculate_sharpe_ratio, 
    calculate_sortino_ratio, 
    download_data, 
    calculate_var, 
    calculate_portfolio_scenarios, 
    calculate_dividend_yield, 
    calculate_max_drawdown,
    calculate_analyst_suggestion
    )

from analyzerportfolio.graphics import (
    compare_portfolio_to_market, 
    simulate_dca, 
    garch, 
    montecarlo, 
    heatmap, 
    probability_cone, 
    drawdown_plot
)

from analyzerportfolio.optimization import (
    markowitz_optimization
)

from analyzerportfolio.ai import (
    newsletter_report,
    get_suggestion,
    monitor_news
)

def test_everything():
    ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
    investments = [100,200,300,300,200,500]
    start_date = '2019-01-01'
    end_date = '2024-08-28'
    market_ticker = '^GSPC'
    risk_free_rate = 0.01
    base_currency = 'EUR'
    openai_key = ""
    
    data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker)

    beta, alpha = calculate_beta_and_alpha(data, ticker, investments, market_ticker)
    sharpe_ratio = calculate_sharpe_ratio(data, ticker, investments, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(data, ticker, investments, risk_free_rate)
    var = calculate_var(data, ticker, investments)
    var_5_h = calculate_var(data,ticker,investments,confidence_level=0.99,time_horizon=5,method='historical')
    dividend_yield = calculate_dividend_yield(ticker, investments)
    max_drawdown = calculate_max_drawdown(data, ticker, investments)
    portfolio_scenarios = calculate_portfolio_scenarios(ticker,investments,base_currency)
    analyst_info = calculate_analyst_suggestion(ticker,investments)

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
    
    compare_portfolio_to_market(data, ticker, investments, market_ticker)
    garch(data, ticker, investments,market_ticker)
    simulate_dca(data, ticker, 1000, 100, 30, [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    montecarlo(data,ticker,investments,250,50,50,market_ticker)
    heatmap(data, ticker, market_ticker)
    probability_cone(data, ticker, investments,750)
    drawdown_plot(data, ticker, investments, market_ticker)
    
    markowitz_portfolio = markowitz_optimization(data, ticker, investments, method = 'volatility')
    print("\n   | Markowitz Optimal Portfolio |   ")
    print("Optimal Weights:", markowitz_portfolio)
    print( ' \n \n \n \n \n ')
    
    
    monitor_news(ticker,delay = 60,loop_forever=True, openai_key=openai_key)
    report = newsletter_report(data,ticker,investments,start_date_report="2024-08-01",openai_key=openai_key)
    print(report)
    print( ' \n \n \n \n \n ')

    suggestion = get_suggestion(data, ticker, investments, openai_key)
    print(suggestion)