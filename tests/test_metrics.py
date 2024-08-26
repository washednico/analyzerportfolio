from portfolioanalyzer.metrics import calculate_beta_and_alpha, calculate_sharpe_ratio, calculate_sortino_ratio, download_data, calculate_var
from portfolioanalyzer.graphics import compare_portfolio_to_market

def test_sortino():
    ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA']
    investments = [100,200,300,300,200]
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    market_index = '^GSPC'
    risk_free_rate = 0.01
    base_currency = 'EUR'
    
    
    data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_index=market_index)
    beta, alpha = calculate_beta_and_alpha(data, ticker, investments, market_index)
    sharpe_ratio = calculate_sharpe_ratio(data, ticker, investments, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(data, ticker, investments, risk_free_rate)
    var = calculate_var(data, ticker, investments)
    
    print("Beta: ", beta)
    print("Alpha: ", alpha)
    print("Sharpe Ratio: ", sharpe_ratio)
    print("Sortino Ratio: ", sortino_ratio)
    print("Value at Risk: ", var)
    
    compare_portfolio_to_market(data, ticker, investments, market_index)




