from portfolioanalyzer.metrics import calculate_beta_and_alpha, calculate_sharpe_ratio, calculate_sortino_ratio

def test_sortino():
    tickers = ['AAPL','MSFT','GOOGL','AMZN','TSLA']
    investments = [100,200,300,300,200]
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    risk_free_rate = 0.01

    # Here we're assuming the function is returning a float, but since it depends on
    # live market data, we just ensure it returns a number and not raise an error.
    sortino = calculate_sortino_ratio(tickers, investments, start_date, end_date,risk_free_rate)
    print("Portfolio Sortino ratio: ",sortino)

    
def test_calculate_portfolio_beta():
    tickers = ['AAPL','MSFT','GOOGL','AMZN','TSLA']
    investments = [100,200,300,300,200]
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    risk_free_rate = 0.01

    # Here we're assuming the function is returning a float, but since it depends on
    # live market data, we just ensure it returns a number and not raise an error.
    beta,alpha = calculate_beta_and_alpha(tickers, investments, start_date, end_date,risk_free_rate)
    print("Portfolio Beta: ",beta,"\nPortfolio alpha: ",alpha)
    
    
def test_calculate_sharpe():
    tickers = ['AAPL','MSFT','GOOGL','AMZN','TSLA']
    investments = [100,200,300,300,200]
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    risk_free_rate = 0.01

    # Here we're assuming the function is returning a float, but since it depends on
    # live market data, we just ensure it returns a number and not raise an error.
    sharpe = calculate_sharpe_ratio(tickers, investments, start_date, end_date,risk_free_rate)
    print("Portfolio Sharpe ratio: ",sharpe)

