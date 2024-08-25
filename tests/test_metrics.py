from portfolioanalyzer.metrics import calculate_beta_and_alpha


def test_calculate_portfolio_beta():
    tickers = ['AAPL']
    weights = [1]
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    risk_free_rate = 0.01

    # Here we're assuming the function is returning a float, but since it depends on
    # live market data, we just ensure it returns a number and not raise an error.
    beta,alpha = calculate_beta_and_alpha(tickers, weights, start_date, end_date,risk_free_rate)
    print(beta, alpha)
    
    assert isinstance(beta, float), "Beta should be a float"