import os
print(f"Running tests from: {os.path.abspath(__file__)}")


from portfolioanalyzer.metrics import calculate_beta


def test_calculate_portfolio_beta():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    weights = [0.4, 0.4, 0.2]
    start_date = '2020-01-01'
    end_date = '2021-01-01'

    # Here we're assuming the function is returning a float, but since it depends on
    # live market data, we just ensure it returns a number and not raise an error.
    beta = calculate_beta(tickers, weights, start_date, end_date)
    print(beta)
    
    assert isinstance(beta, float), "Beta should be a float"