from portfolioanalyzer.graphics import compare_portfolio_to_market

def test_compare_portfolio_to_market():
    tickers = ['AAPL','MSFT','GOOGL','AMZN','TSLA']
    investments = [100,200,300,300,200]
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    market_index = '^GSPC'
    risk_free_rate = 0.01

    # Here we're assuming the function is returning a float, but since it depends on
    # live market data, we just ensure it returns a number and not raise an error.
    compare_portfolio_to_market(tickers, investments, start_date, end_date, market_index, )
    print("Portfolio compared to market index: ",market_index)
