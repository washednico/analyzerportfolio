from analyzerportfolio.metrics import (
    calc_beta
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio
)

def test_everything():
    ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
    investments = [100,200,300,300,200,500]
    start_date = '2019-01-01'
    end_date = '2024-08-28'
    market_ticker = '^GSPC'
    base_currency = 'EUR'
    
    data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker)
    portfolio_df = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", rebalancing_period_days=250)

    beta, alpha = calc_beta(portfolio_df)

    print(beta, alpha)



