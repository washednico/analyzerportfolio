from analyzerportfolio.utils import (
    download_data,
    create_portfolio,
    read_portfolio_composition,
    remove_small_weights,
    update_portfolio
)

from analyzerportfolio.graphics import (
    portfolio_value)

if False:
    def test_everything():
        
        ticker = []
         
        investments = []
        ticker = ["COWZ"]
            
        for i in ticker:
            investments.append(100_000_000/len(ticker))
        
        start_date = '2020-02-01'
        end_date = '2024-08-28'
        market_ticker = '^GSPC'
        base_currency = 'USD'
        risk_free = "DTB3"
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free, use_cache=True,folder_path=r"/Users/nicolafochi/Desktop/test_cache")
        portfolio = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio 1", base_currency=base_currency, rebalancing_period_days=250)
        
        portfolio_value(portfolio)


        start_date = '2019-02-01'
        end_date = '2024-08-28'
        market_ticker = '^GSPC'
        base_currency = 'USD'
        risk_free = "DTB3"
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free, use_cache=True,folder_path=r"/Users/nicolafochi/Desktop/test_cache")
        portfolio = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio 1", base_currency=base_currency, rebalancing_period_days=250)
        
        portfolio_value(portfolio)

        start_date = '2018-02-01'
        end_date = '2024-08-28'
        market_ticker = '^GSPC'
        base_currency = 'USD'
        risk_free = "DTB3"
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free, use_cache=True,folder_path=r"/Users/nicolafochi/Desktop/test_cache")
        portfolio = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio 1", base_currency=base_currency, rebalancing_period_days=250)
        
        portfolio_value(portfolio)

