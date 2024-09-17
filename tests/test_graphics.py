from analyzerportfolio.graphics import (
    portfolio_value,
    garch,
    montecarlo,
    drawdown
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio
)

def test_everything():
    start_date = '2019-01-01'
    end_date = '2024-08-28'
    market_ticker = '^GSPC'
    base_currency = 'EUR'


    ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
    investments = [100,200,300,300,200,500]
    data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker)
    portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio 1", rebalancing_period_days=250)

    ticker = ['AAPL','MSFT','GOOGL']
    investments = [500,300,800]
    portfolio_2 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="3 STOCK", rebalancing_period_days=250)

    ticker = ["VWCE.DE","IGLN.L","IUSN.DE"]
    investments = [500,300,800]
    data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker)
    portfolio_3 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="3 ETF", rebalancing_period_days=250)

    ticker = ["VWCE.DE","IGLN.L"]
    investments = [1300,300]
    portfolio_4 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="2 ETF", rebalancing_period_days=250)

    portfolio_value(portfolio_1)
    portfolio_value([portfolio_1,portfolio_2,portfolio_3,portfolio_4])

    garch(portfolio_1)
    garch([portfolio_1,portfolio_2,portfolio_3,portfolio_4])

    montecarlo(portfolio_1, simulation_length=1000)
    montecarlo([portfolio_1,portfolio_2,portfolio_3,portfolio_4], simulation_length=1000)
    
    drawdown(portfolio_1)
    drawdown([portfolio_1,portfolio_2,portfolio_3,portfolio_4])
    


    



