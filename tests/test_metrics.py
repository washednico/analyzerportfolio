from analyzerportfolio.metrics import (
    calc_beta,
    calc_sharpe,
    calc_sortino,
    calc_scenarios
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
    risk_free = "PCREDIT8"
    rebalancing_period_days = 250
    
    data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free)
    portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1",base_currency=base_currency, rebalancing_period_days=rebalancing_period_days)

    beta, alpha = calc_beta(portfolio_1)
    sharpe = calc_sharpe(portfolio_1)
    sortino = calc_sortino(portfolio_1)
    scenarios = calc_scenarios(portfolio_1)

    print(beta, alpha)
    print(sharpe)
    print(sortino)
    print(scenarios)



