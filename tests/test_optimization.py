from analyzerportfolio.optimization import (
optimize
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio
)

from analyzerportfolio.metrics import (
    c_beta,
    c_sharpe,
    c_sortino,
    c_analyst_scenarios,
    c_analyst_score,
    c_dividend_yield,
    c_VaR
    )


if True:
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
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", base_currency=base_currency, rebalancing_period_days=rebalancing_period_days)

        
        sharpe = c_sharpe(portfolio_1)
        
        portfolio_optimized = optimize(portfolio_1, metric='sharpe')
        sharpe_optimized = c_sharpe(portfolio_optimized)

        print("Sharpe ratio before optimization: ", sharpe)
        print("Sharpe ratio after optimization: ", sharpe_optimized)
        



