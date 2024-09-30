from analyzerportfolio.metrics import (
    c_beta,
    c_sharpe,
    c_sortino,
    c_analyst_scenarios,
    c_analyst_score,
    c_dividend_yield,
    c_VaR,
    c_total_return,
    c_volatility,
    c_max_drawdown
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio
)

if False:
    def test_everything():
        ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
        investments = [100,200,300,300,200,500]
        start_date = '2013-02-01'
        end_date = '2024-08-28'
        market_ticker = '^GSPC'
        base_currency = 'EUR'
        risk_free = "PCREDIT8"
        rebalancing_period_days = 250
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free)
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", base_currency=base_currency, rebalancing_period_days=rebalancing_period_days)

        beta, alpha = c_beta(portfolio_1)
        return_port = c_total_return(portfolio_1)
        volatility = c_volatility(portfolio_1)
        sharpe = c_sharpe(portfolio_1)
        sortino = c_sortino(portfolio_1)
        scenarios = c_analyst_scenarios(portfolio_1)
        score = c_analyst_score(portfolio_1)
        dividend_yield = c_dividend_yield(portfolio_1)
        var_95 = c_VaR(portfolio_1, confidence_level=0.95)
        var_99_30d = c_VaR(portfolio_1, confidence_level=0.95, horizon_days=30)
        var_95_p = c_VaR(portfolio_1, confidence_level=0.95, method="parametric")
        var_99_p_30d = c_VaR(portfolio_1, confidence_level=0.95, horizon_days=30, method="parametric")
        max_drawdown = c_max_drawdown(portfolio_1)


        print(beta, alpha)
        print(return_port)
        print(volatility)
        print(sharpe)
        print(sortino)
        print(scenarios)
        print(score)
        print(dividend_yield)
        print(var_95)
        print(var_99_30d)
        print(var_95_p)
        print(var_99_p_30d)
        print(max_drawdown)




