from analyzerportfolio.optimization import (
optimize,
efficient_frontier
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
    c_VaR,
    c_max_drawdown,
    c_volatility,
    c_info_ratio
    )


if False:
    def test_everything():
        ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
        investments = [100,200,300,300,200,500]
        start_date = '2019-01-01'
        end_date = '2024-08-28'
        market_ticker = '^GSPC'
        base_currency = 'EUR'
        risk_free = "PCREDIT8"
        rebalancing_period_days = 250
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=False, folder_path="/Users/name/Desktop/AP")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", base_currency=base_currency,rebalancing_period_days=rebalancing_period_days)
        
        sharpe = c_sharpe(portfolio_1)
        portfolio_optimized = optimize(portfolio_1, metric='sharpe')
        sharpe_optimized = c_sharpe(portfolio_optimized)
        
        print("Sharpe ratio before optimization: ", sharpe)
        print("Sharpe ratio after optimization: ", sharpe_optimized)

        volatility1 = c_volatility(portfolio_1)
        portfolio_optimized = optimize(portfolio_1, metric='volatility')
        volatility_optimized = c_volatility(portfolio_optimized)

        print("Volatility before optimization: ", volatility1)
        print("Volatility after optimization: ", volatility_optimized)

        drawdown1 = c_max_drawdown(portfolio_1)
        portfolio_optimized = optimize(portfolio_1, metric='drawdown')
        drawdown_optimized = c_max_drawdown(portfolio_optimized)

        print("Max drawdown before optimization: ", drawdown1)
        print("Max drawdown after optimization: ", drawdown_optimized)

        information_ratio1 = c_info_ratio(portfolio_1)
        portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
        info_optimized = c_info_ratio(portfolio_optimized)

        print("Information Ratio before optimization: ", information_ratio1)
        print("Information Ratio optimization: ", info_optimized)

        efficient_frontier(portfolio_1,num_points=10, multi_thread=True, num_threads=3, additional_portfolios=[portfolio_optimized,portfolio_1], colors=["orange","blue"])
