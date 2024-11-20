from analyzerportfolio.optimization import (
optimize,
efficient_frontier
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio,
    read_portfolio_composition,
    remove_small_weights,
    update_portfolio
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

from analyzerportfolio.graphics import (
    portfolio_value,
    garch,
    garch_diff,
    montecarlo,
    pie_chart,
    heatmap,
    distribution_return,
    drawdown
)


import pandas as pd


if False:
    def test_everything():
        
        ticker = []
        investments = []

        ticker = ["CSPX.L",
                  "XZMU.L",
                  "QDVE.DE", "DGRW", "FNDX", "MLPX",
                  "006208.TW","INR.PA","SGLD.L",
                  "DYNF",
                  "LVHI","ARGT"]
        
        
        for i in ticker:
            investments.append(100_000_000/len(ticker))
        
        start_date = '2020-08-27'
        end_date = '2024-10-07'
        market_ticker = 'benchmark'
        base_currency = 'USD'
        risk_free = "PCREDIT8"
        colors= ["orange","blue","red"]
        
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/cache/etf")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Equally Weighted", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
        portoflio__benchmark = create_portfolio(data, tickers=["benchmark"], investments=[100000], market_ticker=market_ticker, name_portfolio="Benchmark", base_currency=base_currency, rebalancing_period_days=1)
        portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
        pie_chart(portfolio_optimized)
        
        portfolio_optimized["portfolio_returns"].to_csv("/Users/nicolafochi/Desktop/cache/etf/portfolio_optimized_returns.csv")



        
        






