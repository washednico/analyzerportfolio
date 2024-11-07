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


if True:
    def test_everything():
        
        ticker = []
        investments = []

        ticker = ['IJM', 'BBY', 'LZB', 'FTT', 'ULTA', '601966', 'ALOS3', 'BLBD', '603090', 'ITSA4', 'SJM', 'SIE', '1504', 'KSCL', 'PETD', 'MMS', 'BETCO',
                   'JEN', 'TEL', '6198', '2881', '601000', 'SLCE3', 'TGT', 'INRETC1', '3569', 'LOUP', '5871', 'NOMD', 'L', 'MRU', '9319',
                   "TKC", "TUL1.F", "TBRG", "300795.SZ", "SRB.L","PLPL3.SA","ADBE", "FSLR", "0K6R.L", "300832.SZ","MAIRE.MI"]
        
        
        for i in ticker:
            investments.append(100_000_000/len(ticker))
        
        start_date = '2020-10-27'
        end_date = '2023-10-07'
        market_ticker = 'benchmark'
        base_currency = 'USD'
        risk_free = "PCREDIT8"
        colors= ["orange","blue","red"]
        
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/stocks_clean")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Equally Weighted", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
        portoflio__benchmark = create_portfolio(data, tickers=["benchmark"], investments=[100000], market_ticker=market_ticker, name_portfolio="Benchmark", base_currency=base_currency, rebalancing_period_days=1)
        
        # Optimization
        portfolio_optimized = optimize(portfolio_1, metric="information_ratio")
        print(c_info_ratio(portfolio_optimized))
        pie_chart(portfolio_optimized)
        

        # OUT OF SAMPLE BENCHMARK
        print("--- out of sample ---")
        tickers_optimized = portfolio_optimized["tickers"]
        investments_optimized = portfolio_optimized["investments"]
        start_date = "2023-10-08"
        end_date = "2024-10-04"
        data = download_data(tickers=tickers_optimized, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/stocks_clean")
        portfolio_optimized_oos = create_portfolio(data, ticker, investments_optimized, market_ticker=market_ticker, name_portfolio="Equally Weighted", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
        print(c_info_ratio(portfolio_optimized_oos))
        portfolio_value(portfolio_optimized_oos)








