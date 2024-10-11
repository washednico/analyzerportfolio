from analyzerportfolio.optimization import (
optimize
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio,
    read_portfolio_composition
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
    garch
)

import os
import pandas as pd


if True:
    def test_everything():
        
        ticker = []
        investments = []
        list_etf = pd.read_csv(r"/Users/nicolafochi/Desktop/cache/etf/etf_info_sort.csv", sep=";")

        for index, row in list_etf.iterrows():
            ticker.append(row["Ticker\n"].split(" ")[0])
            investments.append(100000)

            if len(ticker) == 80:
                break


        start_date = '2022-08-27'
        end_date = '2024-08-28'
        market_ticker = 'benchmark'
        base_currency = 'USD'
        risk_free = "PCREDIT8"
        
        
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/cache/etf")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", base_currency=base_currency, rebalancing_period_days=30)

        information_ratio1 = c_info_ratio(portfolio_1)
        print(information_ratio1)

        portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
        information_ratio_optimized = c_info_ratio(portfolio_optimized)
        print(information_ratio1)
        print(information_ratio_optimized)
        print(read_portfolio_composition(portfolio_optimized,min_value = 0.001))

        garch([portfolio_optimized])
        portfolio_value([portfolio_optimized])



