from analyzerportfolio.optimization import (
optimize,
efficient_frontier
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio,
    read_portfolio_composition,
    remove_small_weights
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
    heatmap
)

import os
import pandas as pd


if True:
    def test_everything():
        
        ticker = []
        investments = []

        if False:
            list_etf = pd.read_csv(r"/Users/nicolafochi/Desktop/cache/etf/etf_info_sort.csv", sep=";")

            for index, row in list_etf.iterrows():
                
                    ticker.append(row["Ticker\n"].split(" ")[0])
                    investments.append(100000)

                    if len(ticker) == 80:
                        break
            
            for etf in ["CSSPX.MI", "EIMI.SW", "SGLD.MI","IBC1.MU"]:
                ticker.append(etf)
                investments.append(100000)

        ticker = ["CSPX.L", "IWDA.L","EIMI.L","IEAC.L", "EQQQ.MI",
                  "XZMU.L","1674.T","CMOD.MI","GSCE.MI","AIGC.MI","AIGE.L",
                  "IMEU.AS", "XDEW.MI", "QDVE.DE", "DGRW", "ITA", "FNDX", "FDVV", "MLPX",
                  "R2US.PA", "EWC", "GDX", "SIL","006208.TW","EZU","BBUS","INR.PA","SGLD.L",
                  "SLV","DYNF","IBC1.MU","IEMB.MI","ECRP.MI","SUOE.MI", "XD9U.DE","XMEU.MI"]
        
        
        for i in ticker:
            investments.append(100000)
        
        start_date = '2020-08-27'
        end_date = '2024-10-07'
        market_ticker = 'benchmark'
        base_currency = 'USD'
        risk_free = "PCREDIT8"
        colors = ["orange"]
        
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/leonardo/Desktop/cache/etf")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
        
        


        information_ratio1 = c_info_ratio(portfolio_1)
        

        portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
        portfolio_optimized = remove_small_weights(portfolio_optimized)
        portfolio_optimized["name"] = "Optimized Portfolio Information Ratio"
        portfolio_optimized_sharpe = optimize(portfolio_1, metric='sharpe')
        portfolio_optimized_sharpe["name"] = "Optimized Portfolio Sharpe Ratio"
        information_ratio_optimized = c_info_ratio(portfolio_optimized)
        print("Information ratio before ", information_ratio1)
        print("Information ratio after ",information_ratio_optimized)
        print("Sharpe ratio after ", c_sharpe(portfolio_optimized))
        print(read_portfolio_composition(portfolio_optimized,min_value = 0.001))
        
        garch([portfolio_optimized,portfolio_optimized_sharpe], plot_difference=True, colors = ["orange","blue"])
        portfolio_value([portfolio_optimized,portfolio_optimized_sharpe],colors = ["orange","blue"])

        montecarlo([portfolio_optimized,portfolio_optimized_sharpe], simulation_length=30)

        garch_diff([portfolio_optimized,portfolio_optimized_sharpe], colors = ["orange","blue"])
        
        heatmap(portfolio_optimized, disassemble=True)
        if False:
            result = efficient_frontier(portfolio_1,num_points=20, multi_thread=True, num_threads=3, additional_portfolios=[portfolio_optimized,portfolio_optimized_sharpe], colors=["orange","blue"])
        
        pie_chart(portfolio_optimized)






