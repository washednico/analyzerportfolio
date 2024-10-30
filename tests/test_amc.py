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

        ticker = ["CSPX.L", "IWDA.L","EIMI.L","IEAC.L", "EQQQ.MI",
                  "XZMU.L","1674.T","CMOD.MI","GSCE.MI","AIGC.MI","AIGE.L",
                  "IMEU.AS", "XDEW.MI", "QDVE.DE", "DGRW", "ITA", "FNDX", "FDVV", "MLPX",
                  "R2US.PA", "EWC", "GDX", "SIL","006208.TW","EZU","BBUS","INR.PA","SGLD.L",
                  "SLV","DYNF","IBC1.MU","USCO.MI","IEMB.MI","ECRP.MI","SUOE.MI", "XD9U.DE",
                  "XMEU.MI","IDMO","LVHI","DXJ","ARGT","EPU","FLCA","DBEU"]
        
        
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
        
        print("Information ratio before optimization", c_info_ratio(portfolio_1))
        print("Sharpe ratio after before optimization ", c_sharpe(portfolio_1))
        portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
        portfolio_optimized = remove_small_weights(portfolio_optimized)
        

        #OUT OF SAMPLE TESTING
        tickers_opt  = portfolio_optimized["tickers"]
        investments_opt = portfolio_optimized["investments"]
        start_date = '2019-10-01'
        end_date = '2020-08-26'
        data = download_data(tickers=tickers_opt, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/oos_cache")
        port_oos = create_portfolio(data, tickers_opt, investments_opt, market_ticker=market_ticker, name_portfolio="OUT OF SAMPLE", base_currency=base_currency, rebalancing_period_days=1)
        portfolio_value(port_oos)
        print("OUT OF SAMPLE TEST: INFO RATIO:",c_info_ratio(port_oos))
        print("OUT OF SAMPLE TEST: SHARPE RATIO:",c_sharpe(port_oos))
        garch_diff(port_oos)
        garch(port_oos, plot_difference=True)

        if False:
            portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
            portfolio_optimized = remove_small_weights(portfolio_optimized)
            print(read_portfolio_composition(portfolio_optimized))
            portfolio_optimized["name"] = "Optimized Portfolio Information"

            print("Information ratio after information optimization",c_info_ratio(portfolio_optimized))
            print("Sharpe ratio after after information optimization ", c_sharpe(portfolio_1))


            portfolio_optimized_sharpe = optimize(portfolio_1, metric='sharpe')
            portfolio_optimized = remove_small_weights(portfolio_optimized)
            portfolio_optimized_sharpe["name"] = "Optimized Portfolio Sharpe"

            print("Information ratio after sharp optimization",c_info_ratio(portfolio_optimized_sharpe))
            print("Sharpe ratio after sharpe optimization ", c_sharpe(portfolio_optimized))

            pie_chart(portfolio_optimized, transparent=True)
            
            
            garch([portfolio_optimized,portfolio_optimized_sharpe,portfolio_1], plot_difference=True, colors = colors)
            
            portfolio_value([portfolio_optimized,portfolio_optimized_sharpe,portfolio_1],colors = colors)

            montecarlo([portfolio_optimized,portfolio_optimized_sharpe,portfolio_1], simulation_length=30)

            garch_diff([portfolio_optimized,portfolio_optimized_sharpe,portfolio_1], colors = colors)
            
            heatmap(portfolio_optimized, disassemble=True)

            distribution_return([portfolio_optimized,portfolio_optimized_sharpe,portfolio_1], colors = colors)

            drawdown([portfolio_optimized,portfolio_optimized_sharpe,portfolio_1], colors = colors)

            if True:
                result = efficient_frontier(portfolio_1,num_points=3, multi_thread=True, num_threads=3, additional_portfolios=[portfolio_optimized,portfolio_optimized_sharpe,portfolio_1,portoflio__benchmark], colors=colors.append("green"))
            
            
        
        






