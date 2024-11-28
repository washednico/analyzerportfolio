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

        ticker = ["1504.TW","EKC.BO","SMIN.L","AXP", "300628.SZ", "JNJ",
                  "6503.T", "6028.T", "002311.SZ","300218.SZ", "ATEA.OL", "SUN.SW",
                  "5904.TWO", "2170.T", "MRK", "IMI.L", "KSCL.NS", "4203.T",
                  "MTX", "TGT", "603040.SS", "NXT.L", "CTRA","TNC","VRTS","ALSN","BBY",
                  "4536.TW","OMC","3705.TW","6113.T","000786.SZ","4766.TW",
                  "2385.TW","7367.T","605166.SS","INDUSTOWER.NS","G","CFT.SW","0728.HK","9942.TW","ALV",
                  "MPLX","LAS-A.TO","9739.T","CS.PA","6754.TW","5871.TW","HAL","600566.SS","6498.T","NVG.LS","LOUP.PA"]
        
        
        for i in ticker:
            investments.append(100_000_000/len(ticker))
        
        start_date = '2020-10-27'
        end_date = '2023-10-07'
        market_ticker = 'benchmark'
        base_currency = 'USD'
        risk_free = "PCREDIT8"
        colors= ["orange","blue","red"]
        
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/stocks2")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Equally Weighted", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
        portoflio__benchmark = create_portfolio(data, tickers=["benchmark"], investments=[100000], market_ticker=market_ticker, name_portfolio="Benchmark", base_currency=base_currency, rebalancing_period_days=1)
        
        # Optimization
        portfolio_optimized = optimize(portfolio_1, metric="information_ratio")
        portfolio_optimized["name"] = "Optimized"
        print(c_info_ratio(portfolio_optimized))
        pie_chart(portfolio_optimized)
        
        if True:
            # OUT OF SAMPLE BENCHMARK
            print("--- out of sample ---")
            tickers_optimized = portfolio_optimized["tickers"]
            investments_optimized = portfolio_optimized["investments"]
            start_date = "2023-10-08"
            end_date = "2024-10-04"
            data = download_data(tickers=tickers_optimized, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/stocks2")
            portfolio_optimized_oos = create_portfolio(data, ticker, investments_optimized, market_ticker=market_ticker, name_portfolio="OPTIMIZED", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
            print(c_info_ratio(portfolio_optimized_oos))
            portfolio_value(portfolio_optimized_oos, colors = ["orange"])


if False:
    def test_everything():
        dictionare_stock = {'EKC.BO': 0.004375700642492835, 'SMIN.L': 0.014145526929102153, '300628.SZ': 0.015723963346357592, 'JNJ': 0.017019324807691086, 
                            '6503.T': 0.008953999614862774, '6028.T': 0.04134650285810199, '002311.SZ': 0.012567041338737414, '300218.SZ': 0.039360505126981804, 
                            'ATEA.OL': 0.008189419140294913, 'SUN.SW': 0.042053690608895314, '5904.TWO': 0.03235712398457033, '2170.T': 0.007008060499517999, 
                            'MRK': 0.019312333579553136, 'IMI.L': 0.011835524498478557, 'KSCL.NS': 0.007951745492326604, '4203.T': 0.017218020785024074, 
                            'MTX': 0.00805844550342327, 'TGT': 0.005014657333006343, '603040.SS': 0.012765281725105488, 'NXT.L': 0.008647365811916184, 
                            'CTRA': 0.006929661281919482, 'TNC': 0.009166941593962398, 'ALSN': 0.004923604554314942, 'BBY': 0.006177939354298765, '4536.TW': 0.02046256515522918, 
                            'OMC': 0.010184793900862878, '3705.TW': 0.07603663085544243, '6113.T': 0.02914723710278478, '000786.SZ': 0.020448810133677487, '4766.TW': 0.03200338831904286, 
                            '2385.TW': 0.037957010201510204, '7367.T': 0.017384676309001024, '605166.SS': 0.08544849707456147, 'INDUSTOWER.NS': 0.012688801082385057, 
                            'G': 0.011380483636401928, '0728.HK': 0.009210102899208974, 'ALV': 0.006310647217016658, 'MPLX': 0.015795883022536306, 'LAS-A.TO': 0.01774923140297777, 
                            '9739.T': 0.036460002192744985, '6754.TW': 0.032196897620725015, 'HAL': 0.002742762623474643, '600566.SS': 0.01999730891595144, '6498.T': 0.05360780311078581, 
                            'NVG.LS': 0.038904723847979265, 'LOUP.PA': 0.052779362964764306}


        start_date = '2020-10-27'
        end_date = '2024-10-07'
        market_ticker = 'benchmark'
        base_currency = 'USD'
        risk_free = "PCREDIT8"

        investments = []
        ticker= []
        for i in dictionare_stock.keys():
            investments.append(dictionare_stock[i])
            ticker.append(i)
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/stocks2")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Stocks_portfolio", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
        portfolio_1["portfolio_returns"].to_csv("/Users/nicolafochi/Desktop/stocks2/portfolio_stocks.csv")
        portfolio_1["market_returns"].to_csv("/Users/nicolafochi/Desktop/stocks2/market_returns.csv")
    