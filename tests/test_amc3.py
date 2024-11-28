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
        ticker = [
    "COWZ", "DFAC", "DGRO", "DIA", "EFA", "IEFA", "IEMG", "IJH", "IJR",
    "ITOT", "IVE", "IVV", "IVW", "IWD", "IWF", "IWM", "IWR", "IXUS",
    "JEPI", "MDY", "QQQ", "QQQM", "QUAL", "RSP", "SCHB", "SCHD", "SCHF",
    "SCHG", "SCHX", "SDY", "SMH", "SPDG", "SPY", "SPYD", "SPYG", "TQQQ",
    "USMV", "VB", "VDE", "VEA", "VEU", "VGK", "VGT", "VHT", "VIG", "VNQ",
    "VO", "VOO", "VT", "VTI", "VTV", "VUG", "VWO", "VXF", "VXUS", "VYM",
    "XLE", "XLF", "XLI", "XLK", "XLV", "XLY"
]
        
        
        for i in ticker:
            investments.append(100_000_000/len(ticker))
        
        start_date = '2024-10-01'
        end_date = '2024-11-27'
        market_ticker = 'benchmark'
        base_currency = 'USD'
        risk_free = "PCREDIT8"
        colors= ["orange","blue","red"]
        
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path=r"/Users/nicolafochi/Desktop/third_round")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Equally Weighted", base_currency=base_currency, exclude_ticker= True, exclude_ticker_time= 7, rebalancing_period_days=1)
        portfolio_value(portfolio_1)
        print(c_info_ratio(portfolio_1))

        info_otpimized_portfolio = optimize(portfolio_1, metric="information_ratio")
        pie_chart(info_otpimized_portfolio)
        portfolio_value(info_otpimized_portfolio)
        print(c_info_ratio(info_otpimized_portfolio))