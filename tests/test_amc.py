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
    c_VaR,
    c_max_drawdown,
    c_volatility,
    c_info_ratio
    )

import os


if True:
    def test_everything():
        
        ticker = []
        investments = []
        for root,dirs,files in os.walk("/Users/nicolafochi/Desktop/index_clean"):
            for file in files:
                if file.endswith(".csv"):
                    ticker.append(file[:-4])
                    investments.append(100000)

        print(ticker)

        start_date = '2023-01-01'
        end_date = '2024-08-28'
        market_ticker = '^GSPC'
        base_currency = 'USD'
        risk_free = "PCREDIT8"
        rebalancing_period_days = 250
        
        
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency, market_ticker=market_ticker, risk_free=risk_free, use_cache=True, folder_path="/Users/nicolafochi/Desktop/index_clean")
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio1", base_currency=base_currency,rebalancing_period_days=rebalancing_period_days)

        information_ratio1 = c_info_ratio(portfolio_1)
        portfolio_optimized = optimize(portfolio_1, metric='information_ratio')
        information_ratio_optimized = c_info_ratio(portfolio_optimized)
        print(information_ratio1)
        print(information_ratio_optimized)

