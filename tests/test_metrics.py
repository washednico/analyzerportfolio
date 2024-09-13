from analyzerportfolio.metrics import (
    calculate_beta_and_alpha, 
    download_data
    )

from analyzerportfolio.graphics import (
    compare_portfolio_to_market, 
    plot_distribution_returns
)



from analyzerportfolio.utils import (
    forming_portfolio
)

def test_everything():
    ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
    investments = [100,200,300,300,200,500]
    start_date = '2019-01-01'
    end_date = '2024-08-28'
    market_ticker = '^GSPC'
    risk_free_rate = 0.01
    base_currency = 'EUR'
    
    data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker)
    portfolio_df = forming_portfolio(data, ticker, investments, market_ticker=market_ticker, rebalancing_period_days=250)

    beta, alpha = calculate_beta_and_alpha(portfolio_df["Portfolio_Returns"], portfolio_df["Market_Returns"])
    compare_portfolio_to_market(portfolio_df["Portfolio_Value"], portfolio_df["Market_Value"])
    plot_distribution_returns(portfolio_df["Portfolio_Returns"])



