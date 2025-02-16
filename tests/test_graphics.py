from analyzerportfolio.graphics import (
    portfolio_value,
    garch,
    montecarlo,
    drawdown,
    heatmap,
    distribution_return,
    simulate_dca,
    probability_cone,
    pie_chart,
    sector_pie,
    country_pie,
    garch_diff
    )

from analyzerportfolio.utils import (
    download_data,
    create_portfolio

)

from analyzerportfolio.config import (
    set_plotly_template,
    get_plotly_template,
    configure_logging
)



if True:

    # User preferences for logging
    #LOG_LEVEL = logging.DEBUG
    #LOG_FILE = 'user_test_log.log'
    #VERBOSE = True
    # Setup logging
    configure_logging()

    def test_everything():
        colors_1 = "orange" #OPTIONAL
        colors_2 = ["orange", "blue"]
        colors_4 = ["orange","blue","purple","red"]  #OPTIONAL

        start_date = '2013-02-01'
        end_date = '2024-08-28'
        market_ticker = '^GSPC'
        base_currency = 'EUR'
        risk_free = "DTB3"


        ticker = ['AAPL','MSFT','GOOGL','AMZN','TSLA','E']
        investments = [100,200,300,300,200,500]
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free)
        portfolio_1 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="Portfolio 1", base_currency=base_currency, rebalancing_period_days=250)

        ticker = ['AAPL','MSFT','GOOGL']
        investments = [500,300,800]
        portfolio_2 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="3 STOCK",base_currency=base_currency, rebalancing_period_days=250)

        ticker = ["VWCE.DE","IGLN.L","IUSN.DE"]
        investments = [500,300,800]
        data = download_data(tickers=ticker, start_date=start_date, end_date=end_date, base_currency=base_currency,market_ticker=market_ticker, risk_free=risk_free)
        portfolio_3 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="3 ETF", base_currency=base_currency, rebalancing_period_days=250)

        ticker = ["VWCE.DE","IGLN.L"]
        investments = [1300,300]
        portfolio_4 = create_portfolio(data, ticker, investments, market_ticker=market_ticker, name_portfolio="2 ETF",base_currency=base_currency, rebalancing_period_days=250)
        
        portfolio_value(portfolio_1, colors=colors_1)

        garch(portfolio_1, colors=colors_1)
        # garch([portfolio_1,portfolio_2,portfolio_3,portfolio_4], colors=colors_4)

        garch_diff([portfolio_1,portfolio_2,portfolio_3,portfolio_4])

        montecarlo(portfolio_1, simulation_length=1000)
        # montecarlo([portfolio_1,portfolio_2,portfolio_3,portfolio_4], simulation_length=1000)
        
        drawdown(portfolio_1, colors=colors_1)
        # drawdown([portfolio_1,portfolio_2,portfolio_3,portfolio_4], colors=colors_4)

        heatmap(portfolio_1)

        distribution_return(portfolio_1, colors=colors_1)
        distribution_return([portfolio_1,portfolio_2,portfolio_3,portfolio_4], colors=colors_4)

        # simulate_dca(portfolio_1, initial_investment=10000, periodic_investment=500, investment_interval=30, colors=colors_1)
        # simulate_dca([portfolio_1,portfolio_2,portfolio_3,portfolio_4], initial_investment=10000, periodic_investment=500, investment_interval=30, colors=colors_4)

        # probability_cone(portfolio_1, time_horizon=1000)






    



