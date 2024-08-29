import openai
import pandas as pd
from portfolioanalyzer import metrics

def newsletter_report(
    price_df: pd.DataFrame,
    tickers: list[str],
    investments: list[float],
    start_date_report: str,
    market_ticker: str = '^GSPC',
    risk_free_rate: float = 0.01,
    openai_key: str = None,
) -> str:
    
    """Generate a newsletter report for a stock portfolio.
    
    Parameters:
    price_df (pd.DataFrame): A DataFrame containing the historical price data of the stocks in the portfolio.
    tickers (list[str]): A list of stock tickers in the portfolio.
    investments (list[float]): A list of investment amounts corresponding to each stock in the portfolio AT FINAL DATE!!!
    start_date_report (str): The start date of the report to calculate returns in the format 'YYYY-MM-DD'.
    market_ticker (str, optional): The ticker symbol of the market index to compare the portfolio against. Default is '^GSPC' (S&P 500).
    risk_free_rate (float, optional): The risk-free rate used in calculating the Sharpe and Sortino ratios. Default is 0.01 (1%).
    openai_key (str, optional): The OpenAI API key to use for generating the report. If not provided, the report will be generated without AI assistance.
    
    Returns:
    str: The generated newsletter report as a string.
    """

    # Calculate portfolio metrics
    beta, alpha = metrics.calculate_beta_and_alpha(price_df, tickers, investments, market_ticker)
    sharpe_ratio = metrics.calculate_sharpe_ratio(price_df, tickers, investments, risk_free_rate)
    sortino_ratio = metrics.calculate_sortino_ratio(price_df, tickers, investments, risk_free_rate)
    var = metrics.calculate_var(price_df, tickers, investments)
    max_drawdown = metrics.calculate_max_drawdown(price_df, tickers, investments)
    dividend_yield = metrics.calculate_dividend_yield(tickers, investments)
    
    # Determine the last available date in the price data
    last_day = price_df.index[-1].strftime('%Y-%m-%d')
    first_metric_day = price_df.index[0].strftime('%Y-%m-%d')
    
    # Calculate the number of shares for each stock based on the final investment amount
    shares = [investment / price_df[ticker].iloc[-1] for ticker, investment in zip(tickers, investments)]
    
    # Calculate initial and final portfolio values
    portfolio_initial_value = sum(price_df[ticker].loc[start_date_report] * share for ticker, share in zip(tickers, shares))
    portfolio_final_value = sum(investments)
    
    # Calculate market and portfolio returns from the start_date_report to the last available day
    market_return = (price_df[market_ticker].loc[last_day] / price_df[market_ticker].loc[start_date_report] - 1) * 100
    portfolio_return = (portfolio_final_value / portfolio_initial_value - 1) * 100
    
    # Calculate individual stock returns and monetary surplus/deficit
    stock_details = ""
    for ticker, investment, share in zip(tickers, investments, shares):
        initial_value = share * price_df[ticker].loc[start_date_report]
        final_value = share * price_df[ticker].iloc[-1]
        stock_return = (final_value / initial_value - 1) * 100
        surplus_or_deficit = final_value - initial_value
        stock_details += (
            f"{ticker}:\n"
            f"  - Initial Value: {initial_value:,.2f}\n"
            f"  - Final Value: {final_value:,.2f}\n"
            f"  - Return: {stock_return:.2f}%\n"
            f"  - Surplus/Deficit: {surplus_or_deficit:,.2f}\n"
        )
    
    # Generate the base report
    report = (
        "Portfolio Newsletter Report\n"
        "---------------------------\n"
        f"Portfolio Metrics (Calculated from {first_metric_day} to {last_day})\n"
        f"- Beta: {beta:.2f}\n"
        f"- Alpha: {alpha:.2f}\n"
        f"- Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"- Sortino Ratio: {sortino_ratio:.2f}\n"
        f"- Value at Risk: {var:.2f}\n"
        f"- Max Drawdown: {max_drawdown*100:.2f}%\n"
        f"- Dividend Yield: {dividend_yield:.2%}\n"
        "\n"
        "---------------------------\n"
        f"Returns (Calculated from {start_date_report} to {last_day})\n"
        f"Portfolio Initial Value: {portfolio_initial_value:,.2f}\n"
        f"Portfolio Final Value: {portfolio_final_value:,.2f}\n"
        f"Portfolio Return: {portfolio_return:.2f}%\n"
        f"Market Return: {market_return:.2f}%\n"
        "\n"
        "---------------------------\n"
        "Individual Stock Performance:\n"
        f"{stock_details}"
    )
    
    # Use OpenAI to generate a summary or commentary if an API key is provided
    if openai_key is not None:
        try:
            openai.api_key = openai_key  # Set the API key
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": f"Here is a financial summary:\n\n{report}\n\nPlease provide a brief commentary on the performance (NOT METRICS) of the portfolio, from {start_date_report} to {last_day} OF THE PORTFOLIO. Response need to be ready to append to a monthly newsletter report."}
                ]
            )
            ai_commentary = completion.choices[0].message.content
            final_report = report + "\n\nGeneral Comments:\n" + ai_commentary
        except Exception as e:
            final_report = report + "\n\n(Note: AI Commentary could not be generated due to an error: " + str(e) + ")"
        return final_report

    return report 