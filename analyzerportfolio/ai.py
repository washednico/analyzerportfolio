import openai
import pandas as pd
from analyzerportfolio.metrics import calculate_portfolio_metrics
import threading
import time
import yfinance as yf
import openai
from datetime import datetime, timedelta

def newsletter_report(
    price_df: pd.DataFrame,
    tickers: list[str],
    investments: list[float],
    start_date_report: str,
    market_ticker: str = '^GSPC',
    risk_free_rate: float = 0.01,
    openai_key: str = None
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
    
    metrics_data = calculate_portfolio_metrics(
        price_df, 
        tickers, 
        investments, 
        start_date_report=start_date_report, 
        investment_at_final_date=True, 
        market_ticker=market_ticker, 
        risk_free_rate=risk_free_rate
    )

    # Generate the base report
    report = (
        "Portfolio Newsletter Report\n"
        "---------------------------\n"
        f"Portfolio Metrics (Calculated from {metrics_data['first_metric_day']} to {metrics_data['last_day']})\n"
        f"- Beta: {metrics_data['beta']:.2f}\n"
        f"- Alpha: {metrics_data['alpha']:.2f}\n"
        f"- Sharpe Ratio: {metrics_data['sharpe_ratio']:.2f}\n"
        f"- Sortino Ratio: {metrics_data['sortino_ratio']:.2f}\n"
        f"- Value at Risk: {metrics_data['var']:.2f}\n"
        f"- Max Drawdown: {metrics_data['max_drawdown']*100:.2f}%\n"
        f"- Dividend Yield: {metrics_data['dividend_yield']:.2%}\n"
        "\n"
        "---------------------------\n"
        f"Returns (Calculated from {start_date_report} to {metrics_data['last_day']})\n"
        f"Portfolio Initial Value: {metrics_data['portfolio_initial_value']:,.2f}\n"
        f"Portfolio Final Value: {metrics_data['portfolio_final_value']:,.2f}\n"
        f"Portfolio Return: {metrics_data['portfolio_return']:.2f}%\n"
        f"Market Return: {metrics_data['market_return']:.2f}%\n"
        "\n"
        "---------------------------\n"
        "Individual Stock Performance:\n"
    )

    for stock in metrics_data['stock_details']:
        report += (
            f"{stock['ticker']}:\n"
            f"  - Initial Value: {stock['initial_value']:,.2f}\n"
            f"  - Final Value: {stock['final_value']:,.2f}\n"
            f"  - Return: {stock['return']:.2f}%\n"
            f"  - Surplus/Deficit: {stock['surplus_or_deficit']:,.2f}\n"
        )
    
    # Use OpenAI to generate a summary or commentary if an API key is provided
    if openai_key is not None:
        try:
            openai.api_key = openai_key
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": f"Here is a financial summary:\n\n{report}\n\nPlease provide a brief commentary on the performance (NOT METRICS) of the portfolio, from {start_date_report} to {metrics_data['last_day']}. Response needs to be ready to append to a newsletter report."}
                ]
            )
            ai_commentary = completion.choices[0].message.content
            final_report = report + "\n\nGeneral Comments:\n" + ai_commentary
        except Exception as e:
            final_report = report + "\n\n(Note: AI Commentary could not be generated due to an error: " + str(e) + ")"
        return final_report

    return report

def get_suggestion(
    price_df: pd.DataFrame,
    tickers: list[str],
    investments: list[float], 
    openai_key: str,
    market_ticker: str = '^GSPC',
    risk_free_rate: float = 0.01
) -> str:
    
    """Receive portfolio suggestions from ChatGPT.
    
    Parameters:
    price_df (pd.DataFrame): A DataFrame containing the historical price data of the stocks in the portfolio.
    tickers (list[str]): A list of stock tickers in the portfolio.
    investments (list[float]): A list of investment amounts corresponding to each stock in the portfolio.
    openai_key (str): The OpenAI API key to use for generating the report.
    market_ticker (str, optional): The ticker symbol of the market index to compare the portfolio against. Default is '^GSPC' (S&P 500).
    risk_free_rate (float, optional): The risk-free rate used in calculating the Sharpe and Sortino ratios. Default is 0.01 (1%).
    
    Returns:
    str: The generated newsletter report as a string.
    """
    
    metrics_data = calculate_portfolio_metrics(
        price_df, 
        tickers, 
        investments, 
        start_date_report=None, 
        investment_at_final_date=False, 
        market_ticker=market_ticker, 
        risk_free_rate=risk_free_rate
    )

    # Generate the base report
    report = (
        "Portfolio Improvement Suggestions\n"
        "---------------------------\n"
        f"Portfolio Metrics (Calculated from {metrics_data['first_metric_day']} to {metrics_data['last_day']})\n"
        f"- Beta: {metrics_data['beta']:.2f}\n"
        f"- Alpha: {metrics_data['alpha']:.2f}\n"
        f"- Sharpe Ratio: {metrics_data['sharpe_ratio']:.2f}\n"
        f"- Sortino Ratio: {metrics_data['sortino_ratio']:.2f}\n"
        f"- Value at Risk: {metrics_data['var']:.2f}\n"
        f"- Max Drawdown: {metrics_data['max_drawdown']*100:.2f}%\n"
        f"- Dividend Yield: {metrics_data['dividend_yield']:.2%}\n"
        "\n"
        "---------------------------\n"
        "Individual Stock Performance:\n"
    )

    for stock in metrics_data['stock_details']:
        report += (
            f"{stock['ticker']}:\n"
            f"  - Initial Value: {stock['initial_value']:,.2f}\n"
            f"  - Final Value: {stock['final_value']:,.2f}\n"
            f"  - Return: {stock['return']:.2f}%\n"
            f"  - Surplus/Deficit: {stock['surplus_or_deficit']:,.2f}\n"
        )
    
    # Use OpenAI to generate suggestions if an API key is provided
    if openai_key is not None:
        try:
            openai.api_key = openai_key
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": f"Here are the portfolio metrics and returns:\n\n{report}\n\nBased on this information, please provide suggestions to improve the portfolio."}
                ]
            )
            ai_suggestions = completion.choices[0].message.content
            final_report = report + "\n\nSuggestions for Improvement:\n" + ai_suggestions
        except Exception as e:
            final_report = report + "\n\n(Note: Suggestions could not be generated due to an error: " + str(e) + ")"
        return final_report

    return report

def monitor_news(
    tickers: list[str],
    openai_key: str = None,
    delay: int = 3600,  # Delay in seconds (default 1 hour)
    loop_forever: bool = True
) -> None:
    """
    Monitor news for the given tickers, optionally analyze it using OpenAI GPT, and run the process in a separate thread.

    Parameters:
    tickers (list[str]): A list of stock tickers to monitor.
    openai_key (str, optional): The OpenAI API key for analyzing news importance and sentiment. Default is None.
    delay (int, optional): The delay between checks in seconds. Default is 3600 seconds (1 hour).
    loop_forever (bool, optional): Whether to run the thread forever or stop when the main one is terminated. Default is True.
    """

    def fetch_and_analyze_news():
        """Fetch and analyze news for the tickers."""
        last_checked = datetime.now() - timedelta(seconds=delay)  # Fetch news from the last delay period

        while True:
            for ticker in tickers:
                # Fetch news from Yahoo Finance
                news = yf.Ticker(ticker).news

                for article in news:
                    pub_date = datetime.fromtimestamp(article['providerPublishTime'])
                    if pub_date > last_checked:
                        title = article['title']
                        link = article['link']
                        summary = article.get('summary', '')


                        if openai_key:
                            # Use OpenAI to evaluate importance and sentiment
                            try:
                                openai.api_key = openai_key
                                completion = openai.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are a financial analyst."},
                                        {"role": "user", "content": f"Evaluate the following news for {ticker}:\n\nTitle: {title}\n\nSummary: {summary}\n\nProvide an importance score (1-10) and the sentiment (positive, neutral, or negative)."}
                                    ]
                                )
                                evaluation = completion.choices[0].message.content
                            
                            except Exception as e:
                                evaluation = "\n\n(Note: Suggestions could not be generated due to an error: " + str(e) + ")"

                        else:
                            # Basic sentiment analysis using TextBlob

                            evaluation = "Not analyzed"

                        print(f"--------------portfolioanalyzer--------------\nRetrieved news for {ticker} at {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\nTitle: {title}\nLink: {link}\n{evaluation}\n")

            last_checked = datetime.now()
            time.sleep(delay)

    # Start the monitoring in a separate thread
    thread = threading.Thread(target=fetch_and_analyze_news, daemon= not loop_forever)
    thread.start()
