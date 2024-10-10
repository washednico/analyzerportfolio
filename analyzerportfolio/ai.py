import datetime
import openai
import threading
import time
import yfinance as yf
from typing import Union, List, Dict



def monitor_news(
    portfolio: Union[Dict, List[Dict]],
    openai_key: str = None,
    delay: int = 3600,  # Delay in seconds (default 1 hour)
    loop_forever: bool = True
) -> None:
    """
    Monitor news for the given portfolios, optionally analyze it using OpenAI GPT, and run the process in a separate thread.
    Monitors are retrieved from Yahoo Finance and analyzed using OpenAI GPT-4 if an API key is provided.

    Parameters:
    portfolio (Union[Dict, List[Dict]]): The portfolio(s) to monitor. Each portfolio should be a dictionary created using the create_portfolio function.
    openai_key (str, optional): The OpenAI API key for analyzing news importance and sentiment. Default is None.
    delay (int, optional): The delay between checks in seconds. Default is 3600 seconds (1 hour).
    loop_forever (bool, optional): Whether to run the thread forever or stop when the main one is terminated. Default is True.
    """

    def fetch_and_analyze_news():
        """Fetch and analyze news for the tickers."""
        last_checked = datetime.now() - datetime.timedelta(seconds=delay)  # Fetch news from the last delay period

        #Filtering tickers to avoid redudant check of the same ticker
        portfolio = [portfolio] if isinstance(portfolio, dict) else portfolio
        filtered_tickers = []
        for i in range(len(portfolio)):
            tickers = portfolio[i]['tickers']
            for ticker in tickers:
                if ticker not in filtered_tickers:
                    filtered_tickers.append(ticker)


        while True:
            for ticker in filtered_tickers:
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

