# yahoo_provider.py

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from .data_provider import DataProvider
from .cache_manager import CacheManager  # Assuming your CacheManager is saved in this file


class YahooFinanceProvider(DataProvider):
    """ 
    Manage yfinance integration. 

    """

    def __init__(self, cache_dir: str = "cache", min_rows: int = 10, base_currency: str = "USD"):
        """
        cache_dir : directory where cache is stored.
        min_rows : minimum number of rows of obs. a ticker need to have. 
        base_currency : your portfolio base currency for conversion checks.
        """
        self.min_rows = min_rows
        self.base_currency = base_currency
        self.CacheManager = CacheManager(cache_dir).get_data()
       

    def fetch_price_history(self, tickers: list[str], start: str = None, end: str = None) -> dict[str, pd.DataFrame]:
        """
        Fetch price history, first from cache with date range check,
        then download missing date ranges at the edges only.
        Merge downloaded data with existing cache and update cache and metadata.

        Params
        ------
        tickers : list of ticker symbols to fetch
        start : str or None, start date for data
        end : str or None, end date for data

        Returns
        -------
        dict of ticker -> pd.DataFrame with price history (Close)
        """
        result = {}

        # Convert start/end to timestamps
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) if end else pd.Timestamp(datetime.today())

        for ticker in tickers:
            cached_df = self.CacheManager._get_from_cache(ticker)
            metadata = self.CacheManager.metadata

            if cached_df is None or cached_df.empty:
                # No cache, full download
                print(f"Downloading full history for {ticker}")
                df = yf.download(ticker, start=start_dt, end=end_dt, auto_adjust=True)['Close'].dropna()
                if not df.empty:
                    metadata = {
                        "base_currency": self.CacheManager.get_currency(ticker),
                        "version": self.CacheManager._get_latest_version(ticker),
                        "cached_at": pd.Timestamp.now().isoformat(),
                        "start_date": df.index.min().isoformat(),
                        "end_date": df.index.max().isoformat(),
                    }
                    self.CacheManager._add_to_cache(ticker, df)
                    self.CacheManager._add_to_metadata(ticker, metadata)
                    result[ticker] = df
                else:
                    print(f"No data downloaded for {ticker}")
                    result[ticker] = pd.DataFrame()
            else:
                # Cache exists, check if need to download edges
                cached_start = cached_df.index.min()
                cached_end = cached_df.index.max()

                to_concat = []

                # Download earlier missing part if any
                if start_dt and start_dt < cached_start:
                    print(f"Downloading earlier data for {ticker} from {start_dt.date()} to {(cached_start - pd.Timedelta(days=1)).date()}")
                    early_data = yf.download(
                        ticker,
                        start=start_dt,
                        end=cached_start - pd.Timedelta(days=1),
                        auto_adjust=True,
                    )['Close'].dropna()
                    if not early_data.empty:
                        to_concat.append(early_data)

                to_concat.append(cached_df)

                # Download later missing part if any
                if end_dt and end_dt > cached_end:
                    print(f"Downloading later data for {ticker} from {(cached_end + pd.Timedelta(days=1)).date()} to {end_dt.date()}")
                    late_data = yf.download(
                        ticker,
                        start=cached_end + pd.Timedelta(days=1),
                        end=end_dt,
                        auto_adjust=True,
                    )['Close'].dropna()
                    if not late_data.empty:
                        to_concat.append(late_data)

                # Merge all parts, sort and drop duplicates
                full_df = pd.concat(to_concat).sort_index().drop_duplicates()

                # Save updated cache and metadata
                metadata = {
                    "base_currency": self.get_currency(ticker),
                    "cached_at": pd.Timestamp.now().isoformat(),
                    "start_date": full_df.index.min().isoformat(),
                    "end_date": full_df.index.max().isoformat(),
                }
                self.save_to_cache(ticker, full_df, ext="csv", metadata=metadata)
                result[ticker] = full_df

        return result

    def get_currency(self, ticker: str) -> str:
        """Fetch the currency of the given ticker. Defaults to USD."""
        try:
            ticker_info = yf.Ticker(ticker).info
            return ticker_info.get('currency', 'USD')
        except Exception:
            return 'USD'

    def fetch_info(self, ticker: str) -> dict:
        try:
            return yf.Ticker(ticker).info
        except Exception as e:
            print(f"Failed to fetch info for {ticker}: {e}")
            return {}

    def currency_conversion(self, currency: str, base_currency: str) -> bool:
        """Check if a currency conversion is needed."""
        return currency != base_currency