# data_provider.py

from abc import ABC, abstractmethod
import os
import pandas as pd
from collections.abc import Sequence
import yfinance as yf

CACHE_DIR = "cache"

class DataProvider(ABC):
    @abstractmethod
    def fetch_price_history(self, tickers: list[str], start: str = None, end: str = None) -> dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def fetch_info(self, ticker: str) -> dict:
        pass