# AnalyzerPortfolio Cache System
#
#
#

import os
import json
import pandas as pd
import logging

class CacheManager:
    """ AnalyzerPortfolio Cache System """

    def __init__(self, cache_dir:str="cache", metadata_path:str = None, ext:str="csv") -> None:
        """ 
        Initialize the cache manager. 

        - cache_dir (str): Cache directory
        - ext (str): The exstenstion of cache files (Default is "csv")
        """
        self.cache_dir = cache_dir
        self.ext = ext.lower()
        
        # Make cache dir iff it does not exist yet
        os.makedirs(cache_dir, exist_ok=True)

        # Load metadata via the specified path (Default is cache_dir/metadata.json)
        self.metadata_path = metadata_path or os.path.join(cache_dir, "metadata.json")

        # Load metadata and initialize in-memory cache
        self.metadata = self._load_metadata()

        # Define empty cache dictonary to store cache (CACHING IN RAM)
        self.cache = {} # Store {ticker: pd.DataFrame}
        
    def _load_metadata(self) -> dict | None : 
        """Load metadata into memory if it exists and is valid, otherwise create an empty one"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Metadata file '{self.metadata_path}' is invalid or corrupted. Resetting to empty metadata. Error: {e}")
                return {}
        else:
            self.metadata = {}
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=4)
                return {}

    def _get_versioned_path(self, key: str) -> str:
        version = self._get_latest_version(key)
        versioned = f"_{version}" if version > 0 else ""
        return os.path.join(self.cache_dir, f"{key}{versioned}.{self.ext}")
    
    def _get_latest_version(self, key: str) -> int | None:
        meta = self.metadata.get(key)
        return meta.get("version", 0) if meta else 0
    
    def _get_base_currency(self, key: str) -> str:
        meta = self.metadata.get(key)
        return meta.get("base_currency", "USD") if meta else "USD"
    
    def _add_to_cache(self, key: str, df: pd.DataFrame) -> None:
        """Add a DataFrame to in-memory cache."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Only pandas DataFrames are supported.")
        self.cache[key.upper()] = df

    def _get_from_cache(self, key: str) -> pd.DataFrame | None:
        """Retrieve a DataFrame from in-memory cache."""
        return self.cache.get(key.upper())
    
    def _add_to_metadata(self, key: str, metadata: dict) -> None:
        ticker = key.upper()
        if ticker in self.metadata:
            self.metadata[ticker].update(metadata)
        else:
            self.metadata[ticker] = metadata

    # ---- ---- ---- ---- #   
    
    def save_to_cache(self, key:str, df:pd.DataFrame, metadata: dict) -> None:
        self._add_to_cache(key, df)
        self._add_to_metadata(key, metadata)
        
    def read_file_data(self, key:str) -> json:
        """ 
        Get file metadata. 
        It expected to be in the cache directory.

        Cache file are assumed to be store as following
        cache/AAPL_version.csv
        In case of no version 
        cache/AAPL.csv

        Params
        -----
        key(str)
        """
        #TODO: Exception management 

        path = self._get_versioned_path(key)
        
        if self.ext == "csv":
            return pd.read_csv(path, index_col=0, parse_dates=True)
        elif self.ext == "xlsx":
            return pd.read_excel(path, index_col=0, parse_dates=True)
        else:
            print(f"No cache found for {key}, check format")
            return None
    
    def save_metadata(self) -> None:
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save metadata to {self.metadata_path}. Error: {e}")

    def update_metadata_from_cache(self) -> None:
        """Update metadata to reflect the current in-memory cache state."""
        for ticker, df in self.cache.items():
            self.metadata[ticker] = {
                #"rows": len(df),
                #"columns": list(df.columns),
                "end_date": df.index.max().isoformat(),
                "start_date": df.index.min().isoformat(),
                "base_currency": self._get_base_currency(ticker) ,
                "version": self._get_latest_version(ticker),
                "cached_at": pd.Timestamp.now().isoformat(),
            }
        self.save_metadata()
        #TODO: save to disk 
        # self.write_file_data(key, df)  # Optional: persist to disk

    def get_cache(self, keys:list) -> dict:
        """ 
        Get ticker(s) data from specified cache folder.
        Currently csv and xlsx are supported. 

        Params
        ------
        tickers (list): List of asset tickers

        Returns
        -------
        
        """
        #TODO: 
        #  - Not sure if define at level class (self) is really needed 
        #  - Add a tickers check (something like download_data/validate_tickers in utils.py)

        for key in keys: 
            df = self.read_file_data(key)
            self._add_to_cache(key, df)

        
        # Update metadata
        self.update_metadata_from_cache()
        return self.cache