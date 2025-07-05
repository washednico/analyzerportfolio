from collections.abc import Sequence  # more flexible than just list
from asset.asset import AssetBase
import pandas as pd 


class Portfolio:
    """
    Represents an investment portfolio composed of multiple assets.

    Attributes:
        name (str): Name of the portfolio.
        base_currency (str): The currency in which the portfolio is valued.
        assets (Sequence[AssetBase]): List of AssetBase-derived objects in the portfolio.
        investments (Sequence[float]): List of invested amounts corresponding to each asset.
        market_benchmark (AssetBase | None): Optional benchmark asset for comparison.
        rebalance_period_days (int | None): Optional rebalancing period in days.
        history (pd.DataFrame): Historical portfolio returns computed from weighted asset returns.
    """
    def __init__(
        self,
        name: str,
        base_currency: str,
        assets: Sequence[AssetBase],
        investments: Sequence[float],
        market_benchmark: AssetBase | None = None,
        rebalance_period_days: int | None = None
    ):
        """
        Initializes the Portfolio object.

        Args:
            name (str): Name of the portfolio.
            base_currency (str): Base currency for valuation.
            assets (Sequence[AssetBase]): List of assets in the portfolio.
            investments (Sequence[float]): Amounts invested in each asset.
            market_benchmark (AssetBase | None): Optional benchmark for comparison.
            rebalance_period_days (int | None): Optional rebalancing interval in days.

        Raises:
            ValueError: If the lengths of assets and investments don't match.
        """

        if len(assets) != len(investments):
            raise ValueError("Assets and investments must have the same length.")

        self.name = name
        self.base_currency = base_currency
        self.assets = assets
        self.investments = investments
        self.market_benchmark = market_benchmark
        self.rebalance_period_days = rebalance_period_days

        self.history: pd.DataFrame = self._construct_portfolio_history()

    def _construct_portfolio_history(self) -> pd.DataFrame:
        """
        Constructs the historical returns of the portfolio by aggregating
        weighted returns of individual assets.

        Returns:
            pd.DataFrame: DataFrame with time series of portfolio returns.
        
        Raises:
            ValueError: If any asset lacks historical data.
        """
        
        # Aggregate historical returns weighted by initial investments
        histories = []
        weights = [inv / sum(self.investments) for inv in self.investments]

        for asset, weight in zip(self.assets, weights):
            if asset.historical_data is None:
                raise ValueError(f"Historical data not loaded for asset {asset.ticker}")
            returns = asset.get_returns()
            weighted_returns = returns * weight
            histories.append(weighted_returns.rename(asset.ticker))

        # Combine into a single DataFrame
        portfolio_returns = pd.concat(histories, axis=1).sum(axis=1).to_frame(name="portfolio_return")

        # Optionally add benchmark comparison
        if self.market_benchmark and self.market_benchmark.historical_data is not None:
            portfolio_returns["benchmark_return"] = self.market_benchmark.get_returns()

        return portfolio_returns