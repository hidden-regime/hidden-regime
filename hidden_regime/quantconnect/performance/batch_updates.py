"""
Batch regime update system for multi-asset portfolios.

This module provides efficient batch updating of regimes for multiple assets,
reducing overhead and improving performance for multi-asset strategies.
"""

from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


class BatchRegimeUpdater:
    """
    Batch updater for multiple asset regime detections.

    Efficiently updates regimes for multiple assets in parallel,
    minimizing redundant operations.
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_parallel: bool = True,
    ):
        """
        Initialize batch updater.

        Args:
            max_workers: Maximum parallel workers
            use_parallel: Whether to use parallel processing
        """
        self.max_workers = max_workers
        self.use_parallel = use_parallel
        self._executor = None

    def batch_update(
        self,
        assets: List[str],
        data_dict: Dict[str, pd.DataFrame],
        pipeline_dict: Dict[str, Any],
        update_func: Any,
    ) -> Dict[str, Any]:
        """
        Update regimes for multiple assets in batch.

        Args:
            assets: List of asset tickers
            data_dict: Dict mapping tickers to DataFrames
            pipeline_dict: Dict mapping tickers to pipelines
            update_func: Function to update single asset regime

        Returns:
            Dict mapping tickers to update results
        """
        if self.use_parallel and len(assets) > 1:
            return self._parallel_update(
                assets, data_dict, pipeline_dict, update_func
            )
        else:
            return self._sequential_update(
                assets, data_dict, pipeline_dict, update_func
            )

    def _sequential_update(
        self,
        assets: List[str],
        data_dict: Dict[str, pd.DataFrame],
        pipeline_dict: Dict[str, Any],
        update_func: Any,
    ) -> Dict[str, Any]:
        """
        Sequential regime updates.

        Args:
            assets: List of tickers
            data_dict: Data for each ticker
            pipeline_dict: Pipelines for each ticker
            update_func: Update function

        Returns:
            Update results
        """
        results = {}

        for ticker in assets:
            if ticker not in data_dict or ticker not in pipeline_dict:
                continue

            try:
                result = update_func(
                    ticker=ticker,
                    data=data_dict[ticker],
                    pipeline=pipeline_dict[ticker],
                )
                results[ticker] = result
            except Exception as e:
                results[ticker] = {"error": str(e)}

        return results

    def _parallel_update(
        self,
        assets: List[str],
        data_dict: Dict[str, pd.DataFrame],
        pipeline_dict: Dict[str, Any],
        update_func: Any,
    ) -> Dict[str, Any]:
        """
        Parallel regime updates using ThreadPoolExecutor.

        Args:
            assets: List of tickers
            data_dict: Data for each ticker
            pipeline_dict: Pipelines for each ticker
            update_func: Update function

        Returns:
            Update results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {}
            for ticker in assets:
                if ticker not in data_dict or ticker not in pipeline_dict:
                    continue

                future = executor.submit(
                    update_func,
                    ticker=ticker,
                    data=data_dict[ticker],
                    pipeline=pipeline_dict[ticker],
                )
                future_to_ticker[future] = ticker

            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results[ticker] = result
                except Exception as e:
                    results[ticker] = {"error": str(e)}

        return results

    def batch_collect_data(
        self,
        assets: List[str],
        data_adapters: Dict[str, Any],
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all adapters in batch.

        Args:
            assets: List of tickers
            data_adapters: Dict of data adapters

        Returns:
            Dict mapping tickers to DataFrames
        """
        data_dict = {}

        for ticker in assets:
            if ticker not in data_adapters:
                continue

            adapter = data_adapters[ticker]
            try:
                if adapter.is_ready():
                    data_dict[ticker] = adapter.to_dataframe()
            except Exception:
                # Skip assets with data issues
                continue

        return data_dict


class OptimizedMultiAssetUpdater:
    """
    Optimized updater for multi-asset strategies.

    Combines batch processing, caching, and intelligent scheduling
    to minimize computation time.
    """

    def __init__(
        self,
        assets: List[str],
        update_schedule: str = "staggered",
        batch_size: int = 4,
    ):
        """
        Initialize optimized updater.

        Args:
            assets: List of asset tickers
            update_schedule: 'staggered', 'synchronized', or 'on_demand'
            batch_size: Assets per batch
        """
        self.assets = assets
        self.update_schedule = update_schedule
        self.batch_size = batch_size

        # Track when each asset was last updated
        self.last_update: Dict[str, int] = {asset: 0 for asset in assets}
        self.update_counter = 0

    def get_assets_to_update(self) -> List[str]:
        """
        Determine which assets need updating based on schedule.

        Returns:
            List of tickers to update
        """
        self.update_counter += 1

        if self.update_schedule == "synchronized":
            # Update all assets together
            return self.assets

        elif self.update_schedule == "staggered":
            # Stagger updates across days to spread load
            # Each asset updates every N days where N = num_assets
            assets_to_update = []
            for i, asset in enumerate(self.assets):
                if (self.update_counter + i) % len(self.assets) == 0:
                    assets_to_update.append(asset)
            return assets_to_update

        elif self.update_schedule == "on_demand":
            # Only update assets that haven't been updated recently
            # (determined externally, return all by default)
            return self.assets

        return self.assets

    def create_batches(self, assets: List[str]) -> List[List[str]]:
        """
        Create batches of assets for parallel processing.

        Args:
            assets: List of all assets to update

        Returns:
            List of asset batches
        """
        batches = []
        for i in range(0, len(assets), self.batch_size):
            batch = assets[i : i + self.batch_size]
            batches.append(batch)
        return batches


def create_batch_updater(
    assets: List[str],
    max_workers: int = 4,
    use_parallel: bool = True,
    update_schedule: str = "staggered",
) -> BatchRegimeUpdater:
    """
    Factory function to create batch regime updater.

    Args:
        assets: List of asset tickers
        max_workers: Maximum parallel workers
        use_parallel: Whether to use parallel processing
        update_schedule: Update scheduling strategy

    Returns:
        BatchRegimeUpdater instance
    """
    return BatchRegimeUpdater(
        max_workers=max_workers,
        use_parallel=use_parallel,
    )
