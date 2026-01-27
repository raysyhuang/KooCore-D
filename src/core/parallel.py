"""
Parallel Processing Module

Provides parallelized screening for improved performance.
Uses ThreadPoolExecutor for I/O-bound tasks (API calls, data fetching).
"""

from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, Optional, TypeVar
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ParallelResult:
    """Result container for parallel processing."""
    success: list[tuple[str, Any]]  # (key, result) pairs
    failed: list[tuple[str, Exception]]  # (key, error) pairs
    duration_seconds: float
    
    @property
    def success_count(self) -> int:
        return len(self.success)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed)
    
    @property
    def total_count(self) -> int:
        return self.success_count + self.failed_count
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count


def parallel_map(
    func: Callable[[str], T],
    items: list[str],
    max_workers: int = 8,
    timeout_per_item: float = 30.0,
    show_progress: bool = True,
    progress_interval: int = 50,
) -> ParallelResult:
    """
    Apply function to items in parallel with error handling.
    
    Args:
        func: Function to apply to each item
        items: List of items (typically ticker symbols)
        max_workers: Maximum parallel threads
        timeout_per_item: Timeout per item in seconds
        show_progress: Whether to print progress
        progress_interval: How often to show progress (every N items)
    
    Returns:
        ParallelResult with successful and failed results
    """
    if not items:
        return ParallelResult(success=[], failed=[], duration_seconds=0.0)
    
    success = []
    failed = []
    start_time = time.time()
    total = len(items)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(func, item): item 
            for item in items
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            completed += 1
            
            try:
                result = future.result(timeout=timeout_per_item)
                success.append((item, result))
            except Exception as e:
                failed.append((item, e))
                logger.debug(f"Failed processing {item}: {e}")
            
            # Progress indicator
            if show_progress and (completed % progress_interval == 0 or completed == total):
                pct = (completed / total) * 100
                print(f"  Progress: {completed}/{total} ({pct:.1f}%) | Success: {len(success)}, Failed: {len(failed)}", end="\r")
    
    if show_progress:
        print()  # New line after progress
    
    duration = time.time() - start_time
    
    return ParallelResult(
        success=success,
        failed=failed,
        duration_seconds=duration
    )


def parallel_batch_map(
    func: Callable[[list[str]], dict[str, Any]],
    items: list[str],
    batch_size: int = 50,
    max_workers: int = 4,
    timeout_per_batch: float = 120.0,
    show_progress: bool = True,
) -> ParallelResult:
    """
    Apply batch function to items in parallel.
    
    Useful for APIs that support batch requests (like yfinance).
    
    Args:
        func: Function that takes a list of items and returns dict[item, result]
        items: List of items to process
        batch_size: Size of each batch
        max_workers: Maximum parallel batch threads
        timeout_per_batch: Timeout per batch in seconds
        show_progress: Whether to print progress
    
    Returns:
        ParallelResult with combined results from all batches
    """
    if not items:
        return ParallelResult(success=[], failed=[], duration_seconds=0.0)
    
    # Split into batches
    batches = [
        items[i:i + batch_size] 
        for i in range(0, len(items), batch_size)
    ]
    
    success = []
    failed = []
    start_time = time.time()
    total_batches = len(batches)
    completed_batches = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(func, batch): batch 
            for batch in batches
        }
        
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            completed_batches += 1
            
            try:
                results = future.result(timeout=timeout_per_batch)
                if isinstance(results, dict):
                    for item, result in results.items():
                        if result is not None:
                            success.append((item, result))
                        else:
                            failed.append((item, ValueError("None result")))
                else:
                    # Assume it's a list of tuples
                    success.extend(results)
            except Exception as e:
                # Mark all items in batch as failed
                for item in batch:
                    failed.append((item, e))
                logger.warning(f"Batch failed ({len(batch)} items): {e}")
            
            if show_progress:
                pct = (completed_batches / total_batches) * 100
                print(f"  Batches: {completed_batches}/{total_batches} ({pct:.1f}%)", end="\r")
    
    if show_progress:
        print()
    
    duration = time.time() - start_time
    
    return ParallelResult(
        success=success,
        failed=failed,
        duration_seconds=duration
    )


class ParallelScreener:
    """
    Parallel screening wrapper for ticker analysis.
    
    Provides a high-level interface for parallel ticker screening
    with automatic retry, caching, and progress tracking.
    """
    
    def __init__(
        self,
        max_workers: int = 8,
        batch_size: int = 50,
        retry_failed: bool = True,
        max_retries: int = 2,
    ):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.retry_failed = retry_failed
        self.max_retries = max_retries
    
    def screen_tickers(
        self,
        tickers: list[str],
        screen_func: Callable[[str], Optional[dict]],
        filter_func: Optional[Callable[[dict], bool]] = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Screen tickers in parallel with optional filtering.
        
        Args:
            tickers: List of tickers to screen
            screen_func: Function that screens a single ticker
            filter_func: Optional function to filter results (returns True to keep)
        
        Returns:
            Tuple of (passed_results, dropped_results)
        """
        logger.info(f"Screening {len(tickers)} tickers with {self.max_workers} workers...")
        
        # Initial parallel screening
        result = parallel_map(
            func=screen_func,
            items=tickers,
            max_workers=self.max_workers,
            show_progress=True,
        )
        
        # Retry failed items
        if self.retry_failed and result.failed:
            failed_items = [item for item, _ in result.failed]
            logger.info(f"Retrying {len(failed_items)} failed items...")
            
            for attempt in range(self.max_retries):
                retry_result = parallel_map(
                    func=screen_func,
                    items=failed_items,
                    max_workers=max(2, self.max_workers // 2),  # Use fewer workers for retries
                    show_progress=False,
                )
                
                # Add recovered results
                result.success.extend(retry_result.success)
                
                # Update failed list
                failed_items = [item for item, _ in retry_result.failed]
                
                if not failed_items:
                    break
            
            # Update final failed count
            result = ParallelResult(
                success=result.success,
                failed=[(item, Exception("Max retries exceeded")) for item in failed_items],
                duration_seconds=result.duration_seconds
            )
        
        logger.info(
            f"Screening complete: {result.success_count} success, "
            f"{result.failed_count} failed ({result.duration_seconds:.1f}s)"
        )
        
        # Separate passed and dropped
        passed = []
        dropped = []
        
        for ticker, data in result.success:
            if data is None:
                dropped.append({"ticker": ticker, "reason": "no_data"})
                continue
            
            if filter_func is not None:
                if filter_func(data):
                    passed.append(data)
                else:
                    dropped.append({"ticker": ticker, "reason": "filtered", "data": data})
            else:
                passed.append(data)
        
        # Add failed items to dropped
        for ticker, error in result.failed:
            dropped.append({"ticker": ticker, "reason": str(error)})
        
        return passed, dropped
    
    def fetch_data_parallel(
        self,
        tickers: list[str],
        fetch_func: Callable[[list[str]], dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Fetch data for multiple tickers in parallel batches.
        
        Args:
            tickers: List of tickers
            fetch_func: Batch fetch function
        
        Returns:
            Dict mapping ticker -> data
        """
        result = parallel_batch_map(
            func=fetch_func,
            items=tickers,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            show_progress=True,
        )
        
        return dict(result.success)


def create_ticker_processor(
    data_dict: dict,
    params: dict,
    process_func: Callable,
) -> Callable[[str], Optional[dict]]:
    """
    Factory function to create a ticker processor for parallel execution.
    
    Args:
        data_dict: Pre-fetched OHLCV data keyed by ticker
        params: Screening parameters
        process_func: Function(ticker, df, params) -> Optional[dict]
    
    Returns:
        Function that can be passed to parallel_map
    """
    def processor(ticker: str) -> Optional[dict]:
        df = data_dict.get(ticker)
        if df is None or df.empty:
            return None
        return process_func(ticker, df, params)
    
    return processor
