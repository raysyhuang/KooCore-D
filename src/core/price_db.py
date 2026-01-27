"""
Permanent Price Database

Stores historical price data permanently - no expiration.
Historical data is immutable: once a trading day closes, that data never changes.

Design principles:
1. Historical data (completed trading days) is stored permanently
2. Polygon.io is primary source, Yahoo Finance is secondary/verification
3. Data is stored per-ticker per-date for efficient queries
4. Supports backtesting, analysis, and model training without re-downloading
"""

from __future__ import annotations

import os
import sqlite3
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Database path
DEFAULT_DB_PATH = "data/prices.db"


class PriceDatabase:
    """
    Permanent storage for historical price data.
    
    Features:
    - No TTL/expiration for historical data
    - Efficient date-range queries
    - Data completeness tracking
    - Source tracking (polygon/yfinance)
    - Verification status
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Main prices table - one row per ticker per date
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adjusted_close REAL,
                    source TEXT DEFAULT 'polygon',
                    verified INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            # Index for efficient date range queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_ticker_date 
                ON daily_prices(ticker, date)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_date 
                ON daily_prices(date)
            """)
            
            # Metadata table - tracks data completeness per ticker
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_metadata (
                    ticker TEXT PRIMARY KEY,
                    first_date TEXT,
                    last_date TEXT,
                    total_records INTEGER DEFAULT 0,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    data_complete INTEGER DEFAULT 0
                )
            """)
            
            # Download log - track what we've downloaded
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    records_downloaded INTEGER,
                    success INTEGER,
                    error_message TEXT,
                    downloaded_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_prices(
        self,
        ticker: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """
        Get price data for a ticker in a date range.
        
        Returns DataFrame with OHLCV columns, indexed by date.
        Returns empty DataFrame if no data found.
        """
        start_str = str(start_date)[:10]
        end_str = str(end_date)[:10]
        
        with self._get_connection() as conn:
            query = """
                SELECT date, open, high, low, close, volume, adjusted_close
                FROM daily_prices
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn, params=(ticker, start_str, end_str))
        
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.index.name = 'Date'
        
        # Rename columns to standard format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adjusted_close': 'Adj Close'
        })
        
        return df
    
    def store_prices(
        self,
        ticker: str,
        df: pd.DataFrame,
        source: str = "polygon",
        verified: bool = False,
    ) -> int:
        """
        Store price data for a ticker.
        
        Args:
            ticker: Stock symbol
            df: DataFrame with OHLCV data, indexed by date
            source: Data source ('polygon', 'yfinance')
            verified: Whether data has been cross-verified
        
        Returns:
            Number of records inserted/updated
        """
        if df.empty:
            return 0
        
        ticker = ticker.upper()
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        records = 0
        
        with self._get_connection() as conn:
            for idx, row in df.iterrows():
                date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
                
                def _get_scalar(*keys):
                    for key in keys:
                        if key not in row:
                            continue
                        value = row[key]
                        if isinstance(value, pd.Series):
                            if value.empty:
                                continue
                            value = value.iloc[0]
                        if pd.isna(value):
                            continue
                        return value
                    return None

                # Handle different column naming conventions
                open_price = _get_scalar("Open", "open")
                high_price = _get_scalar("High", "high")
                low_price = _get_scalar("Low", "low")
                close_price = _get_scalar("Close", "close")
                volume = _get_scalar("Volume", "volume")
                adj_close = _get_scalar("Adj Close", "adjusted_close") or close_price
                
                # Skip invalid rows
                if pd.isna(close_price) or close_price <= 0:
                    continue
                
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO daily_prices 
                        (ticker, date, open, high, low, close, volume, adjusted_close, source, verified, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, date_str, 
                        float(open_price) if not pd.isna(open_price) else None,
                        float(high_price) if not pd.isna(high_price) else None,
                        float(low_price) if not pd.isna(low_price) else None,
                        float(close_price),
                        int(volume) if not pd.isna(volume) else None,
                        float(adj_close) if not pd.isna(adj_close) else None,
                        source, 
                        1 if verified else 0,
                        now
                    ))
                    records += 1
                except Exception as e:
                    logger.debug(f"Failed to store price for {ticker} on {date_str}: {e}")
            
            # Update metadata
            if records > 0:
                conn.execute("""
                    INSERT OR REPLACE INTO ticker_metadata (ticker, first_date, last_date, total_records, last_updated)
                    SELECT 
                        ticker,
                        MIN(date) as first_date,
                        MAX(date) as last_date,
                        COUNT(*) as total_records,
                        ? as last_updated
                    FROM daily_prices
                    WHERE ticker = ?
                    GROUP BY ticker
                """, (now, ticker))
            
            conn.commit()
        
        logger.debug(f"Stored {records} price records for {ticker}")
        return records
    
    def has_data(
        self,
        ticker: str,
        start_date: str | date,
        end_date: str | date,
        min_coverage: float = 0.6
    ) -> bool:
        """
        Check if we have sufficient data for a ticker in date range.
        
        Args:
            ticker: Stock symbol
            start_date: Start date
            end_date: End date  
            min_coverage: Minimum fraction of expected trading days (default 60%)
        
        Returns:
            True if we have sufficient data coverage
        """
        start_str = str(start_date)[:10]
        end_str = str(end_date)[:10]
        
        with self._get_connection() as conn:
            result = conn.execute("""
                SELECT COUNT(*) as count
                FROM daily_prices
                WHERE ticker = ? AND date >= ? AND date <= ?
            """, (ticker, start_str, end_str)).fetchone()
            
            count = result['count'] if result else 0
        
        # Estimate expected trading days (roughly 252 per year, ~70% of calendar days)
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        calendar_days = (end_dt - start_dt).days + 1
        expected_trading_days = int(calendar_days * 0.70)  # ~70% are trading days
        
        return count >= (expected_trading_days * min_coverage)
    
    def get_missing_dates(
        self,
        ticker: str,
        start_date: str | date,
        end_date: str | date,
    ) -> List[str]:
        """
        Get list of dates we're missing data for.
        
        Note: This is an approximation - we can't know all trading days
        without a trading calendar.
        """
        start_str = str(start_date)[:10]
        end_str = str(end_date)[:10]
        
        with self._get_connection() as conn:
            result = conn.execute("""
                SELECT date FROM daily_prices
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            """, (ticker, start_str, end_str)).fetchall()
            
            existing_dates = {row['date'] for row in result}
        
        # Generate all business days in range
        all_dates = pd.bdate_range(start_str, end_str)
        missing = [d.strftime("%Y-%m-%d") for d in all_dates if d.strftime("%Y-%m-%d") not in existing_dates]
        
        return missing
    
    def get_tickers_with_data(self) -> List[str]:
        """Get list of all tickers that have price data stored."""
        with self._get_connection() as conn:
            result = conn.execute("""
                SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker
            """).fetchall()
            return [row['ticker'] for row in result]
    
    def get_ticker_date_range(self, ticker: str) -> tuple[Optional[str], Optional[str]]:
        """Get the date range we have data for a ticker."""
        with self._get_connection() as conn:
            result = conn.execute("""
                SELECT MIN(date) as first_date, MAX(date) as last_date
                FROM daily_prices WHERE ticker = ?
            """, (ticker,)).fetchone()
            
            if result and result['first_date']:
                return result['first_date'], result['last_date']
            return None, None
    
    def log_download(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        source: str,
        records: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Log a download attempt for auditing."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO download_log 
                (ticker, start_date, end_date, source, records_downloaded, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, start_date, end_date, source, records, 1 if success else 0, error))
            conn.commit()
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            total_records = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
            total_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM daily_prices").fetchone()[0]
            
            date_range = conn.execute("""
                SELECT MIN(date) as first, MAX(date) as last FROM daily_prices
            """).fetchone()
            
            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "total_records": total_records,
                "total_tickers": total_tickers,
                "first_date": date_range['first'] if date_range else None,
                "last_date": date_range['last'] if date_range else None,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
                "db_path": str(self.db_path),
            }
    
    def vacuum(self):
        """Optimize database file size."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")


# Global instance
_price_db: Optional[PriceDatabase] = None


def get_price_db(db_path: str = DEFAULT_DB_PATH) -> PriceDatabase:
    """Get or create global price database instance."""
    global _price_db
    if _price_db is None:
        _price_db = PriceDatabase(db_path)
    return _price_db
