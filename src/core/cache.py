"""
Caching Layer for Price and News Data

Provides SQLite and Redis backends to reduce API calls and improve performance.
"""

from __future__ import annotations
import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager
import pandas as pd
import hashlib

logger = logging.getLogger(__name__)

from src.utils.time import utc_now


class CacheBackend:
    """Abstract base class for cache backends."""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    def clear(self) -> bool:
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        raise NotImplementedError


class SQLiteCache(CacheBackend):
    """SQLite-based cache backend for local persistence."""
    
    def __init__(self, db_path: str = "data/price_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            
            # Index for expiration cleanup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache(expires_at)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        # Don't use PARSE_DECLTYPES to avoid timestamp parsing issues
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Check expiration - handle various timestamp formats
            try:
                expires_str = str(row["expires_at"])
                # Handle ISO format with 'T' separator
                if "T" in expires_str:
                    expires_at = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                else:
                    # Handle space-separated format
                    expires_at = datetime.strptime(expires_str, "%Y-%m-%d %H:%M:%S")
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                # If parsing fails, treat as expired
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            
            if utc_now() > expires_at:
                # Expired, delete it
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            
            return json.loads(row["value"])
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            expires_at = utc_now() + timedelta(seconds=ttl_seconds)
            value_json = json.dumps(value, default=str)
            
            with self._get_connection() as conn:
                # Use space-separated format for SQLite compatibility
                expires_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, expires_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, value_json, expires_str)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Cache set failed for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Cache delete failed for {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached data."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?",
                (now_str,)
            )
            conn.commit()
            return cursor.rowcount
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
                (now_str,)
            ).fetchone()[0]
            
            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2)
            }


class RedisCache(CacheBackend):
    """Redis-based cache backend for distributed caching."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.redis_url)
                self._client.ping()  # Test connection
            except ImportError:
                raise ImportError("redis package not installed. Install with: pip install redis")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Redis: {e}")
        return self._client
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get failed for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in Redis with TTL."""
        try:
            value_json = json.dumps(value, default=str)
            self.client.setex(key, ttl_seconds, value_json)
            return True
        except Exception as e:
            logger.error(f"Redis set failed for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete failed for {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached data."""
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self.client.exists(key))
        except Exception:
            return False


class PriceCache:
    """
    High-level cache interface for price data.
    
    Provides smart caching for OHLCV data with:
    - Automatic key generation
    - DataFrame serialization
    - TTL management based on data type
    """
    
    def __init__(
        self,
        backend: str = "sqlite",
        sqlite_path: str = "data/price_cache.db",
        redis_url: str = "redis://localhost:6379/0",
        price_ttl_seconds: int = 3600,
        news_ttl_seconds: int = 1800,
    ):
        self.price_ttl = price_ttl_seconds
        self.news_ttl = news_ttl_seconds
        
        if backend == "sqlite":
            self._cache = SQLiteCache(sqlite_path)
        elif backend == "redis":
            self._cache = RedisCache(redis_url)
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
    
    def _make_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments."""
        key_parts = [prefix] + [str(a) for a in args]
        key_str = ":".join(key_parts)
        # Hash long keys to keep them manageable
        if len(key_str) > 200:
            key_hash = hashlib.md5(key_str.encode()).hexdigest()[:16]
            return f"{prefix}:{key_hash}"
        return key_str
    
    def _make_ticker_key(self, ticker: str, interval: str = "1d") -> str:
        """Generate cache key for ticker (without date range)."""
        return f"prices_v2:{ticker}:{interval}"
    
    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get cached price data for a ticker.
        
        First tries the new ticker-based key (can serve any date range).
        Falls back to old date-range-specific key for backwards compatibility.
        
        Returns DataFrame filtered to requested date range if found, None otherwise.
        """
        # Try new ticker-based cache first (more flexible)
        ticker_key = self._make_ticker_key(ticker, interval)
        cached = self._cache.get(ticker_key)
        
        if cached is not None:
            try:
                df = pd.DataFrame(cached["data"])
                df.index = pd.to_datetime(df.index)
                df.index.name = cached.get("index_name", "Date")
                df = df.sort_index()
                
                # Filter to requested date range
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                df_filtered = df[(df.index >= start_ts) & (df.index <= end_ts)]
                
                if not df_filtered.empty:
                    return df_filtered
            except Exception as e:
                logger.debug(f"Failed to deserialize cached prices for {ticker}: {e}")
        
        # Fallback to old date-range-specific key
        old_key = self._make_key("prices", ticker, start_date, end_date, interval)
        cached = self._cache.get(old_key)
        
        if cached is None:
            return None
        
        try:
            df = pd.DataFrame(cached["data"])
            df.index = pd.to_datetime(df.index)
            df.index.name = cached.get("index_name", "Date")
            df = df.sort_index()
            return df
        except Exception as e:
            logger.debug(f"Failed to deserialize cached prices for {ticker}: {e}")
            return None
    
    def set_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        data: pd.DataFrame,
        interval: str = "1d",
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache price data for a ticker.
        
        Stores data in a ticker-based key (can serve any date range subset).
        Also merges with existing cached data to build up a complete history.
        
        Args:
            ttl_seconds: Override TTL. If None, uses 30 days for historical data
                        (end_date > 3 days ago) or price_ttl for recent data.
        """
        if data.empty:
            return False
        
        # Use ticker-based key (more flexible for serving different date ranges)
        ticker_key = self._make_ticker_key(ticker, interval)
        
        # Try to merge with existing cached data
        existing = self._cache.get(ticker_key)
        if existing is not None:
            try:
                existing_df = pd.DataFrame(existing["data"])
                existing_df.index = pd.to_datetime(existing_df.index)
                
                # Merge: use new data where available, keep old data for other dates
                merged = pd.concat([existing_df, data])
                merged = merged[~merged.index.duplicated(keep='last')]  # Keep newest
                merged = merged.sort_index()
                data = merged
            except Exception:
                pass  # If merge fails, just use new data
        
        # Convert DataFrame to JSON-serializable format
        df_to_cache = data.copy()
        df_to_cache.index = df_to_cache.index.strftime("%Y-%m-%d")  # Convert Timestamps to strings
        
        serialized = {
            "data": df_to_cache.to_dict(),
            "index_name": data.index.name or "Date",
            "ticker": ticker,
            "start": data.index.min().strftime("%Y-%m-%d") if not data.empty else start_date,
            "end": data.index.max().strftime("%Y-%m-%d") if not data.empty else end_date,
            "interval": interval,
            "cached_at": utc_now().isoformat().replace("+00:00", "")
        }
        
        # Determine TTL: use 30 days for historical data, 1 hour for recent
        if ttl_seconds is None:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                is_historical = (utc_now() - end_dt).days > 3
                ttl_seconds = 86400 * 30 if is_historical else self.price_ttl  # 30 days or 1 hour
            except Exception:
                ttl_seconds = self.price_ttl
        
        return self._cache.set(ticker_key, serialized, ttl_seconds)
    
    def get_news(self, ticker: str) -> Optional[list[dict]]:
        """Get cached news for a ticker."""
        key = self._make_key("news", ticker)
        return self._cache.get(key)
    
    def set_news(self, ticker: str, news: list[dict]) -> bool:
        """Cache news for a ticker."""
        key = self._make_key("news", ticker)
        return self._cache.set(key, news, self.news_ttl)
    
    def get_company_info(self, ticker: str) -> Optional[dict]:
        """Get cached company info."""
        key = self._make_key("info", ticker)
        return self._cache.get(key)
    
    def set_company_info(self, ticker: str, info: dict) -> bool:
        """Cache company info (longer TTL)."""
        key = self._make_key("info", ticker)
        return self._cache.set(key, info, 86400)  # 24 hours
    
    def invalidate_ticker(self, ticker: str) -> None:
        """Invalidate all cached data for a ticker."""
        # Note: This is a basic implementation
        # Full pattern-based deletion would require Redis SCAN or SQLite LIKE
        for prefix in ["prices", "news", "info"]:
            key = self._make_key(prefix, ticker)
            self._cache.delete(key)
    
    def cleanup(self) -> int:
        """Cleanup expired entries. Returns count removed."""
        if isinstance(self._cache, SQLiteCache):
            return self._cache.cleanup_expired()
        return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if isinstance(self._cache, SQLiteCache):
            return self._cache.get_stats()
        return {"backend": "redis", "stats": "use redis-cli INFO"}


# Global cache instance (lazy initialization)
_global_cache: Optional[PriceCache] = None


def get_cache(config: Optional[dict] = None) -> PriceCache:
    """Get or create global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        if config is None:
            config = {}
        
        cache_config = config.get("cache", {})
        
        _global_cache = PriceCache(
            backend=cache_config.get("backend", "sqlite"),
            sqlite_path=cache_config.get("sqlite_path", "data/price_cache.db"),
            redis_url=cache_config.get("redis_url", "redis://localhost:6379/0"),
            price_ttl_seconds=cache_config.get("price_ttl_seconds", 3600),
            news_ttl_seconds=cache_config.get("news_ttl_seconds", 1800),
        )
    
    return _global_cache
