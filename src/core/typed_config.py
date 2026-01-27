"""
Typed Configuration Module

Provides type-safe configuration using Pydantic models.
Replaces untyped dict-based configuration for better validation and IDE support.
"""

from __future__ import annotations
from typing import Optional, Literal
from pathlib import Path
import os

try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback to dataclasses if pydantic not installed
    from dataclasses import dataclass, field
    PYDANTIC_AVAILABLE = False
    
    # Create a dummy BaseModel for compatibility
    class BaseModel:
        def dict(self):
            return self.__dict__


if PYDANTIC_AVAILABLE:
    class UniverseConfig(BaseModel):
        """Universe configuration."""
        mode: Literal["SP500", "SP500+NASDAQ100", "SP500+NASDAQ100+R2000"] = "SP500+NASDAQ100+R2000"
        cache_file: Optional[str] = "universe_cache.csv"
        cache_max_age_days: int = Field(default=7, ge=1, le=30)
        manual_include_file: Optional[str] = "tickers/manual_include_tickers.txt"
        r2000_include_file: Optional[str] = "tickers/r2000.txt"
        manual_include_mode: Literal["ALWAYS", "ONLY_IF_IN_UNIVERSE"] = "ALWAYS"


    class LiquidityConfig(BaseModel):
        """Liquidity filter configuration."""
        price_min: float = Field(default=2.0, ge=0.0)
        min_avg_dollar_volume_20d: int = Field(default=50_000_000, ge=0)
        max_5d_return: float = Field(default=0.15, ge=0.0, le=1.0)


    class TechnicalsConfig(BaseModel):
        """Technical analysis configuration."""
        lookback_days: int = Field(default=300, ge=50, le=500)
        price_min: float = Field(default=7.0, ge=0.0)


    class QualityFiltersWeeklyConfig(BaseModel):
        """Weekly scanner quality filters."""
        min_technical_score: float = Field(default=0.0, ge=0.0, le=10.0)


    class QualityFilters30dConfig(BaseModel):
        """30-day screener quality filters."""
        rvol_min: float = Field(default=2.0, ge=0.0)
        atr_pct_min: float = Field(default=4.0, ge=0.0)
        near_high_max_pct: float = Field(default=8.0, ge=0.0, le=100.0)
        rsi_reversal_max: float = Field(default=35.0, ge=0.0, le=100.0)
        breakout_rsi_min: float = Field(default=55.0, ge=0.0, le=100.0)
        reversal_rsi_max: float = Field(default=32.0, ge=0.0, le=100.0)
        reversal_dist_to_high_min_pct: float = Field(default=15.0, ge=0.0, le=100.0)
        min_score: float = Field(default=0.0, ge=0.0)


    class Outputs30dConfig(BaseModel):
        """30-day output configuration."""
        top_n_breakout: int = Field(default=15, ge=1, le=100)
        top_n_reversal: int = Field(default=15, ge=1, le=100)
        top_n_total: int = Field(default=25, ge=1, le=200)


    class AttentionPoolConfig(BaseModel):
        """Attention pool configuration."""
        rvol_min: float = Field(default=1.8, ge=0.0)
        atr_pct_min: float = Field(default=3.5, ge=0.0)
        min_abs_day_move_pct: float = Field(default=3.0, ge=0.0)
        lookback_days: int = Field(default=120, ge=20, le=365)
        chunk_size: int = Field(default=200, ge=10, le=1000)
        enable_intraday: bool = False
        intraday_interval: str = "5m"
        intraday_lookback_days: int = Field(default=5, ge=1, le=30)
        market_open_buffer_min: int = Field(default=20, ge=0, le=60)
        intraday_rvol_min: float = Field(default=2.0, ge=0.0)


    class RegimeGateConfig(BaseModel):
        """Market regime gate configuration."""
        enabled: bool = True
        spy_symbol: str = "SPY"
        vix_symbol: str = "^VIX"
        spy_ma_days: int = Field(default=20, ge=5, le=200)
        vix_max: float = Field(default=25.0, ge=0.0, le=100.0)
        action: Literal["WARN", "BLOCK"] = "WARN"


    class NewsConfig(BaseModel):
        """News fetching configuration."""
        max_items: int = Field(default=25, ge=1, le=100)
        packet_headlines: int = Field(default=12, ge=1, le=50)
        throttle_sec: float = Field(default=0.15, ge=0.0, le=5.0)


    class MoversConfig(BaseModel):
        """Daily movers configuration."""
        enabled: bool = True
        top_n: int = Field(default=50, ge=1, le=200)
        gainers_pct_range: tuple[float, float] = (7.0, 15.0)
        losers_pct_range: tuple[float, float] = (-15.0, -7.0)
        gainers_volume_spike: float = Field(default=2.0, ge=0.0)
        losers_volume_spike: float = Field(default=1.8, ge=0.0)
        close_position_min: float = Field(default=0.75, ge=0.0, le=1.0)
        price_min: float = Field(default=2.0, ge=0.0)
        adv_20d_min: int = Field(default=50_000_000, ge=0)
        cooling_days_required: int = Field(default=1, ge=0, le=10)
        max_age_days: int = Field(default=5, ge=1, le=30)


    class RuntimeConfig(BaseModel):
        """Runtime configuration."""
        method_version: str = "v3.1"
        yf_auto_adjust: bool = False
        threads: bool = True
        polygon_primary: bool = False
        polygon_fallback: bool = True
        polygon_max_workers: int = Field(default=8, ge=1, le=32)
        polygon_intraday: bool = False
        polygon_intraday_interval: int = Field(default=5, ge=1, le=60)
        polygon_intraday_lookback_days: int = Field(default=5, ge=1, le=30)
        block_recent_splits: bool = True
        split_block_days: int = Field(default=7, ge=1, le=30)
        split_lookback_days: int = Field(default=120, ge=30, le=365)
        allow_partial_day_attention: bool = False


    class OutputsConfig(BaseModel):
        """Output configuration."""
        root_dir: str = "outputs"


    class AlertTriggersConfig(BaseModel):
        """Alert trigger configuration."""
        all_three_overlap: bool = True
        weekly_pro30_overlap: bool = True
        high_composite_score: float = Field(default=7.0, ge=0.0, le=10.0)


    class EmailConfig(BaseModel):
        """Email configuration."""
        smtp_host: str = "smtp.gmail.com"
        smtp_port: int = Field(default=587, ge=1, le=65535)
        from_address: Optional[str] = None
        to_addresses: list[str] = Field(default_factory=list)


    class AlertsConfig(BaseModel):
        """Alerts configuration."""
        enabled: bool = False
        channels: list[str] = Field(default_factory=list)
        slack_webhook: Optional[str] = None
        discord_webhook: Optional[str] = None
        email: EmailConfig = Field(default_factory=EmailConfig)
        triggers: AlertTriggersConfig = Field(default_factory=AlertTriggersConfig)


    class CacheConfig(BaseModel):
        """Cache configuration."""
        enabled: bool = True
        backend: Literal["sqlite", "redis"] = "sqlite"
        sqlite_path: str = "data/price_cache.db"
        redis_url: str = "redis://localhost:6379/0"
        price_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
        news_ttl_seconds: int = Field(default=1800, ge=60, le=86400)


    class APIConfig(BaseModel):
        """API server configuration."""
        enabled: bool = False
        host: str = "0.0.0.0"
        port: int = Field(default=8000, ge=1, le=65535)
        cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


    class AppConfig(BaseModel):
        """Main application configuration."""
        universe: UniverseConfig = Field(default_factory=UniverseConfig)
        liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
        technicals: TechnicalsConfig = Field(default_factory=TechnicalsConfig)
        quality_filters_weekly: QualityFiltersWeeklyConfig = Field(default_factory=QualityFiltersWeeklyConfig)
        quality_filters_30d: QualityFilters30dConfig = Field(default_factory=QualityFilters30dConfig)
        outputs_30d: Outputs30dConfig = Field(default_factory=Outputs30dConfig)
        attention_pool: AttentionPoolConfig = Field(default_factory=AttentionPoolConfig)
        regime_gate: RegimeGateConfig = Field(default_factory=RegimeGateConfig)
        news: NewsConfig = Field(default_factory=NewsConfig)
        movers: MoversConfig = Field(default_factory=MoversConfig)
        runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
        outputs: OutputsConfig = Field(default_factory=OutputsConfig)
        alerts: AlertsConfig = Field(default_factory=AlertsConfig)
        cache: CacheConfig = Field(default_factory=CacheConfig)
        api: APIConfig = Field(default_factory=APIConfig)
        
        model_config = ConfigDict(extra="allow")
        
        @classmethod
        def from_yaml(cls, path: str | Path) -> "AppConfig":
            """Load configuration from YAML file."""
            import yaml
            
            path = Path(path)
            if not path.exists():
                return cls()
            
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            return cls(**data)
        
        @classmethod
        def from_dict(cls, data: dict) -> "AppConfig":
            """Create configuration from dictionary."""
            return cls(**data)
        
        def to_dict(self) -> dict:
            """Convert configuration to dictionary."""
            return self.model_dump()
        
        def to_yaml(self, path: str | Path) -> None:
            """Save configuration to YAML file."""
            import yaml
            
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

else:
    # Fallback dataclass implementation (less validation)
    @dataclass
    class AppConfig:
        """Main application configuration (fallback without pydantic)."""
        # Basic nested dicts since we can't use pydantic models
        universe: dict = None
        liquidity: dict = None
        technicals: dict = None
        quality_filters_weekly: dict = None
        quality_filters_30d: dict = None
        outputs_30d: dict = None
        attention_pool: dict = None
        regime_gate: dict = None
        news: dict = None
        movers: dict = None
        runtime: dict = None
        outputs: dict = None
        alerts: dict = None
        cache: dict = None
        api: dict = None
        
        def __post_init__(self):
            # Set defaults
            self.universe = self.universe or {}
            self.liquidity = self.liquidity or {}
            self.technicals = self.technicals or {}
            self.quality_filters_weekly = self.quality_filters_weekly or {}
            self.quality_filters_30d = self.quality_filters_30d or {}
            self.outputs_30d = self.outputs_30d or {}
            self.attention_pool = self.attention_pool or {}
            self.regime_gate = self.regime_gate or {}
            self.news = self.news or {}
            self.movers = self.movers or {}
            self.runtime = self.runtime or {}
            self.outputs = self.outputs or {}
            self.alerts = self.alerts or {}
            self.cache = self.cache or {}
            self.api = self.api or {}
        
        @classmethod
        def from_yaml(cls, path: str | Path) -> "AppConfig":
            """Load configuration from YAML file."""
            import yaml
            
            path = Path(path)
            if not path.exists():
                return cls()
            
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            return cls(**data)
        
        def to_dict(self) -> dict:
            """Convert to dictionary."""
            return {
                "universe": self.universe,
                "liquidity": self.liquidity,
                "technicals": self.technicals,
                "quality_filters_weekly": self.quality_filters_weekly,
                "quality_filters_30d": self.quality_filters_30d,
                "outputs_30d": self.outputs_30d,
                "attention_pool": self.attention_pool,
                "regime_gate": self.regime_gate,
                "news": self.news,
                "movers": self.movers,
                "runtime": self.runtime,
                "outputs": self.outputs,
                "alerts": self.alerts,
                "cache": self.cache,
                "api": self.api,
            }


def load_typed_config(path: str | Path = "config/default.yaml") -> AppConfig:
    """
    Load typed configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file
    
    Returns:
        AppConfig instance with validated settings
    """
    return AppConfig.from_yaml(path)


def get_default_config() -> AppConfig:
    """Get default configuration."""
    return AppConfig()
