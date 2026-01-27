"""
Situation Memory System (Inspired by TradingAgents)

Uses ChromaDB with OpenAI embeddings to store and retrieve similar past
trading situations. This enables the system to learn from historical outcomes.

The memory stores:
- Market conditions at entry
- Technical setup
- Catalyst information
- The actual outcome (hit/miss, return %)
- Lessons learned from that trade

When analyzing new candidates, we retrieve similar past situations to inform
the current decision.
"""

from __future__ import annotations
import os
import json
from typing import Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from src.utils.time import utc_now


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, handling None and invalid types."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ChromaDB is optional - graceful fallback if not installed
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed - memory system disabled. Install with: pip install chromadb")

# OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TradingMemory:
    """
    Vector memory system for trading situations.
    
    Stores past trading setups with their outcomes, enabling retrieval
    of similar situations to inform new decisions.
    """
    
    def __init__(
        self,
        collection_name: str = "trading_memory",
        persist_directory: str = "data/memory",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the trading memory system.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Where to persist the database
            embedding_model: OpenAI embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._openai_client = None
        
        if not CHROMADB_AVAILABLE:
            logger.warning("TradingMemory disabled - chromadb not installed")
            return
        
        if not OPENAI_AVAILABLE:
            logger.warning("TradingMemory disabled - openai not installed")
            return
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB and OpenAI clients."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            return
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Trading situation memory for learning from outcomes"}
        )
        
        # Initialize OpenAI client for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self._openai_client = OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not set - embeddings will fail")
    
    @property
    def is_available(self) -> bool:
        """Check if memory system is available."""
        return self._collection is not None and self._openai_client is not None
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        response = self._openai_client.embeddings.create(
            model=self.embedding_model,
            input=text[:8000],  # Limit input length
        )
        return response.data[0].embedding
    
    def _situation_to_text(self, situation: dict) -> str:
        """Convert situation dict to searchable text."""
        parts = []
        
        # Ticker and basic info
        if ticker := situation.get("ticker"):
            parts.append(f"Ticker: {ticker}")
        if sector := situation.get("sector"):
            parts.append(f"Sector: {sector}")
        
        # Technical setup
        if tech := situation.get("technical"):
            parts.append(f"Technical setup: RSI={tech.get('rsi14', 'N/A')}, "
                        f"near_52w_high={tech.get('within_5pct_52w_high', False)}, "
                        f"volume_spike={tech.get('volume_ratio_3d_to_20d', 'N/A')}")
        
        # Catalyst
        if catalyst := situation.get("catalyst"):
            parts.append(f"Catalyst: {catalyst.get('title', 'None')} on {catalyst.get('date', 'N/A')}")
        
        # Sentiment
        if sentiment := situation.get("sentiment"):
            parts.append(f"Sentiment: {sentiment.get('overall', 'neutral')}")
        
        # Source and scores
        if source := situation.get("source"):
            parts.append(f"Source: {source}")
        if score := situation.get("composite_score"):
            parts.append(f"Composite score: {score}")
        
        # Market conditions
        if market := situation.get("market_conditions"):
            parts.append(f"Market: SPY trend={market.get('spy_trend', 'N/A')}, "
                        f"VIX={market.get('vix', 'N/A')}")
        
        # Headlines
        if headlines := situation.get("headlines"):
            parts.append(f"Headlines: {' | '.join(headlines[:3])}")
        
        return "\n".join(parts)
    
    def add_situation(
        self,
        situation: dict,
        outcome: dict,
        lesson: str = "",
    ) -> str:
        """
        Add a trading situation and its outcome to memory.
        
        Args:
            situation: Dict describing the trading setup
                {
                    "ticker": "AAPL",
                    "entry_date": "2025-01-15",
                    "entry_price": 185.50,
                    "sector": "Technology",
                    "source": "weekly",  # weekly, pro30, movers
                    "composite_score": 7.5,
                    "technical": {...},
                    "catalyst": {...},
                    "sentiment": {...},
                    "market_conditions": {...},
                    "headlines": [...]
                }
            outcome: Dict describing what happened
                {
                    "max_gain_7d": 12.5,  # % gain
                    "max_drawdown_7d": -3.2,  # % loss
                    "close_7d_return": 8.3,  # % return at day 7
                    "hit_10pct": True,
                    "hit_7pct": True,
                    "hit_5pct": True,
                    "days_to_hit_10pct": 3,
                    "outcome_date": "2025-01-22"
                }
            lesson: What we learned from this trade
        
        Returns:
            ID of the stored memory
        """
        if not self.is_available:
            logger.warning("Memory system not available - skipping add")
            return ""
        
        # Generate ID
        ticker = situation.get("ticker", "UNK")
        entry_date = situation.get("entry_date", utc_now().strftime("%Y-%m-%d"))
        memory_id = f"{ticker}_{entry_date}_{utc_now().strftime('%H%M%S')}"
        
        # Convert situation to searchable text
        situation_text = self._situation_to_text(situation)
        
        # Generate embedding
        embedding = self._get_embedding(situation_text)
        
        # Build metadata
        metadata = {
            "ticker": ticker,
            "entry_date": entry_date,
            "source": situation.get("source", "unknown"),
            "composite_score": _safe_float(situation.get("composite_score"), 0.0),
            "sector": situation.get("sector", ""),
            # Outcome fields
            "hit_10pct": outcome.get("hit_10pct", False),
            "hit_7pct": outcome.get("hit_7pct", False),
            "hit_5pct": outcome.get("hit_5pct", False),
            "max_gain_7d": _safe_float(outcome.get("max_gain_7d"), 0.0),
            "max_drawdown_7d": _safe_float(outcome.get("max_drawdown_7d"), 0.0),
            "close_7d_return": _safe_float(outcome.get("close_7d_return"), 0.0),
            "days_to_hit": int(outcome.get("days_to_hit_10pct", 0) or 0),
            "outcome_date": outcome.get("outcome_date", ""),
            "lesson": lesson[:500] if lesson else "",
        }
        
        # Build document text (situation + outcome + lesson)
        document = f"""{situation_text}

OUTCOME:
- Hit 10%: {outcome.get('hit_10pct', False)}
- Max gain (7d): {outcome.get('max_gain_7d', 0):.1f}%
- Max drawdown (7d): {outcome.get('max_drawdown_7d', 0):.1f}%
- Days to hit: {outcome.get('days_to_hit_10pct', 'N/A')}

LESSON LEARNED:
{lesson or 'No lesson recorded'}
"""
        
        # Add to collection
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )
        
        logger.info(f"Added memory: {memory_id}")
        return memory_id
    
    def get_similar_situations(
        self,
        current_situation: dict,
        n_results: int = 3,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """
        Find similar past situations to inform current decision.
        
        Args:
            current_situation: Current trading setup
            n_results: Number of similar situations to return
            min_similarity: Minimum similarity score (0-1)
        
        Returns:
            List of similar situations with their outcomes
        """
        if not self.is_available:
            return []
        
        # Check if we have any memories
        if self._collection.count() == 0:
            logger.debug("No memories stored yet")
            return []
        
        # Convert current situation to text
        situation_text = self._situation_to_text(current_situation)
        
        # Get embedding
        embedding = self._get_embedding(situation_text)
        
        # Query collection
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to list of dicts
        similar = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # ChromaDB uses L2 distance
            
            if similarity < min_similarity:
                continue
            
            metadata = results["metadatas"][0][i]
            similar.append({
                "memory_id": results["ids"][0][i],
                "similarity": similarity,
                "document": results["documents"][0][i],
                "ticker": metadata.get("ticker"),
                "entry_date": metadata.get("entry_date"),
                "source": metadata.get("source"),
                "composite_score": metadata.get("composite_score"),
                "sector": metadata.get("sector"),
                "hit_10pct": metadata.get("hit_10pct"),
                "hit_7pct": metadata.get("hit_7pct"),
                "max_gain_7d": metadata.get("max_gain_7d"),
                "max_drawdown_7d": metadata.get("max_drawdown_7d"),
                "lesson": metadata.get("lesson"),
            })
        
        return similar
    
    def get_hit_rate_for_similar(
        self,
        current_situation: dict,
        n_results: int = 10,
        threshold_pct: float = 10.0,
    ) -> dict:
        """
        Get historical hit rate for similar situations.
        
        Args:
            current_situation: Current trading setup
            n_results: Number of similar situations to analyze
            threshold_pct: Hit threshold (default 10%)
        
        Returns:
            Dict with hit rate and statistics
        """
        similar = self.get_similar_situations(current_situation, n_results=n_results)
        
        if not similar:
            return {
                "n_similar": 0,
                "hit_rate": None,
                "avg_max_gain": None,
                "avg_max_drawdown": None,
                "recommendation": "No similar situations in memory",
            }
        
        # Calculate statistics
        threshold_key = f"hit_{int(threshold_pct)}pct" if threshold_pct in [5, 7, 10] else "hit_10pct"
        hits = [s for s in similar if s.get(threshold_key, False)]
        
        hit_rate = len(hits) / len(similar) if similar else 0
        avg_gain = sum(s.get("max_gain_7d", 0) for s in similar) / len(similar)
        avg_dd = sum(s.get("max_drawdown_7d", 0) for s in similar) / len(similar)
        
        # Build recommendation
        if hit_rate >= 0.5:
            recommendation = f"Similar setups hit {threshold_pct}%+ {hit_rate*100:.0f}% of the time - FAVORABLE"
        elif hit_rate >= 0.3:
            recommendation = f"Similar setups hit {threshold_pct}%+ {hit_rate*100:.0f}% of the time - NEUTRAL"
        else:
            recommendation = f"Similar setups hit {threshold_pct}%+ only {hit_rate*100:.0f}% of the time - CAUTION"
        
        # Add lessons learned
        lessons = [s.get("lesson") for s in similar if s.get("lesson")]
        
        return {
            "n_similar": len(similar),
            "hit_rate": hit_rate,
            "avg_max_gain": avg_gain,
            "avg_max_drawdown": avg_dd,
            "recommendation": recommendation,
            "lessons": lessons[:3],  # Top 3 lessons
            "similar_tickers": [s.get("ticker") for s in similar],
        }
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        if not self.is_available:
            return {"available": False, "count": 0}
        
        count = self._collection.count()
        
        # Get some aggregate stats
        if count > 0:
            all_data = self._collection.get(include=["metadatas"])
            metadatas = all_data.get("metadatas", [])
            
            hits = sum(1 for m in metadatas if m.get("hit_10pct", False))
            sources = {}
            for m in metadatas:
                src = m.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            
            return {
                "available": True,
                "count": count,
                "hit_rate_10pct": hits / count if count > 0 else 0,
                "by_source": sources,
            }
        
        return {"available": True, "count": 0}
    
    def clear(self):
        """Clear all memories (use with caution)."""
        if not self.is_available:
            return
        
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"description": "Trading situation memory for learning from outcomes"}
        )
        logger.info("Memory cleared")


# Global singleton instance
_memory_instance: Optional[TradingMemory] = None


def get_trading_memory() -> TradingMemory:
    """Get the global trading memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = TradingMemory()
    return _memory_instance


def build_situation_from_packet(packet: dict, source: str = "weekly") -> dict:
    """Convert a scanner packet to a situation dict for memory."""
    return {
        "ticker": packet.get("ticker"),
        "entry_date": packet.get("asof_date", utc_now().strftime("%Y-%m-%d")),
        "entry_price": packet.get("current_price", packet.get("price")),
        "sector": packet.get("sector", ""),
        "source": source,
        "composite_score": packet.get("composite_score", packet.get("technical_score", 0)),
        "technical": packet.get("evidence", {}).get("technical", packet.get("technical_evidence", {})),
        "catalyst": packet.get("primary_catalyst", {}),
        "sentiment": packet.get("evidence", {}).get("sentiment", {}),
        "market_conditions": {},  # Could add SPY/VIX data here
        "headlines": [h.get("title", str(h)) for h in packet.get("news", [])[:5]],
    }


def enrich_packet_with_memory(packet: dict, memory: TradingMemory = None) -> dict:
    """
    Enrich a packet with historical memory insights.
    
    Adds a 'memory_insights' field with similar past situations
    and their outcomes to help inform the current decision.
    """
    if memory is None:
        memory = get_trading_memory()
    
    if not memory.is_available:
        return packet
    
    # Build situation from packet
    situation = build_situation_from_packet(packet)
    
    # Get similar situations
    insights = memory.get_hit_rate_for_similar(situation)
    
    # Add to packet
    packet["memory_insights"] = insights
    
    return packet
