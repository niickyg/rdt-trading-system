"""
News Sentiment Pre-filter for Scanner Signals

Checks news sentiment for scanned symbols and flags signals with
strongly negative news. Per RDT methodology, price action is king
and news is secondary — so we WARN but don't BLOCK signals.

Uses the existing NewsSentimentAnalyzer (Finnhub) when available.
Returns neutral sentiment when no data source is configured.

Results are cached for 15 minutes per symbol to avoid API spam.
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger


# Existing sentiment infrastructure (Finnhub-based)
try:
    from shared.news_sentiment import get_news_analyzer, NewsSentimentAnalyzer
    FINNHUB_ANALYZER_AVAILABLE = True
except ImportError:
    FINNHUB_ANALYZER_AVAILABLE = False
    logger.debug("Finnhub news analyzer not available")



class NewsFilter:
    """
    Lightweight news sentiment filter for scanner signals.

    Checks headlines for positive/negative keywords and returns
    a sentiment assessment. Integrates with the existing
    NewsSentimentAnalyzer when Finnhub API key is configured.
    """

    # Sentiment keywords (aligned with shared/news_sentiment.py)
    POSITIVE_KEYWORDS = [
        'beat', 'beats', 'exceeds', 'surpass', 'upgrade', 'upgraded', 'buy',
        'bullish', 'growth', 'profit', 'gains', 'surge', 'soar', 'rally',
        'record', 'breakthrough', 'innovative', 'strong', 'positive',
        'outperform', 'raises', 'dividend', 'acquisition', 'partnership',
        'expansion', 'momentum', 'optimistic', 'confident', 'success',
    ]

    NEGATIVE_KEYWORDS = [
        'miss', 'misses', 'below', 'downgrade', 'downgraded', 'sell',
        'bearish', 'loss', 'losses', 'decline', 'drop', 'fall', 'plunge',
        'crash', 'warning', 'concern', 'risk', 'weak', 'negative',
        'underperform', 'cuts', 'layoff', 'lawsuit', 'investigation',
        'fraud', 'scandal', 'bankruptcy', 'default', 'recession', 'fear',
    ]

    VERY_NEGATIVE_KEYWORDS = [
        'fraud', 'scandal', 'bankruptcy', 'investigation', 'sec', 'fbi',
        'criminal', 'indictment', 'recall', 'death', 'fatal',
    ]

    def __init__(self, cache_ttl_minutes: int = 15):
        """
        Initialize news filter.

        Args:
            cache_ttl_minutes: How long to cache results per symbol (default 15 min)
        """
        self._cache: Dict[str, Tuple[Dict, float]] = {}
        self._cache_ttl_seconds = cache_ttl_minutes * 60
        self._finnhub_analyzer: Optional[NewsSentimentAnalyzer] = None

        # Try to use existing Finnhub analyzer
        if FINNHUB_ANALYZER_AVAILABLE:
            try:
                self._finnhub_analyzer = get_news_analyzer()
                if self._finnhub_analyzer.api_key:
                    logger.info("NewsFilter using Finnhub analyzer")
                else:
                    logger.info("NewsFilter: no Finnhub API key — returning neutral sentiment")
                    self._finnhub_analyzer = None
            except Exception as e:
                logger.debug(f"Could not initialize Finnhub analyzer: {e}")
                self._finnhub_analyzer = None

        if self._finnhub_analyzer is None:
            logger.info("NewsFilter: no data source available — sentiment checks return neutral")

    def check_symbol(self, symbol: str) -> Dict:
        """
        Check news sentiment for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            dict with keys:
                - has_negative_news: bool — True if strongly negative sentiment
                - sentiment_score: float — overall score from -1 to 1
                - headlines: list of str — recent headline strings
        """
        # Check cache first
        if symbol in self._cache:
            cached_result, cached_at = self._cache[symbol]
            if time.time() - cached_at < self._cache_ttl_seconds:
                return cached_result

        # Fetch and analyze
        result = self._analyze(symbol)

        # Store in cache
        self._cache[symbol] = (result, time.time())

        return result

    def _analyze(self, symbol: str) -> Dict:
        """Analyze sentiment for a symbol using available data source."""
        # Use Finnhub analyzer if available
        if self._finnhub_analyzer is not None:
            return self._analyze_via_finnhub(symbol)

        # No data source available — return neutral
        return {
            'has_negative_news': False,
            'sentiment_score': 0.0,
            'headlines': [],
        }

    def _analyze_via_finnhub(self, symbol: str) -> Dict:
        """Use existing Finnhub-based NewsSentimentAnalyzer."""
        try:
            sentiment = self._finnhub_analyzer.get_sentiment_sync(symbol)
            headlines = [item.headline for item in sentiment.latest_news[:5]]

            return {
                'has_negative_news': sentiment.overall_sentiment < -0.5,
                'sentiment_score': round(sentiment.overall_sentiment, 3),
                'headlines': headlines,
            }
        except Exception as e:
            logger.debug(f"Finnhub sentiment check failed for {symbol}: {e}")
            return {
                'has_negative_news': False,
                'sentiment_score': 0.0,
                'headlines': [],
            }

    def _score_headline(self, text: str) -> float:
        """
        Score a headline's sentiment using keyword matching.

        Returns:
            Score from -1 (very negative) to 1 (very positive)
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        # Check for very negative keywords first
        for keyword in self.VERY_NEGATIVE_KEYWORDS:
            if keyword in text_lower:
                return -0.9

        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        score = (positive_count - negative_count) / max(total, 1)
        return max(-1.0, min(1.0, score))

    def clear_cache(self):
        """Clear the sentiment cache."""
        self._cache.clear()


# Module-level singleton
_news_filter: Optional[NewsFilter] = None
_news_filter_lock = threading.Lock()


def get_news_filter(cache_ttl_minutes: int = 15) -> NewsFilter:
    """Get the global NewsFilter singleton."""
    global _news_filter
    if _news_filter is None:
        with _news_filter_lock:
            if _news_filter is None:
                _news_filter = NewsFilter(cache_ttl_minutes=cache_ttl_minutes)
    return _news_filter
