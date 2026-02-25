"""
News and Sentiment Analysis Module

Integrates news sentiment to improve trade filtering:
- Fetches news from Finnhub (free tier)
- Analyzes sentiment using keyword scoring
- Filters out stocks with negative news
- Tracks earnings dates
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from loguru import logger

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - news features disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class NewsItem:
    """Single news item"""
    headline: str
    summary: str
    source: str
    datetime: datetime
    symbol: str
    sentiment_score: float  # -1 to 1
    url: Optional[str] = None
    category: Optional[str] = None


@dataclass
class SentimentResult:
    """Sentiment analysis result for a symbol"""
    symbol: str
    overall_sentiment: float  # -1 (very negative) to 1 (very positive)
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    has_earnings_soon: bool
    earnings_date: Optional[datetime] = None
    latest_news: List[NewsItem] = None
    recommendation: str = "neutral"  # "bullish", "bearish", "neutral", "avoid"

    def __post_init__(self):
        if self.latest_news is None:
            self.latest_news = []


class NewsSentimentAnalyzer:
    """
    News and sentiment analyzer using Finnhub API

    Features:
    - Real-time news fetching
    - Keyword-based sentiment scoring
    - Earnings calendar integration
    - Caching to minimize API calls
    """

    # Sentiment keywords
    POSITIVE_KEYWORDS = [
        'beat', 'beats', 'exceeds', 'surpass', 'upgrade', 'upgraded', 'buy',
        'bullish', 'growth', 'profit', 'gains', 'surge', 'soar', 'rally',
        'record', 'breakthrough', 'innovative', 'strong', 'positive',
        'outperform', 'raises', 'dividend', 'acquisition', 'partnership',
        'expansion', 'momentum', 'optimistic', 'confident', 'success'
    ]

    NEGATIVE_KEYWORDS = [
        'miss', 'misses', 'below', 'downgrade', 'downgraded', 'sell',
        'bearish', 'loss', 'losses', 'decline', 'drop', 'fall', 'plunge',
        'crash', 'warning', 'concern', 'risk', 'weak', 'negative',
        'underperform', 'cuts', 'layoff', 'lawsuit', 'investigation',
        'fraud', 'scandal', 'bankruptcy', 'default', 'recession', 'fear'
    ]

    VERY_NEGATIVE_KEYWORDS = [
        'fraud', 'scandal', 'bankruptcy', 'investigation', 'sec', 'fbi',
        'criminal', 'indictment', 'lawsuit', 'recall', 'death', 'fatal'
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_minutes: int = 15
    ):
        """
        Initialize news analyzer

        Args:
            api_key: Finnhub API key (or set FINNHUB_API_KEY env var)
            cache_dir: Directory for caching results
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.api_key = api_key or os.environ.get('FINNHUB_API_KEY', '')
        self.cache_dir = cache_dir or Path.home() / ".rdt-trading" / "news_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

        self.base_url = "https://finnhub.io/api/v1"

        # In-memory cache
        self._sentiment_cache: Dict[str, Tuple[SentimentResult, datetime]] = {}
        self._earnings_cache: Dict[str, Tuple[Optional[datetime], datetime]] = {}

        if not self.api_key:
            logger.warning(
                "No Finnhub API key - using limited sentiment analysis. "
                "Set FINNHUB_API_KEY for full features."
            )

    async def get_sentiment(self, symbol: str) -> SentimentResult:
        """
        Get sentiment analysis for a symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            SentimentResult with analysis
        """
        # Check cache
        if symbol in self._sentiment_cache:
            result, cached_time = self._sentiment_cache[symbol]
            if datetime.now() - cached_time < self.cache_ttl:
                return result

        # Fetch fresh data
        news_items = await self._fetch_news(symbol)
        earnings_date = await self._fetch_earnings_date(symbol)

        # Analyze sentiment
        result = self._analyze_sentiment(symbol, news_items, earnings_date)

        # Cache result
        self._sentiment_cache[symbol] = (result, datetime.now())

        return result

    def get_sentiment_sync(self, symbol: str) -> SentimentResult:
        """Synchronous version of get_sentiment"""
        # Check cache
        if symbol in self._sentiment_cache:
            result, cached_time = self._sentiment_cache[symbol]
            if datetime.now() - cached_time < self.cache_ttl:
                return result

        # Fetch fresh data
        news_items = self._fetch_news_sync(symbol)
        earnings_date = self._fetch_earnings_date_sync(symbol)

        # Analyze sentiment
        result = self._analyze_sentiment(symbol, news_items, earnings_date)

        # Cache result
        self._sentiment_cache[symbol] = (result, datetime.now())

        return result

    async def _fetch_news(self, symbol: str) -> List[NewsItem]:
        """Fetch news from Finnhub API"""
        if not self.api_key or not AIOHTTP_AVAILABLE:
            return []

        try:
            today = datetime.now()
            week_ago = today - timedelta(days=7)

            url = f"{self.base_url}/company-news"
            params = {
                "symbol": symbol,
                "from": week_ago.strftime("%Y-%m-%d"),
                "to": today.strftime("%Y-%m-%d"),
                "token": self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_news(symbol, data)
                    else:
                        logger.warning(f"News API error for {symbol}: {resp.status}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def _fetch_news_sync(self, symbol: str) -> List[NewsItem]:
        """Synchronous news fetch"""
        if not self.api_key or not REQUESTS_AVAILABLE:
            return []

        try:
            today = datetime.now()
            week_ago = today - timedelta(days=7)

            url = f"{self.base_url}/company-news"
            params = {
                "symbol": symbol,
                "from": week_ago.strftime("%Y-%m-%d"),
                "to": today.strftime("%Y-%m-%d"),
                "token": self.api_key
            }

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return self._parse_news(symbol, resp.json())
            return []

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    async def _fetch_earnings_date(self, symbol: str) -> Optional[datetime]:
        """Fetch next earnings date"""
        # Check cache
        if symbol in self._earnings_cache:
            date, cached_time = self._earnings_cache[symbol]
            if datetime.now() - cached_time < timedelta(hours=24):
                return date

        if not self.api_key or not AIOHTTP_AVAILABLE:
            return None

        try:
            today = datetime.now()
            future = today + timedelta(days=90)

            url = f"{self.base_url}/calendar/earnings"
            params = {
                "symbol": symbol,
                "from": today.strftime("%Y-%m-%d"),
                "to": future.strftime("%Y-%m-%d"),
                "token": self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        earnings = data.get("earningsCalendar", [])

                        if earnings:
                            date_str = earnings[0].get("date")
                            if date_str:
                                date = datetime.strptime(date_str, "%Y-%m-%d")
                                self._earnings_cache[symbol] = (date, datetime.now())
                                return date

            self._earnings_cache[symbol] = (None, datetime.now())
            return None

        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return None

    def _fetch_earnings_date_sync(self, symbol: str) -> Optional[datetime]:
        """Synchronous earnings date fetch"""
        # Check cache
        if symbol in self._earnings_cache:
            date, cached_time = self._earnings_cache[symbol]
            if datetime.now() - cached_time < timedelta(hours=24):
                return date

        if not self.api_key or not REQUESTS_AVAILABLE:
            return None

        try:
            today = datetime.now()
            future = today + timedelta(days=90)

            url = f"{self.base_url}/calendar/earnings"
            params = {
                "symbol": symbol,
                "from": today.strftime("%Y-%m-%d"),
                "to": future.strftime("%Y-%m-%d"),
                "token": self.api_key
            }

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                earnings = data.get("earningsCalendar", [])

                if earnings:
                    date_str = earnings[0].get("date")
                    if date_str:
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                        self._earnings_cache[symbol] = (date, datetime.now())
                        return date

            self._earnings_cache[symbol] = (None, datetime.now())
            return None

        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return None

    def _parse_news(self, symbol: str, data: List[Dict]) -> List[NewsItem]:
        """Parse news response into NewsItem objects"""
        items = []

        for article in data[:20]:  # Limit to 20 articles
            try:
                headline = article.get("headline", "")
                summary = article.get("summary", "")

                # Calculate sentiment score
                sentiment = self._score_text(headline + " " + summary)

                items.append(NewsItem(
                    headline=headline,
                    summary=summary,
                    source=article.get("source", "Unknown"),
                    datetime=datetime.fromtimestamp(article.get("datetime", 0)),
                    symbol=symbol,
                    sentiment_score=sentiment,
                    url=article.get("url"),
                    category=article.get("category")
                ))
            except Exception as e:
                logger.debug(f"Error parsing news article: {e}")
                continue

        return items

    def _score_text(self, text: str) -> float:
        """
        Score text sentiment using keyword matching

        Returns:
            Score from -1 (very negative) to 1 (very positive)
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        # Check for very negative keywords first
        for keyword in self.VERY_NEGATIVE_KEYWORDS:
            if keyword in text_lower:
                return -0.9  # Very negative

        # Count positive and negative keywords
        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        # Calculate score
        score = (positive_count - negative_count) / max(total, 1)
        return max(-1.0, min(1.0, score))

    def _analyze_sentiment(
        self,
        symbol: str,
        news_items: List[NewsItem],
        earnings_date: Optional[datetime]
    ) -> SentimentResult:
        """Analyze overall sentiment for a symbol"""

        # Check earnings proximity
        has_earnings_soon = False
        if earnings_date:
            days_to_earnings = (earnings_date - datetime.now()).days
            has_earnings_soon = -2 <= days_to_earnings <= 5  # 2 days before to 5 days after

        if not news_items:
            # No news - neutral sentiment
            recommendation = "avoid" if has_earnings_soon else "neutral"
            return SentimentResult(
                symbol=symbol,
                overall_sentiment=0.0,
                news_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                has_earnings_soon=has_earnings_soon,
                earnings_date=earnings_date,
                latest_news=[],
                recommendation=recommendation
            )

        # Categorize news
        positive_count = sum(1 for n in news_items if n.sentiment_score > 0.2)
        negative_count = sum(1 for n in news_items if n.sentiment_score < -0.2)
        neutral_count = len(news_items) - positive_count - negative_count

        # Calculate overall sentiment (weighted by recency)
        total_weight = 0
        weighted_sentiment = 0

        now = datetime.now()
        for item in news_items:
            # More recent news has higher weight
            age_hours = (now - item.datetime).total_seconds() / 3600
            weight = max(0.1, 1.0 - (age_hours / 168))  # Decay over 1 week

            weighted_sentiment += item.sentiment_score * weight
            total_weight += weight

        overall_sentiment = weighted_sentiment / max(total_weight, 1)

        # Determine recommendation
        if has_earnings_soon:
            recommendation = "avoid"
        elif overall_sentiment < -0.3 or negative_count > positive_count * 2:
            recommendation = "avoid"
        elif overall_sentiment < -0.1:
            recommendation = "bearish"
        elif overall_sentiment > 0.3:
            recommendation = "bullish"
        elif overall_sentiment > 0.1:
            recommendation = "bullish"
        else:
            recommendation = "neutral"

        return SentimentResult(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            news_count=len(news_items),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            has_earnings_soon=has_earnings_soon,
            earnings_date=earnings_date,
            latest_news=news_items[:5],  # Keep top 5
            recommendation=recommendation
        )

    def should_trade(self, symbol: str, direction: str = "long") -> Tuple[bool, str]:
        """
        Check if we should trade this symbol based on news

        Args:
            symbol: Stock ticker
            direction: "long" or "short"

        Returns:
            (should_trade, reason)
        """
        try:
            sentiment = self.get_sentiment_sync(symbol)

            # Avoid trading near earnings
            if sentiment.has_earnings_soon:
                days = (sentiment.earnings_date - datetime.now()).days if sentiment.earnings_date else 0
                return False, f"Earnings in {days} days - avoid"

            # Check sentiment alignment with direction
            if direction == "long":
                if sentiment.recommendation == "avoid":
                    return False, f"Negative news sentiment ({sentiment.overall_sentiment:.2f})"
                if sentiment.recommendation == "bearish":
                    return False, f"Bearish news - not suitable for long"
                return True, f"News OK (sentiment: {sentiment.overall_sentiment:.2f})"

            else:  # short
                if sentiment.recommendation == "avoid":
                    return False, f"Uncertain news environment"
                if sentiment.recommendation == "bullish":
                    return False, f"Bullish news - not suitable for short"
                return True, f"News OK for short (sentiment: {sentiment.overall_sentiment:.2f})"

        except Exception as e:
            logger.error(f"Error checking news for {symbol}: {e}")
            return True, "News check failed - proceeding with caution"

    def clear_cache(self):
        """Clear all cached data"""
        self._sentiment_cache.clear()
        self._earnings_cache.clear()


# Global instance
_analyzer: Optional[NewsSentimentAnalyzer] = None


def get_news_analyzer() -> NewsSentimentAnalyzer:
    """Get global news analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = NewsSentimentAnalyzer()
    return _analyzer
