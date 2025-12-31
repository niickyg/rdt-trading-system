"""
Schwab API Trading Client
Implements AbstractBroker interface for live trading via Schwab API
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal
from loguru import logger
import requests

from brokers.base import (
    AbstractBroker, Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, OrderStatus
)
from brokers.schwab.auth import SchwabAuth


class SchwabClient(AbstractBroker):
    """
    Schwab API Trading Client

    Implements the AbstractBroker interface for live trading through
    Charles Schwab's Trader API.

    Requires:
        - Schwab developer account (https://developer.schwab.com/)
        - App credentials (app_key, app_secret)
        - thinkorswim enabled on trading account
    """

    BASE_URL = "https://api.schwabapi.com/trader/v1"
    MARKET_DATA_URL = "https://api.schwabapi.com/marketdata/v1"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        callback_url: str = "https://localhost:8080",
        account_number: Optional[str] = None,
        token_path: Optional[str] = None
    ):
        """
        Initialize Schwab client

        Args:
            app_key: Schwab API app key
            app_secret: Schwab API app secret
            callback_url: OAuth callback URL
            account_number: Specific account to use (optional)
            token_path: Path to store OAuth tokens
        """
        self.auth = SchwabAuth(app_key, app_secret, callback_url, token_path)
        self._account_number = account_number
        self._accounts: List[Dict] = []

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

        logger.info("SchwabClient initialized")

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict]:
        """Make authenticated API request"""
        self._rate_limit()

        headers = self.auth.get_auth_header()
        if not headers:
            logger.error("Not authenticated")
            return None

        # Merge headers
        if "headers" in kwargs:
            kwargs["headers"].update(headers)
        else:
            kwargs["headers"] = headers

        url = f"{base_url or self.BASE_URL}{endpoint}"

        try:
            response = requests.request(method, url, timeout=30, **kwargs)

            if response.status_code == 401:
                # Token may have expired, try refresh
                logger.warning("Got 401, attempting token refresh")
                if self.auth.refresh_access_token():
                    kwargs["headers"] = self.auth.get_auth_header()
                    response = requests.request(method, url, timeout=30, **kwargs)

            if response.status_code >= 400:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

            if response.content:
                return response.json()
            return {}

        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def connect(self) -> bool:
        """
        Connect to Schwab API
        Runs OAuth flow if not authenticated
        """
        if not self.auth.is_authenticated:
            logger.info("Not authenticated, starting OAuth flow")
            if not self.auth.authorize_interactive():
                return False

        # Fetch accounts
        self._accounts = self._fetch_accounts()
        if not self._accounts:
            logger.error("Failed to fetch accounts")
            return False

        # Set default account if not specified
        if not self._account_number and self._accounts:
            self._account_number = self._accounts[0]["accountNumber"]
            logger.info(f"Using account: {self._account_number}")

        return True

    def disconnect(self):
        """Disconnect from API"""
        logger.info("Disconnected from Schwab API")

    def _fetch_accounts(self) -> List[Dict]:
        """Fetch all linked accounts"""
        result = self._request("GET", "/accounts")
        if result:
            return result.get("accounts", [])
        return []

    @property
    def account_hash(self) -> Optional[str]:
        """Get encrypted account hash for API calls"""
        for account in self._accounts:
            if account.get("accountNumber") == self._account_number:
                return account.get("hashValue")
        return None

    # ==================== Order Methods ====================

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY"
    ) -> Optional[Order]:
        """Place an order through Schwab API"""

        if not self.account_hash:
            logger.error("No account selected")
            return None

        # Build order payload
        order_payload = {
            "orderType": self._convert_order_type(order_type),
            "session": "NORMAL",
            "duration": time_in_force,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": self._convert_side(side),
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": "EQUITY"
                }
            }]
        }

        # Add price for limit orders
        if order_type == OrderType.LIMIT and price:
            order_payload["price"] = str(price)

        # Add stop price for stop orders
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price:
            order_payload["stopPrice"] = str(stop_price)
            if order_type == OrderType.STOP_LIMIT and price:
                order_payload["price"] = str(price)

        # Submit order
        result = self._request(
            "POST",
            f"/accounts/{self.account_hash}/orders",
            json=order_payload
        )

        if result is not None:
            # Schwab returns order ID in location header
            # For now, create order object with pending status
            order = Order(
                order_id=f"schwab_{int(time.time()*1000)}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.PENDING,
                price=price,
                stop_price=stop_price,
                created_at=datetime.now()
            )
            logger.info(f"Order placed: {order}")
            return order

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.account_hash:
            return False

        # Extract actual order ID if prefixed
        actual_id = order_id.replace("schwab_", "")

        result = self._request(
            "DELETE",
            f"/accounts/{self.account_hash}/orders/{actual_id}"
        )

        return result is not None

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        if not self.account_hash:
            return None

        actual_id = order_id.replace("schwab_", "")

        result = self._request(
            "GET",
            f"/accounts/{self.account_hash}/orders/{actual_id}"
        )

        if result:
            return self._parse_order(result)
        return None

    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        if not self.account_hash:
            return []

        result = self._request(
            "GET",
            f"/accounts/{self.account_hash}/orders",
            params={"status": "QUEUED,ACCEPTED,WORKING"}
        )

        if result:
            return [self._parse_order(o) for o in result.get("orders", [])]
        return []

    # ==================== Position Methods ====================

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        if not self.account_hash:
            return {}

        result = self._request(
            "GET",
            f"/accounts/{self.account_hash}",
            params={"fields": "positions"}
        )

        positions = {}
        if result:
            for pos in result.get("securitiesAccount", {}).get("positions", []):
                symbol = pos.get("instrument", {}).get("symbol")
                if symbol:
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=int(pos.get("longQuantity", 0) - pos.get("shortQuantity", 0)),
                        avg_price=float(pos.get("averagePrice", 0)),
                        current_price=float(pos.get("currentPrice", 0)),
                        market_value=float(pos.get("marketValue", 0)),
                        unrealized_pnl=float(pos.get("currentDayProfitLoss", 0)),
                        unrealized_pnl_pct=float(pos.get("currentDayProfitLossPercentage", 0))
                    )

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        positions = self.get_positions()
        return positions.get(symbol)

    # ==================== Quote Methods ====================

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote for a symbol"""
        result = self._request(
            "GET",
            f"/{symbol}/quotes",
            base_url=self.MARKET_DATA_URL
        )

        if result and symbol in result:
            data = result[symbol].get("quote", {})
            return Quote(
                symbol=symbol,
                bid=float(data.get("bidPrice", 0)),
                ask=float(data.get("askPrice", 0)),
                last=float(data.get("lastPrice", 0)),
                volume=int(data.get("totalVolume", 0)),
                timestamp=datetime.now()
            )

        return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols"""
        quotes = {}
        # Schwab API supports batch quotes
        symbols_str = ",".join(symbols)

        result = self._request(
            "GET",
            f"/quotes",
            base_url=self.MARKET_DATA_URL,
            params={"symbols": symbols_str}
        )

        if result:
            for symbol, data in result.items():
                quote_data = data.get("quote", {})
                quotes[symbol] = Quote(
                    symbol=symbol,
                    bid=float(quote_data.get("bidPrice", 0)),
                    ask=float(quote_data.get("askPrice", 0)),
                    last=float(quote_data.get("lastPrice", 0)),
                    volume=int(quote_data.get("totalVolume", 0)),
                    timestamp=datetime.now()
                )

        return quotes

    # ==================== Account Methods ====================

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        if not self.account_hash:
            return None

        result = self._request(
            "GET",
            f"/accounts/{self.account_hash}",
            params={"fields": "positions"}
        )

        if result:
            account = result.get("securitiesAccount", {})
            balances = account.get("currentBalances", {})

            return AccountInfo(
                account_id=self._account_number,
                buying_power=float(balances.get("buyingPower", 0)),
                cash=float(balances.get("cashBalance", 0)),
                equity=float(balances.get("equity", 0)),
                day_trades_remaining=5 - int(account.get("roundTrips", 0)),
                pattern_day_trader=account.get("isPatternDayTrader", False)
            )

        return None

    def get_buying_power(self) -> float:
        """Get current buying power"""
        info = self.get_account_info()
        return info.buying_power if info else 0.0

    # ==================== Helper Methods ====================

    def _convert_side(self, side: OrderSide) -> str:
        """Convert OrderSide to Schwab instruction"""
        mapping = {
            OrderSide.BUY: "BUY",
            OrderSide.SELL: "SELL",
            OrderSide.BUY_TO_COVER: "BUY_TO_COVER",
            OrderSide.SELL_SHORT: "SELL_SHORT"
        }
        return mapping.get(side, "BUY")

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType to Schwab order type"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP",
            OrderType.STOP_LIMIT: "STOP_LIMIT"
        }
        return mapping.get(order_type, "MARKET")

    def _parse_order(self, data: Dict) -> Order:
        """Parse Schwab order response into Order object"""
        leg = data.get("orderLegCollection", [{}])[0]

        # Map Schwab status to our status
        status_map = {
            "QUEUED": OrderStatus.PENDING,
            "ACCEPTED": OrderStatus.PENDING,
            "WORKING": OrderStatus.OPEN,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }

        # Map instruction to side
        side_map = {
            "BUY": OrderSide.BUY,
            "SELL": OrderSide.SELL,
            "BUY_TO_COVER": OrderSide.BUY_TO_COVER,
            "SELL_SHORT": OrderSide.SELL_SHORT
        }

        # Map order type
        type_map = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP,
            "STOP_LIMIT": OrderType.STOP_LIMIT
        }

        return Order(
            order_id=f"schwab_{data.get('orderId', '')}",
            symbol=leg.get("instrument", {}).get("symbol", ""),
            side=side_map.get(leg.get("instruction", ""), OrderSide.BUY),
            quantity=int(leg.get("quantity", 0)),
            order_type=type_map.get(data.get("orderType", ""), OrderType.MARKET),
            status=status_map.get(data.get("status", ""), OrderStatus.PENDING),
            price=float(data.get("price", 0)) or None,
            stop_price=float(data.get("stopPrice", 0)) or None,
            filled_quantity=int(data.get("filledQuantity", 0)),
            avg_fill_price=float(data.get("averagePrice", 0)) or None,
            created_at=datetime.now()  # Would parse from response
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated"""
        return self.auth.is_authenticated and bool(self._accounts)
