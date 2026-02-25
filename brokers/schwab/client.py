"""
Schwab API Trading Client
Implements BrokerInterface for live trading via Schwab API
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
import requests

from brokers.broker_interface import (
    BrokerInterface, Order, Position, Quote, AccountInfo,
    OrderSide, OrderType, OrderStatus,
    BrokerError, AuthenticationError, ConnectionError, OrderError,
    InsufficientFundsError
)
from brokers.schwab.auth import SchwabAuth


class SchwabClient(BrokerInterface):
    """
    Schwab API Trading Client

    Implements the BrokerInterface for live trading through
    Charles Schwab's Trader API.

    Requires:
        - Schwab developer account (https://developer.schwab.com/)
        - App credentials (app_key, app_secret)
        - thinkorswim enabled on trading account

    Example:
        client = SchwabClient(app_key="...", app_secret="...")
        if client.connect():
            account = client.get_account()
            order = client.place_order("AAPL", OrderSide.BUY, 10)
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
        Initialize Schwab client.

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
        self._connected = False

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

        # Order tracking
        self._orders: Dict[str, Order] = {}

        logger.info("SchwabClient initialized")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
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
        """Make authenticated API request."""
        self._rate_limit()

        headers = self.auth.get_auth_header()
        if not headers:
            raise AuthenticationError("Not authenticated - token missing or expired")

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
                else:
                    raise AuthenticationError(
                        "Authentication failed - token refresh failed"
                    )

            if response.status_code == 400:
                error_text = response.text
                logger.error(f"Bad request: {error_text}")
                raise OrderError(f"Bad request: {error_text}")

            if response.status_code == 403:
                raise AuthenticationError(
                    "Access denied - check API permissions"
                )

            if response.status_code == 404:
                return None

            if response.status_code >= 400:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise BrokerError(f"API error: {response.status_code}")

            if response.content:
                return response.json()
            return {}

        except requests.exceptions.Timeout:
            raise ConnectionError("Request timeout")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except (AuthenticationError, BrokerError):
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise BrokerError(f"Request failed: {e}")

    # ==================== Connection Methods ====================

    def connect(self) -> bool:
        """
        Connect to Schwab API.
        Runs OAuth flow if not authenticated.

        Returns:
            True if connection successful.
        """
        try:
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
                self._account_number = self._accounts[0].get("accountNumber")
                logger.info(f"Using account: {self._account_number}")

            self._connected = True
            logger.info("Connected to Schwab API")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from API."""
        self._connected = False
        logger.info("Disconnected from Schwab API")

    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self._connected and self.auth.is_authenticated and bool(self._accounts)

    def _fetch_accounts(self) -> List[Dict]:
        """Fetch all linked accounts."""
        try:
            result = self._request("GET", "/accounts")
            if result:
                return result.get("accounts", [])
            return []
        except Exception as e:
            logger.error(f"Failed to fetch accounts: {e}")
            return []

    @property
    def account_hash(self) -> Optional[str]:
        """Get encrypted account hash for API calls."""
        for account in self._accounts:
            if account.get("accountNumber") == self._account_number:
                return account.get("hashValue")
        return None

    # ==================== Account Methods ====================

    def get_account(self) -> AccountInfo:
        """Get account information."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            raise BrokerError("No account selected")

        result = self._request(
            "GET",
            f"/accounts/{self.account_hash}",
            params={"fields": "positions"}
        )

        if not result:
            raise BrokerError("Failed to get account info")

        account = result.get("securitiesAccount", {})
        balances = account.get("currentBalances", {})

        return AccountInfo(
            account_id=self._account_number or "",
            buying_power=float(balances.get("buyingPower", 0)),
            cash=float(balances.get("cashBalance", 0)),
            equity=float(balances.get("equity", 0)),
            day_trades_remaining=5 - int(account.get("roundTrips", 0)),
            pattern_day_trader=account.get("isPatternDayTrader", False)
        )

    # Legacy alias
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information (legacy alias)."""
        try:
            return self.get_account()
        except Exception:
            return None

    def get_buying_power(self) -> float:
        """Get current buying power."""
        try:
            info = self.get_account()
            return info.buying_power
        except Exception:
            return 0.0

    # ==================== Position Methods ====================

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

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
                    quantity = int(
                        pos.get("longQuantity", 0) - pos.get("shortQuantity", 0)
                    )
                    avg_price = float(pos.get("averagePrice", 0))
                    current_price = float(pos.get("currentPrice", 0))
                    market_value = float(pos.get("marketValue", 0))
                    unrealized_pnl = float(pos.get("currentDayProfitLoss", 0))
                    unrealized_pnl_pct = float(
                        pos.get("currentDayProfitLossPercentage", 0)
                    )

                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_cost=avg_price,
                        current_price=current_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct
                    )

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        return positions.get(symbol)

    # ==================== Order Methods ====================

    @property
    def supports_extended_hours(self) -> bool:
        """Schwab supports extended hours trading."""
        return True

    def _convert_session(self, session: str) -> str:
        """
        Convert session parameter to Schwab session value.

        Args:
            session: Our session value ('regular', 'premarket', 'afterhours', 'extended')

        Returns:
            Schwab session value ('NORMAL', 'AM', 'PM', 'SEAMLESS')
        """
        session_map = {
            'regular': 'NORMAL',
            'premarket': 'AM',
            'afterhours': 'PM',
            'extended': 'SEAMLESS'  # SEAMLESS covers both pre-market and after-hours
        }
        return session_map.get(session, 'NORMAL')

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        session: str = "regular"
    ) -> Order:
        """Place an order through Schwab API with extended hours support."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            raise BrokerError("No account selected")

        # Validate order
        is_valid, error_msg = self.validate_order(
            symbol, side, quantity, order_type, price, stop_price
        )
        if not is_valid:
            raise OrderError(error_msg)

        # Extended hours requires limit orders (market orders not allowed)
        if session != 'regular' and order_type == OrderType.MARKET:
            raise OrderError(
                "Extended hours trading requires limit orders. "
                "Market orders are not allowed during pre-market or after-hours."
            )

        # Extended hours requires limit price
        if session != 'regular' and price is None:
            raise OrderError(
                "Extended hours trading requires a limit price."
            )

        # Build order payload
        schwab_session = self._convert_session(session)
        order_payload = {
            "orderType": self._convert_order_type(order_type),
            "session": schwab_session,
            "duration": time_in_force,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": self._convert_side(side),
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol.upper(),
                    "assetType": "EQUITY"
                }
            }]
        }

        # Add price for limit orders
        if order_type == OrderType.LIMIT and price:
            order_payload["price"] = str(round(price, 2))

        # Add stop price for stop orders
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price:
            order_payload["stopPrice"] = str(round(stop_price, 2))
            if order_type == OrderType.STOP_LIMIT and price:
                order_payload["price"] = str(round(price, 2))

        # Log extended hours order
        if session != 'regular':
            logger.info(
                f"Placing extended hours order: {symbol} {side.value} {quantity} "
                f"session={session} (Schwab: {schwab_session})"
            )

        # Submit order
        try:
            result = self._request(
                "POST",
                f"/accounts/{self.account_hash}/orders",
                json=order_payload
            )

            # Schwab returns order ID in location header on success
            order_id = f"schwab_{int(time.time()*1000)}"

            order = Order(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.PENDING,
                price=price,
                stop_price=stop_price,
                created_at=datetime.now(),
                time_in_force=time_in_force,
                session=session
            )

            self._orders[order_id] = order
            logger.info(f"Order placed: {order}")
            return order

        except BrokerError:
            raise
        except Exception as e:
            raise OrderError(f"Order placement failed: {e}")

    def get_extended_hours_quote(self, symbol: str) -> 'Quote':
        """
        Get extended hours quote for a symbol from Schwab.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote with extended hours data.
        """
        try:
            # Schwab provides extended hours data in the regular quote endpoint
            result = self._request(
                "GET",
                f"/{symbol.upper()}/quotes",
                base_url=self.MARKET_DATA_URL
            )

            if result and symbol.upper() in result:
                data = result[symbol.upper()].get("quote", {})

                # Determine current session
                from utils.timezone import get_extended_hours_session
                current_session = get_extended_hours_session()

                # Extract extended hours data from Schwab response
                # Schwab provides extended hours in 'extendedChange' fields
                extended_last = float(data.get("mark", data.get("lastPrice", 0)))
                extended_bid = float(data.get("bidPrice", 0))
                extended_ask = float(data.get("askPrice", 0))

                # Pre/post market specific fields
                if current_session == 'premarket':
                    extended_last = float(data.get("mark", extended_last))
                elif current_session == 'afterhours':
                    extended_last = float(data.get("postMarketPrice", data.get("mark", extended_last)))

                quote = Quote(
                    symbol=symbol.upper(),
                    bid=float(data.get("bidPrice", 0)),
                    ask=float(data.get("askPrice", 0)),
                    last=float(data.get("lastPrice", 0)),
                    volume=int(data.get("totalVolume", 0)),
                    timestamp=datetime.now(),
                    bid_size=int(data.get("bidSize", 0)),
                    ask_size=int(data.get("askSize", 0)),
                    high=float(data.get("highPrice", 0)),
                    low=float(data.get("lowPrice", 0)),
                    open=float(data.get("openPrice", 0)),
                    prev_close=float(data.get("closePrice", 0)),
                    # Extended hours fields
                    extended_hours=current_session in ('premarket', 'afterhours'),
                    session=current_session,
                    extended_bid=extended_bid,
                    extended_ask=extended_ask,
                    extended_last=extended_last,
                    extended_volume=int(data.get("totalVolume", 0))
                )

                return quote

            raise BrokerError(f"No extended hours quote data for {symbol}")

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get extended hours quote for {symbol}: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            return False

        # Extract actual order ID if prefixed
        actual_id = order_id.replace("schwab_", "")

        try:
            self._request(
                "DELETE",
                f"/accounts/{self.account_hash}/orders/{actual_id}"
            )

            # Update local order status
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            return None

        actual_id = order_id.replace("schwab_", "")

        try:
            result = self._request(
                "GET",
                f"/accounts/{self.account_hash}/orders/{actual_id}"
            )

            if result:
                return self._parse_order(result)
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")

        # Return cached order if API fails
        return self._orders.get(order_id)

    # Legacy alias
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order status (legacy alias)."""
        return self.get_order_status(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            return []

        try:
            result = self._request(
                "GET",
                f"/accounts/{self.account_hash}/orders",
                params={"status": "QUEUED,ACCEPTED,WORKING"}
            )

            if result:
                return [self._parse_order(o) for o in result.get("orders", [])]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")

        return []

    # ==================== Quote Methods ====================

    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol."""
        try:
            result = self._request(
                "GET",
                f"/{symbol.upper()}/quotes",
                base_url=self.MARKET_DATA_URL
            )

            if result and symbol.upper() in result:
                data = result[symbol.upper()].get("quote", {})
                return Quote(
                    symbol=symbol.upper(),
                    bid=float(data.get("bidPrice", 0)),
                    ask=float(data.get("askPrice", 0)),
                    last=float(data.get("lastPrice", 0)),
                    volume=int(data.get("totalVolume", 0)),
                    timestamp=datetime.now(),
                    bid_size=int(data.get("bidSize", 0)),
                    ask_size=int(data.get("askSize", 0)),
                    high=float(data.get("highPrice", 0)),
                    low=float(data.get("lowPrice", 0)),
                    open=float(data.get("openPrice", 0)),
                    prev_close=float(data.get("closePrice", 0))
                )

            raise BrokerError(f"No quote data for {symbol}")

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get quote for {symbol}: {e}")

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        quotes = {}

        if not symbols:
            return quotes

        # Schwab API supports batch quotes
        symbols_upper = [s.upper() for s in symbols]
        symbols_str = ",".join(symbols_upper)

        try:
            result = self._request(
                "GET",
                "/quotes",
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
                        timestamp=datetime.now(),
                        bid_size=int(quote_data.get("bidSize", 0)),
                        ask_size=int(quote_data.get("askSize", 0)),
                        high=float(quote_data.get("highPrice", 0)),
                        low=float(quote_data.get("lowPrice", 0)),
                        open=float(quote_data.get("openPrice", 0)),
                        prev_close=float(quote_data.get("closePrice", 0))
                    )

        except Exception as e:
            logger.warning(f"Batch quote failed, falling back to individual: {e}")
            # Fallback to individual quotes
            for symbol in symbols:
                try:
                    quotes[symbol] = self.get_quote(symbol)
                except Exception:
                    pass

        return quotes

    # ==================== Helper Methods ====================

    def _convert_side(self, side: OrderSide) -> str:
        """Convert OrderSide to Schwab instruction."""
        mapping = {
            OrderSide.BUY: "BUY",
            OrderSide.SELL: "SELL",
            OrderSide.BUY_TO_COVER: "BUY_TO_COVER",
            OrderSide.SELL_SHORT: "SELL_SHORT"
        }
        return mapping.get(side, "BUY")

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType to Schwab order type."""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP",
            OrderType.STOP_LIMIT: "STOP_LIMIT",
            OrderType.TRAILING_STOP: "TRAILING_STOP"
        }
        return mapping.get(order_type, "MARKET")

    def _parse_order(self, data: Dict) -> Order:
        """Parse Schwab order response into Order object."""
        leg = data.get("orderLegCollection", [{}])[0]

        # Map Schwab status to our status
        status_map = {
            "QUEUED": OrderStatus.PENDING,
            "ACCEPTED": OrderStatus.PENDING,
            "WORKING": OrderStatus.OPEN,
            "PENDING_ACTIVATION": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
            "REPLACED": OrderStatus.CANCELLED,
            "PENDING_CANCEL": OrderStatus.PENDING,
            "PENDING_REPLACE": OrderStatus.PENDING
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
            "STOP_LIMIT": OrderType.STOP_LIMIT,
            "TRAILING_STOP": OrderType.TRAILING_STOP
        }

        order_id = f"schwab_{data.get('orderId', '')}"
        schwab_status = data.get("status", "PENDING")

        return Order(
            order_id=order_id,
            symbol=leg.get("instrument", {}).get("symbol", ""),
            side=side_map.get(leg.get("instruction", ""), OrderSide.BUY),
            quantity=int(leg.get("quantity", 0)),
            order_type=type_map.get(data.get("orderType", ""), OrderType.MARKET),
            status=status_map.get(schwab_status, OrderStatus.PENDING),
            price=float(data.get("price", 0)) or None,
            stop_price=float(data.get("stopPrice", 0)) or None,
            filled_quantity=int(data.get("filledQuantity", 0)),
            avg_fill_price=float(data.get("averagePrice", 0)) or None,
            created_at=datetime.now(),
            broker_order_id=str(data.get("orderId", ""))
        )

    # ==================== Advanced Order Methods ====================

    def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float],
        take_profit_price: float,
        stop_loss_price: float,
        entry_type: OrderType = OrderType.LIMIT
    ) -> tuple:
        """
        Place a bracket order through Schwab API.

        Schwab supports bracket orders via orderStrategyType='TRIGGER'.

        Args:
            symbol: Stock ticker symbol
            side: OrderSide.BUY for long, OrderSide.SELL_SHORT for short
            quantity: Number of shares
            entry_price: Entry limit price (None for market order)
            take_profit_price: Profit target price
            stop_loss_price: Stop loss trigger price
            entry_type: Entry order type (MARKET or LIMIT)

        Returns:
            Tuple of (entry_order, take_profit_order, stop_loss_order)
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            raise BrokerError("No account selected")

        # Determine exit side
        if side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            exit_side = "SELL"
        else:
            exit_side = "BUY_TO_COVER"

        # Build entry order
        entry_instruction = self._convert_side(side)
        entry_order_type = self._convert_order_type(entry_type)

        # Build bracket order payload (Schwab uses TRIGGER orderStrategyType)
        order_payload = {
            "orderType": entry_order_type,
            "session": "NORMAL",
            "duration": "GTC",
            "orderStrategyType": "TRIGGER",
            "orderLegCollection": [{
                "instruction": entry_instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol.upper(),
                    "assetType": "EQUITY"
                }
            }],
            "childOrderStrategies": [
                # OCO group for take profit and stop loss
                {
                    "orderStrategyType": "OCO",
                    "childOrderStrategies": [
                        # Take profit order
                        {
                            "orderType": "LIMIT",
                            "session": "NORMAL",
                            "duration": "GTC",
                            "price": str(round(take_profit_price, 2)),
                            "orderStrategyType": "SINGLE",
                            "orderLegCollection": [{
                                "instruction": exit_side,
                                "quantity": quantity,
                                "instrument": {
                                    "symbol": symbol.upper(),
                                    "assetType": "EQUITY"
                                }
                            }]
                        },
                        # Stop loss order
                        {
                            "orderType": "STOP",
                            "session": "NORMAL",
                            "duration": "GTC",
                            "stopPrice": str(round(stop_loss_price, 2)),
                            "orderStrategyType": "SINGLE",
                            "orderLegCollection": [{
                                "instruction": exit_side,
                                "quantity": quantity,
                                "instrument": {
                                    "symbol": symbol.upper(),
                                    "assetType": "EQUITY"
                                }
                            }]
                        }
                    ]
                }
            ]
        }

        # Add entry price for limit orders
        if entry_type == OrderType.LIMIT and entry_price:
            order_payload["price"] = str(round(entry_price, 2))

        try:
            result = self._request(
                "POST",
                f"/accounts/{self.account_hash}/orders",
                json=order_payload
            )

            # Create order objects
            timestamp = int(time.time() * 1000)
            entry_order_id = f"schwab_bracket_{timestamp}_entry"
            tp_order_id = f"schwab_bracket_{timestamp}_tp"
            sl_order_id = f"schwab_bracket_{timestamp}_sl"

            exit_order_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY_TO_COVER

            entry_order = Order(
                order_id=entry_order_id,
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=entry_type,
                price=entry_price,
                status=OrderStatus.PENDING,
                created_at=datetime.now()
            )

            take_profit_order = Order(
                order_id=tp_order_id,
                symbol=symbol.upper(),
                side=exit_order_side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=take_profit_price,
                status=OrderStatus.PENDING,
                created_at=datetime.now()
            )

            stop_loss_order = Order(
                order_id=sl_order_id,
                symbol=symbol.upper(),
                side=exit_order_side,
                quantity=quantity,
                order_type=OrderType.STOP,
                stop_price=stop_loss_price,
                status=OrderStatus.PENDING,
                created_at=datetime.now()
            )

            self._orders[entry_order_id] = entry_order
            self._orders[tp_order_id] = take_profit_order
            self._orders[sl_order_id] = stop_loss_order

            logger.info(
                f"Bracket order placed: {symbol} - Entry: {entry_price or 'MKT'}, "
                f"TP: {take_profit_price}, SL: {stop_loss_price}"
            )

            return entry_order, take_profit_order, stop_loss_order

        except BrokerError:
            raise
        except Exception as e:
            raise OrderError(f"Bracket order failed: {e}")

    def place_trailing_stop(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        activation_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Order:
        """
        Place a trailing stop order through Schwab API.

        Args:
            symbol: Stock ticker symbol
            side: SELL for long exit, BUY_TO_COVER for short exit
            quantity: Number of shares
            trail_amount: Trail by fixed dollar amount
            trail_percent: Trail by percentage
            activation_price: Price at which trailing begins (optional)
            time_in_force: Order duration

        Returns:
            Order object with trailing stop details
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            raise BrokerError("No account selected")

        if trail_amount is None and trail_percent is None:
            raise OrderError("Either trail_amount or trail_percent must be specified")

        instruction = self._convert_side(side)

        # Build trailing stop payload
        order_payload = {
            "orderType": "TRAILING_STOP",
            "session": "NORMAL",
            "duration": time_in_force,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol.upper(),
                    "assetType": "EQUITY"
                }
            }]
        }

        # Set trailing type and amount
        if trail_percent is not None:
            order_payload["stopPriceLinkBasis"] = "LAST"
            order_payload["stopPriceLinkType"] = "PERCENT"
            order_payload["stopPriceOffset"] = trail_percent
        else:
            order_payload["stopPriceLinkBasis"] = "LAST"
            order_payload["stopPriceLinkType"] = "VALUE"
            order_payload["stopPriceOffset"] = trail_amount

        try:
            result = self._request(
                "POST",
                f"/accounts/{self.account_hash}/orders",
                json=order_payload
            )

            order_id = f"schwab_trail_{int(time.time() * 1000)}"

            order = Order(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side,
                quantity=quantity,
                order_type=OrderType.TRAILING_STOP,
                stop_price=trail_amount or (trail_percent if trail_percent else 0),
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                time_in_force=time_in_force
            )

            self._orders[order_id] = order
            logger.info(
                f"Trailing stop placed: {symbol} - Trail: "
                f"${trail_amount or 'N/A'} / {trail_percent or 'N/A'}%"
            )

            return order

        except BrokerError:
            raise
        except Exception as e:
            raise OrderError(f"Trailing stop order failed: {e}")

    def place_oco_order(
        self,
        symbol: str,
        orders: List[Dict]
    ) -> List[Order]:
        """
        Place OCO (one-cancels-other) orders through Schwab API.

        Args:
            symbol: Stock ticker symbol
            orders: List of order specifications

        Returns:
            List of Order objects linked as OCO
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            raise BrokerError("No account selected")

        if len(orders) < 2:
            raise OrderError("OCO requires at least 2 orders")

        # Build child order strategies
        child_orders = []
        for spec in orders:
            order_side = spec.get("side", OrderSide.SELL)
            order_type = spec.get("order_type", OrderType.MARKET)
            quantity = spec.get("quantity", 1)
            price = spec.get("price")
            stop_price = spec.get("stop_price")

            child_order = {
                "orderType": self._convert_order_type(order_type),
                "session": "NORMAL",
                "duration": spec.get("time_in_force", "GTC"),
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [{
                    "instruction": self._convert_side(order_side),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol.upper(),
                        "assetType": "EQUITY"
                    }
                }]
            }

            if order_type == OrderType.LIMIT and price:
                child_order["price"] = str(round(price, 2))
            elif order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price:
                child_order["stopPrice"] = str(round(stop_price, 2))
                if order_type == OrderType.STOP_LIMIT and price:
                    child_order["price"] = str(round(price, 2))

            child_orders.append(child_order)

        # Build OCO order payload
        order_payload = {
            "orderStrategyType": "OCO",
            "childOrderStrategies": child_orders
        }

        try:
            result = self._request(
                "POST",
                f"/accounts/{self.account_hash}/orders",
                json=order_payload
            )

            # Create order objects
            timestamp = int(time.time() * 1000)
            result_orders = []

            for i, spec in enumerate(orders):
                order_id = f"schwab_oco_{timestamp}_{i}"
                order = Order(
                    order_id=order_id,
                    symbol=symbol.upper(),
                    side=spec.get("side", OrderSide.SELL),
                    quantity=spec.get("quantity", 1),
                    order_type=spec.get("order_type", OrderType.MARKET),
                    price=spec.get("price"),
                    stop_price=spec.get("stop_price"),
                    status=OrderStatus.PENDING,
                    created_at=datetime.now()
                )
                self._orders[order_id] = order
                result_orders.append(order)

            logger.info(f"OCO order placed: {symbol} with {len(result_orders)} orders")

            return result_orders

        except BrokerError:
            raise
        except Exception as e:
            raise OrderError(f"OCO order failed: {e}")

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """
        Modify an existing order through Schwab API.

        Schwab supports order replacement (cancel and replace).

        Args:
            order_id: Order ID to modify
            price: New limit price (optional)
            stop_price: New stop trigger price (optional)
            quantity: New quantity (optional)

        Returns:
            True if modification successful
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Schwab API")

        if not self.account_hash:
            return False

        # Get existing order
        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found for modification")
            return False

        existing_order = self._orders[order_id]
        actual_id = order_id.replace("schwab_", "").split("_")[0]

        # Build replacement order payload
        order_payload = {
            "orderType": self._convert_order_type(existing_order.order_type),
            "session": "NORMAL",
            "duration": existing_order.time_in_force,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": self._convert_side(existing_order.side),
                "quantity": quantity or existing_order.quantity,
                "instrument": {
                    "symbol": existing_order.symbol,
                    "assetType": "EQUITY"
                }
            }]
        }

        # Update prices
        if existing_order.order_type == OrderType.LIMIT:
            new_price = price or existing_order.price
            if new_price:
                order_payload["price"] = str(round(new_price, 2))

        if existing_order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            new_stop = stop_price or existing_order.stop_price
            if new_stop:
                order_payload["stopPrice"] = str(round(new_stop, 2))
            if existing_order.order_type == OrderType.STOP_LIMIT:
                new_price = price or existing_order.price
                if new_price:
                    order_payload["price"] = str(round(new_price, 2))

        try:
            # Schwab uses PUT for order replacement
            self._request(
                "PUT",
                f"/accounts/{self.account_hash}/orders/{actual_id}",
                json=order_payload
            )

            # Update local order
            if price is not None:
                existing_order.price = price
            if stop_price is not None:
                existing_order.stop_price = stop_price
            if quantity is not None:
                existing_order.quantity = quantity

            logger.info(f"Order {order_id} modified successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False

    # Legacy property (for backward compatibility)
    @property
    def avg_price(self) -> float:
        """Backward compatibility property."""
        return 0.0
