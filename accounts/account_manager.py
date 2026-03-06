"""
Account Manager for Multiple Trading Accounts.

Provides centralized management of trading accounts across multiple brokers.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from accounts.models import (
    TradingAccount,
    BrokerType,
    AccountStore,
    get_account_store
)
from brokers import get_broker, BrokerInterface, BrokerError


class AccountNotFoundError(Exception):
    """Raised when an account is not found."""
    pass


class AccountValidationError(Exception):
    """Raised when account validation fails."""
    pass


class AccountManager:
    """
    Manages multiple trading accounts for a user.

    Provides methods to:
    - Add, remove, and update accounts
    - Set default account
    - Connect to brokers
    - Cache broker connections

    Example:
        manager = AccountManager(user_id="user123")

        # Add accounts
        account_id = manager.add_account(
            name="Main Trading",
            broker_type="schwab",
            credentials={"app_key": "...", "app_secret": "..."}
        )

        # Get broker for trading
        broker = manager.get_broker(account_id)
        broker.connect()
        positions = broker.get_positions()
    """

    def __init__(
        self,
        user_id: str,
        store: Optional[AccountStore] = None
    ):
        """
        Initialize account manager for a user.

        Args:
            user_id: The user's ID
            store: Optional custom account store
        """
        self.user_id = user_id
        self._store = store or get_account_store()
        self._broker_cache: Dict[str, BrokerInterface] = {}

        logger.info(f"AccountManager initialized for user: {user_id}")

    def add_account(
        self,
        name: str,
        broker_type: str,
        credentials: Dict[str, Any],
        is_default: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new trading account.

        Args:
            name: Display name for the account
            broker_type: Type of broker ("paper", "schwab", "ibkr")
            credentials: Broker-specific credentials
            is_default: Whether to set as default account
            metadata: Additional account metadata

        Returns:
            The new account's ID

        Raises:
            AccountValidationError: If validation fails
        """
        # Validate broker type
        try:
            broker_enum = BrokerType(broker_type.lower())
        except ValueError:
            raise AccountValidationError(
                f"Invalid broker type: {broker_type}. "
                f"Valid types: {[t.value for t in BrokerType]}"
            )

        # Validate name
        if not name or len(name.strip()) < 1:
            raise AccountValidationError("Account name is required")

        if len(name) > 100:
            raise AccountValidationError("Account name must be 100 characters or less")

        # Validate credentials based on broker type
        self._validate_credentials(broker_enum, credentials)

        # If this is the first account or is_default, handle default status
        user_accounts = self._store.get_by_user(self.user_id)

        if not user_accounts:
            is_default = True  # First account is always default
        elif is_default:
            # Remove default from other accounts
            for acc in user_accounts:
                if acc.is_default:
                    acc.is_default = False
                    self._store.update(acc)

        # Create account
        account = TradingAccount.create(
            user_id=self.user_id,
            name=name.strip(),
            broker_type=broker_enum,
            credentials=credentials,
            is_default=is_default,
            metadata=metadata
        )

        self._store.add(account)
        logger.info(f"Added account: {account.name} ({account.id}) for user {self.user_id}")

        return account.id

    def _validate_credentials(
        self,
        broker_type: BrokerType,
        credentials: Dict[str, Any]
    ) -> None:
        """
        Validate credentials for a broker type.

        Args:
            broker_type: Type of broker
            credentials: Credentials to validate

        Raises:
            AccountValidationError: If validation fails
        """
        if broker_type == BrokerType.SCHWAB:
            required = ["app_key", "app_secret"]
            for field in required:
                if field not in credentials or not credentials[field]:
                    raise AccountValidationError(
                        f"Schwab account requires {field}"
                    )

        elif broker_type == BrokerType.IBKR:
            # IBKR can use defaults, but if specified, validate
            if "port" in credentials:
                port = credentials["port"]
                if not isinstance(port, int) or port < 1 or port > 65535:
                    raise AccountValidationError(
                        "IBKR port must be a valid port number"
                    )

        elif broker_type == BrokerType.PAPER:
            # Paper trading has optional initial_balance
            if "initial_balance" in credentials:
                balance = credentials["initial_balance"]
                if not isinstance(balance, (int, float)) or balance < 0:
                    raise AccountValidationError(
                        "Paper account initial_balance must be a positive number"
                    )

    def remove_account(self, account_id: str) -> bool:
        """
        Remove a trading account.

        Args:
            account_id: ID of account to remove

        Returns:
            True if removed successfully

        Raises:
            AccountNotFoundError: If account not found
        """
        account = self._store.get(account_id)

        if not account:
            raise AccountNotFoundError(f"Account not found: {account_id}")

        if account.user_id != self.user_id:
            raise AccountNotFoundError(f"Account not found: {account_id}")

        # Disconnect broker if cached
        if account_id in self._broker_cache:
            try:
                self._broker_cache[account_id].disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting broker: {e}")
            del self._broker_cache[account_id]

        # If removing default, set another account as default
        was_default = account.is_default
        result = self._store.delete(account_id)

        if was_default and result:
            remaining = self._store.get_by_user(self.user_id)
            if remaining:
                remaining[0].is_default = True
                self._store.update(remaining[0])
                logger.info(f"Set {remaining[0].name} as new default account")

        logger.info(f"Removed account: {account_id}")
        return result

    def get_account(self, account_id: str) -> TradingAccount:
        """
        Get account details.

        Args:
            account_id: ID of account to retrieve

        Returns:
            TradingAccount instance

        Raises:
            AccountNotFoundError: If account not found
        """
        account = self._store.get(account_id)

        if not account or account.user_id != self.user_id:
            raise AccountNotFoundError(f"Account not found: {account_id}")

        return account

    def get_all_accounts(self) -> List[TradingAccount]:
        """
        Get all accounts for the user.

        Returns:
            List of TradingAccount instances
        """
        return self._store.get_by_user(self.user_id)

    def get_active_accounts(self) -> List[TradingAccount]:
        """
        Get all active accounts for the user.

        Returns:
            List of active TradingAccount instances
        """
        return [
            acc for acc in self._store.get_by_user(self.user_id)
            if acc.is_active
        ]

    def set_default_account(self, account_id: str) -> bool:
        """
        Set the default trading account.

        Args:
            account_id: ID of account to set as default

        Returns:
            True if set successfully

        Raises:
            AccountNotFoundError: If account not found
        """
        account = self._store.get(account_id)

        if not account or account.user_id != self.user_id:
            raise AccountNotFoundError(f"Account not found: {account_id}")

        # Remove default from other accounts
        for acc in self._store.get_by_user(self.user_id):
            if acc.is_default and acc.id != account_id:
                acc.is_default = False
                self._store.update(acc)

        # Set new default
        account.is_default = True
        self._store.update(account)

        logger.info(f"Set default account: {account.name} ({account_id})")
        return True

    def get_default_account(self) -> Optional[TradingAccount]:
        """
        Get the default trading account.

        Returns:
            Default TradingAccount or None if no accounts
        """
        return self._store.get_default(self.user_id)

    def update_account(
        self,
        account_id: str,
        name: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TradingAccount:
        """
        Update account details.

        Args:
            account_id: ID of account to update
            name: New display name
            credentials: New credentials (will be encrypted)
            is_active: Active status
            metadata: Additional metadata to merge

        Returns:
            Updated TradingAccount

        Raises:
            AccountNotFoundError: If account not found
        """
        account = self._store.get(account_id)

        if not account or account.user_id != self.user_id:
            raise AccountNotFoundError(f"Account not found: {account_id}")

        if name is not None:
            if len(name.strip()) < 1:
                raise AccountValidationError("Account name is required")
            if len(name) > 100:
                raise AccountValidationError("Account name must be 100 characters or less")
            account.name = name.strip()

        if credentials is not None:
            self._validate_credentials(account.broker_type, credentials)
            account.update_credentials(credentials)
            # Clear broker cache when credentials change
            if account_id in self._broker_cache:
                try:
                    self._broker_cache[account_id].disconnect()
                except Exception:
                    pass
                del self._broker_cache[account_id]

        if is_active is not None:
            account.is_active = is_active

        if metadata is not None:
            account.metadata.update(metadata)

        self._store.update(account)
        logger.info(f"Updated account: {account.name} ({account_id})")

        return account

    def get_broker(
        self,
        account_id: Optional[str] = None,
        connect: bool = True
    ) -> BrokerInterface:
        """
        Get broker instance for an account.

        Creates and caches broker instances. Optionally connects automatically.

        Args:
            account_id: Account ID (uses default if not specified)
            connect: Whether to connect the broker automatically

        Returns:
            BrokerInterface instance

        Raises:
            AccountNotFoundError: If account not found
            BrokerError: If connection fails
        """
        if account_id is None:
            account = self.get_default_account()
            if not account:
                raise AccountNotFoundError("No default account configured")
            account_id = account.id
        else:
            account = self.get_account(account_id)

        if not account.is_active:
            raise AccountNotFoundError(f"Account {account_id} is not active")

        # Check cache first
        if account_id in self._broker_cache:
            broker = self._broker_cache[account_id]
            if connect and not broker.is_connected:
                broker.connect()
                account.mark_connected()
                self._store.update(account)
            return broker

        # Create new broker instance
        credentials = account.get_credentials()

        try:
            broker = get_broker(
                account.broker_type.value,
                **credentials
            )

            if connect:
                broker.connect()
                account.mark_connected()
                self._store.update(account)

            # Cache the broker
            self._broker_cache[account_id] = broker

            logger.info(f"Created broker for account: {account.name}")
            return broker

        except BrokerError as e:
            logger.error(f"Failed to create broker for {account.name}: {e}")
            raise

    def test_connection(self, account_id: str) -> Dict[str, Any]:
        """
        Test connection to an account's broker.

        Args:
            account_id: Account ID to test

        Returns:
            Dictionary with connection status and details
        """
        account = self.get_account(account_id)

        try:
            broker = self.get_broker(account_id, connect=True)
            account_info = broker.get_account()

            return {
                "success": True,
                "account_id": account_id,
                "broker_type": account.broker_type.value,
                "connected": broker.is_connected,
                "equity": account_info.equity,
                "buying_power": account_info.buying_power,
                "tested_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Connection test failed for account {account_id}: {e}")
            return {
                "success": False,
                "account_id": account_id,
                "broker_type": account.broker_type.value,
                "error": "Connection test failed",
                "tested_at": datetime.utcnow().isoformat()
            }

    def disconnect_all(self) -> None:
        """Disconnect all cached brokers."""
        for account_id, broker in self._broker_cache.items():
            try:
                broker.disconnect()
                logger.debug(f"Disconnected broker for account {account_id}")
            except Exception as e:
                logger.warning(f"Error disconnecting broker {account_id}: {e}")

        self._broker_cache.clear()
        logger.info("Disconnected all brokers")

    def get_account_summary(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary information for an account.

        Args:
            account_id: Account ID (uses default if not specified)

        Returns:
            Dictionary with account summary
        """
        if account_id is None:
            account = self.get_default_account()
            if not account:
                return {"error": "No default account"}
            account_id = account.id
        else:
            account = self.get_account(account_id)

        result = {
            "id": account.id,
            "name": account.name,
            "broker_type": account.broker_type.value,
            "is_default": account.is_default,
            "is_active": account.is_active,
            "created_at": account.created_at.isoformat(),
            "last_connected_at": (
                account.last_connected_at.isoformat()
                if account.last_connected_at else None
            )
        }

        # Try to get live account data
        try:
            broker = self.get_broker(account_id)
            if broker.is_connected:
                account_info = broker.get_account()
                positions = broker.get_positions()

                result.update({
                    "equity": account_info.equity,
                    "buying_power": account_info.buying_power,
                    "cash": account_info.cash,
                    "positions_count": len(positions),
                    "positions_value": account_info.positions_value,
                    "daily_pnl": account_info.daily_pnl,
                    "day_trades_remaining": account_info.day_trades_remaining,
                    "connected": True
                })
            else:
                result["connected"] = False
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            result["connected"] = False
            result["error"] = "Failed to retrieve summary"

        return result

    def get_all_account_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summary information for all accounts.

        Returns:
            List of account summaries
        """
        summaries = []
        for account in self.get_all_accounts():
            try:
                summary = self.get_account_summary(account.id)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error getting summary for account {account.id}: {e}")
                summaries.append({
                    "id": account.id,
                    "name": account.name,
                    "broker_type": account.broker_type.value,
                    "is_default": account.is_default,
                    "is_active": account.is_active,
                    "error": "Failed to retrieve summary"
                })

        return summaries
