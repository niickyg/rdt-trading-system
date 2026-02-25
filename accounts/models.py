"""
Trading Account Database Models.

Provides secure storage for multiple trading accounts with encrypted credentials.
"""

import os
import json
import base64
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


class BrokerType(str, Enum):
    """Supported broker types."""
    PAPER = "paper"
    SCHWAB = "schwab"
    IBKR = "ibkr"


class CredentialEncryption:
    """
    Handles encryption/decryption of sensitive credential data.

    Uses Fernet symmetric encryption with a key derived from a master password.
    The master password should be stored securely (e.g., environment variable).
    """

    _instance = None
    _fernet = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'CredentialEncryption':
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize encryption with master key."""
        self._init_encryption()

    def _init_encryption(self):
        """Initialize Fernet encryption from environment."""
        master_key = os.environ.get('RDT_CREDENTIAL_KEY')

        if not master_key:
            # Fall back to SECRET_KEY, but it MUST be set
            secret_key = os.environ.get('SECRET_KEY')
            if not secret_key:
                raise RuntimeError(
                    "Neither RDT_CREDENTIAL_KEY nor SECRET_KEY is set. "
                    "Set one of these environment variables before starting the application."
                )
            salt = os.environ.get('RDT_CREDENTIAL_SALT', 'rdt-trading-salt').encode()

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
            logger.warning(
                "Using derived credential key from SECRET_KEY. Set RDT_CREDENTIAL_KEY in production."
            )
        else:
            # Use provided master key (should be Fernet key format)
            key = master_key.encode() if isinstance(master_key, str) else master_key

        self._fernet = Fernet(key)

    def encrypt(self, data: Dict[str, Any]) -> str:
        """
        Encrypt credential data.

        Args:
            data: Dictionary of credentials to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        try:
            json_data = json.dumps(data)
            encrypted = self._fernet.encrypt(json_data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError("Failed to encrypt credentials")

    def decrypt(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt credential data.

        Args:
            encrypted_data: Base64-encoded encrypted string

        Returns:
            Dictionary of decrypted credentials
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt credentials")


@dataclass
class TradingAccount:
    """
    Represents a trading account with encrypted credentials.

    Attributes:
        id: Unique account identifier
        user_id: Owner's user ID
        name: Display name for the account
        broker_type: Type of broker (paper, schwab, ibkr)
        is_default: Whether this is the default account
        is_active: Whether the account is active for trading
        credentials_encrypted: Encrypted credential data
        created_at: Account creation timestamp
        updated_at: Last update timestamp
        last_connected_at: Last successful connection timestamp
        metadata: Additional account metadata
    """
    id: str
    user_id: str
    name: str
    broker_type: BrokerType
    is_default: bool = False
    is_active: bool = True
    credentials_encrypted: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_connected_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique account ID."""
        return f"acc_{secrets.token_hex(12)}"

    @classmethod
    def create(
        cls,
        user_id: str,
        name: str,
        broker_type: BrokerType,
        credentials: Dict[str, Any],
        is_default: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'TradingAccount':
        """
        Create a new trading account with encrypted credentials.

        Args:
            user_id: Owner's user ID
            name: Display name for the account
            broker_type: Type of broker
            credentials: Plain credentials to encrypt
            is_default: Whether this should be the default account
            metadata: Additional metadata

        Returns:
            New TradingAccount instance
        """
        encryption = CredentialEncryption.get_instance()
        encrypted_creds = encryption.encrypt(credentials)

        return cls(
            id=cls.generate_id(),
            user_id=user_id,
            name=name,
            broker_type=broker_type,
            is_default=is_default,
            is_active=True,
            credentials_encrypted=encrypted_creds,
            metadata=metadata or {}
        )

    def get_credentials(self) -> Dict[str, Any]:
        """
        Decrypt and return credentials.

        Returns:
            Decrypted credential dictionary
        """
        if not self.credentials_encrypted:
            return {}

        encryption = CredentialEncryption.get_instance()
        return encryption.decrypt(self.credentials_encrypted)

    def update_credentials(self, credentials: Dict[str, Any]) -> None:
        """
        Update credentials with new encrypted values.

        Args:
            credentials: New credentials to encrypt and store
        """
        encryption = CredentialEncryption.get_instance()
        self.credentials_encrypted = encryption.encrypt(credentials)
        self.updated_at = datetime.utcnow()

    def mark_connected(self) -> None:
        """Mark the account as recently connected."""
        self.last_connected_at = datetime.utcnow()

    def to_dict(self, include_credentials: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Args:
            include_credentials: Whether to include decrypted credentials (dangerous!)

        Returns:
            Dictionary representation
        """
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "broker_type": self.broker_type.value,
            "is_default": self.is_default,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_connected_at": (
                self.last_connected_at.isoformat()
                if self.last_connected_at else None
            ),
            "metadata": self.metadata
        }

        if include_credentials:
            result["credentials"] = self.get_credentials()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingAccount':
        """
        Create instance from dictionary.

        Args:
            data: Dictionary with account data

        Returns:
            TradingAccount instance
        """
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data["name"],
            broker_type=BrokerType(data["broker_type"]),
            is_default=data.get("is_default", False),
            is_active=data.get("is_active", True),
            credentials_encrypted=data.get("credentials_encrypted", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.utcnow()),
            last_connected_at=datetime.fromisoformat(data["last_connected_at"]) if data.get("last_connected_at") else None,
            metadata=data.get("metadata", {})
        )


class AccountStore:
    """
    Persistence layer for trading accounts.

    This implementation uses JSON file storage for simplicity.
    Can be extended to use SQLAlchemy, Redis, or other backends.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize account store.

        Args:
            storage_path: Path to JSON storage file (default: ~/.rdt/accounts.json)
        """
        if storage_path is None:
            home_dir = os.path.expanduser("~")
            rdt_dir = os.path.join(home_dir, ".rdt")
            os.makedirs(rdt_dir, exist_ok=True)
            storage_path = os.path.join(rdt_dir, "accounts.json")

        self.storage_path = storage_path
        self._accounts: Dict[str, TradingAccount] = {}
        self._load()

    def _load(self) -> None:
        """Load accounts from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                for account_data in data.get("accounts", []):
                    account = TradingAccount.from_dict(account_data)
                    self._accounts[account.id] = account

                logger.debug(f"Loaded {len(self._accounts)} accounts from storage")
            except Exception as e:
                logger.error(f"Failed to load accounts: {e}")
                self._accounts = {}

    def _save(self) -> None:
        """Save accounts to storage."""
        try:
            data = {
                "accounts": [
                    {
                        **account.to_dict(),
                        "credentials_encrypted": account.credentials_encrypted
                    }
                    for account in self._accounts.values()
                ]
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._accounts)} accounts to storage")
        except Exception as e:
            logger.error(f"Failed to save accounts: {e}")
            raise

    def add(self, account: TradingAccount) -> None:
        """Add an account to storage."""
        self._accounts[account.id] = account
        self._save()

    def get(self, account_id: str) -> Optional[TradingAccount]:
        """Get an account by ID."""
        return self._accounts.get(account_id)

    def get_by_user(self, user_id: str) -> list[TradingAccount]:
        """Get all accounts for a user."""
        return [
            acc for acc in self._accounts.values()
            if acc.user_id == user_id
        ]

    def update(self, account: TradingAccount) -> None:
        """Update an account."""
        if account.id in self._accounts:
            account.updated_at = datetime.utcnow()
            self._accounts[account.id] = account
            self._save()

    def delete(self, account_id: str) -> bool:
        """Delete an account."""
        if account_id in self._accounts:
            del self._accounts[account_id]
            self._save()
            return True
        return False

    def get_default(self, user_id: str) -> Optional[TradingAccount]:
        """Get the default account for a user."""
        user_accounts = self.get_by_user(user_id)
        for acc in user_accounts:
            if acc.is_default:
                return acc
        # Return first active account if no default
        for acc in user_accounts:
            if acc.is_active:
                return acc
        return None


# Singleton store instance
_account_store: Optional[AccountStore] = None


def get_account_store(storage_path: Optional[str] = None) -> AccountStore:
    """Get the singleton account store instance."""
    global _account_store
    if _account_store is None:
        _account_store = AccountStore(storage_path)
    return _account_store
