"""
Schwab OAuth Authentication Handler
Manages OAuth 2.0 flow for Schwab API access
"""

import os
import json
import time
import webbrowser
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
from dataclasses import dataclass
from loguru import logger
import requests


@dataclass
class TokenData:
    """OAuth token data"""
    access_token: str
    refresh_token: str
    expires_at: datetime
    token_type: str = "Bearer"
    scope: str = ""

    def is_expired(self) -> bool:
        """Check if access token is expired (with 5 min buffer)"""
        return datetime.now() >= (self.expires_at - timedelta(minutes=5))

    def to_dict(self) -> Dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at.isoformat(),
            "token_type": self.token_type,
            "scope": self.scope
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TokenData":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=datetime.fromisoformat(data["expires_at"]),
            token_type=data.get("token_type", "Bearer"),
            scope=data.get("scope", "")
        )


class SchwabAuth:
    """
    Schwab OAuth 2.0 Authentication Handler

    Usage:
        auth = SchwabAuth(app_key, app_secret, callback_url)
        token = auth.get_valid_token()  # Returns valid access token
    """

    BASE_AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
    TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        callback_url: str = "https://localhost:8080",
        token_path: Optional[str] = None
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.callback_url = callback_url

        # Default token storage path
        if token_path is None:
            token_path = Path.home() / ".rdt-trading" / "schwab_token.json"
        self.token_path = Path(token_path)
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

        self._token: Optional[TokenData] = None
        self._load_token()

    def _load_token(self):
        """Load token from file if exists"""
        if self.token_path.exists():
            try:
                with open(self.token_path, 'r') as f:
                    data = json.load(f)
                self._token = TokenData.from_dict(data)
                logger.debug("Loaded token from file")
            except Exception as e:
                logger.warning(f"Failed to load token: {e}")
                self._token = None

    def _save_token(self):
        """Save token to file"""
        if self._token:
            try:
                with open(self.token_path, 'w') as f:
                    json.dump(self._token.to_dict(), f, indent=2)
                # Secure permissions
                os.chmod(self.token_path, 0o600)
                logger.debug("Saved token to file")
            except Exception as e:
                logger.error(f"Failed to save token: {e}")

    def get_authorization_url(self) -> str:
        """Generate OAuth authorization URL"""
        params = {
            "client_id": self.app_key,
            "redirect_uri": self.callback_url,
            "response_type": "code",
            "scope": "api"  # Schwab API scope
        }
        url = f"{self.BASE_AUTH_URL}?{urllib.parse.urlencode(params)}"
        return url

    def authorize_interactive(self) -> bool:
        """
        Run interactive OAuth flow
        Opens browser for user to authorize, then prompts for callback URL
        """
        auth_url = self.get_authorization_url()

        print("\n" + "="*60)
        print("SCHWAB OAUTH AUTHORIZATION")
        print("="*60)
        print("\n1. Opening browser to Schwab authorization page...")
        print("2. Log in with your Schwab credentials")
        print("3. Authorize the application")
        print("4. Copy the FULL redirect URL from your browser")
        print("\n" + "="*60)

        # Try to open browser
        try:
            webbrowser.open(auth_url)
        except Exception:
            print(f"\nCould not open browser. Please visit:\n{auth_url}")

        # Wait for user to paste callback URL
        print("\nAfter authorizing, paste the full redirect URL here:")
        callback_response = input("> ").strip()

        # Extract authorization code from callback URL
        try:
            parsed = urllib.parse.urlparse(callback_response)
            params = urllib.parse.parse_qs(parsed.query)
            auth_code = params.get("code", [None])[0]

            if not auth_code:
                logger.error("No authorization code found in URL")
                return False

            # Exchange code for tokens
            return self.exchange_code_for_token(auth_code)

        except Exception as e:
            logger.error(f"Failed to parse callback URL: {e}")
            return False

    def exchange_code_for_token(self, auth_code: str) -> bool:
        """Exchange authorization code for access token"""
        try:
            import base64

            # Basic auth header
            credentials = base64.b64encode(
                f"{self.app_key}:{self.app_secret}".encode()
            ).decode()

            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            data = {
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": self.callback_url
            }

            response = requests.post(
                self.TOKEN_URL,
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return False

            token_data = response.json()
            self._token = TokenData(
                access_token=token_data["access_token"],
                refresh_token=token_data["refresh_token"],
                expires_at=datetime.now() + timedelta(seconds=token_data["expires_in"]),
                token_type=token_data.get("token_type", "Bearer"),
                scope=token_data.get("scope", "")
            )

            self._save_token()
            logger.info("Successfully obtained access token")
            return True

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False

    def refresh_access_token(self, max_retries: int = 3) -> bool:
        """
        Refresh the access token using refresh token with retry logic.

        Uses exponential backoff with jitter for retries on transient failures.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            True if refresh successful, False otherwise
        """
        import random

        if not self._token or not self._token.refresh_token:
            logger.error("No refresh token available")
            return False

        import base64
        credentials = base64.b64encode(
            f"{self.app_key}:{self.app_secret}".encode()
        ).decode()

        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._token.refresh_token
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.TOKEN_URL,
                    headers=headers,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    token_data = response.json()
                    self._token = TokenData(
                        access_token=token_data["access_token"],
                        refresh_token=token_data.get("refresh_token", self._token.refresh_token),
                        expires_at=datetime.now() + timedelta(seconds=token_data["expires_in"]),
                        token_type=token_data.get("token_type", "Bearer"),
                        scope=token_data.get("scope", "")
                    )

                    self._save_token()
                    logger.info("Successfully refreshed access token")
                    return True

                # Non-retriable errors (auth failures)
                if response.status_code in (400, 401, 403):
                    logger.error(
                        f"Token refresh failed with non-retriable error: "
                        f"{response.status_code} - {response.text}"
                    )
                    # Refresh token may be invalid, need re-authorization
                    self._token = None
                    return False

                # Retriable server errors
                if response.status_code >= 500:
                    last_error = f"Server error {response.status_code}"
                    if attempt < max_retries - 1:
                        delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                        logger.warning(
                            f"Token refresh attempt {attempt + 1}/{max_retries} failed: "
                            f"{response.status_code}. Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue

                logger.error(f"Token refresh failed: {response.status_code}")
                return False

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                    logger.warning(
                        f"Token refresh timeout (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                    logger.warning(
                        f"Token refresh connection error (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
            except Exception as e:
                logger.error(f"Token refresh error: {e}")
                return False

        logger.error(f"Token refresh failed after {max_retries} attempts: {last_error}")
        return False

    def should_refresh_proactively(self, buffer_minutes: int = 10) -> bool:
        """
        Check if token should be proactively refreshed.

        Proactive refresh helps prevent request failures due to token expiry
        during long-running operations.

        Args:
            buffer_minutes: Minutes before expiry to trigger proactive refresh

        Returns:
            True if token should be refreshed proactively
        """
        if self._token is None:
            return False

        return datetime.now() >= (self._token.expires_at - timedelta(minutes=buffer_minutes))

    def get_valid_token(self, proactive_refresh: bool = True) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.

        Supports proactive refresh to refresh tokens before they expire,
        preventing failures during long-running operations.

        Args:
            proactive_refresh: If True, refresh token proactively 10 min before expiry

        Returns:
            Valid access token or None if authentication required
        """
        if self._token is None:
            logger.warning("No token available - authorization required")
            return None

        # Check for expired token (includes 5 min buffer via is_expired)
        if self._token.is_expired():
            logger.info("Token expired, refreshing...")
            if not self.refresh_access_token():
                logger.warning("Token refresh failed - re-authorization required")
                return None
        # Proactive refresh: refresh before token expires to prevent failures
        elif proactive_refresh and self.should_refresh_proactively():
            minutes_left = (self._token.expires_at - datetime.now()).total_seconds() / 60
            logger.info(
                f"Proactively refreshing token ({minutes_left:.1f} min until expiry)"
            )
            # Don't fail if proactive refresh fails - token is still valid
            if not self.refresh_access_token():
                logger.warning("Proactive token refresh failed, will retry on next request")

        return self._token.access_token

    def get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get authorization header for API requests"""
        token = self.get_valid_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return None

    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication"""
        return self.get_valid_token() is not None

    def logout(self):
        """Clear stored tokens"""
        self._token = None
        if self.token_path.exists():
            self.token_path.unlink()
        logger.info("Logged out, tokens cleared")
