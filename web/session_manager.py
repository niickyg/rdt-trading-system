"""
Session Management for RDT Trading System Dashboard

Provides:
- Active user session tracking
- Session revocation (single and bulk)
- Session activity tracking
- Session expiry management
"""

import os
import secrets
import hashlib
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from sqlalchemy import and_
from loguru import logger

# Import database utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database.models import UserSession
from web.auth import get_db_session


# Configuration
SESSION_EXPIRY_DAYS = int(os.environ.get('SESSION_EXPIRY_DAYS', '30'))
SESSION_TOKEN_LENGTH = 64


@dataclass
class SessionInfo:
    """Session information for display."""
    id: int
    session_token_hash: str
    ip_address: str
    ip_masked: str
    user_agent: str
    device_info: Dict[str, str]
    created_at: datetime
    last_activity: datetime
    is_current: bool
    is_active: bool


def generate_session_token() -> str:
    """Generate a secure random session token."""
    return secrets.token_urlsafe(SESSION_TOKEN_LENGTH)


def hash_session_token(token: str) -> str:
    """Hash a session token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def mask_ip_address(ip_address: str) -> str:
    """
    Partially mask an IP address for privacy.

    IPv4: 192.168.1.100 -> 192.168.xxx.xxx
    IPv6: 2001:0db8:85a3::8a2e:0370:7334 -> 2001:0db8:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx
    """
    if not ip_address:
        return 'Unknown'

    if ':' in ip_address:
        # IPv6
        parts = ip_address.split(':')
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}:xxxx:xxxx"
        return ip_address
    else:
        # IPv4
        parts = ip_address.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        return ip_address


def parse_user_agent(user_agent: str) -> Dict[str, str]:
    """
    Parse user agent string to extract device/browser info.

    Returns dict with: browser, browser_version, os, device_type
    """
    if not user_agent:
        return {
            'browser': 'Unknown',
            'browser_version': '',
            'os': 'Unknown',
            'device_type': 'Unknown'
        }

    ua_lower = user_agent.lower()

    # Detect browser
    browser = 'Unknown'
    browser_version = ''

    if 'firefox/' in ua_lower:
        browser = 'Firefox'
        try:
            idx = ua_lower.index('firefox/')
            version_part = user_agent[idx + 8:idx + 20].split()[0]
            browser_version = version_part.split('.')[0]
        except (ValueError, IndexError):
            pass
    elif 'edg/' in ua_lower:
        browser = 'Edge'
        try:
            idx = ua_lower.index('edg/')
            version_part = user_agent[idx + 4:idx + 16].split()[0]
            browser_version = version_part.split('.')[0]
        except (ValueError, IndexError):
            pass
    elif 'chrome/' in ua_lower and 'chromium/' not in ua_lower:
        browser = 'Chrome'
        try:
            idx = ua_lower.index('chrome/')
            version_part = user_agent[idx + 7:idx + 19].split()[0]
            browser_version = version_part.split('.')[0]
        except (ValueError, IndexError):
            pass
    elif 'safari/' in ua_lower and 'chrome/' not in ua_lower:
        browser = 'Safari'
        try:
            if 'version/' in ua_lower:
                idx = ua_lower.index('version/')
                version_part = user_agent[idx + 8:idx + 20].split()[0]
                browser_version = version_part.split('.')[0]
        except (ValueError, IndexError):
            pass
    elif 'opera/' in ua_lower or 'opr/' in ua_lower:
        browser = 'Opera'
    elif 'msie' in ua_lower or 'trident/' in ua_lower:
        browser = 'Internet Explorer'

    # Detect OS
    os_name = 'Unknown'
    if 'windows nt 10' in ua_lower:
        os_name = 'Windows 10/11'
    elif 'windows nt 6.3' in ua_lower:
        os_name = 'Windows 8.1'
    elif 'windows nt 6.2' in ua_lower:
        os_name = 'Windows 8'
    elif 'windows nt 6.1' in ua_lower:
        os_name = 'Windows 7'
    elif 'windows' in ua_lower:
        os_name = 'Windows'
    elif 'mac os x' in ua_lower:
        os_name = 'macOS'
    elif 'linux' in ua_lower:
        if 'android' in ua_lower:
            os_name = 'Android'
        else:
            os_name = 'Linux'
    elif 'iphone' in ua_lower or 'ipad' in ua_lower:
        os_name = 'iOS'

    # Detect device type
    device_type = 'Desktop'
    if 'mobile' in ua_lower or 'android' in ua_lower and 'mobile' in ua_lower:
        device_type = 'Mobile'
    elif 'tablet' in ua_lower or 'ipad' in ua_lower:
        device_type = 'Tablet'

    return {
        'browser': browser,
        'browser_version': browser_version,
        'os': os_name,
        'device_type': device_type
    }


class SessionManager:
    """
    Manages user sessions for the dashboard.

    Tracks active sessions, handles revocation, and manages session expiry.
    """

    def __init__(self):
        self.expiry_days = SESSION_EXPIRY_DAYS

    def create_session(
        self,
        user_id: int,
        ip_address: str,
        user_agent: str
    ) -> str:
        """
        Create a new session for a user.

        Args:
            user_id: The user's ID
            ip_address: Client IP address
            user_agent: Client user agent string

        Returns:
            Session token (unhashed) to be stored in client cookie
        """
        session = get_db_session()

        # Generate session token
        token = generate_session_token()
        token_hash = hash_session_token(token)

        # Parse user agent for device info
        device_info = parse_user_agent(user_agent)
        device_info_str = f"{device_info['browser']} on {device_info['os']}"

        # Create session record
        user_session = UserSession(
            user_id=user_id,
            session_token=token_hash,
            ip_address=ip_address,
            user_agent=user_agent[:500] if user_agent else None,  # Truncate long UAs
            device_info=device_info_str,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            is_active=True
        )

        try:
            session.add(user_session)
            session.commit()
            logger.info(f"Session created for user {user_id} from {mask_ip_address(ip_address)}")
            return token
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating session: {e}")
            raise

    def validate_session(self, token: str) -> Optional[int]:
        """
        Validate a session token and return the user ID if valid.

        Args:
            token: The session token from client cookie

        Returns:
            User ID if session is valid, None otherwise
        """
        if not token:
            return None

        session = get_db_session()
        try:
            token_hash = hash_session_token(token)

            # Find active session
            user_session = session.query(UserSession).filter(
                and_(
                    UserSession.session_token == token_hash,
                    UserSession.is_active == True
                )
            ).first()

            if not user_session:
                return None

            # Check expiry
            expiry_time = user_session.last_activity + timedelta(days=self.expiry_days)
            if datetime.utcnow() > expiry_time:
                # Session expired, mark as inactive
                user_session.is_active = False
                session.commit()
                logger.info(f"Session expired for user {user_session.user_id}")
                return None

            return user_session.user_id
        finally:
            session.close()

    def update_activity(self, token: str, ip_address: str = None) -> bool:
        """
        Update the last activity timestamp for a session.

        Args:
            token: The session token
            ip_address: Optional new IP address

        Returns:
            True if session was updated, False otherwise
        """
        if not token:
            return False

        session = get_db_session()
        try:
            token_hash = hash_session_token(token)

            user_session = session.query(UserSession).filter(
                and_(
                    UserSession.session_token == token_hash,
                    UserSession.is_active == True
                )
            ).first()

            if not user_session:
                return False

            user_session.last_activity = datetime.utcnow()
            if ip_address:
                user_session.ip_address = ip_address

            try:
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating session activity: {e}")
                return False
        finally:
            session.close()

    def get_user_sessions(
        self,
        user_id: int,
        current_token: str = None
    ) -> List[SessionInfo]:
        """
        Get all active sessions for a user.

        Args:
            user_id: The user's ID
            current_token: The current session token to mark as current

        Returns:
            List of SessionInfo objects
        """
        session = get_db_session()
        current_token_hash = hash_session_token(current_token) if current_token else None

        # Get all active sessions for user
        user_sessions = session.query(UserSession).filter(
            and_(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            )
        ).order_by(UserSession.last_activity.desc()).all()

        sessions = []
        for us in user_sessions:
            # Check if session is expired
            expiry_time = us.last_activity + timedelta(days=self.expiry_days)
            is_expired = datetime.utcnow() > expiry_time

            if is_expired:
                # Mark as inactive
                us.is_active = False
                continue

            device_info = parse_user_agent(us.user_agent)

            sessions.append(SessionInfo(
                id=us.id,
                session_token_hash=us.session_token[:16] + '...',  # Truncated for display
                ip_address=us.ip_address,
                ip_masked=mask_ip_address(us.ip_address),
                user_agent=us.user_agent or 'Unknown',
                device_info=device_info,
                created_at=us.created_at,
                last_activity=us.last_activity,
                is_current=(us.session_token == current_token_hash),
                is_active=us.is_active
            ))

        try:
            session.commit()  # Commit any inactive markings
        except Exception as e:
            session.rollback()
            logger.error(f"Error committing session status updates: {e}")

        return sessions

    def revoke_session(self, session_id: int, user_id: int) -> bool:
        """
        Revoke a specific session.

        Args:
            session_id: The session ID to revoke
            user_id: The user ID (for authorization check)

        Returns:
            True if session was revoked, False otherwise
        """
        session = get_db_session()

        user_session = session.query(UserSession).filter(
            and_(
                UserSession.id == session_id,
                UserSession.user_id == user_id,
                UserSession.is_active == True
            )
        ).first()

        if not user_session:
            return False

        user_session.is_active = False

        try:
            session.commit()
            logger.info(f"Session {session_id} revoked for user {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error revoking session: {e}")
            return False

    def revoke_all_except_current(self, user_id: int, current_token: str) -> int:
        """
        Revoke all sessions for a user except the current one.

        Args:
            user_id: The user's ID
            current_token: The current session token to keep

        Returns:
            Number of sessions revoked
        """
        session = get_db_session()
        current_token_hash = hash_session_token(current_token) if current_token else None

        # Find all other active sessions
        user_sessions = session.query(UserSession).filter(
            and_(
                UserSession.user_id == user_id,
                UserSession.is_active == True,
                UserSession.session_token != current_token_hash
            )
        ).all()

        count = 0
        for us in user_sessions:
            us.is_active = False
            count += 1

        try:
            session.commit()
            logger.info(f"Revoked {count} sessions for user {user_id}")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error revoking sessions: {e}")
            return 0

    def delete_session(self, token: str) -> bool:
        """
        Delete a session (used on logout).

        Args:
            token: The session token to delete

        Returns:
            True if session was deleted, False otherwise
        """
        if not token:
            return False

        session = get_db_session()
        token_hash = hash_session_token(token)

        user_session = session.query(UserSession).filter(
            UserSession.session_token == token_hash
        ).first()

        if not user_session:
            return False

        user_session.is_active = False

        try:
            session.commit()
            logger.info(f"Session deleted for user {user_session.user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting session: {e}")
            return False

    def cleanup_expired_sessions(self, days_old: int = None) -> int:
        """
        Clean up expired sessions from the database.

        Args:
            days_old: Number of days after which to consider sessions expired.
                     Defaults to SESSION_EXPIRY_DAYS.

        Returns:
            Number of sessions cleaned up
        """
        if days_old is None:
            days_old = self.expiry_days

        session = get_db_session()
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        # Mark old sessions as inactive
        result = session.query(UserSession).filter(
            and_(
                UserSession.last_activity < cutoff_date,
                UserSession.is_active == True
            )
        ).update({UserSession.is_active: False})

        try:
            session.commit()
            logger.info(f"Cleaned up {result} expired sessions")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up sessions: {e}")
            return 0

    def get_session_count(self, user_id: int) -> int:
        """
        Get the count of active sessions for a user.

        Args:
            user_id: The user's ID

        Returns:
            Number of active sessions
        """
        session = get_db_session()

        return session.query(UserSession).filter(
            and_(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            )
        ).count()


# Global session manager instance
_session_manager: Optional[SessionManager] = None
_session_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance (thread-safe)."""
    global _session_manager
    if _session_manager is None:
        with _session_manager_lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    return _session_manager
