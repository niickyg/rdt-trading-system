"""
Authentication module for RDT Trading System Dashboard

Provides:
- User authentication with Flask-Login
- Password hashing with werkzeug.security
- Login/logout routes
- Session management with session tracking
- Session activity middleware
"""

import os
from datetime import datetime, timedelta
from functools import wraps

from urllib.parse import urlparse
from flask import Blueprint, render_template, redirect, url_for, request, flash, current_app, g, make_response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

# Import User model
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database.models import User, UserSession, Base


# Create auth blueprint
auth_bp = Blueprint('auth', __name__)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access the dashboard.'
login_manager.login_message_category = 'info'

# Database session (will be initialized in init_auth)
_db_session = None

# Session cookie name
SESSION_COOKIE_NAME = 'rdt_session_token'

# Brute force protection settings
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

# Password policy settings
MIN_PASSWORD_LENGTH = 12
PASSWORD_REQUIREMENTS = {
    'min_length': MIN_PASSWORD_LENGTH,
    'require_uppercase': True,
    'require_lowercase': True,
    'require_digit': True,
    'require_special': True,
}


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password meets security requirements.

    Requirements:
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    import re

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"

    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"

    if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;\'`~]', password):
        return False, "Password must contain at least one special character"

    return True, ""


def check_account_lockout(user) -> tuple[bool, str]:
    """Check if account is locked due to too many failed login attempts.

    Args:
        user: User object to check

    Returns:
        Tuple of (is_locked, message)
    """
    if user.locked_until and user.locked_until > datetime.utcnow():
        remaining = (user.locked_until - datetime.utcnow()).total_seconds() // 60
        return True, f"Account locked due to too many failed attempts. Try again in {int(remaining) + 1} minutes."
    return False, ""


def record_failed_login(user, session):
    """Record a failed login attempt and lock account if threshold exceeded.

    Args:
        user: User object
        session: Database session
    """
    user.failed_login_attempts = (user.failed_login_attempts or 0) + 1

    if user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
        user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        logger.warning(f"Account locked for user {user.username} after {MAX_LOGIN_ATTEMPTS} failed attempts")

    session.commit()


def reset_failed_login(user, session):
    """Reset failed login counter on successful login.

    Args:
        user: User object
        session: Database session
    """
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.utcnow()
    session.commit()


def get_db_path():
    """Get the database path"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(base_dir, 'data', 'database')
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'rdt_auth.db')


def get_db_session():
    """Get or create database session"""
    global _db_session
    if _db_session is None:
        db_path = get_db_path()
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        _db_session = Session()
    return _db_session


def get_client_ip():
    """Get client IP address from request, handling proxies"""
    # Check for X-Forwarded-For header (when behind proxy/load balancer)
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    # Check for X-Real-IP header
    if request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    # Fall back to remote_addr
    return request.remote_addr


def get_session_token():
    """Get the session token from cookies"""
    return request.cookies.get(SESSION_COOKIE_NAME)


def set_session_cookie(response, token, remember=False):
    """Set the session cookie on a response with security attributes."""
    max_age = 30 * 24 * 60 * 60 if remember else None  # 30 days if remember, else session cookie
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,  # Prevent JavaScript access
        secure=not current_app.debug,  # HTTPS only in production
        samesite='Lax',  # CSRF protection
        path='/'  # Valid for all paths
    )
    return response


def clear_session_cookie(response):
    """Clear the session cookie from a response.

    Uses matching attributes to ensure cookie is properly cleared.
    """
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path='/',
        samesite='Lax'
    )
    return response


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    session = get_db_session()
    try:
        user = session.query(User).filter_by(id=int(user_id)).first()
        return user
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None
    finally:
        session.close()


def init_auth(app):
    """Initialize authentication for the Flask app"""
    login_manager.init_app(app)
    app.register_blueprint(auth_bp)

    # Register session activity middleware
    @app.before_request
    def update_session_activity():
        """Update session activity on each request"""
        if current_user.is_authenticated:
            token = get_session_token()
            if token:
                try:
                    from web.session_manager import get_session_manager
                    session_manager = get_session_manager()

                    # Validate session is still active
                    user_id = session_manager.validate_session(token)
                    if user_id is None or user_id != current_user.id:
                        # Session is invalid or expired, log out user
                        logout_user()
                        return redirect(url_for('auth.login'))

                    # Update activity
                    session_manager.update_activity(token, get_client_ip())

                    # Store token in g for later use
                    g.session_token = token
                except Exception as e:
                    logger.error(f"Error updating session activity: {e}")

    # Note: Admin user should be created via scripts/create_admin.py with environment variables
    # DO NOT create default admin with hardcoded credentials

    logger.info("Dashboard authentication initialized")


# =============================================================================
# AUTH ROUTES
# =============================================================================

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    # Redirect if already logged in
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    error = None

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False) == 'on'

        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            session = get_db_session()
            try:
                user = session.query(User).filter_by(username=username).first()

                if user is None:
                    error = 'Invalid username or password.'
                    logger.warning(f"Login attempt with unknown username: {username}")
                elif not user.is_active:
                    error = 'This account has been deactivated.'
                    logger.warning(f"Login attempt for deactivated user: {username}")
                else:
                    # Check for account lockout (brute force protection)
                    is_locked, lock_message = check_account_lockout(user)
                    if is_locked:
                        error = lock_message
                        logger.warning(f"Login attempt for locked account: {username}")
                    elif not check_password_hash(user.password_hash, password):
                        error = 'Invalid username or password.'
                        logger.warning(f"Failed login attempt for user: {username}")
                        # Record failed attempt (may trigger lockout)
                        record_failed_login(user, session)
                    else:
                        # Successful login - reset failed attempts
                        reset_failed_login(user, session)

                        login_user(user, remember=remember)
                        logger.info(f"User logged in: {username}")

                        # Create session record
                        try:
                            from web.session_manager import get_session_manager
                            session_manager = get_session_manager()
                            session_token = session_manager.create_session(
                                user_id=user.id,
                                ip_address=get_client_ip(),
                                user_agent=request.headers.get('User-Agent', '')
                            )

                            # Determine redirect URL
                            next_page = request.args.get('next')
                            redirect_url = url_for('dashboard')
                            if next_page:
                                parsed = urlparse(next_page)
                                if not parsed.netloc and next_page.startswith('/'):
                                    redirect_url = next_page

                            # Create response with session cookie
                            response = make_response(redirect(redirect_url))
                            set_session_cookie(response, session_token, remember=remember)
                            return response

                        except Exception as e:
                            logger.error(f"Error creating session: {e}")
                            # Continue without session tracking if it fails
                            next_page = request.args.get('next')
                            fallback_url = url_for('dashboard')
                            if next_page:
                                parsed = urlparse(next_page)
                                if not parsed.netloc and next_page.startswith('/'):
                                    fallback_url = next_page
                            return redirect(fallback_url)
            finally:
                session.close()

    return render_template('login.html', error=error)


@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    username = current_user.username

    # Delete session record
    try:
        from web.session_manager import get_session_manager
        session_manager = get_session_manager()
        session_token = get_session_token()
        if session_token:
            session_manager.delete_session(session_token)
    except Exception as e:
        logger.error(f"Error deleting session: {e}")

    logout_user()
    logger.info(f"User logged out: {username}")
    flash('You have been logged out successfully.', 'success')

    # Create response and clear session cookie
    response = make_response(redirect(url_for('auth.login')))
    clear_session_cookie(response)
    return response


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hash_password(password):
    """Generate password hash"""
    return generate_password_hash(password)


def verify_password(password_hash, password):
    """Verify password against hash"""
    return check_password_hash(password_hash, password)


def create_user(username, password, email=None, is_admin=False, skip_password_validation=False):
    """Create a new user.

    Args:
        username: Username for the new user
        password: Password (must meet security requirements)
        email: Optional email address
        is_admin: Whether user should have admin privileges
        skip_password_validation: Skip password validation (for migrations only)

    Raises:
        ValueError: If username/email exists or password doesn't meet requirements
    """
    # Validate password meets security requirements
    if not skip_password_validation:
        is_valid, error = validate_password(password)
        if not is_valid:
            raise ValueError(error)

    session = get_db_session()
    try:
        # Check if username already exists
        existing = session.query(User).filter_by(username=username).first()
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        # Check if email already exists (if provided)
        if email:
            existing = session.query(User).filter_by(email=email).first()
            if existing:
                raise ValueError(f"Email '{email}' already exists")

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_active=True,
            is_admin=is_admin,
            created_at=datetime.utcnow()
        )

        session.add(user)
        session.commit()

        logger.info(f"New user created: {username}")
        return user
    finally:
        session.close()


def change_password(user_id, new_password, skip_password_validation=False):
    """Change user password.

    Args:
        user_id: ID of the user
        new_password: New password (must meet security requirements)
        skip_password_validation: Skip password validation (for migrations only)

    Raises:
        ValueError: If user not found or password doesn't meet requirements
    """
    # Validate password meets security requirements
    if not skip_password_validation:
        is_valid, error = validate_password(new_password)
        if not is_valid:
            raise ValueError(error)

    session = get_db_session()
    try:
        user = session.query(User).filter_by(id=user_id).first()

        if user is None:
            raise ValueError("User not found")

        user.password_hash = generate_password_hash(new_password)
        session.commit()

        logger.info(f"Password changed for user: {user.username}")
    finally:
        session.close()
