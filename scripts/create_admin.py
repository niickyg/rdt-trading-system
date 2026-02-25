#!/usr/bin/env python3
"""
Create admin user from environment variables.

Usage:
    RDT_ADMIN_USERNAME=admin \
    RDT_ADMIN_EMAIL=admin@example.com \
    RDT_ADMIN_PASSWORD=SecurePass123! \
    python scripts/create_admin.py

Environment Variables:
    RDT_ADMIN_USERNAME: Admin username (required)
    RDT_ADMIN_EMAIL: Admin email address (required)
    RDT_ADMIN_PASSWORD: Admin password - min 12 chars with complexity (required)
"""
import os
import sys
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from werkzeug.security import generate_password_hash
from data.database.connection import get_db_manager
from data.database.models import User


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password meets security requirements."""
    if len(password) < 12:
        return False, "Password must be at least 12 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"

    # Check against common passwords
    common_passwords = {
        'password1234', 'password123!', '123456789012', 'qwerty123456',
        'admin1234567', 'administrator', 'letmein12345'
    }
    if password.lower() in common_passwords:
        return False, "Password is too common"

    return True, ""


def validate_email(email: str) -> tuple[bool, str]:
    """Validate email format."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    return True, ""


def validate_username(username: str) -> tuple[bool, str]:
    """Validate username format."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 50:
        return False, "Username must be at most 50 characters"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, ""


def create_admin():
    """Create admin user from environment variables."""
    username = os.environ.get('RDT_ADMIN_USERNAME')
    email = os.environ.get('RDT_ADMIN_EMAIL')
    password = os.environ.get('RDT_ADMIN_PASSWORD')

    # Check required environment variables
    missing = []
    if not username:
        missing.append('RDT_ADMIN_USERNAME')
    if not email:
        missing.append('RDT_ADMIN_EMAIL')
    if not password:
        missing.append('RDT_ADMIN_PASSWORD')

    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("\nUsage:")
        print("  RDT_ADMIN_USERNAME=admin \\")
        print("  RDT_ADMIN_EMAIL=admin@example.com \\")
        print("  RDT_ADMIN_PASSWORD=SecurePass123! \\")
        print("  python scripts/create_admin.py")
        sys.exit(1)

    # Validate username
    valid, error = validate_username(username)
    if not valid:
        print(f"ERROR: {error}")
        sys.exit(1)

    # Validate email
    valid, error = validate_email(email)
    if not valid:
        print(f"ERROR: {error}")
        sys.exit(1)

    # Validate password
    valid, error = validate_password(password)
    if not valid:
        print(f"ERROR: {error}")
        print("\nPassword requirements:")
        print("  - At least 12 characters")
        print("  - At least one uppercase letter")
        print("  - At least one lowercase letter")
        print("  - At least one digit")
        print("  - At least one special character (!@#$%^&*(),.?\":{}|<>)")
        sys.exit(1)

    # Get database manager
    try:
        db = get_db_manager()
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    # Create admin user
    try:
        with db.get_session() as session:
            # Check if username already exists
            existing = session.query(User).filter_by(username=username).first()
            if existing:
                print(f"Admin user '{username}' already exists")
                print("To update the password, use the password reset functionality.")
                return

            # Check if email already exists
            existing_email = session.query(User).filter_by(email=email).first()
            if existing_email:
                print(f"ERROR: Email '{email}' is already in use by another user")
                sys.exit(1)

            # Create admin user
            admin = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                is_active=True,
                is_admin=True,
                created_at=datetime.utcnow(),
                failed_login_attempts=0,
                locked_until=None
            )
            session.add(admin)
            session.commit()

            print(f"Admin user '{username}' created successfully")
            print(f"  Email: {email}")
            print(f"  Admin: Yes")
            print(f"  Active: Yes")

    except Exception as e:
        print(f"ERROR: Failed to create admin user: {e}")
        sys.exit(1)


if __name__ == '__main__':
    create_admin()
