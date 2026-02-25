#!/usr/bin/env python3
"""
Create API user from environment variables.

Usage:
    RDT_API_USER_EMAIL=user@example.com \
    RDT_API_USER_TIER=pro \
    python scripts/create_api_user.py

Environment Variables:
    RDT_API_USER_EMAIL: User email address (required)
    RDT_API_USER_TIER: Subscription tier - free, basic, pro, elite (default: free)
    RDT_API_USER_DURATION_DAYS: Subscription duration in days (optional, default: no expiry)
"""
import os
import sys
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.v1.auth import api_key_manager, SubscriptionTier


def validate_email(email: str) -> tuple[bool, str]:
    """Validate email format."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    return True, ""


def create_api_user():
    """Create API user from environment variables."""
    email = os.environ.get('RDT_API_USER_EMAIL')
    tier_str = os.environ.get('RDT_API_USER_TIER', 'free').lower()
    duration_days_str = os.environ.get('RDT_API_USER_DURATION_DAYS')

    # Check required environment variables
    if not email:
        print("ERROR: Missing required environment variable: RDT_API_USER_EMAIL")
        print("\nUsage:")
        print("  RDT_API_USER_EMAIL=user@example.com \\")
        print("  RDT_API_USER_TIER=pro \\")
        print("  python scripts/create_api_user.py")
        print("\nAvailable tiers: free, basic, pro, elite")
        sys.exit(1)

    # Validate email
    valid, error = validate_email(email)
    if not valid:
        print(f"ERROR: {error}")
        sys.exit(1)

    # Validate tier
    valid_tiers = ['free', 'basic', 'pro', 'elite']
    if tier_str not in valid_tiers:
        print(f"ERROR: Invalid tier '{tier_str}'. Must be one of: {', '.join(valid_tiers)}")
        sys.exit(1)

    tier = SubscriptionTier(tier_str)

    # Parse duration days
    duration_days = None
    if duration_days_str:
        try:
            duration_days = int(duration_days_str)
            if duration_days <= 0:
                print("ERROR: RDT_API_USER_DURATION_DAYS must be a positive integer")
                sys.exit(1)
        except ValueError:
            print(f"ERROR: RDT_API_USER_DURATION_DAYS must be an integer, got '{duration_days_str}'")
            sys.exit(1)

    # Initialize API key manager
    try:
        api_key_manager.initialize()
    except Exception as e:
        print(f"ERROR: Failed to initialize API key manager: {e}")
        sys.exit(1)

    # Create API user
    try:
        user = api_key_manager.create_user(
            email=email,
            tier=tier,
            duration_days=duration_days
        )

        print("API user created successfully!")
        print("-" * 50)
        print(f"  Email: {user.email}")
        print(f"  User ID: {user.user_id}")
        print(f"  Tier: {user.subscription_tier.value}")
        print(f"  Rate Limit: {user.rate_limit} requests/hour")
        if user.expires_at:
            print(f"  Expires: {user.expires_at.isoformat()}")
        else:
            print(f"  Expires: Never")
        print("-" * 50)
        print(f"  API Key: {user.api_key}")
        print("-" * 50)
        print("\nIMPORTANT: Save the API key securely. It cannot be retrieved later.")

    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to create API user: {e}")
        sys.exit(1)


if __name__ == '__main__':
    create_api_user()
