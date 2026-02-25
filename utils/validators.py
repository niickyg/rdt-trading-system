"""
Input Validation Utilities for RDT Trading System

Provides comprehensive input validation for:
- Stock symbols
- Prices and quantities
- Email addresses
- API keys
- General input sanitization

All validators return a tuple of (is_valid, sanitized_value_or_error_message)
"""

import re
import html
from typing import Tuple, Union, Optional
from decimal import Decimal, InvalidOperation


# Valid stock symbol pattern: 1-5 uppercase letters, optionally followed by a dot and 1-2 letters (for class shares)
SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')

# Email pattern (basic RFC 5322 compliant)
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# API key pattern: prefix_random_string format
# Examples: rdt_live_abc123, rdt_test_xyz789, sk_live_xxx
API_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_]{3,10}_[a-zA-Z0-9]{16,64}$')

# Dangerous characters for HTML/JS injection
DANGEROUS_CHARS_PATTERN = re.compile(r'[<>"\'\x00-\x1f\x7f-\x9f]')

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)', re.IGNORECASE),
    re.compile(r'(--|;|/\*|\*/|@@|@)', re.IGNORECASE),
    re.compile(r'(\bOR\b.*=.*\bOR\b|\bAND\b.*=.*\bAND\b)', re.IGNORECASE),
]

# XSS patterns
XSS_PATTERNS = [
    re.compile(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', re.IGNORECASE),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'on\w+\s*=', re.IGNORECASE),
    re.compile(r'<\s*iframe', re.IGNORECASE),
    re.compile(r'<\s*embed', re.IGNORECASE),
    re.compile(r'<\s*object', re.IGNORECASE),
]


class ValidationError(Exception):
    """Custom exception for validation errors"""

    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

    def to_dict(self):
        result = {'error': self.message}
        if self.field:
            result['field'] = self.field
        return result


def validate_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Validate a stock ticker symbol.

    Valid symbols are 1-5 uppercase letters, optionally followed by a dot
    and 1-2 letters for share class (e.g., BRK.A, BRK.B).

    Args:
        symbol: The stock ticker symbol to validate

    Returns:
        Tuple of (is_valid, sanitized_symbol or error_message)

    Examples:
        >>> validate_symbol("AAPL")
        (True, "AAPL")
        >>> validate_symbol("brk.a")
        (True, "BRK.A")
        >>> validate_symbol("TOOLONG123")
        (False, "Invalid symbol format. Must be 1-5 letters, optionally followed by .XX")
    """
    if symbol is None:
        return False, "Symbol is required"

    # Clean and uppercase
    symbol = str(symbol).strip().upper()

    if not symbol:
        return False, "Symbol cannot be empty"

    if len(symbol) > 8:  # Max length: XXXXX.XX
        return False, "Symbol is too long (max 8 characters)"

    if not SYMBOL_PATTERN.match(symbol):
        return False, "Invalid symbol format. Must be 1-5 letters, optionally followed by .XX"

    return True, symbol


def validate_price(price: Union[str, int, float, Decimal],
                   min_price: float = 0.0001,
                   max_price: float = 1000000.0,
                   allow_zero: bool = False) -> Tuple[bool, Union[float, str]]:
    """
    Validate a price value.

    Args:
        price: The price to validate (can be string, int, float, or Decimal)
        min_price: Minimum allowed price (default 0.0001)
        max_price: Maximum allowed price (default 1,000,000)
        allow_zero: Whether to allow zero as a valid price (default False)

    Returns:
        Tuple of (is_valid, sanitized_price as float or error_message)

    Examples:
        >>> validate_price(150.50)
        (True, 150.5)
        >>> validate_price("-10")
        (False, "Price must be positive")
        >>> validate_price("abc")
        (False, "Invalid price format")
    """
    if price is None:
        return False, "Price is required"

    try:
        # Convert to Decimal for precision, then to float
        if isinstance(price, str):
            price = price.strip().replace(',', '')
            if not price:
                return False, "Price cannot be empty"

        price_decimal = Decimal(str(price))
        price_float = float(price_decimal)

    except (InvalidOperation, ValueError, TypeError):
        return False, "Invalid price format"

    # Check for special float values
    if price_float != price_float:  # NaN check
        return False, "Price cannot be NaN"

    if price_float == float('inf') or price_float == float('-inf'):
        return False, "Price cannot be infinite"

    # Check bounds
    if not allow_zero and price_float <= 0:
        return False, "Price must be positive"

    if allow_zero and price_float < 0:
        return False, "Price cannot be negative"

    if price_float < min_price and not (allow_zero and price_float == 0):
        return False, f"Price must be at least {min_price}"

    if price_float > max_price:
        return False, f"Price cannot exceed {max_price:,.2f}"

    # Round to reasonable precision (max 4 decimal places for prices)
    price_float = round(price_float, 4)

    return True, price_float


def validate_quantity(qty: Union[str, int, float],
                      min_qty: int = 1,
                      max_qty: int = 10000000) -> Tuple[bool, Union[int, str]]:
    """
    Validate a quantity (shares/units).

    Args:
        qty: The quantity to validate
        min_qty: Minimum allowed quantity (default 1)
        max_qty: Maximum allowed quantity (default 10,000,000)

    Returns:
        Tuple of (is_valid, sanitized_quantity as int or error_message)

    Examples:
        >>> validate_quantity(100)
        (True, 100)
        >>> validate_quantity("50.5")
        (True, 50)  # Truncated to integer
        >>> validate_quantity(-10)
        (False, "Quantity must be at least 1")
    """
    if qty is None:
        return False, "Quantity is required"

    try:
        if isinstance(qty, str):
            qty = qty.strip().replace(',', '')
            if not qty:
                return False, "Quantity cannot be empty"

        # Convert to float first (to handle decimals), then to int
        qty_float = float(qty)
        qty_int = int(qty_float)

    except (ValueError, TypeError):
        return False, "Invalid quantity format"

    # Check for special values
    if qty_float != qty_float:  # NaN check
        return False, "Quantity cannot be NaN"

    if qty_int < min_qty:
        return False, f"Quantity must be at least {min_qty}"

    if qty_int > max_qty:
        return False, f"Quantity cannot exceed {max_qty:,}"

    return True, qty_int


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate an email address.

    Args:
        email: The email address to validate

    Returns:
        Tuple of (is_valid, sanitized_email or error_message)

    Examples:
        >>> validate_email("user@example.com")
        (True, "user@example.com")
        >>> validate_email("invalid-email")
        (False, "Invalid email format")
    """
    if email is None:
        return False, "Email is required"

    email = str(email).strip().lower()

    if not email:
        return False, "Email cannot be empty"

    if len(email) > 254:  # RFC 5321 limit
        return False, "Email address is too long"

    if not EMAIL_PATTERN.match(email):
        return False, "Invalid email format"

    # Additional checks
    local_part, domain = email.rsplit('@', 1)

    if len(local_part) > 64:  # RFC 5321 local part limit
        return False, "Email local part is too long"

    if '..' in email:
        return False, "Email cannot contain consecutive dots"

    if local_part.startswith('.') or local_part.endswith('.'):
        return False, "Email local part cannot start or end with a dot"

    return True, email


def validate_api_key(key: str) -> Tuple[bool, str]:
    """
    Validate an API key format.

    Expected format: prefix_random_string
    - Prefix: 3-10 alphanumeric characters with underscores
    - Random string: 16-64 alphanumeric characters

    Args:
        key: The API key to validate

    Returns:
        Tuple of (is_valid, sanitized_key or error_message)

    Examples:
        >>> validate_api_key("rdt_live_a1b2c3d4e5f6g7h8i9j0")
        (True, "rdt_live_a1b2c3d4e5f6g7h8i9j0")
        >>> validate_api_key("short")
        (False, "Invalid API key format")
    """
    if key is None:
        return False, "API key is required"

    key = str(key).strip()

    if not key:
        return False, "API key cannot be empty"

    if len(key) < 20:
        return False, "API key is too short"

    if len(key) > 128:
        return False, "API key is too long"

    # Check for invalid characters
    if not re.match(r'^[a-zA-Z0-9_]+$', key):
        return False, "API key contains invalid characters"

    # Check format (prefix_value)
    if '_' not in key:
        return False, "Invalid API key format"

    # Ensure at least one underscore separating prefix and value
    parts = key.split('_')
    if len(parts) < 2:
        return False, "Invalid API key format"

    # The random part should be the last segment
    random_part = parts[-1]
    if len(random_part) < 16:
        return False, "Invalid API key format"

    return True, key


def sanitize_input(text: str,
                   max_length: int = 1000,
                   allow_newlines: bool = False,
                   strip_html: bool = True) -> Tuple[bool, str]:
    """
    Sanitize general text input by removing dangerous characters.

    Args:
        text: The text to sanitize
        max_length: Maximum allowed length (default 1000)
        allow_newlines: Whether to preserve newline characters (default False)
        strip_html: Whether to escape HTML entities (default True)

    Returns:
        Tuple of (is_valid, sanitized_text or error_message)

    Examples:
        >>> sanitize_input("Hello, World!")
        (True, "Hello, World!")
        >>> sanitize_input("<script>alert('xss')</script>")
        (True, "&lt;script&gt;alert('xss')&lt;/script&gt;")
    """
    if text is None:
        return True, ""

    text = str(text)

    # Check length first
    if len(text) > max_length:
        return False, f"Input exceeds maximum length of {max_length} characters"

    # Remove null bytes
    text = text.replace('\x00', '')

    # Handle newlines
    if not allow_newlines:
        text = text.replace('\n', ' ').replace('\r', ' ')

    # Strip leading/trailing whitespace
    text = text.strip()

    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)

    # Check for SQL injection attempts
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(text):
            return False, "Input contains potentially dangerous content"

    # Check for XSS attempts
    for pattern in XSS_PATTERNS:
        if pattern.search(text):
            # Don't reject, but sanitize by escaping HTML
            pass

    # Escape HTML entities if requested
    if strip_html:
        text = html.escape(text, quote=True)

    return True, text


def validate_direction(direction: str) -> Tuple[bool, str]:
    """
    Validate trade direction.

    Args:
        direction: Trade direction ('long' or 'short')

    Returns:
        Tuple of (is_valid, sanitized_direction or error_message)
    """
    if direction is None:
        return False, "Direction is required"

    direction = str(direction).strip().lower()

    if direction not in ('long', 'short'):
        return False, "Direction must be 'long' or 'short'"

    return True, direction


def validate_condition(condition: str) -> Tuple[bool, str]:
    """
    Validate alert condition type.

    Args:
        condition: Alert condition type

    Returns:
        Tuple of (is_valid, sanitized_condition or error_message)
    """
    valid_conditions = {'rrs_above', 'rrs_below', 'price_above', 'price_below'}

    if condition is None:
        return False, "Condition is required"

    condition = str(condition).strip().lower()

    if condition not in valid_conditions:
        return False, f"Condition must be one of: {', '.join(valid_conditions)}"

    return True, condition


def validate_notification_method(method: str) -> Tuple[bool, str]:
    """
    Validate notification method.

    Args:
        method: Notification method ('email', 'sms', 'webhook', 'both')

    Returns:
        Tuple of (is_valid, sanitized_method or error_message)
    """
    valid_methods = {'email', 'sms', 'webhook', 'both'}

    if method is None:
        return False, "Notification method is required"

    method = str(method).strip().lower()

    if method not in valid_methods:
        return False, f"Notification method must be one of: {', '.join(valid_methods)}"

    return True, method


def validate_integer(value: Union[str, int, float],
                     field_name: str = "Value",
                     min_val: Optional[int] = None,
                     max_val: Optional[int] = None) -> Tuple[bool, Union[int, str]]:
    """
    Validate an integer value with optional bounds.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)

    Returns:
        Tuple of (is_valid, sanitized_int or error_message)
    """
    if value is None:
        return False, f"{field_name} is required"

    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return False, f"{field_name} cannot be empty"

        int_val = int(float(value))

    except (ValueError, TypeError):
        return False, f"Invalid {field_name.lower()} format"

    if min_val is not None and int_val < min_val:
        return False, f"{field_name} must be at least {min_val}"

    if max_val is not None and int_val > max_val:
        return False, f"{field_name} cannot exceed {max_val}"

    return True, int_val


def validate_float(value: Union[str, int, float],
                   field_name: str = "Value",
                   min_val: Optional[float] = None,
                   max_val: Optional[float] = None,
                   decimal_places: int = 4) -> Tuple[bool, Union[float, str]]:
    """
    Validate a float value with optional bounds.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        decimal_places: Number of decimal places to round to

    Returns:
        Tuple of (is_valid, sanitized_float or error_message)
    """
    if value is None:
        return False, f"{field_name} is required"

    try:
        if isinstance(value, str):
            value = value.strip().replace(',', '')
            if not value:
                return False, f"{field_name} cannot be empty"

        float_val = round(float(value), decimal_places)

    except (ValueError, TypeError):
        return False, f"Invalid {field_name.lower()} format"

    # Check for special values
    if float_val != float_val:  # NaN
        return False, f"{field_name} cannot be NaN"

    if float_val == float('inf') or float_val == float('-inf'):
        return False, f"{field_name} cannot be infinite"

    if min_val is not None and float_val < min_val:
        return False, f"{field_name} must be at least {min_val}"

    if max_val is not None and float_val > max_val:
        return False, f"{field_name} cannot exceed {max_val}"

    return True, float_val


def validate_json_body(data: dict, required_fields: list) -> Tuple[bool, Union[dict, str]]:
    """
    Validate that a JSON body contains all required fields.

    Args:
        data: The JSON data dictionary
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, data or error_message)
    """
    if data is None:
        return False, "Request body is required"

    if not isinstance(data, dict):
        return False, "Request body must be a JSON object"

    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    return True, data
