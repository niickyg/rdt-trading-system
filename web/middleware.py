"""
Request Validation Middleware for RDT Trading System

Provides:
- Content-Type validation for API requests
- Request size limits
- Rate limiting support
- Security headers
- Request logging
"""

import time
from functools import wraps
from flask import request, jsonify, g, Response, current_app
from loguru import logger


# Configuration defaults
DEFAULT_MAX_CONTENT_LENGTH = 1024 * 1024  # 1 MB
DEFAULT_JSON_MAX_LENGTH = 100 * 1024  # 100 KB for JSON
DEFAULT_ALLOWED_CONTENT_TYPES = {
    'application/json',
    'application/x-www-form-urlencoded',
    'multipart/form-data',
}


class RequestValidationMiddleware:
    """
    WSGI middleware for request validation and security.

    Provides:
    - Content-Type validation
    - Request size limits
    - Security headers injection
    - Request timing
    """

    def __init__(self, app, config=None):
        """
        Initialize middleware.

        Args:
            app: Flask application instance
            config: Optional configuration dictionary
        """
        self.app = app
        self.config = config or {}

        # Get configuration with defaults
        self.max_content_length = self.config.get(
            'max_content_length', DEFAULT_MAX_CONTENT_LENGTH
        )
        self.json_max_length = self.config.get(
            'json_max_length', DEFAULT_JSON_MAX_LENGTH
        )
        self.allowed_content_types = self.config.get(
            'allowed_content_types', DEFAULT_ALLOWED_CONTENT_TYPES
        )

        # Register before/after request handlers
        self.register_handlers()

    def register_handlers(self):
        """Register request handlers with the Flask app"""

        @self.app.before_request
        def before_request():
            """Pre-request validation and setup"""
            # Store request start time
            g.request_start_time = time.time()

            # Skip validation for static files and health checks
            if self._skip_validation():
                return None

            # Validate Content-Type for non-GET requests with body
            if request.method in ('POST', 'PUT', 'PATCH'):
                error = self._validate_content_type()
                if error:
                    return error

                # Validate request size
                error = self._validate_content_length()
                if error:
                    return error

            return None

        @self.app.after_request
        def after_request(response):
            """Post-request processing"""
            # Add security headers
            response = self._add_security_headers(response)

            # Log request timing
            if hasattr(g, 'request_start_time'):
                duration = time.time() - g.request_start_time
                if duration > 1.0:  # Log slow requests
                    logger.warning(
                        f"Slow request: {request.method} {request.path} "
                        f"took {duration:.2f}s"
                    )

            return response

    def _skip_validation(self):
        """Check if validation should be skipped for this request"""
        skip_paths = [
            '/static/',
            '/health',
            '/metrics',
            '/favicon.ico',
        ]
        return any(request.path.startswith(path) for path in skip_paths)

    def _validate_content_type(self):
        """
        Validate Content-Type header for requests with body.

        Returns:
            Error response or None if valid
        """
        # Allow empty body for GET-like requests
        if not request.data and not request.form and not request.files:
            return None

        content_type = request.content_type
        if not content_type:
            # For form submissions without explicit content type, allow it
            if request.form:
                return None
            return jsonify({
                'error': 'Content-Type header is required',
                'code': 'MISSING_CONTENT_TYPE'
            }), 415

        # Extract base content type (without charset etc.)
        base_content_type = content_type.split(';')[0].strip().lower()

        if base_content_type not in self.allowed_content_types:
            return jsonify({
                'error': f'Unsupported Content-Type: {base_content_type}',
                'allowed': list(self.allowed_content_types),
                'code': 'UNSUPPORTED_CONTENT_TYPE'
            }), 415

        return None

    def _validate_content_length(self):
        """
        Validate request body size.

        Returns:
            Error response or None if valid
        """
        content_length = request.content_length

        if content_length is None:
            # No content length header, check actual data
            if request.data:
                content_length = len(request.data)

        if content_length is None:
            return None

        # Check against max content length
        if content_length > self.max_content_length:
            return jsonify({
                'error': 'Request body too large',
                'max_size': self.max_content_length,
                'received_size': content_length,
                'code': 'REQUEST_TOO_LARGE'
            }), 413

        # Check JSON-specific limit
        content_type = request.content_type or ''
        if 'application/json' in content_type.lower():
            if content_length > self.json_max_length:
                return jsonify({
                    'error': 'JSON body too large',
                    'max_size': self.json_max_length,
                    'received_size': content_length,
                    'code': 'JSON_TOO_LARGE'
                }), 413

        return None

    def _add_security_headers(self, response):
        """
        Add security headers to response.

        Args:
            response: Flask response object

        Returns:
            Response with security headers
        """
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'

        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'

        # XSS protection (legacy browsers)
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # HTTP Strict Transport Security
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Content Security Policy (basic)
        # Note: Customize based on your application needs
        if 'text/html' in response.content_type:
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' wss: ws:; "
                "frame-ancestors 'self';"
            )
            response.headers['Content-Security-Policy'] = csp

        # Referrer policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Permissions policy
        response.headers['Permissions-Policy'] = (
            'accelerometer=(), camera=(), geolocation=(), gyroscope=(), '
            'magnetometer=(), microphone=(), payment=(), usb=()'
        )

        return response


def init_request_validation(app, config=None):
    """
    Initialize request validation middleware for a Flask app.

    Args:
        app: Flask application instance
        config: Optional configuration dictionary with:
            - max_content_length: Maximum request body size
            - json_max_length: Maximum JSON body size
            - allowed_content_types: Set of allowed Content-Types

    Returns:
        RequestValidationMiddleware instance
    """
    # Set Flask's max content length
    max_length = (config or {}).get('max_content_length', DEFAULT_MAX_CONTENT_LENGTH)
    app.config['MAX_CONTENT_LENGTH'] = max_length

    middleware = RequestValidationMiddleware(app, config)
    logger.info("Request validation middleware initialized")

    return middleware


# =============================================================================
# DECORATOR-BASED VALIDATORS
# =============================================================================

def require_json(f):
    """
    Decorator to require JSON Content-Type and body.

    Usage:
        @app.route('/api/endpoint', methods=['POST'])
        @require_json
        def endpoint():
            data = request.get_json()
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'code': 'JSON_REQUIRED'
            }), 415

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({
                'error': 'Invalid JSON body',
                'code': 'INVALID_JSON'
            }), 400

        return f(*args, **kwargs)
    return decorated


def require_fields(*required_fields):
    """
    Decorator to require specific fields in JSON body.

    Usage:
        @app.route('/api/endpoint', methods=['POST'])
        @require_json
        @require_fields('symbol', 'price', 'quantity')
        def endpoint():
            data = request.get_json()
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({
                    'error': 'Request body is required',
                    'code': 'BODY_REQUIRED'
                }), 400

            missing = []
            for field in required_fields:
                if field not in data or data[field] is None:
                    missing.append(field)

            if missing:
                return jsonify({
                    'error': f'Missing required fields: {", ".join(missing)}',
                    'missing_fields': missing,
                    'code': 'MISSING_FIELDS'
                }), 400

            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_request_size(max_size=None):
    """
    Decorator to validate request body size.

    Args:
        max_size: Maximum allowed size in bytes (default: 100KB)

    Usage:
        @app.route('/api/endpoint', methods=['POST'])
        @validate_request_size(50 * 1024)  # 50KB max
        def endpoint():
            ...
    """
    if max_size is None:
        max_size = DEFAULT_JSON_MAX_LENGTH

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            content_length = request.content_length or 0
            if request.data:
                content_length = len(request.data)

            if content_length > max_size:
                return jsonify({
                    'error': 'Request body too large',
                    'max_size': max_size,
                    'received_size': content_length,
                    'code': 'REQUEST_TOO_LARGE'
                }), 413

            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_content_type(*allowed_types):
    """
    Decorator to validate Content-Type header.

    Args:
        *allowed_types: Allowed Content-Type values

    Usage:
        @app.route('/api/endpoint', methods=['POST'])
        @validate_content_type('application/json', 'application/xml')
        def endpoint():
            ...
    """
    if not allowed_types:
        allowed_types = ('application/json',)

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            content_type = request.content_type
            if not content_type:
                if request.data or request.form:
                    return jsonify({
                        'error': 'Content-Type header is required',
                        'code': 'MISSING_CONTENT_TYPE'
                    }), 415
                return f(*args, **kwargs)

            # Extract base content type
            base_type = content_type.split(';')[0].strip().lower()

            if base_type not in [t.lower() for t in allowed_types]:
                return jsonify({
                    'error': f'Unsupported Content-Type: {base_type}',
                    'allowed': list(allowed_types),
                    'code': 'UNSUPPORTED_CONTENT_TYPE'
                }), 415

            return f(*args, **kwargs)
        return decorated
    return decorator


# =============================================================================
# INPUT SANITIZATION HELPERS
# =============================================================================

def get_json_or_error():
    """
    Get JSON from request or return error response.

    Returns:
        Tuple of (data, error_response)
        data is the JSON data if valid, None otherwise
        error_response is a tuple (response, status_code) if error, None otherwise

    Usage:
        data, error = get_json_or_error()
        if error:
            return error
        # Use data...
    """
    if not request.is_json:
        return None, (jsonify({
            'error': 'Content-Type must be application/json',
            'code': 'JSON_REQUIRED'
        }), 415)

    try:
        data = request.get_json()
        if data is None:
            return None, (jsonify({
                'error': 'Request body is required',
                'code': 'BODY_REQUIRED'
            }), 400)
        return data, None
    except Exception as e:
        return None, (jsonify({
            'error': 'Invalid JSON body',
            'details': str(e),
            'code': 'INVALID_JSON'
        }), 400)


def get_validated_json(required_fields=None, optional_fields=None):
    """
    Get and validate JSON from request.

    Args:
        required_fields: List of required field names
        optional_fields: List of optional field names (filters out unknown fields)

    Returns:
        Tuple of (validated_data, error_response)

    Usage:
        data, error = get_validated_json(
            required_fields=['symbol', 'price'],
            optional_fields=['quantity', 'note']
        )
        if error:
            return error
        # Use data...
    """
    data, error = get_json_or_error()
    if error:
        return None, error

    if not isinstance(data, dict):
        return None, (jsonify({
            'error': 'Request body must be a JSON object',
            'code': 'INVALID_BODY_TYPE'
        }), 400)

    # Check required fields
    if required_fields:
        missing = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing.append(field)
        if missing:
            return None, (jsonify({
                'error': f'Missing required fields: {", ".join(missing)}',
                'missing_fields': missing,
                'code': 'MISSING_FIELDS'
            }), 400)

    # Filter to allowed fields if specified
    if optional_fields is not None:
        allowed_fields = set(required_fields or []) | set(optional_fields)
        data = {k: v for k, v in data.items() if k in allowed_fields}

    return data, None


def get_query_param(name, type_func=str, default=None, required=False):
    """
    Get and validate a query parameter.

    Args:
        name: Parameter name
        type_func: Type conversion function (str, int, float, bool)
        default: Default value if not provided
        required: Whether the parameter is required

    Returns:
        Tuple of (value, error_response)

    Usage:
        days, error = get_query_param('days', int, default=30)
        if error:
            return error
    """
    value = request.args.get(name)

    if value is None:
        if required:
            return None, (jsonify({
                'error': f'Missing required query parameter: {name}',
                'code': 'MISSING_PARAMETER'
            }), 400)
        return default, None

    try:
        if type_func == bool:
            # Handle boolean specially
            value = value.lower() in ('true', '1', 'yes', 'on')
        else:
            value = type_func(value)
        return value, None
    except (ValueError, TypeError):
        return None, (jsonify({
            'error': f'Invalid value for parameter: {name}',
            'expected_type': type_func.__name__,
            'code': 'INVALID_PARAMETER'
        }), 400)
