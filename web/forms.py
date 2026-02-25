"""
WTForms for RDT Trading System

Provides form classes with validation for:
- User authentication (login)
- User settings and preferences
- Alert configuration
- Position management

Uses Flask-WTF for CSRF protection and WTForms validators.
"""

from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    PasswordField,
    BooleanField,
    SelectField,
    IntegerField,
    DecimalField,
    TextAreaField,
    EmailField,
    TelField,
    HiddenField,
)
from wtforms.validators import (
    DataRequired,
    Email,
    Length,
    NumberRange,
    Optional,
    Regexp,
    ValidationError,
    EqualTo,
)
import re


# Custom validators
def validate_stock_symbol(form, field):
    """Validate stock ticker symbol format"""
    if not field.data:
        return

    symbol = field.data.strip().upper()
    pattern = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')

    if not pattern.match(symbol):
        raise ValidationError(
            'Invalid symbol format. Must be 1-5 letters, optionally followed by .XX'
        )


def validate_positive_number(form, field):
    """Validate that a number is positive"""
    if field.data is not None and field.data <= 0:
        raise ValidationError('Value must be positive')


def validate_api_key_format(form, field):
    """Validate API key format"""
    if not field.data:
        return

    key = field.data.strip()
    if len(key) < 20:
        raise ValidationError('API key is too short')

    if not re.match(r'^[a-zA-Z0-9_]+$', key):
        raise ValidationError('API key contains invalid characters')

    if '_' not in key:
        raise ValidationError('Invalid API key format')


def validate_phone_number(form, field):
    """Validate phone number format"""
    if not field.data:
        return

    # Remove common formatting characters
    phone = re.sub(r'[\s\-\(\)\.\+]', '', field.data)

    if not phone.isdigit():
        raise ValidationError('Phone number must contain only digits')

    if len(phone) < 10 or len(phone) > 15:
        raise ValidationError('Phone number must be 10-15 digits')


def validate_no_xss(form, field):
    """Validate that field doesn't contain XSS patterns"""
    if not field.data:
        return

    xss_patterns = [
        r'<script\b',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe',
        r'<embed',
        r'<object',
    ]

    text = field.data.lower()
    for pattern in xss_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValidationError('Input contains potentially dangerous content')


# =============================================================================
# LOGIN FORM
# =============================================================================

class LoginForm(FlaskForm):
    """Login form with CSRF protection"""

    username = StringField(
        'Username',
        validators=[
            DataRequired(message='Username is required'),
            Length(min=3, max=64, message='Username must be 3-64 characters'),
            Regexp(
                r'^[a-zA-Z0-9_]+$',
                message='Username can only contain letters, numbers, and underscores'
            ),
        ],
        render_kw={
            'placeholder': 'Enter your username',
            'autocomplete': 'username',
            'class': 'form-control',
        }
    )

    password = PasswordField(
        'Password',
        validators=[
            DataRequired(message='Password is required'),
            Length(min=6, max=128, message='Password must be 6-128 characters'),
        ],
        render_kw={
            'placeholder': 'Enter your password',
            'autocomplete': 'current-password',
            'class': 'form-control',
        }
    )

    remember = BooleanField(
        'Remember me for 30 days',
        default=False
    )


# =============================================================================
# REGISTRATION FORM
# =============================================================================

class RegistrationForm(FlaskForm):
    """User registration form"""

    username = StringField(
        'Username',
        validators=[
            DataRequired(message='Username is required'),
            Length(min=3, max=64, message='Username must be 3-64 characters'),
            Regexp(
                r'^[a-zA-Z0-9_]+$',
                message='Username can only contain letters, numbers, and underscores'
            ),
        ],
        render_kw={
            'placeholder': 'Choose a username',
            'autocomplete': 'username',
            'class': 'form-control',
        }
    )

    email = EmailField(
        'Email',
        validators=[
            DataRequired(message='Email is required'),
            Email(message='Please enter a valid email address'),
            Length(max=254, message='Email is too long'),
        ],
        render_kw={
            'placeholder': 'your@email.com',
            'autocomplete': 'email',
            'class': 'form-control',
        }
    )

    password = PasswordField(
        'Password',
        validators=[
            DataRequired(message='Password is required'),
            Length(min=8, max=128, message='Password must be at least 8 characters'),
            Regexp(
                r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)',
                message='Password must contain at least one uppercase letter, one lowercase letter, and one number'
            ),
        ],
        render_kw={
            'placeholder': 'Create a strong password',
            'autocomplete': 'new-password',
            'class': 'form-control',
        }
    )

    confirm_password = PasswordField(
        'Confirm Password',
        validators=[
            DataRequired(message='Please confirm your password'),
            EqualTo('password', message='Passwords must match'),
        ],
        render_kw={
            'placeholder': 'Confirm your password',
            'autocomplete': 'new-password',
            'class': 'form-control',
        }
    )

    terms_accepted = BooleanField(
        'I agree to the Terms of Service and Privacy Policy',
        validators=[
            DataRequired(message='You must accept the terms to register'),
        ]
    )


# =============================================================================
# SETTINGS FORM
# =============================================================================

class SettingsForm(FlaskForm):
    """User settings form for preferences"""

    # Account Information
    full_name = StringField(
        'Full Name',
        validators=[
            Optional(),
            Length(max=100, message='Name is too long'),
            validate_no_xss,
        ],
        render_kw={
            'placeholder': 'John Doe',
            'class': 'form-control',
        }
    )

    email = EmailField(
        'Email Address',
        validators=[
            DataRequired(message='Email is required'),
            Email(message='Please enter a valid email address'),
            Length(max=254, message='Email is too long'),
        ],
        render_kw={
            'placeholder': 'john@example.com',
            'class': 'form-control',
        }
    )

    phone = TelField(
        'Phone Number',
        validators=[
            Optional(),
            validate_phone_number,
        ],
        render_kw={
            'placeholder': '+1 (555) 123-4567',
            'class': 'form-control',
        }
    )

    timezone = SelectField(
        'Timezone',
        choices=[
            ('America/New_York', 'Eastern Time (ET)'),
            ('America/Chicago', 'Central Time (CT)'),
            ('America/Denver', 'Mountain Time (MT)'),
            ('America/Los_Angeles', 'Pacific Time (PT)'),
            ('UTC', 'UTC'),
        ],
        default='America/New_York',
        render_kw={'class': 'form-control'},
    )

    # Trading Preferences
    default_position_size = IntegerField(
        'Default Position Size ($)',
        validators=[
            Optional(),
            NumberRange(min=100, max=10000000, message='Position size must be between $100 and $10,000,000'),
        ],
        render_kw={
            'placeholder': '10000',
            'class': 'form-control',
        }
    )

    max_positions = IntegerField(
        'Max Open Positions',
        validators=[
            Optional(),
            NumberRange(min=1, max=100, message='Max positions must be between 1 and 100'),
        ],
        default=5,
        render_kw={
            'placeholder': '5',
            'class': 'form-control',
        }
    )

    risk_per_trade = DecimalField(
        'Risk Per Trade (%)',
        validators=[
            Optional(),
            NumberRange(min=0.1, max=10, message='Risk must be between 0.1% and 10%'),
        ],
        default=2.0,
        places=1,
        render_kw={
            'placeholder': '2.0',
            'class': 'form-control',
        }
    )

    min_rrs = DecimalField(
        'Minimum RRS Threshold',
        validators=[
            Optional(),
            NumberRange(min=0, max=10, message='RRS threshold must be between 0 and 10'),
        ],
        default=1.5,
        places=1,
        render_kw={
            'placeholder': '1.5',
            'class': 'form-control',
        }
    )

    preferred_direction = SelectField(
        'Preferred Trade Direction',
        choices=[
            ('both', 'Both (Long & Short)'),
            ('long', 'Long Only'),
            ('short', 'Short Only'),
        ],
        default='both',
        render_kw={'class': 'form-control'},
    )

    signal_strength = SelectField(
        'Minimum Signal Strength',
        choices=[
            ('weak', 'All Signals'),
            ('moderate', 'Moderate & Strong'),
            ('strong', 'Strong Only'),
        ],
        default='moderate',
        render_kw={'class': 'form-control'},
    )

    # Notification Settings
    email_notifications = BooleanField('Email Notifications', default=True)
    sms_notifications = BooleanField('SMS Notifications', default=False)
    signal_alerts = BooleanField('New Signal Alerts', default=True)
    price_alerts = BooleanField('Price Target Alerts', default=True)
    daily_summary = BooleanField('Daily Summary', default=True)


# =============================================================================
# ALERT CONFIG FORM
# =============================================================================

class AlertConfigForm(FlaskForm):
    """Form for configuring price and RRS alerts"""

    symbol = StringField(
        'Symbol',
        validators=[
            DataRequired(message='Symbol is required'),
            Length(min=1, max=8, message='Symbol must be 1-8 characters'),
            validate_stock_symbol,
        ],
        render_kw={
            'placeholder': 'e.g., AAPL, MSFT, TSLA',
            'class': 'form-control',
        }
    )

    condition = SelectField(
        'Condition',
        choices=[
            ('', 'Select condition...'),
            ('rrs_above', 'RRS Above'),
            ('rrs_below', 'RRS Below'),
            ('price_above', 'Price Above'),
            ('price_below', 'Price Below'),
        ],
        validators=[
            DataRequired(message='Condition is required'),
        ],
        render_kw={'class': 'form-control'},
    )

    value = DecimalField(
        'Value',
        validators=[
            DataRequired(message='Value is required'),
            NumberRange(min=-100, max=1000000, message='Value out of range'),
        ],
        places=2,
        render_kw={
            'placeholder': '0.00',
            'class': 'form-control',
        }
    )

    notification_method = SelectField(
        'Notification Method',
        choices=[
            ('email', 'Email'),
            ('sms', 'SMS'),
            ('both', 'Email & SMS'),
            ('webhook', 'Webhook'),
        ],
        default='email',
        render_kw={'class': 'form-control'},
    )

    note = StringField(
        'Note (optional)',
        validators=[
            Optional(),
            Length(max=200, message='Note is too long (max 200 characters)'),
            validate_no_xss,
        ],
        render_kw={
            'placeholder': 'Add a note to remember why you set this alert',
            'class': 'form-control',
        }
    )


# =============================================================================
# POSITION FORM
# =============================================================================

class PositionForm(FlaskForm):
    """Form for opening/editing positions"""

    symbol = StringField(
        'Symbol',
        validators=[
            DataRequired(message='Symbol is required'),
            Length(min=1, max=8, message='Symbol must be 1-8 characters'),
            validate_stock_symbol,
        ],
        render_kw={
            'placeholder': 'e.g., AAPL',
            'class': 'form-control',
        }
    )

    direction = SelectField(
        'Direction',
        choices=[
            ('long', 'Long'),
            ('short', 'Short'),
        ],
        validators=[
            DataRequired(message='Direction is required'),
        ],
        render_kw={'class': 'form-control'},
    )

    entry_price = DecimalField(
        'Entry Price',
        validators=[
            DataRequired(message='Entry price is required'),
            NumberRange(min=0.0001, max=1000000, message='Invalid entry price'),
            validate_positive_number,
        ],
        places=4,
        render_kw={
            'placeholder': '0.00',
            'class': 'form-control',
        }
    )

    shares = IntegerField(
        'Shares',
        validators=[
            DataRequired(message='Number of shares is required'),
            NumberRange(min=1, max=10000000, message='Shares must be between 1 and 10,000,000'),
        ],
        render_kw={
            'placeholder': '100',
            'class': 'form-control',
        }
    )

    stop_price = DecimalField(
        'Stop Price',
        validators=[
            DataRequired(message='Stop price is required'),
            NumberRange(min=0.0001, max=1000000, message='Invalid stop price'),
            validate_positive_number,
        ],
        places=4,
        render_kw={
            'placeholder': '0.00',
            'class': 'form-control',
        }
    )

    target_price = DecimalField(
        'Target Price',
        validators=[
            DataRequired(message='Target price is required'),
            NumberRange(min=0.0001, max=1000000, message='Invalid target price'),
            validate_positive_number,
        ],
        places=4,
        render_kw={
            'placeholder': '0.00',
            'class': 'form-control',
        }
    )

    rrs_at_entry = DecimalField(
        'RRS at Entry (optional)',
        validators=[
            Optional(),
            NumberRange(min=-20, max=20, message='RRS must be between -20 and 20'),
        ],
        places=2,
        render_kw={
            'placeholder': '0.00',
            'class': 'form-control',
        }
    )

    note = TextAreaField(
        'Trade Note (optional)',
        validators=[
            Optional(),
            Length(max=500, message='Note is too long (max 500 characters)'),
            validate_no_xss,
        ],
        render_kw={
            'placeholder': 'Entry rationale, market conditions, etc.',
            'class': 'form-control',
            'rows': 3,
        }
    )


# =============================================================================
# BACKTEST FORM
# =============================================================================

class BacktestForm(FlaskForm):
    """Form for configuring backtest parameters"""

    symbols = StringField(
        'Symbols (comma-separated)',
        validators=[
            Optional(),
            Length(max=500, message='Too many symbols'),
            validate_no_xss,
        ],
        render_kw={
            'placeholder': 'AAPL, MSFT, GOOGL (leave blank for default watchlist)',
            'class': 'form-control',
        }
    )

    days = IntegerField(
        'Lookback Period (days)',
        validators=[
            DataRequired(message='Lookback period is required'),
            NumberRange(min=30, max=730, message='Period must be between 30 and 730 days'),
        ],
        default=365,
        render_kw={
            'placeholder': '365',
            'class': 'form-control',
        }
    )

    rrs_threshold = DecimalField(
        'RRS Threshold',
        validators=[
            DataRequired(message='RRS threshold is required'),
            NumberRange(min=0.5, max=5, message='Threshold must be between 0.5 and 5'),
        ],
        default=1.75,
        places=2,
        render_kw={
            'placeholder': '1.75',
            'class': 'form-control',
        }
    )

    stop_atr_mult = DecimalField(
        'Stop ATR Multiplier',
        validators=[
            DataRequired(message='Stop ATR multiplier is required'),
            NumberRange(min=0.25, max=3, message='Multiplier must be between 0.25 and 3'),
        ],
        default=0.75,
        places=2,
        render_kw={
            'placeholder': '0.75',
            'class': 'form-control',
        }
    )

    target_atr_mult = DecimalField(
        'Target ATR Multiplier',
        validators=[
            DataRequired(message='Target ATR multiplier is required'),
            NumberRange(min=0.5, max=5, message='Multiplier must be between 0.5 and 5'),
        ],
        default=1.5,
        places=2,
        render_kw={
            'placeholder': '1.5',
            'class': 'form-control',
        }
    )

    risk_per_trade = DecimalField(
        'Risk Per Trade (%)',
        validators=[
            DataRequired(message='Risk per trade is required'),
            NumberRange(min=0.5, max=5, message='Risk must be between 0.5% and 5%'),
        ],
        default=2.0,
        places=1,
        render_kw={
            'placeholder': '2.0',
            'class': 'form-control',
        }
    )

    initial_capital = IntegerField(
        'Initial Capital ($)',
        validators=[
            Optional(),
            NumberRange(min=1000, max=10000000, message='Capital must be between $1,000 and $10,000,000'),
        ],
        default=25000,
        render_kw={
            'placeholder': '25000',
            'class': 'form-control',
        }
    )


# =============================================================================
# PASSWORD CHANGE FORM
# =============================================================================

class PasswordChangeForm(FlaskForm):
    """Form for changing user password"""

    current_password = PasswordField(
        'Current Password',
        validators=[
            DataRequired(message='Current password is required'),
        ],
        render_kw={
            'placeholder': 'Enter current password',
            'autocomplete': 'current-password',
            'class': 'form-control',
        }
    )

    new_password = PasswordField(
        'New Password',
        validators=[
            DataRequired(message='New password is required'),
            Length(min=8, max=128, message='Password must be at least 8 characters'),
            Regexp(
                r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)',
                message='Password must contain at least one uppercase letter, one lowercase letter, and one number'
            ),
        ],
        render_kw={
            'placeholder': 'Enter new password',
            'autocomplete': 'new-password',
            'class': 'form-control',
        }
    )

    confirm_password = PasswordField(
        'Confirm New Password',
        validators=[
            DataRequired(message='Please confirm your new password'),
            EqualTo('new_password', message='Passwords must match'),
        ],
        render_kw={
            'placeholder': 'Confirm new password',
            'autocomplete': 'new-password',
            'class': 'form-control',
        }
    )


# =============================================================================
# API KEY FORM
# =============================================================================

class APIKeyForm(FlaskForm):
    """Form for API key management"""

    key_name = StringField(
        'Key Name',
        validators=[
            DataRequired(message='Key name is required'),
            Length(min=1, max=100, message='Name must be 1-100 characters'),
            validate_no_xss,
        ],
        render_kw={
            'placeholder': 'My Trading Bot',
            'class': 'form-control',
        }
    )

    permissions = SelectField(
        'Permissions',
        choices=[
            ('read', 'Read Only'),
            ('write', 'Read & Write'),
            ('admin', 'Full Access'),
        ],
        default='read',
        render_kw={'class': 'form-control'},
    )

    expires_in_days = SelectField(
        'Expiration',
        choices=[
            ('30', '30 Days'),
            ('90', '90 Days'),
            ('180', '180 Days'),
            ('365', '1 Year'),
            ('0', 'Never'),
        ],
        default='90',
        render_kw={'class': 'form-control'},
    )
