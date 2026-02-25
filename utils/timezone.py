"""
Timezone Utilities for RDT Trading System

Provides proper Eastern Time (ET) handling for US market hours.
US markets operate in Eastern Time, so all market-related time calculations
should use these utilities.

Uses Python 3.9+ zoneinfo for timezone handling.
"""

from datetime import datetime, date, time, timedelta
from typing import Optional, Set
from zoneinfo import ZoneInfo

# Eastern Time zone
EASTERN_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# US Market Hours (in Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Pre-market and after-hours
PRE_MARKET_OPEN_HOUR = 4
PRE_MARKET_OPEN_MINUTE = 0
AFTER_HOURS_CLOSE_HOUR = 20
AFTER_HOURS_CLOSE_MINUTE = 0

# US Market Holidays for 2024-2026
# These are the days when US stock markets (NYSE, NASDAQ) are closed
# Source: NYSE Holiday Calendar
US_MARKET_HOLIDAYS: Set[date] = {
    # 2024 Holidays
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 15),   # Martin Luther King Jr. Day
    date(2024, 2, 19),   # Presidents Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 27),   # Memorial Day
    date(2024, 6, 19),   # Juneteenth
    date(2024, 7, 4),    # Independence Day
    date(2024, 9, 2),    # Labor Day
    date(2024, 11, 28),  # Thanksgiving Day
    date(2024, 12, 25),  # Christmas Day

    # 2025 Holidays
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # Martin Luther King Jr. Day
    date(2025, 2, 17),   # Presidents Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving Day
    date(2025, 12, 25),  # Christmas Day

    # 2026 Holidays
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed - July 4 is Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25),  # Christmas Day
}

# Early close days (market closes at 1:00 PM ET)
# Day before or after major holidays
US_EARLY_CLOSE_DAYS: Set[date] = {
    # 2024
    date(2024, 7, 3),    # Day before Independence Day
    date(2024, 11, 29),  # Day after Thanksgiving
    date(2024, 12, 24),  # Christmas Eve

    # 2025
    date(2025, 7, 3),    # Day before Independence Day
    date(2025, 11, 28),  # Day after Thanksgiving
    date(2025, 12, 24),  # Christmas Eve

    # 2026
    date(2026, 11, 27),  # Day after Thanksgiving
    date(2026, 12, 24),  # Christmas Eve
}


def get_eastern_time() -> datetime:
    """
    Get the current time in Eastern Time (ET).

    Returns:
        datetime: Current datetime with Eastern timezone info
    """
    return datetime.now(EASTERN_TZ)


def to_eastern(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Convert any datetime to Eastern Time.

    Args:
        dt: A datetime object (can be naive, UTC, or any timezone)

    Returns:
        datetime: The datetime converted to Eastern Time, or None if input is None

    Notes:
        - If the datetime is naive (no timezone), it's assumed to be in local time
          and will be localized before conversion
        - If the datetime has timezone info, it will be converted properly
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Naive datetime - assume it's already in Eastern Time
        return dt.replace(tzinfo=EASTERN_TZ)

    # Convert to Eastern Time
    return dt.astimezone(EASTERN_TZ)


def is_trading_day(check_date: Optional[date] = None) -> bool:
    """
    Check if a given date is a trading day (not weekend, not holiday).

    Args:
        check_date: The date to check. Defaults to today in Eastern Time.

    Returns:
        bool: True if it's a trading day, False otherwise
    """
    if check_date is None:
        check_date = get_eastern_time().date()

    # Check if it's a weekend (Saturday=5, Sunday=6)
    if check_date.weekday() >= 5:
        return False

    # Check if it's a market holiday
    if check_date in US_MARKET_HOLIDAYS:
        return False

    return True


def is_early_close_day(check_date: Optional[date] = None) -> bool:
    """
    Check if a given date is an early close day (market closes at 1:00 PM ET).

    Args:
        check_date: The date to check. Defaults to today in Eastern Time.

    Returns:
        bool: True if it's an early close day, False otherwise
    """
    if check_date is None:
        check_date = get_eastern_time().date()

    return check_date in US_EARLY_CLOSE_DAYS


def get_market_open_time(for_date: Optional[date] = None) -> datetime:
    """
    Get the market open time (9:30 AM ET) for a given date.

    Args:
        for_date: The date to get market open time for. Defaults to today.

    Returns:
        datetime: Market open time in Eastern Time with timezone info
    """
    if for_date is None:
        for_date = get_eastern_time().date()

    return datetime(
        for_date.year,
        for_date.month,
        for_date.day,
        MARKET_OPEN_HOUR,
        MARKET_OPEN_MINUTE,
        tzinfo=EASTERN_TZ
    )


def get_market_close_time(for_date: Optional[date] = None) -> datetime:
    """
    Get the market close time for a given date.
    Regular close is 4:00 PM ET, early close is 1:00 PM ET.

    Args:
        for_date: The date to get market close time for. Defaults to today.

    Returns:
        datetime: Market close time in Eastern Time with timezone info
    """
    if for_date is None:
        for_date = get_eastern_time().date()

    # Check for early close days
    if is_early_close_day(for_date):
        close_hour = 13  # 1:00 PM
        close_minute = 0
    else:
        close_hour = MARKET_CLOSE_HOUR
        close_minute = MARKET_CLOSE_MINUTE

    return datetime(
        for_date.year,
        for_date.month,
        for_date.day,
        close_hour,
        close_minute,
        tzinfo=EASTERN_TZ
    )


def is_market_open(check_time: Optional[datetime] = None) -> bool:
    """
    Check if the US stock market is currently open.

    Args:
        check_time: The time to check. Defaults to current Eastern Time.

    Returns:
        bool: True if market is open, False otherwise

    Notes:
        - Checks if it's a trading day (not weekend/holiday)
        - Checks if current time is between 9:30 AM and 4:00 PM ET
        - Accounts for early close days (1:00 PM close)
    """
    if check_time is None:
        check_time = get_eastern_time()
    else:
        check_time = to_eastern(check_time)

    # Check if it's a trading day
    if not is_trading_day(check_time.date()):
        return False

    # Get market hours for this date
    market_open = get_market_open_time(check_time.date())
    market_close = get_market_close_time(check_time.date())

    # Check if current time is within market hours
    return market_open <= check_time < market_close


def is_pre_market(check_time: Optional[datetime] = None) -> bool:
    """
    Check if it's currently pre-market hours (4:00 AM - 9:30 AM ET).

    Args:
        check_time: The time to check. Defaults to current Eastern Time.

    Returns:
        bool: True if in pre-market hours, False otherwise
    """
    if check_time is None:
        check_time = get_eastern_time()
    else:
        check_time = to_eastern(check_time)

    # Check if it's a trading day
    if not is_trading_day(check_time.date()):
        return False

    # Pre-market hours: 4:00 AM - 9:30 AM ET
    pre_market_open = datetime(
        check_time.year, check_time.month, check_time.day,
        PRE_MARKET_OPEN_HOUR, PRE_MARKET_OPEN_MINUTE,
        tzinfo=EASTERN_TZ
    )
    market_open = get_market_open_time(check_time.date())

    return pre_market_open <= check_time < market_open


def is_after_hours(check_time: Optional[datetime] = None) -> bool:
    """
    Check if it's currently after-hours trading (4:00 PM - 8:00 PM ET).

    Args:
        check_time: The time to check. Defaults to current Eastern Time.

    Returns:
        bool: True if in after-hours, False otherwise
    """
    if check_time is None:
        check_time = get_eastern_time()
    else:
        check_time = to_eastern(check_time)

    # Check if it's a trading day
    if not is_trading_day(check_time.date()):
        return False

    # After-hours: market close - 8:00 PM ET
    market_close = get_market_close_time(check_time.date())
    after_hours_close = datetime(
        check_time.year, check_time.month, check_time.day,
        AFTER_HOURS_CLOSE_HOUR, AFTER_HOURS_CLOSE_MINUTE,
        tzinfo=EASTERN_TZ
    )

    return market_close <= check_time < after_hours_close


# Alias functions for extended hours trading support
def is_premarket(check_time: Optional[datetime] = None) -> bool:
    """
    Check if in pre-market hours (4:00-9:30 AM ET).

    Alias for is_pre_market() for consistency with naming conventions.

    Args:
        check_time: The time to check. Defaults to current Eastern Time.

    Returns:
        bool: True if in pre-market hours, False otherwise
    """
    return is_pre_market(check_time)


def is_afterhours(check_time: Optional[datetime] = None) -> bool:
    """
    Check if in after-hours trading (4:00-8:00 PM ET).

    Alias for is_after_hours() for consistency with naming conventions.

    Args:
        check_time: The time to check. Defaults to current Eastern Time.

    Returns:
        bool: True if in after-hours, False otherwise
    """
    return is_after_hours(check_time)


def is_extended_hours(check_time: Optional[datetime] = None) -> bool:
    """
    Check if in extended hours trading (pre-market or after-hours).

    Extended hours include:
    - Pre-market: 4:00 AM - 9:30 AM ET
    - After-hours: 4:00 PM - 8:00 PM ET

    Args:
        check_time: The time to check. Defaults to current Eastern Time.

    Returns:
        bool: True if in extended hours (pre-market or after-hours), False otherwise
    """
    return is_pre_market(check_time) or is_after_hours(check_time)


def get_extended_hours_session(check_time: Optional[datetime] = None) -> str:
    """
    Get the current extended hours trading session.

    Args:
        check_time: The time to check. Defaults to current Eastern Time.

    Returns:
        str: One of 'premarket', 'regular', 'afterhours', or 'closed'
    """
    if check_time is None:
        check_time = get_eastern_time()
    else:
        check_time = to_eastern(check_time)

    # Check if it's a trading day first
    if not is_trading_day(check_time.date()):
        return 'closed'

    # Check each session in order
    if is_pre_market(check_time):
        return 'premarket'
    elif is_market_open(check_time):
        return 'regular'
    elif is_after_hours(check_time):
        return 'afterhours'
    else:
        return 'closed'


def get_premarket_open_time(for_date: Optional[date] = None) -> datetime:
    """
    Get the pre-market open time (4:00 AM ET) for a given date.

    Args:
        for_date: The date to get pre-market open time for. Defaults to today.

    Returns:
        datetime: Pre-market open time in Eastern Time with timezone info
    """
    if for_date is None:
        for_date = get_eastern_time().date()

    return datetime(
        for_date.year,
        for_date.month,
        for_date.day,
        PRE_MARKET_OPEN_HOUR,
        PRE_MARKET_OPEN_MINUTE,
        tzinfo=EASTERN_TZ
    )


def get_afterhours_close_time(for_date: Optional[date] = None) -> datetime:
    """
    Get the after-hours close time (8:00 PM ET) for a given date.

    Args:
        for_date: The date to get after-hours close time for. Defaults to today.

    Returns:
        datetime: After-hours close time in Eastern Time with timezone info
    """
    if for_date is None:
        for_date = get_eastern_time().date()

    return datetime(
        for_date.year,
        for_date.month,
        for_date.day,
        AFTER_HOURS_CLOSE_HOUR,
        AFTER_HOURS_CLOSE_MINUTE,
        tzinfo=EASTERN_TZ
    )


def format_timestamp(dt: Optional[datetime] = None, include_tz: bool = True) -> str:
    """
    Format a datetime as an ISO 8601 string with timezone info.

    Args:
        dt: The datetime to format. Defaults to current Eastern Time.
        include_tz: Whether to include timezone info in the output.

    Returns:
        str: ISO 8601 formatted datetime string

    Examples:
        >>> format_timestamp()
        '2024-01-15T10:30:45-05:00'
    """
    if dt is None:
        dt = get_eastern_time()
    else:
        dt = to_eastern(dt)

    if include_tz:
        return dt.isoformat()
    else:
        return dt.strftime("%Y-%m-%dT%H:%M:%S")


def get_next_trading_day(from_date: Optional[date] = None) -> date:
    """
    Get the next trading day after a given date.

    Args:
        from_date: The starting date. Defaults to today in Eastern Time.

    Returns:
        date: The next trading day
    """
    if from_date is None:
        from_date = get_eastern_time().date()

    next_day = from_date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)

    return next_day


def get_previous_trading_day(from_date: Optional[date] = None) -> date:
    """
    Get the previous trading day before a given date.

    Args:
        from_date: The starting date. Defaults to today in Eastern Time.

    Returns:
        date: The previous trading day
    """
    if from_date is None:
        from_date = get_eastern_time().date()

    prev_day = from_date - timedelta(days=1)
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)

    return prev_day


def get_market_status() -> str:
    """
    Get a human-readable market status string.

    Returns:
        str: One of 'open', 'closed', 'pre-market', 'after-hours', 'holiday', 'weekend'
    """
    now = get_eastern_time()
    today = now.date()

    # Check for weekend
    if today.weekday() >= 5:
        return 'weekend'

    # Check for holiday
    if today in US_MARKET_HOLIDAYS:
        return 'holiday'

    # Check market hours
    if is_market_open(now):
        return 'open'
    elif is_pre_market(now):
        return 'pre-market'
    elif is_after_hours(now):
        return 'after-hours'
    else:
        return 'closed'


def time_until_market_open() -> Optional[timedelta]:
    """
    Get the time until market opens.

    Returns:
        timedelta: Time until market opens, or None if market is already open

    Notes:
        If the market is closed for today, returns time until next trading day open.
    """
    now = get_eastern_time()

    if is_market_open(now):
        return None

    # Find the next trading day's market open
    if is_trading_day(now.date()) and now < get_market_open_time(now.date()):
        # Market opens later today
        next_open = get_market_open_time(now.date())
    else:
        # Market opens on the next trading day
        next_trading_day = get_next_trading_day(now.date())
        next_open = get_market_open_time(next_trading_day)

    return next_open - now


def time_until_market_close() -> Optional[timedelta]:
    """
    Get the time until market closes.

    Returns:
        timedelta: Time until market closes, or None if market is already closed
    """
    now = get_eastern_time()

    if not is_market_open(now):
        return None

    market_close = get_market_close_time(now.date())
    return market_close - now
