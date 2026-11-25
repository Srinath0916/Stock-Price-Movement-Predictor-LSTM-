"""
US Market Trading Calendar and Session Utility.
Accurately accounts for NYSE / NASDAQ market holidays and weekend closures.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Union
from zoneinfo import ZoneInfo
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    USMartinLutherKingJr,
    USPresidentsDay,
    GoodFriday,
    USMemorialDay,
    USLaborDay,
    USThanksgivingDay,
)


class USTradingCalendar(AbstractHolidayCalendar):
    """
    Official US Equity Market (NYSE/NASDAQ) Holiday Calendar:
    1. New Year's Day
    2. Martin Luther King, Jr. Day (3rd Monday of January)
    3. Washington's Birthday / Presidents' Day (3rd Monday of February)
    4. Good Friday (Friday before Easter)
    5. Memorial Day (Last Monday of May)
    6. Juneteenth National Independence Day (June 19, observed since 2021)
    7. Independence Day (July 4, observed on nearest weekday)
    8. Labor Day (1st Monday of September)
    9. Thanksgiving Day (4th Thursday of November)
    10. Christmas Day (December 25, observed on nearest weekday)
    """
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2021-06-19"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_calendar = USTradingCalendar()
# Cache holidays for a broad 40-year period (1995 to 2035)
_holidays_index = _calendar.holidays(start="1995-01-01", end="2035-12-31")
_holidays_set = set(pd.to_datetime(_holidays_index).normalize())


def is_trading_day(dt: Union[str, datetime, pd.Timestamp]) -> bool:
    """Returns True if the given date is a weekday and not an official US market holiday."""
    ts = pd.to_datetime(dt).normalize()
    # 0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday
    if ts.weekday() >= 5:
        return False
    return ts not in _holidays_set


def get_next_trading_session(dt: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Calculates the exact next trading session following `dt`, skipping
    weekends and all official US equity market holidays.
    """
    current = pd.to_datetime(dt).normalize()
    candidate = current + pd.Timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += pd.Timedelta(days=1)
    return candidate


def get_market_status(dt: Optional[datetime] = None) -> Dict[str, Union[str, bool]]:
    """
    Evaluates current US equity market status according to NYSE / NASDAQ calendar:
    - Checks weekends
    - Checks official NYSE holidays
    - Checks trading hours (Pre-market 04:00-09:30 ET, Regular 09:30-16:00 ET, After-hours 16:00-20:00 ET, Closed overnight)
    """
    ny_tz = ZoneInfo("America/New_York")
    now_ny = dt.astimezone(ny_tz) if dt and dt.tzinfo else (dt.replace(tzinfo=ny_tz) if dt else datetime.now(ny_tz))

    # Normalize date
    date_normalized = pd.to_datetime(now_ny.date())
    is_weekend = now_ny.weekday() >= 5
    is_holiday = date_normalized in _holidays_set

    # Current minute of day
    minute_of_day = now_ny.hour * 60 + now_ny.minute

    if is_weekend:
        status_code = "CLOSED_WEEKEND"
        status_label = "Market Closed (Weekend)"
        is_open = False
    elif is_holiday:
        status_code = "CLOSED_HOLIDAY"
        status_label = "Market Closed (US Holiday)"
        is_open = False
    elif 570 <= minute_of_day < 960:
        status_code = "OPEN_REGULAR"
        status_label = "Market Open (Delayed 15m)"
        is_open = True
    elif 240 <= minute_of_day < 570:
        status_code = "PRE_MARKET"
        status_label = "Pre-Market (Regular opens 9:30 AM ET)"
        is_open = False
    elif 960 <= minute_of_day < 1200:
        status_code = "AFTER_HOURS"
        status_label = "After-Hours Session"
        is_open = False
    else:
        status_code = "CLOSED_OVERNIGHT"
        status_label = "Market Closed (Overnight)"
        is_open = False

    next_session = get_next_trading_session(date_normalized) if not is_open else date_normalized

    return {
        "statusCode": status_code,
        "statusLabel": status_label,
        "isOpen": is_open,
        "currentTimeET": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "currentDate": now_ny.strftime("%Y-%m-%d"),
        "nextTradingSession": next_session.strftime("%Y-%m-%d"),
    }
