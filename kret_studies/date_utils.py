import datetime as dt
from dateutil.relativedelta import relativedelta
import logging

logger = logging.getLogger(__name__)


def get_start_end_dates(period: str, end_offset: int = 1) -> tuple[str, str]:
    """
    Given a period string (e.g., '1d', '1mo', '1y', etc.) and a count,
    returns a tuple of (start_date, end_date) as ISO strings.
    End date defaults to yesterday.
    """
    today = dt.date.today()
    end_date = today - dt.timedelta(days=end_offset)
    if period.endswith("d"):
        days = int(period[:-1])
        start_date = end_date - dt.timedelta(days=days - 1)
    elif period.endswith("mo"):
        months = int(period[:-2])
        start_date = end_date - relativedelta(months=months - 1)
    elif period.endswith("y"):
        years = int(period[:-1])
        start_date = end_date - relativedelta(years=years - 1)
    elif period == "ytd":
        start_date = dt.date(end_date.year, 1, 1)
    elif period == "max":
        # Arbitrary early date
        start_date = dt.date(1900, 1, 1)
    else:
        raise ValueError(f"Unsupported period: {period}")
    logger.info(f"IN: {period=}, {end_offset=}, {today=}. OUT: {start_date=}, {end_date=}")
    return start_date.isoformat(), end_date.isoformat()
