import datetime as dt
import logging

from dateutil.relativedelta import relativedelta
from pandas.core.indexes.datetimes import DatetimeIndex

logger = logging.getLogger(__name__)


def get_start_end_dates(period: str, end_offset: int = 1) -> tuple[str, str]:
    """
    Calculates the start and end dates for a given period ending at a specified offset from today.

    Args:
        period (str): The period string, such as '1d' (1 day), '1mo' (1 month), '1y' (1 year), 'ytd' (year-to-date), or 'max' (from 1900-01-01).
        end_offset (int, optional): The number of days before today to use as the end date. Defaults to 1 (yesterday).

    Returns:
        tuple[str, str]: A tuple containing the start date and end date as ISO format strings (YYYY-MM-DD).

    Raises:
        ValueError: If the period string is not supported.

    Examples:
        >>> get_start_end_dates('1mo')
        ('2024-05-07', '2024-06-06')
        >>> get_start_end_dates('ytd')
        ('2024-01-01', '2024-06-06')
    """

    today = dt.date.today()
    end_date = today - dt.timedelta(days=end_offset)
    if period.endswith("d"):
        days = int(period[:-1])
        start_date = end_date - dt.timedelta(days=days)
    elif period.endswith("mo"):
        months = int(period[:-2])
        start_date = end_date - relativedelta(months=months)
    elif period.endswith("y"):
        years = int(period[:-1])
        start_date = end_date - relativedelta(years=years)
    elif period == "ytd":
        start_date = dt.date(end_date.year, 1, 1)
    elif period == "max":
        # Arbitrary early date
        start_date = dt.date(1900, 1, 1)
    else:
        raise ValueError(f"Unsupported period: {period}")
    logger.info(f"IN: {period=}, {end_offset=}, {today=}. OUT: {start_date=}, {end_date=}")
    return start_date.isoformat(), end_date.isoformat()


def pd_to_eastern(col: DatetimeIndex):
    return col.tz_localize("US/Eastern") if col.tz is None else col.tz_convert("US/Eastern")
