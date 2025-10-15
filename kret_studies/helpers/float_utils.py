from collections.abc import Iterable
import numpy as np
import pandas as pd


def get_precision(data_range: list[float | int] | np.ndarray):
    """
    Determines an appropriate string format for numerical data based on its range,
    using 10th and 90th percentiles to decide scaling (K, M, B) and decimal places.
    If all values are ints, returns a format with no decimal precision, just the suffix.
    """
    if not isinstance(data_range, np.ndarray):
        data_range = np.array(data_range, dtype=float)  # Ensure float for percentile calculations

    if data_range.size == 0:
        return ""
    if data_range.size < 2:  # Need at least 2 elements for percentiles to be meaningful for this heuristic
        return ".2f"  # Default for very small arrays

    # Check if all original values are ints (before conversion to float)
    all_ints = False
    try:
        # If input is ndarray, check dtype or all values
        if isinstance(data_range, np.ndarray):
            all_ints = np.issubdtype(data_range.dtype, np.integer) or np.all(np.equal(np.mod(data_range, 1), 0))
        else:
            all_ints = all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in data_range)
    except Exception:
        all_ints = False

    if all_ints:
        return ""

    p10 = np.percentile(data_range, 10)
    p90 = np.percentile(data_range, 90)
    max_abs_val_in_percentile_range = max(abs(p10), abs(p90))

    if np.isclose(max_abs_val_in_percentile_range, 0.0):
        return ".0f"

    suffix = ""
    scale_factor = 1.0
    if max_abs_val_in_percentile_range >= 1_000_000_000 - 0.5:
        suffix = "B"
        scale_factor = 1_000_000_000
    elif max_abs_val_in_percentile_range >= 1_000_000 - 0.5:
        suffix = "M"
        scale_factor = 1_000_000
    elif max_abs_val_in_percentile_range >= 1_000 - 0.5:
        suffix = "K"
        scale_factor = 1_000

    scaled_p10 = p10 / scale_factor
    scaled_p90 = p90 / scale_factor
    effective_spread = abs(scaled_p90 - scaled_p10)
    precision = 0

    higher_precision_cutoff = 5.0
    if max_abs_val_in_percentile_range <= higher_precision_cutoff:
        if effective_spread < 0.001:
            precision = 4
        elif effective_spread < 0.01:
            precision = 3
        elif effective_spread < 0.1:
            precision = 2
        else:
            precision = 2
    else:
        if effective_spread < 0.01:
            precision = 3
        elif effective_spread < 0.1:
            precision = 2
        elif effective_spread < 1.0:
            precision = 1
        else:
            precision = 0
        if (effective_spread >= 10.0 and precision == 1) or (effective_spread >= 100.0 and precision == 2):
            precision = 0

    return f".{precision}f{suffix}"


def notable_number(num: float) -> bool:
    """
    Returns True if the number is 'notable' (e.g., round, simple, or a power of ten),
    scaling up to very large numbers (up to 1e12).
    Notable numbers include:
      - Powers of ten (10, 100, 1000, ...)
      - Multiples of powers of ten (e.g., 1_000_000, 10_000_000)
      - Numbers less than 10 (simple small numbers)
    """
    if num < 10:
        return True
    if num == 0:
        return True
    # Check if num is a power of ten
    if num == 10 ** int(np.log10(num)):
        return True
    # Check if num is a multiple of a power of ten (e.g., 100_000_000, 500_000_000)
    exponent = int(np.log10(num))
    power_of_ten = 10**exponent
    if num % power_of_ten == 0:
        return True
    return False


def smart_round(values: pd.DataFrame | Iterable[float | int], max_decimals: int = 2):
    """
    Given a DataFrame or iterable of numbers & and a max_decimals value, return an int recommending
    the number of decimal places to use for rounding.
    """
    if isinstance(values, pd.DataFrame):
        values = values.select_dtypes(include=[np.number]).values.flatten()
    values = np.asarray(values)

    if values.size == 0:
        return 0

    # Compute the 10th and 90th percentiles
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))

    # Determine the effective spread
    effective_spread = abs(p90 - p10)

    # Heuristic: more precision for smaller ranges
    calc_dec: int
    if effective_spread < 0.001:
        calc_dec = 4
    elif effective_spread < 0.01:
        calc_dec = 3
    elif effective_spread < 0.1:
        calc_dec = 2
    elif effective_spread < 1.0:
        calc_dec = 1
    else:
        calc_dec = 0
    # Ensure we do not exceed the max_decimals limit
    calc_dec = min(calc_dec, max_decimals)

    return calc_dec
