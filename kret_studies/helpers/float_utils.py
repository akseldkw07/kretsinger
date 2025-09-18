from collections.abc import Iterable
import numpy as np
import pandas as pd


def get_precision(data_range: list[float] | np.ndarray):
    """
    Determines an appropriate string format for numerical data based on its range,
    using 10th and 90th percentiles to decide scaling (K, M, B) and decimal places.
    This function helps in displaying numbers with appropriate granularity and units
    (thousands, millions, billions).

    Args:
        data_range (list or np.ndarray): A 1D array or list of numbers.

    Returns:
        str: A format string (e.g., ".2f", ".1fM", ".0fK").
             Returns "" for empty data or if percentiles cannot be computed meaningfully.
    """
    if not isinstance(data_range, np.ndarray):
        data_range = np.array(data_range, dtype=float)  # Ensure float for percentile calculations

    if data_range.size == 0:
        return ""
    if data_range.size < 2:  # Need at least 2 elements for percentiles to be meaningful for this heuristic
        return ".2f"  # Default for very small arrays

    p10 = np.percentile(data_range, 10)
    p90 = np.percentile(data_range, 90)

    # Use max_abs_val of the percentile range to determine the overall scale
    max_abs_val_in_percentile_range = max(abs(p10), abs(p90))  # type: ignore

    # Handle the trivial case where the range is essentially zero
    if np.isclose(max_abs_val_in_percentile_range, 0.0):
        return ".0f"  # If all relevant values are zero or very close to zero

    suffix = ""
    scale_factor = 1.0

    # Determine the scaling factor (K, M, B) based on the absolute magnitude
    # We use a slightly smaller threshold (e.g., 999.5 for K) to ensure values like 1000.0 are formatted as 1.0K
    if max_abs_val_in_percentile_range >= 1_000_000_000 - 0.5:
        suffix = "B"
        scale_factor = 1_000_000_000
    elif max_abs_val_in_percentile_range >= 1_000_000 - 0.5:
        suffix = "M"
        scale_factor = 1_000_000
    elif max_abs_val_in_percentile_range >= 1_000 - 0.5:
        suffix = "K"
        scale_factor = 1_000

    # Determine decimal places based on the *scaled* spread and magnitude
    scaled_p10 = p10 / scale_factor
    scaled_p90 = p90 / scale_factor

    # Consider the 'effective' spread of the numbers after scaling
    effective_spread = abs(scaled_p90 - scaled_p10)

    precision = 0  # Default to no decimal places

    # --- REFINED HEURISTICS FOR DECIMAL PLACES ---
    # This section is updated to favor more precision for small ranges.

    higher_precision_cutoff = 5.0
    if max_abs_val_in_percentile_range <= higher_precision_cutoff:
        # For numbers between -1 and 1 (like correlations)
        if effective_spread < 0.001:  # E.g., 0.0001 to 0.0005
            precision = 4
        elif effective_spread < 0.01:  # E.g., 0.001 to 0.009
            precision = 3
        elif effective_spread < 0.1:  # E.g., 0.01 to 0.09
            precision = 2
        else:  # E.g., 0.1 to 0.9 or -0.5 to 0.9
            precision = 2
    else:
        # If the values are large enough to be scaled (i.e., K, M, B), or just large numbers (e.g., 100-999)
        if effective_spread < 0.01:  # e.g., 1.23456M and 1.23457M
            precision = 3  # High precision for very tight ranges
        elif effective_spread < 0.1:  # e.g., 1.23M and 1.29M
            precision = 2
        elif effective_spread < 1.0:  # e.g., 1.2M and 1.9M
            precision = 1
        else:  # Spread is 1.0 or more (e.g., 1.2M to 8.5M, or 10M to 90M)
            precision = 0

        # Add a check to avoid showing .0 for very large whole numbers if not explicitly needed
        # and precision ended up as 1. For example, 12.0M, when 12M would be fine.
        # This is a bit subjective but can improve readability for some datasets.
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
