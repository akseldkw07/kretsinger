import numpy as np


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
    max_abs_val_in_percentile_range = max(abs(p10), abs(p90))

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

    # Use max of scaled percentiles to guide decimal places, or a representative value if spread is small
    representative_scaled_val = max(abs(scaled_p10), abs(scaled_p90))

    # Consider the 'effective' spread of the numbers after scaling
    effective_spread = abs(scaled_p90 - scaled_p10)

    precision = 0  # Default to no decimal places

    # Heuristics for decimal places:
    # 1. If the numbers are very small (e.g., between -1 and 1 before scaling)
    if max_abs_val_in_percentile_range < 1.0:
        if effective_spread < 0.01:  # E.g., 0.001 to 0.005
            precision = 4
        elif effective_spread < 0.1:  # E.g., 0.01 to 0.05
            precision = 3
        else:  # E.g., -0.5 to 0.9
            precision = 2
    # 2. If the values are large enough to be scaled (i.e., K, M, B), or just large numbers (e.g., 100-999)
    else:
        # If the spread of the scaled values is very small (e.g., values are almost identical)
        if effective_spread < 0.1:  # e.g., 1.234M and 1.2345M (spread 0.0005M)
            precision = 3  # High precision for very tight ranges
        elif effective_spread < 1.0:  # e.g., 1.2M and 1.9M (spread 0.7M)
            precision = 2
        elif effective_spread < 10.0:  # e.g., 1.2M and 8.5M (spread 7.3M)
            precision = 1
        else:  # Spread is 10 or more (e.g., 10M to 90M)
            precision = 0

        # Ensure we don't show .0 for whole numbers if the spread is very large and they should be integer-like
        if effective_spread > 100 and precision == 1:
            precision = 0

    return f".{precision}f{suffix}"
