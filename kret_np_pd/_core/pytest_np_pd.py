import numpy as np
import pandas as pd
import pytest
import torch

from ..UTILS_np_pd import NP_PD_Utils


class TestMaskAnd:
    """Test the mask_and method which performs logical AND across multiple boolean masks."""

    def test_mask_and_two_numpy_arrays(self):
        """Test AND operation with two numpy arrays."""
        mask1 = np.array([True, False, True, False])
        mask2 = np.array([True, True, False, False])

        result = NP_PD_Utils.mask_and(mask1, mask2)
        expected = np.array([True, False, False, False])

        np.testing.assert_array_equal(result, expected)

    def test_mask_and_three_numpy_arrays(self):
        """Test AND operation with three numpy arrays."""
        mask1 = np.array([True, True, True, False])
        mask2 = np.array([True, True, False, False])
        mask3 = np.array([True, False, True, False])

        result = NP_PD_Utils.mask_and(mask1, mask2, mask3)
        expected = np.array([True, False, False, False])

        np.testing.assert_array_equal(result, expected)

    def test_mask_and_two_dataframes(self):
        """Test AND operation with two pandas DataFrames."""
        mask1 = pd.DataFrame({"a": [True, False, True], "b": [True, True, False]})
        mask2 = pd.DataFrame({"a": [True, True, False], "b": [True, False, True]})

        result = NP_PD_Utils.mask_and(mask1, mask2)
        # Each row must be all True in both DataFrames
        expected = np.array([True, False, False])

        np.testing.assert_array_equal(result, expected)

    def test_mask_and_mixed_numpy_and_dataframe(self):
        """Test AND operation with mixed numpy array and DataFrame."""
        mask1 = np.array([True, False, True, True])
        mask2 = pd.DataFrame({"a": [True, True, False, True], "b": [True, False, True, True]})

        result = NP_PD_Utils.mask_and(mask1, mask2)
        expected = np.array([True, False, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_mask_and_single_mask(self):
        """Test AND operation with a single mask."""
        mask = np.array([True, False, True])

        result = NP_PD_Utils.mask_and(mask)
        expected = np.array([True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_mask_and_all_true(self):
        """Test AND operation where all masks are True."""
        mask1 = np.array([True, True, True])
        mask2 = np.array([True, True, True])

        result = NP_PD_Utils.mask_and(mask1, mask2)
        expected = np.array([True, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_mask_and_all_false(self):
        """Test AND operation where all masks are False."""
        mask1 = np.array([False, False, False])
        mask2 = np.array([False, False, False])

        result = NP_PD_Utils.mask_and(mask1, mask2)
        expected = np.array([False, False, False])

        np.testing.assert_array_equal(result, expected)

    def test_mask_and_integer_conversion(self):
        """Test that integer arrays are converted to boolean."""
        mask1 = np.array([1, 0, 1])
        mask2 = np.array([1, 1, 0])

        result = NP_PD_Utils.mask_and(mask1, mask2)
        expected = np.array([True, False, False])

        np.testing.assert_array_equal(result, expected)


class TestMaskOr:
    """Test the mask_or method which performs logical OR across multiple boolean masks."""

    def test_mask_or_two_numpy_arrays(self):
        """Test OR operation with two numpy arrays."""
        mask1 = np.array([True, False, True, False])
        mask2 = np.array([False, False, False, False])

        result = NP_PD_Utils.mask_or(mask1, mask2)
        expected = np.array([True, False, True, False])

        np.testing.assert_array_equal(result, expected)

    def test_mask_or_three_numpy_arrays(self):
        """Test OR operation with three numpy arrays."""
        mask1 = np.array([False, False, False, False])
        mask2 = np.array([True, False, False, False])
        mask3 = np.array([False, False, True, False])

        result = NP_PD_Utils.mask_or(mask1, mask2, mask3)
        expected = np.array([True, False, True, False])

        np.testing.assert_array_equal(result, expected)

    def test_mask_or_two_dataframes(self):
        """Test OR operation with two pandas DataFrames."""
        mask1 = pd.DataFrame({"a": [False, False, False], "b": [False, True, False]})
        mask2 = pd.DataFrame({"a": [False, False, True], "b": [False, False, True]})

        result = NP_PD_Utils.mask_or(mask1, mask2)
        # Row is True if ANY value in that row is True across all DataFrames
        expected = np.array([False, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_mask_or_mixed_numpy_and_dataframe(self):
        """Test OR operation with mixed numpy array and DataFrame."""
        mask1 = np.array([False, False, False, True])
        mask2 = pd.DataFrame({"a": [True, False, False, False], "b": [False, False, True, False]})

        result = NP_PD_Utils.mask_or(mask1, mask2)
        expected = np.array([True, False, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_mask_or_single_mask(self):
        """Test OR operation with a single mask."""
        mask = np.array([True, False, True])

        result = NP_PD_Utils.mask_or(mask)
        expected = np.array([True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_mask_or_all_true(self):
        """Test OR operation where all masks are True."""
        mask1 = np.array([True, True, True])
        mask2 = np.array([True, True, True])

        result = NP_PD_Utils.mask_or(mask1, mask2)
        expected = np.array([True, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_mask_or_all_false(self):
        """Test OR operation where all masks are False."""
        mask1 = np.array([False, False, False])
        mask2 = np.array([False, False, False])

        result = NP_PD_Utils.mask_or(mask1, mask2)
        expected = np.array([False, False, False])

        np.testing.assert_array_equal(result, expected)

    def test_mask_or_integer_conversion(self):
        """Test that integer arrays are converted to boolean."""
        mask1 = np.array([0, 0, 0])
        mask2 = np.array([0, 1, 0])

        result = NP_PD_Utils.mask_or(mask1, mask2)
        expected = np.array([False, True, False])

        np.testing.assert_array_equal(result, expected)


class TestNanFilter:
    """Test the nan_filter method which creates masks for NaN values."""

    def test_nan_filter_single_numpy_array_any(self):
        """Test nan_filter with a single numpy array and 'any' mode."""
        arr = np.array([1.0, np.nan, 3.0, np.nan])

        result = NP_PD_Utils.nan_filter(arr, how="any")
        expected = np.array([False, True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_single_numpy_array_all(self):
        """Test nan_filter with a single numpy array and 'all' mode."""
        arr = np.array([1.0, np.nan, 3.0])

        result = NP_PD_Utils.nan_filter(arr, how="all")
        expected = np.array([False, True, False])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_dataframe_any(self):
        """Test nan_filter with a DataFrame and 'any' mode."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})

        result = NP_PD_Utils.nan_filter(df, how="any")
        # Row has NaN in any column
        expected = np.array([False, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_dataframe_all(self):
        """Test nan_filter with a DataFrame and 'all' mode."""
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [4.0, np.nan, np.nan]})

        result = NP_PD_Utils.nan_filter(df, how="all")
        # Row has NaN in all columns
        expected = np.array([False, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_series_any(self):
        """Test nan_filter with a pandas Series and 'any' mode."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan])

        result = NP_PD_Utils.nan_filter(series, how="any")
        expected = np.array([False, True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_series_all(self):
        """Test nan_filter with a pandas Series and 'all' mode."""
        series = pd.Series([1.0, np.nan, 3.0])

        result = NP_PD_Utils.nan_filter(series, how="all")
        expected = np.array([False, True, False])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_torch_tensor_conversion(self):
        """Test that torch tensors are converted to numpy arrays."""
        tensor = torch.tensor([1.0, float("nan"), 3.0])

        result = NP_PD_Utils.nan_filter(tensor, how="any")
        expected = np.array([False, True, False])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_multiple_inputs_any(self):
        """Test nan_filter with multiple inputs and 'any' mode."""
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([4.0, 5.0, np.nan])

        result = NP_PD_Utils.nan_filter(arr1, arr2, how="any")
        # Row is True if any input has NaN
        expected = np.array([False, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_multiple_inputs_all(self):
        """Test nan_filter with multiple inputs and 'all' mode."""
        arr1 = np.array([1.0, np.nan, np.nan])
        arr2 = np.array([4.0, np.nan, np.nan])

        result = NP_PD_Utils.nan_filter(arr1, arr2, how="all")
        # Row is True if all inputs have NaN
        expected = np.array([False, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_no_nan_values(self):
        """Test nan_filter with data containing no NaN values."""
        arr = np.array([1.0, 2.0, 3.0])

        result = NP_PD_Utils.nan_filter(arr, how="any")
        expected = np.array([False, False, False])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_all_nan_values(self):
        """Test nan_filter with data containing all NaN values."""
        arr = np.array([np.nan, np.nan, np.nan])

        result = NP_PD_Utils.nan_filter(arr, how="any")
        expected = np.array([True, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_dataframe_and_series_any(self):
        """Test nan_filter with mixed DataFrame and Series and 'any' mode."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        series = pd.Series([7.0, 8.0, np.nan])

        result = NP_PD_Utils.nan_filter(df, series, how="any")
        expected = np.array([False, True, True])

        np.testing.assert_array_equal(result, expected)

    def test_nan_filter_default_how_is_any(self):
        """Test that default 'how' parameter is 'any'."""
        arr = np.array([1.0, np.nan, 3.0])

        result_explicit = NP_PD_Utils.nan_filter(arr, how="any")
        result_default = NP_PD_Utils.nan_filter(arr)

        np.testing.assert_array_equal(result_explicit, result_default)

    def test_nan_filter_with_integers(self):
        """Test nan_filter with integer arrays (should not have NaN values)."""
        arr = np.array([1, 2, 3], dtype=int)

        result = NP_PD_Utils.nan_filter(arr, how="any")
        expected = np.array([False, False, False])

        np.testing.assert_array_equal(result, expected)


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_mask_operations_with_nan_filter(self):
        """Test combining nan_filter with mask_and."""
        arr1 = np.array([1.0, np.nan, 3.0, 4.0])
        arr2 = np.array([5.0, 6.0, np.nan, 8.0])

        nan_mask = NP_PD_Utils.nan_filter(arr1, arr2, how="any")
        # Create additional condition mask
        condition_mask = np.array([True, True, True, False])

        # Combine masks
        combined = NP_PD_Utils.mask_and(nan_mask, condition_mask)
        expected = np.array([False, True, True, False])

        np.testing.assert_array_equal(combined, expected)

    def test_complex_dataframe_operation(self):
        """Test complex operation on a real DataFrame."""
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [5.0, np.nan, 7.0, 8.0], "z": [9.0, 10.0, 11.0, np.nan]})

        # Filter rows with any NaN
        nan_mask = NP_PD_Utils.nan_filter(df, how="any")

        # Get rows without NaN
        clean_df = df[~nan_mask]

        assert clean_df.isna().sum().sum() == 0, "DataFrame should have no NaN values after filtering"
        assert len(clean_df) == 1, f"Expected 1 row after filtering, got {len(clean_df)}"
        assert clean_df.index.tolist() == [0], f"Expected indices [0], got {clean_df.index.tolist()}"

    def test_torch_tensor_with_mask_operations(self):
        """Test torch tensor conversion combined with mask operations."""
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        mask_array = np.array([True, False, True])

        nan_mask = NP_PD_Utils.nan_filter(tensor, how="any")
        combined = NP_PD_Utils.mask_or(nan_mask, mask_array)
        expected = np.array([True, True, True])

        np.testing.assert_array_equal(combined, expected)


class TestIsClose:
    """Test the is_close method for comparing arrays and DataFrames."""

    def test_is_close_numpy_arrays(self):
        """Test is_close with two numpy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0, np.nan])
        arr2 = np.array([1.0, 2.00001, 3.1, np.nan])

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-04)
        expected = np.array([True, True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_is_close_dataframes(self):
        """Test is_close with two pandas DataFrames."""
        df1 = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0001, np.nan], "b": [4.0, 5.1, 6.0]})

        result = NP_PD_Utils.is_close(df1, df2, rtol=1e-03)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"a": [True, True, True], "b": [True, False, True]})

        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
