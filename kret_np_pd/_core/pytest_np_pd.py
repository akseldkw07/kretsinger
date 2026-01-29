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

    # -------------------------------------------------------------------------
    # Basic numpy array tests
    # -------------------------------------------------------------------------

    def test_is_close_numpy_arrays(self):
        """Test is_close with two numpy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0, np.nan])
        arr2 = np.array([1.0, 2.00001, 3.1, np.nan])

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-04)
        expected = np.array([True, True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_identical(self):
        """Test is_close with identical numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = NP_PD_Utils.is_close(arr, arr)
        expected = np.array([True, True, True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_completely_different(self):
        """Test is_close with completely different values."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([100.0, 200.0, 300.0])
        result = NP_PD_Utils.is_close(arr1, arr2)
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_2d_arrays(self):
        """Test is_close with 2D numpy arrays."""
        arr1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr2 = np.array([[1.0, 2.00001], [3.1, 4.0]])
        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-04)
        expected = np.array([[True, True], [False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_zeros(self):
        """Test is_close with zero values."""
        arr1 = np.array([0.0, 0.0, 1e-10])
        arr2 = np.array([0.0, 1e-10, 0.0])
        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-05)
        # np.isclose uses both rtol and atol, default atol=1e-08
        expected = np.isclose(arr1, arr2, rtol=1e-05, equal_nan=True)
        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_negative_values(self):
        """Test is_close with negative values."""
        arr1 = np.array([-1.0, -2.0, -3.0])
        arr2 = np.array([-1.0, -2.00001, -3.1])
        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-04)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_inf_values(self):
        """Test is_close with infinity values."""
        arr1 = np.array([np.inf, -np.inf, 1.0])
        arr2 = np.array([np.inf, -np.inf, 1.0])
        result = NP_PD_Utils.is_close(arr1, arr2)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_inf_vs_finite(self):
        """Test is_close with inf compared to finite values."""
        arr1 = np.array([np.inf, 1.0])
        arr2 = np.array([1e308, 1.0])
        result = NP_PD_Utils.is_close(arr1, arr2)
        expected = np.array([False, True])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # DataFrame tests
    # -------------------------------------------------------------------------

    def test_is_close_dataframes(self):
        """Test is_close with two pandas DataFrames."""
        df1 = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0001, np.nan], "b": [4.0, 5.1, 6.0]})

        result = NP_PD_Utils.is_close(df1, df2, rtol=1e-03)
        assert isinstance(result, pd.DataFrame)
        # rtol=1e-03: 2.0 vs 2.0001 is close, 5.0 vs 5.1 (2% diff) is not close
        expected = pd.DataFrame({"a": [1, 1, 1], "b": [1, 0, 1]})

        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_dataframes_identical(self):
        """Test is_close with identical DataFrames."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = NP_PD_Utils.is_close(df, df)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]})
        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_dataframes_single_column(self):
        """Test is_close with single-column DataFrames that do NOT auto-coerce."""
        df1 = pd.DataFrame({"col": [1.0, 2.0, 3.0], "col2": [1.0, 1.0, 1.0]})
        df2 = pd.DataFrame({"col": [1.0, 2.00001, 3.1], "col2": [1.0, 1.0, 1.0]})
        result = NP_PD_Utils.is_close(df1, df2, rtol=1e-04)
        assert isinstance(result, pd.DataFrame)
        # 3.0 vs 3.1 is ~3.3% diff, not close with rtol=1e-04
        expected = pd.DataFrame({"col": [1, 1, 0], "col2": [1, 1, 1]})
        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_dataframes_many_columns(self):
        """Test is_close with DataFrames having many columns."""
        df1 = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0], "e": [5.0]})
        df2 = pd.DataFrame({"a": [1.0], "b": [2.1], "c": [3.0], "d": [4.1], "e": [5.0]})
        result = NP_PD_Utils.is_close(df1, df2, rtol=1e-03)
        assert isinstance(result, pd.DataFrame)
        # 2.0 vs 2.1 is 5% diff, 4.0 vs 4.1 is 2.5% diff - both exceed rtol=0.1%
        expected = pd.DataFrame({"a": [1], "b": [0], "c": [1], "d": [0], "e": [1]})
        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_dataframes_with_nan_in_all_columns(self):
        """Test is_close with NaN in all columns."""
        df1 = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, 1.0]})
        df2 = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, 1.0]})
        result = NP_PD_Utils.is_close(df1, df2, nan_true=True)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"a": [1, 1], "b": [1, 1]})
        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_dataframes_different_column_names(self):
        """Test is_close raises error when DataFrames have different column names."""
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0], "c": [3.0, 4.0]})  # 'c' instead of 'b'

        with pytest.raises(AssertionError, match="same columns"):
            NP_PD_Utils.is_close(df1, df2)

    # -------------------------------------------------------------------------
    # pd.Series tests
    # -------------------------------------------------------------------------

    def test_is_close_two_series(self):
        """Test is_close with two pandas Series (compares by position, not index)."""
        s1 = pd.Series([1.0, 2.0, 3.0])
        s2 = pd.Series([1.0, 2.00001, 3.1])

        result = NP_PD_Utils.is_close(s1, s2, rtol=1e-04)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_series_different_index(self):
        """Test is_close with Series having different indices (compares by position)."""
        s1 = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        s2 = pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30])

        result = NP_PD_Utils.is_close(s1, s2)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_series_and_numpy(self):
        """Test is_close with pd.Series and np.ndarray (mixed ARR_TYPE)."""
        s = pd.Series([1.0, 2.0, 3.0])
        arr = np.array([1.0, 2.0, 3.0])

        result = NP_PD_Utils.is_close(s, arr)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_numpy_and_series(self):
        """Test is_close with np.ndarray and pd.Series (order reversed)."""
        arr = np.array([1.0, 2.0, 3.0])
        s = pd.Series([1.0, 2.00001, 3.1])

        result = NP_PD_Utils.is_close(arr, s, rtol=1e-04)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # torch.Tensor tests
    # -------------------------------------------------------------------------

    def test_is_close_two_tensors_1d(self):
        """Test is_close with two 1D torch tensors (auto-coerces to ndarray)."""
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.00001, 3.1])

        result = NP_PD_Utils.is_close(t1, t2, rtol=1e-04)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_two_tensors_2d(self):
        """Test is_close with two 2D torch tensors (no auto-coerce, uses tensor branch)."""
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([[1.0, 2.00001], [3.1, 4.0]])

        result = NP_PD_Utils.is_close(t1, t2, rtol=1e-04)
        expected = np.array([[True, True], [False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_tensor_with_nan(self):
        """Test is_close with torch tensors containing NaN."""
        t1 = torch.tensor([1.0, float("nan"), 3.0])
        t2 = torch.tensor([1.0, float("nan"), 3.0])

        result = NP_PD_Utils.is_close(t1, t2, nan_true=True)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # pd.Categorical tests
    # -------------------------------------------------------------------------

    def test_is_close_two_categoricals(self):
        """Test is_close with two pd.Categorical (auto-coerces to ndarray via codes)."""
        cat1 = pd.Categorical(["a", "b", "c"])
        cat2 = pd.Categorical(["a", "b", "c"])

        result = NP_PD_Utils.is_close(cat1, cat2)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_categorical_different_values(self):
        """Test is_close with categoricals having different underlying codes."""
        cat1 = pd.Categorical(["a", "b", "c"])
        cat2 = pd.Categorical(["a", "a", "c"])

        result = NP_PD_Utils.is_close(cat1, cat2)
        # Compares codes: cat1=[0,1,2], cat2=[0,0,1] (different category order possible)
        # With same categories: a=0, b=1, c=2 for both
        # cat1 codes: [0, 1, 2], cat2 codes: [0, 0, 1]
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Filter parameter tests
    # -------------------------------------------------------------------------

    def test_is_close_with_filter_numpy(self):
        """Test is_close with filter on numpy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0, 4.0])
        arr2 = np.array([1.0, 2.0, 3.0, 4.0])
        filt = np.array([True, False, True, False])

        result = NP_PD_Utils.is_close(arr1, arr2, filt=filt)
        # Positions with filter=False return -1
        expected = np.array([True, -1, True, -1])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_with_filter_all_false(self):
        """Test is_close with all-false filter."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([100.0, 200.0, 300.0])  # Would be False without filter
        filt = np.array([False, False, False])

        result = NP_PD_Utils.is_close(arr1, arr2, filt=filt)
        expected = np.array([-1, -1, -1])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_with_filter_all_true(self):
        """Test is_close with all-true filter (same as no filter)."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0001, 3.1])
        filt = np.array([True, True, True])

        result_with_filter = NP_PD_Utils.is_close(arr1, arr2, filt=filt, rtol=1e-03)
        result_without_filter = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-03)
        np.testing.assert_array_equal(result_with_filter, result_without_filter)

    def test_is_close_with_filter_dataframe(self):
        """Test is_close with filter on DataFrames."""
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        filt = np.array([True, False, True])

        result = NP_PD_Utils.is_close(df1, df2, filt=filt)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"a": [1, -1, 1], "b": [1, -1, 1]})
        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_with_filter_mixed_results(self):
        """Test is_close with filter where filtered positions differ."""
        arr1 = np.array([1.0, 999.0, 3.0, 999.0])
        arr2 = np.array([1.0, 1.0, 3.0, 1.0])
        filt = np.array([True, False, True, False])

        result = NP_PD_Utils.is_close(arr1, arr2, filt=filt)
        # Even though positions 1 and 3 differ, they're filtered out
        expected = np.array([True, -1, True, -1])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # nan_true parameter tests
    # -------------------------------------------------------------------------

    def test_is_close_nan_true_default(self):
        """Test that nan_true=True is the default behavior."""
        arr1 = np.array([np.nan, np.nan, 1.0])
        arr2 = np.array([np.nan, 1.0, np.nan])

        result = NP_PD_Utils.is_close(arr1, arr2)
        # NaN == NaN is True, NaN vs non-NaN is False
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_nan_true_explicit(self):
        """Test is_close with nan_true=True explicitly set."""
        arr1 = np.array([np.nan, np.nan, 1.0])
        arr2 = np.array([np.nan, np.nan, 1.0])

        result = NP_PD_Utils.is_close(arr1, arr2, nan_true=True)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_nan_false(self):
        """Test is_close with nan_true=False."""
        arr1 = np.array([np.nan, np.nan, 1.0])
        arr2 = np.array([np.nan, np.nan, 1.0])

        result = NP_PD_Utils.is_close(arr1, arr2, nan_true=False)
        # NaN comparisons are False when nan_true=False
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_nan_false_dataframe(self):
        """Test is_close with nan_true=False on DataFrames."""
        df1 = pd.DataFrame({"a": [np.nan, 1.0], "b": [2.0, np.nan]})
        df2 = pd.DataFrame({"a": [np.nan, 1.0], "b": [2.0, np.nan]})

        result = NP_PD_Utils.is_close(df1, df2, nan_true=False)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_all_nan(self):
        """Test is_close when all values are NaN."""
        arr1 = np.array([np.nan, np.nan, np.nan])
        arr2 = np.array([np.nan, np.nan, np.nan])

        result_true = NP_PD_Utils.is_close(arr1, arr2, nan_true=True)
        result_false = NP_PD_Utils.is_close(arr1, arr2, nan_true=False)

        np.testing.assert_array_equal(result_true, np.array([True, True, True]))
        np.testing.assert_array_equal(result_false, np.array([False, False, False]))

    # -------------------------------------------------------------------------
    # rtol parameter tests
    # -------------------------------------------------------------------------

    def test_is_close_rtol_tight(self):
        """Test is_close with tight relative tolerance."""
        arr1 = np.array([1.0, 1.0, 1.0])
        arr2 = np.array([1.0, 1.000001, 1.00001])

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-07)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_rtol_loose(self):
        """Test is_close with loose relative tolerance."""
        arr1 = np.array([1.0, 1.0, 1.0])
        arr2 = np.array([1.0, 1.01, 1.1])

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=0.05)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_rtol_very_loose(self):
        """Test is_close with very loose relative tolerance."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.5, 2.5, 3.5])

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=0.5)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_rtol_zero(self):
        """Test is_close with zero tolerance (exact match only)."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0 + 1e-15, 3.0])

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=0.0)
        # Note: np.isclose still has default atol=1e-08, so very small diffs may still match
        expected = np.isclose(arr1, arr2, rtol=0.0, equal_nan=True)
        np.testing.assert_array_equal(result, expected)

    def test_is_close_rtol_with_large_values(self):
        """Test is_close rtol with large values."""
        arr1 = np.array([1e10, 1e10])
        arr2 = np.array([1e10, 1.00001e10])

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-06)
        expected = np.array([True, False])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # try_coerce_np parameter tests
    # -------------------------------------------------------------------------

    def test_is_close_coerce_np_single_col_dataframe_to_array(self):
        """Test is_close with try_coerce_np on single-column DataFrame to 1D array."""
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        arr = np.array([1.0, 2.0, 3.0])

        result = NP_PD_Utils.is_close(df, arr, try_coerce_np=True)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_coerce_np_both_single_col_dataframes(self):
        """Test is_close with try_coerce_np on two single-column DataFrames with different column names."""
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"b": [1.0, 2.00001, 3.1]})

        result = NP_PD_Utils.is_close(df1, df2, try_coerce_np=True, rtol=1e-04)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_coerce_np_array_to_dataframe(self):
        """Test is_close with try_coerce_np swapping order (array first, DataFrame second)."""
        arr = np.array([1.0, 2.0, 3.0])
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})

        result = NP_PD_Utils.is_close(arr, df, try_coerce_np=True)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_coerce_np_with_filter(self):
        """Test is_close with try_coerce_np and filter combined."""
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        arr = np.array([1.0, 999.0, 3.0])
        filt = np.array([True, False, True])

        result = NP_PD_Utils.is_close(df, arr, filt=filt, try_coerce_np=True)
        expected = np.array([True, -1, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_coerce_np_with_nan(self):
        """Test is_close with try_coerce_np and NaN values."""
        df = pd.DataFrame({"col": [1.0, np.nan, 3.0]})
        arr = np.array([1.0, np.nan, 3.0])

        result = NP_PD_Utils.is_close(df, arr, try_coerce_np=True, nan_true=True)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_coerce_np_false_mixed_types(self):
        """Test that try_coerce_np=False raises error on mixed types.

        Note: Shape check happens before type check, so we get AssertionError for shape mismatch
        since single-col DataFrame has shape (3,1) vs array shape (3,).
        """
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        arr = np.array([1.0, 2.0, 3.0])

        with pytest.raises(AssertionError, match="Shapes must match"):
            NP_PD_Utils.is_close(df, arr, try_coerce_np=False)

    def test_is_close_coerce_np_1d_tensor_to_array(self):
        """Test is_close with try_coerce_np on 1D tensor and array."""
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = np.array([1.0, 2.0, 3.0])

        result = NP_PD_Utils.is_close(t, arr, try_coerce_np=True)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Auto-coercion tests (_should_coerce_np logic)
    # -------------------------------------------------------------------------

    def test_is_close_auto_coerce_1d_dataframe(self):
        """Test that 1D (single-column) DataFrame auto-coerces when try_coerce_np=None."""
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"b": [1.0, 2.0, 3.0]})

        # With try_coerce_np=None, single-col DataFrames should auto-coerce
        result = NP_PD_Utils.is_close(df1, df2)  # try_coerce_np defaults to None
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_auto_coerce_1d_tensor(self):
        """Test that 1D tensor auto-coerces when try_coerce_np=None."""
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = np.array([1.0, 2.0, 3.0])

        result = NP_PD_Utils.is_close(t, arr)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_auto_coerce_categorical(self):
        """Test that pd.Categorical always auto-coerces when try_coerce_np=None."""
        cat = pd.Categorical(["a", "b", "c"])
        arr = np.array([0, 1, 2])  # codes

        result = NP_PD_Utils.is_close(cat, arr)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_no_auto_coerce_2d_dataframe(self):
        """Test that 2D (multi-column) DataFrame does NOT auto-coerce."""
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        result = NP_PD_Utils.is_close(df1, df2)
        assert isinstance(result, pd.DataFrame)
        expected = pd.DataFrame({"a": [1, 1], "b": [1, 1]})
        pd.testing.assert_frame_equal(result, expected)

    def test_is_close_no_auto_coerce_2d_tensor(self):
        """Test that 2D tensor does NOT auto-coerce (uses tensor branch)."""
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = NP_PD_Utils.is_close(t1, t2)
        expected = np.array([[True, True], [True, True]])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Error case tests
    # -------------------------------------------------------------------------

    def test_is_close_shape_mismatch_numpy(self):
        """Test is_close raises error on shape mismatch for numpy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0])

        with pytest.raises(AssertionError, match="Shapes must match"):
            NP_PD_Utils.is_close(arr1, arr2)

    def test_is_close_shape_mismatch_dataframe_rows(self):
        """Test is_close raises error on row count mismatch for DataFrames."""
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0]})

        with pytest.raises(AssertionError, match="Shapes must match"):
            NP_PD_Utils.is_close(df1, df2)

    def test_is_close_shape_mismatch_dataframe_cols(self):
        """Test is_close raises error on column count mismatch for DataFrames."""
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0]})

        with pytest.raises(AssertionError, match="Shapes must match"):
            NP_PD_Utils.is_close(df1, df2)

    def test_is_close_type_mismatch_without_coerce(self):
        """Test is_close raises error on type mismatch without coercion.

        Note: Shape check happens before type check, so we get AssertionError for shape mismatch.
        """
        arr = np.array([1.0, 2.0, 3.0])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})  # 2D so no auto-coerce

        with pytest.raises(AssertionError, match="Shapes must match"):
            NP_PD_Utils.is_close(arr, df, try_coerce_np=False)

    def test_is_close_2d_array_mismatch(self):
        """Test is_close raises error on 2D array shape mismatch."""
        arr1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr2 = np.array([[1.0], [2.0]])

        with pytest.raises(AssertionError, match="Shapes must match"):
            NP_PD_Utils.is_close(arr1, arr2)

    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------

    def test_is_close_single_element(self):
        """Test is_close with single-element arrays."""
        arr1 = np.array([1.0])
        arr2 = np.array([1.0])

        result = NP_PD_Utils.is_close(arr1, arr2)
        expected = np.array([True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_empty_arrays(self):
        """Test is_close with empty arrays."""
        arr1 = np.array([])
        arr2 = np.array([])

        result = NP_PD_Utils.is_close(arr1, arr2)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_large_arrays(self):
        """Test is_close with large arrays."""
        np.random.seed(42)
        arr1 = np.random.randn(10000)
        arr2 = arr1 + np.random.randn(10000) * 1e-10

        result = NP_PD_Utils.is_close(arr1, arr2, rtol=1e-05)
        assert isinstance(result, np.ndarray)
        assert result.all(), "All values should be close with small noise"

    def test_is_close_integer_input_numpy(self):
        """Test is_close with integer numpy arrays (auto-converted to float comparison)."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])

        result = NP_PD_Utils.is_close(arr1, arr2)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_mixed_int_float(self):
        """Test is_close with mixed integer and float arrays."""
        arr1 = np.array([1, 2, 3], dtype=int)
        arr2 = np.array([1.0, 2.0, 3.0], dtype=float)

        result = NP_PD_Utils.is_close(arr1, arr2)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Combined parameter tests
    # -------------------------------------------------------------------------

    def test_is_close_all_params_numpy(self):
        """Test is_close with all parameters on numpy arrays."""
        arr1 = np.array([1.0, np.nan, 3.0, 4.0])
        arr2 = np.array([1.01, np.nan, 3.0, 4.0])
        filt = np.array([True, True, False, True])

        result = NP_PD_Utils.is_close(arr1, arr2, filt=filt, nan_true=True, rtol=0.05)
        # pos 0: 1.0 vs 1.01 with rtol=0.05 -> True
        # pos 1: nan vs nan with nan_true=True -> True
        # pos 2: filtered out -> -1
        # pos 3: 4.0 vs 4.0 -> True
        expected = np.array([True, True, -1, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_close_all_params_dataframe(self):
        """Test is_close with all parameters on DataFrames."""
        df1 = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.01, np.nan, 3.0], "b": [4.5, 5.0, 6.0]})  # 4.5 is >10% diff from 4.0
        filt = np.array([True, True, False])

        result = NP_PD_Utils.is_close(df1, df2, filt=filt, nan_true=True, rtol=0.05)
        # pos 0 col a: 1.0 vs 1.01 with rtol=0.05 -> True (1%)
        # pos 0 col b: 4.0 vs 4.5 with rtol=0.05 -> False (12.5% > 5%)
        # pos 1: nan vs nan with nan_true=True -> True
        # pos 2: filtered out -> -1
        expected = pd.DataFrame({"a": [1, 1, -1], "b": [0, 1, -1]})
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
