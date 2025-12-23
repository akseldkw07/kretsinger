import pandas as pd


class PD_Convenience_utils:
    @classmethod
    def move_columns(cls, df: pd.DataFrame, start: list[str] | None = None, end: list[str] | None = None):
        """
        Return a DataFrame with the specified columns moved to the start and/or end.
        Args:
            df: The DataFrame.
            start: List of column names to move to the beginning.
            end: List of column names to move to the end.
        Returns:
            A new DataFrame with columns reordered.
        """
        start = [col for col in (start or []) if col in df.columns]
        end = [col for col in (end or []) if col in df.columns and col not in start]

        middle = [col for col in df.columns if col not in start and col not in end]
        new_order = start + middle + end
        return df[new_order]
