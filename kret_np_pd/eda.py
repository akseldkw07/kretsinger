"""
See kret_matplotlib/eda.py for EDA utils related to matplotlib. This file is for EDA utils related to pandas and plotly.
"""

import pandas as pd

from .dataset_to_table import PD_Display_Utils


class EDA_Utils:
    @classmethod
    def count_unique(cls, df: pd.DataFrame):
        """
        Count the number of unique values in each column of the DataFrame.
        """
        ret = df.nunique()
        ret = ret.to_frame().T

        PD_Display_Utils.dtt([ret])
