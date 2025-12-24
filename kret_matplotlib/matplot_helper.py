from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from kret_type_hints.typed_cls import Heatmap_Params_TD

from ..kret_utils.float_utils import FloatPrecisionUtils

rwg = ["red", "white", "green"]
wg = ["white", "green"]
wr = ["red", "white"]

red_green_centered = mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", rwg)
white_green = mcolors.LinearSegmentedColormap.from_list("WhiteGreen", wg)
white_red = mcolors.LinearSegmentedColormap.from_list("WhiteRed", wr)

try:
    plt.colormaps.register(cmap=red_green_centered, name="RedWhiteGreen")
    plt.colormaps.register(cmap=white_green, name="WhiteGreen")
    plt.colormaps.register(cmap=white_red, name="WhiteRed")
except ValueError:
    # Re-registering raises ex
    pass


class KretMatplotHelper:
    @classmethod
    def _generate_heatmap_colors(cls, df: pd.DataFrame) -> Heatmap_Params_TD:
        df_min = float(df.min(axis=None))  # type: ignore
        df_max = float(df.max(axis=None))  # type: ignore
        abs_max = max(abs(df_min), abs(df_max))

        if df_min >= 0 and df_max >= 0:
            return {"vmin": 0, "vmax": abs_max, "cmap": white_green}
        if df_min <= 0 and df_max <= 0:
            return {"vmin": -abs_max, "vmax": 0, "cmap": white_red}
        if df_min < 0 and df_max >= 0:
            return {"vmin": -abs_max, "vmax": abs_max, "cmap": red_green_centered}
        else:
            raise ValueError(f"{df_min=} {df_max=}, {abs_max=}")

    @classmethod
    def _generate_heatmap_params(cls, df: pd.DataFrame):
        colors = cls._generate_heatmap_colors(df)

        fmt = FloatPrecisionUtils.get_precision(df.values.flatten())

        ret = colors | {"fmt": fmt}
        return ret
