from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex

from kret_type_hints.typed_cls import (
    Background_gradient_TypedDict,
    Format_TypedDict,
    Heatmap_Params_TD,
    Sns_Heatmap_TypedDict,
)
from kret_utils.float_utils import FloatPrecisionUtils


class KretMatplotHelper:

    rwg = ["red", "white", "green"]
    wg = ["white", "green"]
    wr = ["red", "white"]

    red_green_centered = mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", rwg)
    white_green = mcolors.LinearSegmentedColormap.from_list("WhiteGreen", wg)
    white_red = mcolors.LinearSegmentedColormap.from_list("WhiteRed", wr)
    sns_heatmap_defaults: Sns_Heatmap_TypedDict = {
        "annot": True,
        "cmap": red_green_centered,
        "linewidths": 0.1,
        "cbar": True,
    }
    background_grad_defaults: Background_gradient_TypedDict = {"cmap": red_green_centered, "axis": None}
    format_defaults: Format_TypedDict = {"formatter": "{:.2f}", "decimal": ".", "thousands": "_", "na_rep": "NaN"}

    @classmethod
    def _generate_heatmap_colors(cls, df: pd.DataFrame) -> Heatmap_Params_TD:
        df_min = float(df.min(axis=None))  # type: ignore
        df_max = float(df.max(axis=None))  # type: ignore
        abs_max = max(abs(df_min), abs(df_max))

        if df_min >= 0 and df_max >= 0:
            return {"vmin": 0, "vmax": abs_max, "cmap": cls.white_green}
        if df_min <= 0 and df_max <= 0:
            return {"vmin": -abs_max, "vmax": 0, "cmap": cls.white_red}
        if df_min < 0 and df_max >= 0:
            return {"vmin": -abs_max, "vmax": abs_max, "cmap": cls.red_green_centered}
        else:
            raise ValueError(f"{df_min=} {df_max=}, {abs_max=}")

    @classmethod
    def _generate_heatmap_params(cls, df: pd.DataFrame):
        vminmax_colormap = cls._generate_heatmap_colors(df)

        fmt = FloatPrecisionUtils.get_precision(df.values.flatten())

        ret = vminmax_colormap | {"fmt": fmt}
        return ret

    @classmethod
    def _legend_html(
        cls, *, cmap, vmin: float, vmax: float, vcenter: float | None = None, n: int = 256, height: int = 14, **kwargs
    ) -> str:
        if vcenter is not None and vmin < vcenter < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Sample the colormap in *normalized* space so the gradient matches the mapping.
        xs = np.linspace(vmin, vmax, n)
        colors = [to_hex(cmap(norm(x))) for x in xs]
        gradient = ", ".join(colors)

        # Position the 0 marker correctly if present
        zero_marker = ""
        if vcenter is not None and vmin < vcenter < vmax:
            pct = 100.0 * (vcenter - vmin) / (vmax - vmin)
            zero_marker = f"""
            <div style="position: relative; height: 0;">
            <div style="position:absolute; left:{pct:.2f}%; transform: translateX(-50%);
                        top:-{height+6}px; font-size: 11px; color:#444;">0</div>
            <div style="position:absolute; left:{pct:.2f}%; transform: translateX(-50%);
                        top:-{height}px; width:2px; height:{height+2}px; background:#444;"></div>
            </div>
            """

        return f"""
        <div style="margin-top:6px; margin-bottom:12px;">
        <div style="height:{height}px; border:1px solid #ccc;
                    background: linear-gradient(to right, {gradient});"></div>
        {zero_marker}
        <div style="display:flex; justify-content:space-between; font-family: monospace; font-size: 14px; color: #FFFFFF;">
            <span>{vmin:.2f}</span>
            <span>{vmax:.2f}</span>
        </div>
        </div>
        """


try:
    # TODO this seems slow, maybe lazy registration?
    plt.colormaps.register(cmap=KretMatplotHelper.red_green_centered, name="RedWhiteGreen")
    plt.colormaps.register(cmap=KretMatplotHelper.white_green, name="WhiteGreen")
    plt.colormaps.register(cmap=KretMatplotHelper.white_red, name="WhiteRed")
except ValueError:
    # Re-registering raises ex
    pass
