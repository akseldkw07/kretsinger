import matplotlib.colors as mcolors

from .typed_cls_mpl import Background_gradient_TypedDict, Format_TypedDict, Sns_Heatmap_TypedDict


class MPLConstants:

    rwg = ["red", "white", "green"]
    wg = ["white", "green"]
    wr = ["red", "white"]

    red_green_centered = mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", rwg)
    white_green = mcolors.LinearSegmentedColormap.from_list("WhiteGreen", wg)
    white_red = mcolors.LinearSegmentedColormap.from_list("WhiteRed", wr)


class MPLDefaults:
    sns_heatmap_defaults: Sns_Heatmap_TypedDict = {
        "annot": True,
        "cmap": MPLConstants.red_green_centered,
        "linewidths": 0.1,
        "cbar": True,
    }
    background_grad_defaults: Background_gradient_TypedDict = {"cmap": MPLConstants.red_green_centered, "axis": None}
    format_defaults: Format_TypedDict = {"formatter": "{:.2f}", "decimal": ".", "thousands": "_", "na_rep": "NaN"}
