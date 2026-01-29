import typing as t

from kret_decorators.class_property import classproperty

from .typed_cls_mpl import *

if t.TYPE_CHECKING:
    import matplotlib.colors as mcolors


class MPLConstants:

    rwg = ["red", "white", "green"]
    wg = ["white", "green"]
    wr = ["red", "white"]

    # Lazy — only imports matplotlib when first accessed
    @classproperty
    def red_green_centered(cls) -> "mcolors.LinearSegmentedColormap":
        if not hasattr(cls, "_red_green_centered"):
            import matplotlib.colors as mcolors

            cls._red_green_centered = mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", cls.rwg)
        return cls._red_green_centered

    @classproperty
    def white_green(cls) -> "mcolors.LinearSegmentedColormap":
        if not hasattr(cls, "_white_green"):
            import matplotlib.colors as mcolors

            cls._white_green = mcolors.LinearSegmentedColormap.from_list("WhiteGreen", cls.wg)
        return cls._white_green

    @classproperty
    def white_red(cls) -> "mcolors.LinearSegmentedColormap":
        if not hasattr(cls, "_white_red"):
            import matplotlib.colors as mcolors

            cls._white_red = mcolors.LinearSegmentedColormap.from_list("WhiteRed", cls.wr)
        return cls._white_red


class MPLDefaults:
    @classproperty
    def sns_heatmap_defaults(cls) -> Sns_Heatmap_TypedDict:
        return {
            "annot": True,
            "cmap": MPLConstants.red_green_centered,
            "linewidths": 0.1,
            "cbar": True,
        }

    @classproperty
    def background_grad_defaults(cls) -> Background_gradient_TypedDict:
        return {"cmap": MPLConstants.red_green_centered, "axis": None}

    PX_HIST_DEFAULTS: PxHistogram_TypedDict = {"marginal": "box", "barmode": "overlay", "barnorm": None}
    # GO_HIST_DEFAULTS: GoHistogram_TypedDict = {"marginal": "box", "barmode": "overlay", "barnorm": None}

    format_defaults: Format_TypedDict = {"formatter": "{:.2f}", "decimal": ".", "thousands": "_", "na_rep": "NaN"}
