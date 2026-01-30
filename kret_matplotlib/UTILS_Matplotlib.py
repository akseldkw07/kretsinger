from .classification_viz import ClassificationVizUtils
from .eda import EDA_Utils
from .plot_heatmap import HeatmapUtils
from .sklearn_model_viz import SklearnModelVizUtils
from .subplot_utils import PlotlySubplotHelper, SubplotHelper


class UTILS_Plotting(
    HeatmapUtils,
    SubplotHelper,
    PlotlySubplotHelper,
    SklearnModelVizUtils,
    ClassificationVizUtils,
    EDA_Utils,
):
    """
    Utility class for common plotting functions using Matplotlib and Seaborn.
    """
