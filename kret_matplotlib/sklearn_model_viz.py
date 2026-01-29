import numpy as np
from sklearn.decomposition import PCA

from .subplot_utils import SubplotHelper


class SklearnModelVizUtils:
    @classmethod
    def plot_pca_curve(cls, model: PCA):
        cum_sum = np.cumsum(model.explained_variance_ratio_) * 100
        comp = [n for n in range(len(cum_sum))]

        fig, ax = SubplotHelper.subplots(nrows=1, ncols=1)

        ax.plot(comp, cum_sum, marker="o", linestyle="--", color="b")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance (%)")
        ax.set_title("PCA Explained Variance Curve")

        return fig
