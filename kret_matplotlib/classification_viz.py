import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .subplot_utils import SubplotHelper


class ClassificationVizUtils:
    """Plotting utilities for binary classification evaluation."""

    # region Confusion Matrix

    @classmethod
    def plot_confusion_matrix(
        cls,
        cm_df: pd.DataFrame,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        fmt: str = "d",
        annot_fontsize: int = 14,
    ) -> Figure:
        """
        Plot a confusion matrix heatmap from a labeled DataFrame.

        Args:
            cm_df: DataFrame from ClassificationEvalUtils.confusion_matrix_df()
                   (rows=Actual, columns=Predicted)
            title: Plot title
            cmap: Colormap name
            fmt: Annotation format ('d' for integers, '.2%' for normalized)
            annot_fontsize: Font size for cell annotations
        """
        fig, ax = SubplotHelper.subplots(nrows=1, ncols=1, width_per=7, height_per=6)

        sns.heatmap(
            cm_df,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            linewidths=0.5,
            cbar=True,
            ax=ax,
            annot_kws={"fontsize": annot_fontsize},
        )
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(cm_df.index.name or "Actual", fontsize=14)
        ax.set_xlabel(cm_df.columns.name or "Predicted", fontsize=14)

        fig.tight_layout()
        return fig

    # endregion

    # region ROC Curve

    @classmethod
    def plot_roc_curve(
        cls,
        roc_df: pd.DataFrame,
        auc_score: float | None = None,
        title: str = "ROC Curve",
    ) -> Figure:
        """
        Plot ROC curve from a DataFrame with columns: fpr, tpr, threshold.

        Args:
            roc_df: DataFrame from ClassificationEvalUtils.roc_curve_df()
            auc_score: AUC score to display in legend (optional)
            title: Plot title
        """
        fig, ax = SubplotHelper.subplots(nrows=1, ncols=1, width_per=7, height_per=6)

        label = f"ROC (AUC = {auc_score:.4f})" if auc_score is not None else "ROC"
        ax.plot(roc_df["fpr"], roc_df["tpr"], color="blue", lw=2, label=label)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

        SubplotHelper.set_title_label(ax, title=title, xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.legend(loc="lower right", fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        fig.tight_layout()
        return fig

    # endregion

    # region PR Curve

    @classmethod
    def plot_pr_curve(
        cls,
        pr_df: pd.DataFrame,
        title: str = "Precision-Recall Curve",
    ) -> Figure:
        """
        Plot Precision-Recall curve from a DataFrame with columns: precision, recall, threshold.

        Args:
            pr_df: DataFrame from ClassificationEvalUtils.pr_curve_df()
            title: Plot title
        """
        fig, ax = SubplotHelper.subplots(nrows=1, ncols=1, width_per=7, height_per=6)

        ax.plot(pr_df["recall"], pr_df["precision"], color="blue", lw=2, label="PR Curve")

        SubplotHelper.set_title_label(ax, title=title, xlabel="Recall", ylabel="Precision")
        ax.legend(loc="upper right", fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        fig.tight_layout()
        return fig

    # endregion

    # region Combined Dashboard

    @classmethod
    def plot_binary_eval_dashboard(
        cls,
        cm_df: pd.DataFrame,
        roc_df: pd.DataFrame,
        pr_df: pd.DataFrame,
        auc_score: float | None = None,
        title: str = "Binary Classification Evaluation",
    ) -> Figure:
        """
        Plot a 1x3 dashboard with confusion matrix, ROC curve, and PR curve.

        Args:
            cm_df: Confusion matrix DataFrame
            roc_df: ROC curve DataFrame
            pr_df: PR curve DataFrame
            auc_score: AUC score for ROC legend
            title: Overall figure title
        """
        fig, axes = SubplotHelper.subplots(nrows=1, ncols=3, width_per=7, height_per=6)

        # 1. Confusion Matrix
        ax_cm = axes[0]
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.5,
            cbar=False,
            ax=ax_cm,
            annot_kws={"fontsize": 14},
        )
        ax_cm.set_title("Confusion Matrix", fontsize=14)
        ax_cm.set_ylabel(cm_df.index.name or "Actual", fontsize=12)
        ax_cm.set_xlabel(cm_df.columns.name or "Predicted", fontsize=12)

        # 2. ROC Curve
        ax_roc = axes[1]
        roc_label = f"ROC (AUC = {auc_score:.4f})" if auc_score is not None else "ROC"
        ax_roc.plot(roc_df["fpr"], roc_df["tpr"], color="blue", lw=2, label=roc_label)
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        ax_roc.set_title("ROC Curve", fontsize=14)
        ax_roc.set_xlabel("False Positive Rate", fontsize=12)
        ax_roc.set_ylabel("True Positive Rate", fontsize=12)
        ax_roc.legend(loc="lower right", fontsize=10)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])

        # 3. PR Curve
        ax_pr = axes[2]
        ax_pr.plot(pr_df["recall"], pr_df["precision"], color="blue", lw=2, label="PR Curve")
        ax_pr.set_title("Precision-Recall Curve", fontsize=14)
        ax_pr.set_xlabel("Recall", fontsize=12)
        ax_pr.set_ylabel("Precision", fontsize=12)
        ax_pr.legend(loc="upper right", fontsize=10)
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])

        fig.suptitle(title, fontsize=18, y=1.02)
        fig.tight_layout()
        return fig

    # endregion
