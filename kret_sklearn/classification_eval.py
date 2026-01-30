import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryEvalResult:
    """Container for binary classification evaluation results."""

    confusion_matrix: pd.DataFrame
    classification_report: pd.DataFrame
    roc_curve: pd.DataFrame  # columns: fpr, tpr, thresholds
    pr_curve: pd.DataFrame  # columns: precision, recall, thresholds
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def summary(self) -> pd.DataFrame:
        """Single-row DataFrame with scalar metrics."""
        return pd.DataFrame(
            {
                "accuracy": [self.accuracy],
                "precision": [self.precision],
                "recall": [self.recall],
                "f1": [self.f1],
                "roc_auc": [self.roc_auc],
            }
        )

    def confusion_heatmap(self, ax: Axes | None = None, **kwargs):
        """
        Plot confusion matrix heatmap using sklearn's ConfusionMatrixDisplay.
        TODO allow passing kwargs to disp.plot()
        """

        disp = ConfusionMatrixDisplay(
            confusion_matrix=self.confusion_matrix.values, display_labels=self.confusion_matrix.index
        )
        return disp.plot(ax=ax, **kwargs)


class ClassificationEvalUtils:
    """Utility class for binary classification evaluation using sklearn metrics."""

    @classmethod
    def binary_eval(
        cls,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_prob: ArrayLike | None = None,
        labels: tuple[str, str] = ("Negative", "Positive"),
    ) -> BinaryEvalResult:
        """
        Run a full binary classification evaluation.

        Args:
            y_true: Ground truth binary labels (0/1)
            y_pred: Predicted binary labels (0/1)
            y_prob: Predicted probabilities for the positive class (optional, needed for ROC/PR curves)
            labels: Display names for (negative, positive) classes

        Returns:
            BinaryEvalResult with all computed metrics and curves
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Confusion matrix as labeled DataFrame
        cm = cls.confusion_matrix_df(y_true, y_pred, labels=labels)

        # Classification report as DataFrame
        cr = cls.classification_report_df(y_true, y_pred, target_names=list(labels))

        # Scalar metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Probability-dependent metrics
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            roc_df = cls.roc_curve_df(y_true, y_prob)
            pr_df = cls.pr_curve_df(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
        else:
            roc_df = pd.DataFrame(columns=["fpr", "tpr", "threshold"])
            pr_df = pd.DataFrame(columns=["precision", "recall", "threshold"])
            auc = float("nan")

        return BinaryEvalResult(
            confusion_matrix=cm,
            classification_report=cr,
            roc_curve=roc_df,
            pr_curve=pr_df,
            roc_auc=float(auc),
            accuracy=float(acc),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
        )

    # region Individual Metrics

    @staticmethod
    def confusion_matrix_df(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        labels: tuple[str, str] = ("Negative", "Positive"),
        normalize: t.Literal["true", "pred", "all"] | None = None,
    ) -> pd.DataFrame:
        """
        Compute confusion matrix and return as a labeled DataFrame.

        Rows = Actual, Columns = Predicted.
        """
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        idx = pd.Index(list(labels), name="Actual")
        col = pd.Index(list(labels), name="Predicted")
        return pd.DataFrame(cm, index=idx, columns=col)

    @staticmethod
    def classification_report_df(
        y_true: ArrayLike,
        y_pred: ArrayLike,
        target_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute sklearn classification_report and return as a DataFrame.
        """
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        return pd.DataFrame(report).T

    @staticmethod
    def roc_curve_df(y_true: ArrayLike, y_prob: ArrayLike) -> pd.DataFrame:
        """Compute ROC curve and return as a DataFrame with columns: fpr, tpr, threshold."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    @staticmethod
    def pr_curve_df(y_true: ArrayLike, y_prob: ArrayLike) -> pd.DataFrame:
        """Compute Precision-Recall curve and return as a DataFrame."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve returns n+1 precision/recall values but n thresholds
        # Pad thresholds with NaN to align lengths
        thresholds_padded = np.append(thresholds, np.nan)
        return pd.DataFrame({"precision": precision, "recall": recall, "threshold": thresholds_padded})

    # endregion
