from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def model_diagnostics(y_true, y_pred, X):
    """
    Compute regression diagnostics with robust handling of inputs.

    - Coerces X to a numeric 2D DataFrame
    - Aligns y_true/y_pred with X and drops NaNs/inf rows
    - Computes R^2, adjusted R^2, Jarque–Bera (stat & p), Durbin–Watson,
      Breusch–Pagan (stat & p), and VIF per feature
    - Returns a metrics DataFrame and a diagnostics figure
    """
    # --- Coerce inputs ---
    y_true = pd.Series(np.asarray(y_true).reshape(-1), name="y_true")
    y_pred = pd.Series(np.asarray(y_pred).reshape(-1), name="y_pred")

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        X_df = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])

    # keep only numeric columns for statsmodels tests
    X_df = X_df.select_dtypes(include=[np.number])

    # Align and drop problematic rows
    df = pd.concat([y_true, y_pred, X_df], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    X_num = df.drop(columns=["y_true", "y_pred"])  # may be empty

    residuals = y_true - y_pred

    # Compute OLS bias terms
    X_line = sm.add_constant(y_true)
    line_model = sm.OLS(y_pred, X_line).fit()
    intercept, slope = line_model.params[0], line_model.params[1]

    # --- Metrics ---
    r2 = r2_score(y_true, y_pred)
    # guard adjusted R^2 when no features are provided
    p = X_num.shape[1]
    n = len(y_true)
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan

    # Normality of residuals
    jb_stat, jb_p, _, _ = jarque_bera(residuals)

    # Autocorrelation (DW)
    dw_stat = durbin_watson(residuals)

    # Standardized residuals
    resid_std = (residuals - residuals.mean()) / (residuals.std(ddof=1) if residuals.std(ddof=1) != 0 else 1.0)

    # Heteroskedasticity (Breusch–Pagan). Requires >=1 regressor.
    if p >= 1:
        # statsmodels is fine with a numpy array or DataFrame; add constant robustly
        exog = sm.add_constant(X_num.to_numpy(), has_constant="add")
        lm_stat, lm_p, _, _ = het_breuschpagan(residuals, exog)
    else:
        lm_stat, lm_p = np.nan, np.nan

    # VIF per feature (only if we have >=2 columns to regress upon; with 1 it still works but can be large)
    if p >= 1:
        vif_vals = [variance_inflation_factor(X_num.to_numpy(), i) for i in range(p)]
        vif = pd.DataFrame({"feature": X_num.columns, "VIF": vif_vals})
        mean_vif = float(np.nanmean(vif["VIF"]))
    else:
        vif = pd.DataFrame({"feature": [], "VIF": []})
        mean_vif = np.nan

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "R²",
                "Adj R²",
                "Durbin–Watson",
                "JB statistic",
                "JB p-value",
                "BP statistic",
                "BP p-value",
                "Mean VIF",
            ],
            "Value": [r2, adj_r2, dw_stat, jb_stat, jb_p, lm_stat, lm_p, mean_vif],
        }
    )
    return {
        "metrics": metrics_df,
        "residuals": residuals,
        "vif": vif,
        "slope": slope,
        "intercept": intercept,
        "resid_std": resid_std,
    }
