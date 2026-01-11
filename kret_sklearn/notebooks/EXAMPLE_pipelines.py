import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from kret_np_pd.UTILS_np_pd import NP_PD_Utils as UKS_NP_PD
from kret_sklearn.custom_transformers import DateTimeSinCosNormalizer, RegressionResidualAdder
from kret_sklearn.pd_pipeline import PipelinePD
from kret_utils.constants_kret import KretConstants


def get_beijing_pipeline():
    float_cols = ["pm2.5", "year", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    date_cols = ["month", "day", "hour"]
    wind_cols = ["cbwd"]

    date_time_normalizer = DateTimeSinCosNormalizer(
        datetime_cols={"month": 12, "day": 31, "hour": 24}
    )  # Normalize 'month' and 'hour' columns
    power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    wind_encoder = OrdinalEncoder()  # needs to be "learned" in a downstream nn.Embedding

    column_transform = ColumnTransformer(
        transformers=[
            ("datetime", date_time_normalizer, date_cols),
            ("scaler", power_transformer, float_cols),
            ("windlabel", wind_encoder, wind_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        verbose=True,
    )
    pipeline_x = PipelinePD(steps=[("column_transform", column_transform)])
    pipeline_y = PipelinePD(steps=[("scaler", power_transformer)])

    return pipeline_x, pipeline_y


def load_and_clean():
    df_load = FunctionTransformer(func=pd.read_csv, validate=False)
    custom_cleanup = FunctionTransformer(func=UKS_NP_PD.data_cleanup, validate=False, kw_args={"ret": True})
    pipeline_load_and_clean = PipelinePD(
        steps=[
            ("df_load", df_load),
            ("cleanup_custom", custom_cleanup),
        ]
    )
    df = pipeline_load_and_clean.fit_transform_df(KretConstants.DATA_DIR / "medical_cost.csv")
    features, target = UKS_NP_PD.pop_label_and_drop(df, label_col="charges", drop_cols=["Id"])
    x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.15, random_state=0)
    return x_train, x_val, y_train, y_val


def scale_and_regress(x_train, x_val, y_train, y_val):
    float_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "region"]
    regressor = RegressionResidualAdder("ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5})
    power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    column_transform = ColumnTransformer(
        transformers=[("scaler", power_transformer, float_cols), ("onehot", one_hot, cat_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
        verbose=True,
    )
    steps = [("column_transform", column_transform), ("ols", regressor)]
    pipeline_scale_ols = PipelinePD(steps=steps)

    x_train = pipeline_scale_ols.fit_transform_df(x_train, y_train)
    x_val = pipeline_scale_ols.transform_df(x_val)
    y_hat_train_ols = x_train.pop("y_hat")
    resid_train = y_train - y_hat_train_ols
    y_hat_val_ols = x_val.pop("y_hat")
    resid_val = y_val - y_hat_val_ols
    UKS_NP_PD.dtt([x_train, x_val, resid_train, resid_val], n=2)
