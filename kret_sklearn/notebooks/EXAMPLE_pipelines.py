from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    PowerTransformer,
)

from kret_sklearn.custom_transformers import (
    DateTimeSinCosNormalizer,
)
from kret_sklearn.pd_pipeline import PipelinePD


def get_beijing_pipeline():
    float_cols = ["pm2.5", "year", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    date_cols = ["month", "day", "hour"]
    wind_cols = ["cbwd"]

    date_time_normalizer = DateTimeSinCosNormalizer(
        datetime_cols={"month": 12, "day": 31, "hour": 24}
    )  # Normalize 'month' and 'hour' columns
    power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    wind_encoder = OrdinalEncoder()

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
