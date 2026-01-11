# autoflake: skip_file
import time

start_time = time.time()
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

from .constants_sklearn import SklearnDefaults
from .custom_transformers import DateTimeSinCosNormalizer, MissingValueRemover, PandasColumnOrderBase
from .pd_pipeline import PipelinePD
from .UTILS_sklearn import UTILS_sklearn

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
