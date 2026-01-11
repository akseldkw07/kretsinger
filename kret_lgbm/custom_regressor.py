"""
Subclass LGBM Regressor and LGBMClassifier to create custom versions.
"""

from lightgbm import LGBMClassifier, LGBMRegressor


class CustomRegressor(LGBMRegressor): ...


class CustomClassifier(LGBMClassifier): ...
