"""
Subclass LGBM Regressor and LGBMClassifier to create custom versions.
"""

from lightgbm import LGBMRegressor, LGBMClassifier


class CustomRegressor(LGBMRegressor): ...


class CustomClassifier(LGBMClassifier): ...
