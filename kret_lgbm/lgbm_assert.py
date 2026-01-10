import warnings

FRAC_CHECK = ["bagging_fraction", "feature_fraction"]


class LGBM_Assertions:
    @classmethod
    def is_valid_params(cls, params: dict):
        for key in FRAC_CHECK:
            low = 0.0
            high = 1.0
            if (val := params.get(key)) is not None and (val < low or val > high):
                warnings.warn(f"{key} expected between {low} and {high}. Got {val}")
                return False

        return True
