def get_ols_formula(feature_cols: list[str] | str, label_col: str | list[str]) -> str:
    if isinstance(label_col, list):
        label_col = label_col[0]
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    ret = f"{label_col} ~ {' + '.join(feature_cols)}"
    return ret
