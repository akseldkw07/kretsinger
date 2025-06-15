def get_ols_formula(feature_cols: list[str], label_col: str | list[str]) -> str:
    if isinstance(label_col, list):
        label_col = label_col[0]
    ret = f"{label_col} ~ {' + '.join(feature_cols)}"
    return ret
