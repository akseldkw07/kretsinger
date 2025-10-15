from IPython.display import display_html
import pandas as pd
import numpy as np


def dataset_to_table(*dfs: pd.DataFrame | np.ndarray, names: list[str] | None = None, spacing: int = 10):
    """
    Display a list of pandas DataFrames side-by-side in Jupyter.

    Args:
        dfs (t.Iterable[pd.DataFrame]): The DataFrames to display.
        names (list[str], optional): Titles for each DataFrame.
        spacing (int, optional): Horizontal spacing in pixels.
    """
    html_str = ""
    for i, df in enumerate(dfs):
        title = f"<h3>{names[i]}</h3>" if names and i < len(names) else ""
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        html_table = df.to_html()
        html_str += (
            f'<div style="display:inline-block; vertical-align:top; margin-right:{spacing}px">{title}{html_table}</div>'
        )
    display_html(html_str, raw=True)


def q_to_expanded_grid(Q: pd.DataFrame | np.ndarray, nrows: int = 4, ncols: int = 4) -> pd.DataFrame:
    """
    Expand a Q-table for a nrows x ncols grid into a (3*nrows) x (3*ncols) DataFrame
    where each original state is represented by a 3x3 block with positions:

        [  ,  up,   ]
        [ left, NaN, right]
        [  , down,  ]

    - Q shape expected: (nrows*ncols, >=4) (FrozenLake action order: 0=left,1=down,2=right,3=up)
    - Returned DataFrame is filled with np.nan except the placed action values.
    """
    if isinstance(Q, pd.DataFrame):
        Q_arr = Q.values
    else:
        Q_arr = np.asarray(Q)

    n_states = nrows * ncols
    if Q_arr.shape[0] != n_states:
        raise ValueError(f"Expected Q to have {n_states} rows (states), got {Q_arr.shape[0]}")
    if Q_arr.shape[1] < 4:
        raise ValueError(f"Expected Q to have at least 4 action columns, got {Q_arr.shape[1]}")

    out_rows = 3 * nrows
    out_cols = 3 * ncols
    out = np.full((out_rows, out_cols), np.nan, dtype=float)

    def to_rc(s: int) -> tuple[int, int]:
        return divmod(s, ncols)

    for s in range(n_states):
        r, c = to_rc(s)
        br = 3 * r
        bc = 3 * c
        # mapping inside 3x3 block
        # up (action 3) -> (0,1)
        out[br + 0, bc + 1] = float(Q_arr[s, 3])
        # left (action 0) -> (1,0)
        out[br + 1, bc + 0] = float(Q_arr[s, 0])
        # center state left as NaN -> out[br+1, bc+1]
        # right (action 2) -> (1,2)
        out[br + 1, bc + 2] = float(Q_arr[s, 2])
        # down (action 1) -> (2,1)
        out[br + 2, bc + 1] = float(Q_arr[s, 1])

    df = pd.DataFrame(out)
    return df
