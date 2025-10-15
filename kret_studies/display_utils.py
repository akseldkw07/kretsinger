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


def q_df_to_spatial(Q: pd.DataFrame | np.ndarray, nrows: int = 4, ncols: int = 4) -> dict:
    """
    Convert a Q-table (shape: n_states x n_actions) where states are numbered row-major
    from 0..(nrows*ncols-1) into a dict of 4 DataFrames (left, down, right, up), each of
    shape (nrows, ncols). For each state s and action a, place the Q[s,a] value into the
    neighbor cell that would be reached by taking action a from state s.

    Rules / assumptions:
    - Actions are interpreted as: 0=left, 1=down, 2=right, 3=up (matching the FrozenLake/env convention)
    - The returned DataFrames use np.nan for:
        * cells that are not a target of any (s,a) pair (i.e., unused cells), and
        * the original state cells themselves (so the heatmap does not color the state's own cell).
    - Q can be a numpy array or pandas DataFrame. If DataFrame, columns correspond to actions.

    Returns:
        dict with keys ['left','down','right','up'] and pandas.DataFrame values of shape (nrows,ncols)

    Example usage:
        spatial = q_df_to_spatial(Q)
        left_df = spatial['left']
    """
    # Normalize input to numpy array
    if isinstance(Q, pd.DataFrame):
        Q_arr = Q.values
    else:
        Q_arr = np.asarray(Q)

    n_states = nrows * ncols
    if Q_arr.shape[0] != n_states:
        raise ValueError(f"Expected Q to have {n_states} rows (states), got {Q_arr.shape[0]}")

    # Ensure there are 4 actions (left,down,right,up) or at least up to 4
    n_actions = Q_arr.shape[1]
    if n_actions < 4:
        raise ValueError(f"Expected Q to have at least 4 action columns, got {n_actions}")

    # Initialize output DataFrames with NaNs
    templates = {
        k: pd.DataFrame(np.full((nrows, ncols), np.nan), columns=[*range(ncols)], index=[*range(nrows)])
        for k in ("left", "down", "right", "up")
    }

    # helper: convert state index to (r,c)
    def to_rc(s: int) -> tuple[int, int]:
        return divmod(s, ncols)

    # mapping action -> target offset (dr, dc) and key
    action_map = {
        0: (0, -1, "left"),
        1: (1, 0, "down"),
        2: (0, 1, "right"),
        3: (-1, 0, "up"),
    }

    for s in range(n_states):
        r, c = to_rc(s)
        # Set the state's own cell to NaN explicitly (already NaN in templates)
        for a in range(min(4, n_actions)):
            dr, dc, key = action_map[a]
            tr, tc = r + dr, c + dc
            # check in bounds
            if 0 <= tr < nrows and 0 <= tc < ncols:
                # place Q[s,a] into the target cell position
                templates[key].iat[tr, tc] = Q_arr[s, a]

    # Optionally convert column/index labels to something prettier (like left, down, ... keep numeric indices now)
    return templates
