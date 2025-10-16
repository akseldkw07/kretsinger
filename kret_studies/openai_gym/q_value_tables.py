import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches


def expanded_heatmap_with_state_borders(
    df: pd.DataFrame | pd.io.formats.style.Styler,
    ax=None,
    fmt: str = ".3f",
    cmap: str = "WhiteGreen",
    annotate: bool = True,
    line_width: float = 2.0,
    line_color: str = "black",
    cbar: bool = True,
):
    """
    Plot a heatmap for the expanded grid DataFrame and overlay thick borders between
    each state (3x3) block. Accepts either the DataFrame produced by `q_to_expanded_grid`
    or a pandas Styler (in which case the underlying .data is used).

    Parameters:
      df: expanded DataFrame (MultiIndex rows/cols) or Styler
      ax: matplotlib Axes (optional)
      fmt: annotation format
      cmap: colormap
      annotate: whether to write numbers in cells
      line_width: width (in points) of separators between states
      line_color: color of separators
      cbar: whether to show colorbar
    """

    # accept a Styler too
    # Styler stores the underlying DataFrame in the .data attribute; handle gracefully
    df_data: pd.DataFrame = df if isinstance(df, pd.DataFrame) else df.data  # type: ignore

    # ensure numeric matrix
    mat = df_data.astype(float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # plot heatmap
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=fmt,
        cbar=cbar,
        linewidths=0.3,
        linecolor="lightgrey",
        annot_kws={"fontsize": 8},
    )
    # also reduce tick label size
    ax.tick_params(axis="both", which="major", labelsize=8)

    # compute block boundaries (top-level MultiIndex values)
    try:
        n_block_rows = len(df_data.index.get_level_values(0).unique())
        n_block_cols = len(df_data.columns.get_level_values(0).unique())
    except Exception:
        # fallback: assume 3x3 blocks with shape divisible by 3
        n_block_rows = df_data.shape[0] // 3
        n_block_cols = df_data.shape[1] // 3
    total_rows = df_data.shape[0]
    total_cols = df_data.shape[1]

    # draw horizontal lines between blocks
    for i in range(1, n_block_rows):
        y = i * 3
        ax.hlines(y, xmin=0, xmax=total_cols, colors=line_color, linewidth=line_width, zorder=5)

    # draw vertical lines between blocks
    for j in range(1, n_block_cols):
        x = j * 3
        ax.vlines(x, ymin=0, ymax=total_rows, colors=line_color, linewidth=line_width, zorder=5)

    # draw outer border
    ax.add_patch(
        patches.Rectangle(
            (0, 0), total_cols, total_rows, fill=False, edgecolor=line_color, linewidth=line_width, zorder=6
        )
    )

    # tweak ticks/labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return ax


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
    # Build MultiIndex for rows and columns to indicate original row/col and sub-position
    row_tuples = []
    subrows = ["up", "", "down"]
    for r in range(nrows):
        for sr in subrows:
            row_tuples.append((r, sr))
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=["row", "subrow"])

    col_tuples = []
    subcols = ["left", "", "right"]
    for c in range(ncols):
        for sc in subcols:
            col_tuples.append((c, sc))
    col_index = pd.MultiIndex.from_tuples(col_tuples, names=["col", "subcol"])

    df.index = row_index
    df.columns = col_index
    return df


def style_expanded_grid(df: pd.DataFrame, line_width: str = "2px", color: str = "black") -> pd.io.formats.style.Styler:
    """
    Return a pandas Styler that applies a dark border between each original state block
    in an expanded grid produced by `q_to_expanded_grid`.

    Expects df to have a MultiIndex on rows (row, subrow) and columns (col, subcol) with
    3 subrows ['up','mid','down'] and 3 subcols ['left','mid','right'].
    """
    styler = df.style

    # compute boundaries: after every 3 rows/cols we want a thicker border
    # number of top-level rows/cols (original grid dims)
    len(df.index.get_level_values(0).unique())
    len(df.columns.get_level_values(0).unique())

    # build CSS rules per cell
    def _border_style_cell(cell_row_index: int, cell_col_index: int) -> str:
        styles = {}
        # column multiindex tuple
        col_tup = df.columns[cell_col_index]
        subcol = col_tup[1]
        if subcol == "right":
            styles["border-right"] = f"{line_width} solid {color}"
        if subcol == "left":
            styles["border-left"] = f"{line_width} solid {color}"

        row_tup = df.index[cell_row_index]
        subrow = row_tup[1]
        if subrow == "down":
            styles["border-bottom"] = f"{line_width} solid {color}"
        if subrow == "up":
            styles["border-top"] = f"{line_width} solid {color}"

        if not styles:
            return ""
        return "; ".join(f"{k}: {v}" for k, v in styles.items())

    # apply styles cellwise using applymap which passes the value; we need indices so we'll build an array
    style_matrix = [[_border_style_cell(r, c) for c in range(df.shape[1])] for r in range(df.shape[0])]

    def _apply_row_style(row):
        # map the row name (a MultiIndex tuple) to integer position
        pos = df.index.get_loc(row.name)
        # get_loc may return a slice or int; coerce to int if needed
        if isinstance(pos, slice):
            # if a slice, take the start index
            pos = pos.start or 0
        return style_matrix[int(pos)]

    styler = styler.apply(_apply_row_style, axis=1)
    return styler
