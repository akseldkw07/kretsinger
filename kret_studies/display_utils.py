from IPython.display import display_html


def display_side_by_side(dfs, names=None, spacing=10):
    """
    Display a list of pandas DataFrames side-by-side in Jupyter.

    Args:
        dfs (list[pd.DataFrame]): The DataFrames to display.
        names (list[str], optional): Titles for each DataFrame.
        spacing (int, optional): Horizontal spacing in pixels.
    """
    html_str = ""
    for i, df in enumerate(dfs):
        title = f"<h3>{names[i]}</h3>" if names and i < len(names) else ""
        html_table = df.to_html()
        html_str += (
            f'<div style="display:inline-block; vertical-align:top; margin-right:{spacing}px">{title}{html_table}</div>'
        )
    display_html(html_str, raw=True)
