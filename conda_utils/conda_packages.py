from __future__ import annotations


class CondaUtils:
    CONDA_PACKAGES_CORE = [
        "numpy",
        "pandas",
        "pyarrow",
        "python-dotenv",
        "jupyter",
        "notebook",
        "jupyterlab",
        "polars",
        "pre-commit",
        "black",
        "flake8",
    ]
    CONDA_PACKAGES_FINANCE = ["yfinance", "pandas-datareader", "ccy", "forex-python"]
    CONDA_PACKAGES_VISUALIZATION = [
        "plotly",
        "bokeh",
        "seaborn",
        "matplotlib",
        "dash",
        "panel",
        "dash-bootstrap-components",
        "dash-core-components",
        "dash-html-components",
        "ipywidgets",
        "pygraphviz",
        "graphviz",
        "drawdata",  # https://python.plainenglish.io/drawdata-the-python-library-you-didnt-know-you-needed-b8c2f2ff328b
    ]
    CONDA_PACKAGES_STATS = [
        "abess",  # don't add, pin to numpy 1.26
        "scikit-learn",
        "statsmodels",
        "scipy",
    ]
    CONDA_PACKAGES_NLP = ["nltk", "gensim", "spacy", "textblob", "transformers"]
    CONDA_PACKAGES_REDDIT = ["praw", "vaderSentiment"]
    CONDA_PACKAGES_ML = [
        "scikit-learn",
        "tensorflow",
        "keras",
        "pytorch",
        "fastai",
        "xgboost",
        "lightgbm",
        "catboost",
        "cvxpy",
        "pulp",
        "fbprophet",
        "sympy",
        "networkx",
        "pydot",
        "pydotplus",
    ]
