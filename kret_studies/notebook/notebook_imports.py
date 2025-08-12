from __future__ import annotations

import os
import re
import sys
import typing as t
from pathlib import Path
from IPython.display import display, HTML
from pprint import pformat
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import io
import requests
from .source_env_vars import source_zsh_env
from .nb_setup import load_dotenv_file

# kaggle
import kagglehub
from kagglehub import KaggleDatasetAdapter


# statsmodels and scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats  # For statistical tests and plots
from scipy.stats import f

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors


# sklearn
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sns.set_theme()

# source env variables
load_dotenv_file()
source_zsh_env()
