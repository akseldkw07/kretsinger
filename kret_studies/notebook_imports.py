from __future__ import annotations

import os
import re
import sys
import typing as t
from pathlib import Path
from IPython.display import display, HTML

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import io
import requests

# region market data
import yfinance as yf

# statsmodels and scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats  # For statistical tests and plots
from scipy.stats import f

# region matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors


# region sklearn
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# endregion
sns.set_theme()
