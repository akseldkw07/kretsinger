from __future__ import annotations

import os
import re
import sys
import typing as t
from pathlib import Path
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats  # For statistical tests and plots

# region sklearn
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# endregion
sns.set_theme()
