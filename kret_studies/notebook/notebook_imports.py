from __future__ import annotations

import io
import os
import re
import sys
import typing
import typing as t
from pathlib import Path
from pprint import pformat

# huggingface
import datasets

# kaggle
import kagglehub

# lightgbm
import lightgbm as lgbm

# matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import requests
import scipy.stats as stats  # For statistical tests and plots
import seaborn as sns

# sklearn
import sklearn

# statsmodels and scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

# pytorch
import torch.nn as nn

# misc
import tqdm
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from IPython.display import HTML, Markdown, display
from kagglehub import KaggleDatasetAdapter
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.axes import Axes
from scipy.stats import f
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .nb_setup import load_dotenv_file
from .source_env_vars import source_zsh_env

sns.set_theme()
