# isort: skip_file
from __future__ import annotations

import io
import re
import typing
import typing as t
from pprint import pformat
import json

# filesystem
import shutil
from urllib.request import urlretrieve
import zipfile
from pathlib import Path
import sys
import os

# networkx
import networkx as nx
import graphviz
import osmnx as ox

# huggingface
import datasets
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download, snapshot_download, list_datasets
import huggingface_hub

# kaggle
import kagglehub
from kagglehub import KaggleDatasetAdapter

# lightgbm
import lightgbm as lgbm
from lightgbm import LGBMClassifier, LGBMRegressor

# matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# openai gym
import gymnasium as gym

# Language
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation
from sentence_transformers.readers import STSBenchmarkDataReader
import spacy
import gensim
import gensim.downloader as gensim_api

# sklearn
import sklearn
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# statsmodels, scipy, pymc
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats  # For statistical tests and plots
import seaborn as sns
from scipy.stats import f
from scipy.stats import t as student_t, gamma as gamma_dist

"""# pymc
import pymc as pm
from pymc import math as pmmath
import arviz as az
import xarray
import pytensor
import pytensor.tensor as pt"""

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset

# numpy, pandas
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# misc
from tqdm.auto import tqdm as tqdm_auto
from tqdm.notebook import tqdm as tqdm_notebook
from IPython.display import HTML, Markdown, display
import polars as pl
import requests
from dataclasses import dataclass
from math import sqrt, log10

# local imports
from .nb_setup import load_dotenv_file
from .source_env_vars import source_zsh_env
from .wandb_utils import start_wandb_run, WANDB_PROJECT_NAME, WANDB_TEAM_NAME
from kret_studies.kret_torch import DEVICE, DEVICE_TORCH_STR

DEVICE_TORCH = DEVICE

sns.set_theme()
