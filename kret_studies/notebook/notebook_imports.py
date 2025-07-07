from __future__ import annotations
from .source_env_vars import source_zsh_env


import seaborn as sns

# region market data

# statsmodels and scipy
import scipy.stats as stats  # For statistical tests and plots

# region matplotlib


# region sklearn

# endregion
sns.set_theme()

# source env variables

source_zsh_env()
