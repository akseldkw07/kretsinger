# autoflake: skip_file
import time

start_time = time.time()

import datasets
import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download, list_datasets, snapshot_download
from torch.utils.data import DataLoader, Dataset, StackDataset, TensorDataset, random_split
from torch.utils.data.dataset import Subset
from torchmetrics.functional import (
    accuracy as accuracy_pt,
    f1_score as f1_score_pt,
    precision as precision_pt,
    r2_score as r2_score_pt,
    recall as recall_pt,
)

from .tensor_ds_custom import TensorDatasetCustom
from .UTILS_torch import KRET_TORCH_UTILS as UKS_TORCH_UTILS

# from torchvision import datasets, transforms

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
