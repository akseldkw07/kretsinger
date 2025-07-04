import pandas as pd
import hashlib
import json
import os
import typing as t
from kret_studies.typed_cls import *
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yfinance")
INDEX_PATH = os.path.join(DATA_DIR, "index.json")


def _json_friendly(d):
    # Remove non-serializable items
    def convert(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [convert(i) for i in v]
        if isinstance(v, dict):
            return {str(k): convert(val) for k, val in v.items()}
        return str(v)

    return {k: convert(v) for k, v in d.items()}


def _get_args_hash(args):
    args_json = json.dumps(args, sort_keys=True)
    return hashlib.sha256(args_json.encode()).hexdigest()


def download(ticker: str, start: str, **kwargs: t.Unpack[Download_TypedDict]):
    """
    Wrapper for yfinance multi.download with local caching.
    Checks if data for the given arguments exists locally (using a json index),
    loads it if so, otherwise queries remote, saves, and updates the index.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    args = _json_friendly(kwargs)
    args_hash = _get_args_hash(args)

    # Load or create index
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            index = json.load(f)
    else:
        index = {}

    # Check if data exists
    if args_hash in index:
        data_path = index[args_hash]["path"]
        if os.path.exists(data_path):
            logger.info(f"Loading cached data from {data_path}")
            return pd.read_parquet(data_path)

    # If not, query remote
    logger.info("Querying remote with yfinance.multi.download...")

    df = t.cast(pd.DataFrame, yf.download(**kwargs))
    # Save data
    fname = f"yf_{args_hash}.parquet"
    data_path = os.path.join(DATA_DIR, fname)
    df.to_parquet(data_path)
    # Update index
    index[args_hash] = {"args": args, "path": data_path}
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Saved new data to {data_path}")
    return df
