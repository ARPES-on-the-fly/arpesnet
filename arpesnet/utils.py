from typing import Callable
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm


def load_config(config: dict | str | Path) -> dict:
    """Load the configuration file.

    Args:
        config (dict | str | Path): the configuration dictionary or path to yaml file.

    Returns:
        dict: the configuration dictionary
    """
    if isinstance(config, (str, Path)):
        config = Path(config).absolute()
        if not config.is_file():
            raise FileNotFoundError(f"Could not find the file {config}")
        config = yaml.safe_load(open(config))
    elif not isinstance(config, dict):
        raise ValueError("config must be a path to a yaml file or a dictionary")
    return config


def load_torch_datasets(
    path: str | Path,
    datasets: str | list = "all",
    config: dict | str | Path = None,
    verbose: bool = True,
    transform: Callable = lambda x: x,
    pbar=True,
) -> torch.Tensor:
    """Load datasets stored as torch tensors from disk.

    Args:
        path: the path to the directory containing the datasets. If None the path is taken from the
            configuration file. Ignored if datasets is "train" or "test"
        datasets: the list of datasets to load. if "train" or "test" it searches the config file
            for path and datasets to load.

    Returns:
        torch.Tensor: The loaded datasets concatenated along the first axis.
    """
    if config is None:
        config = {}
    config = load_config(config)
    path = Path(path)
    if not path.is_dir():
        path = Path(config["paths"]["root"]) / path
    if not path.is_dir():
        raise FileNotFoundError(f"No directory under the path {path}")

    if datasets == "train":
        datasets = config["datasets"].get("train", "all")
    elif datasets == "test":
        datasets = config["datasets"].get("ttestrain", "all")

    if datasets == "all":
        datasets = [f for f in path.iterdir() if f.is_file() and f.suffix == ".pt"]
    else:
        if isinstance(datasets, str):
            datasets = [datasets]
        datasets = [(path / f"{f}").with_suffix(".pt") for f in datasets]
    missing_files = [str(f.stem) for f in datasets if not f.is_file()]

    if len(missing_files) > 0:
        s = "\n".join(missing_files)
        raise FileNotFoundError(f"Could not find {len(missing_files)}/{len(datasets)} files {s}.")
    if len(datasets) == 0:
        raise FileNotFoundError(f"No datasets found in {path}")
    to_load = [f for f in datasets if f.is_file()]

    tds = torch.cat(
        [transform(torch.load(f)) for f in tqdm(to_load, desc="loading data", disable=not pbar)],
        dim=0,
    )
    if verbose:
        print(f"Loaded {len(datasets)} datasets from {path}")
        print(f"Data: {tds.shape[0]} samples | shape {tds.shape[1:]} | dtype {tds.dtype}")
    return tds


def sec2min(sec) -> str:
    """
    Convert seconds to minutes in the format "MM:SS" or "HH:MM:SS".

    Args:
        sec (int): Number of seconds.

    Returns:
        str: Time in the format "MM:SS" or "HH:MM:SS".
    """
    if sec < 3600:
        return f"{sec // 60:2.0f}:{sec % 60:02.0f}"
    else:
        return f"{sec // 3600:2.0f}:{(sec % 3600) / 60:02.0f}:{sec % 60:02.0f}"
