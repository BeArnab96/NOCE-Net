"""
noce_net.config
===============
Loading of the NOCE-Net configuration dictionary (``model_configs``).

All model, training, evaluation and export entry points read their
hyper-parameters from ``model_configs.yaml`` at the root of the
``noce-net`` folder. The resulting dictionary is passed unchanged to
``BatteryModel`` and to the training/evaluation functions.
"""
import os
from pathlib import Path
from typing import Optional, Union

import torch
import yaml

#: Default location of the configuration file (``noce-net/model_configs.yaml``).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "model_configs.yaml"


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Resolve the torch device used for computation.

    Args:
        device: ``"cuda"``, ``"cpu"``, a specific device string such as
            ``"cuda:1"``, or ``None``/``"auto"`` to pick CUDA when available.

    Returns:
        The resolved ``torch.device``.
    """
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_config(path: Optional[Union[str, os.PathLike]] = None,
                device: Optional[str] = None) -> dict:
    """
    Load ``model_configs`` from a YAML file and attach the runtime device.

    Args:
        path: path to the YAML file; defaults to ``DEFAULT_CONFIG_PATH``.
        device: device specification forwarded to :func:`get_device`.

    Returns:
        Configuration dictionary. Besides the YAML entries it contains
        ``device`` (a ``torch.device``) and ``num_patches``
        (``len(patch)``).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        model_configs = yaml.safe_load(f)
    model_configs["device"] = get_device(device)
    model_configs["num_patches"] = len(model_configs["patch"])
    return model_configs
