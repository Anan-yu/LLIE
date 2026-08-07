"""Checkpoint selection helpers shared by evaluation entry points."""

from collections.abc import Mapping

import torch


def select_network_state(checkpoint, param_key="auto"):
    """Select and normalize a network state dictionary.

    ``auto`` prefers EMA weights because CAME validation and best-checkpoint
    selection evaluate the EMA network whenever it is enabled.  Raw state
    dictionaries remain supported for legacy checkpoints.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping.")

    selected_key = param_key
    if param_key == "auto":
        if "params_ema" in checkpoint:
            selected_key = "params_ema"
        elif "params" in checkpoint:
            selected_key = "params"
        elif checkpoint and all(
            torch.is_tensor(value) for value in checkpoint.values()
        ):
            selected_key = "state_dict"
        else:
            raise KeyError(
                "Checkpoint contains neither 'params_ema' nor 'params'."
            )

    if selected_key == "state_dict":
        state_dict = checkpoint
    else:
        if selected_key not in checkpoint:
            available = ", ".join(str(key) for key in checkpoint.keys())
            raise KeyError(
                f"Requested checkpoint key '{selected_key}' is unavailable; "
                f"available keys: {available}."
            )
        state_dict = checkpoint[selected_key]

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Checkpoint entry '{selected_key}' is not a state dictionary."
        )
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized[normalized_key] = value
    return normalized, selected_key
