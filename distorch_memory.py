"""Pure helpers for DisTorch dtype-safe moves and ComfyUI load-list accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _is_floating_dtype(dtype: Any) -> bool:
    return bool(dtype is not None and getattr(dtype, "is_floating_point", False))


def get_compute_dtype(model: Any, state_dict: Mapping[str, Any] | None = None):
    """Return a safe floating compute dtype without treating quantized storage as compute."""
    manual_cast_dtype = getattr(model, "manual_cast_dtype", None)
    if _is_floating_dtype(manual_cast_dtype):
        return manual_cast_dtype

    populations = {}
    resolved_state_dict = state_dict if state_dict is not None else model.state_dict()
    for value in resolved_state_dict.values():
        dtype = getattr(value, "dtype", None)
        if not _is_floating_dtype(dtype):
            continue
        numel = value.numel() if hasattr(value, "numel") else 0
        populations[dtype] = populations.get(dtype, 0) + numel

    return max(populations, key=lambda dtype: populations[dtype]) if populations else None


def move_module_to(module: Any, device: Any) -> Any:
    """Move a module without flattening mixed storage/scale/bias dtypes."""
    return module.to(device=device)


def unpack_load_item(item):
    """Return effective memory and payload for current or legacy ComfyUI load-list entries."""
    if len(item) == 5:
        module_offload_mem, module_mem, module_name, module_object, params = item
        effective_mem = (
            max(module_offload_mem, module_mem)
            if isinstance(module_offload_mem, (int, float))
            else module_mem
        )
        return effective_mem, module_name, module_object, params
    if len(item) == 4:
        return item[0], item[1], item[2], item[3]
    raise ValueError(f"Unsupported ComfyUI load-list entry with {len(item)} fields")


def total_load_memory(loading: Iterable) -> int | float:
    """Sum the exact effective memory values used for DisTorch block distribution."""
    return sum(unpack_load_item(item)[0] for item in loading)
