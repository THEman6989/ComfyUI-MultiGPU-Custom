"""Pure helpers for DisTorch dtype-safe moves and ComfyUI load-list accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LoadItem:
    """Normalized ComfyUI load-list entry with resident and transient memory kept separate."""

    module_mem: int | float
    module_offload_mem: int | float
    module_name: str
    module: Any
    params: Any


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


def normalize_load_item(item) -> LoadItem:
    """Normalize current or legacy ComfyUI load-list entries without conflating memory roles."""
    if len(item) == 5:
        module_offload_mem, module_mem, module_name, module_object, params = item
        return LoadItem(module_mem, module_offload_mem, module_name, module_object, params)
    if len(item) == 4:
        module_mem, module_name, module_object, params = item
        return LoadItem(module_mem, module_mem, module_name, module_object, params)
    raise ValueError(f"Unsupported ComfyUI load-list entry with {len(item)} fields")


def unpack_load_item(item):
    """Return resident module bytes and payload for static DisTorch placement."""
    record = normalize_load_item(item)
    return record.module_mem, record.module_name, record.module, record.params


def total_resident_memory(loading: Iterable) -> int | float:
    """Sum bytes that remain resident across static DisTorch target devices."""
    return sum(normalize_load_item(item).module_mem for item in loading)


def transient_headroom(loading: Iterable) -> int | float:
    """Return ComfyUI's peak per-module offload buffer, not cumulative residency."""
    return max((normalize_load_item(item).module_offload_mem for item in loading), default=0)
