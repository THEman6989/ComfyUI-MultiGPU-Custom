"""Helpers for keeping DisTorch objects on ComfyUI's legacy ModelPatcher."""

from __future__ import annotations

import copy


_DISTORCH_MARKERS = ("_distorch_v2_meta", "_distorch_block_assignments")


def get_model_patcher(value):
    """Return the model patcher owned by a common ComfyUI model container."""
    if value is None:
        return None
    if callable(getattr(value, "is_dynamic", None)) and hasattr(value, "model"):
        return value

    patcher = getattr(value, "patcher", None)
    if patcher is not None and callable(getattr(patcher, "is_dynamic", None)):
        return patcher

    patcher = getattr(value, "control_model_wrapped", None)
    if patcher is not None and callable(getattr(patcher, "is_dynamic", None)):
        return patcher
    return None


def _legacy_clone_for_loader_output(patcher):
    """Convert an exclusive loader output without reloading or sharing a branch."""
    if not patcher.is_dynamic():
        return patcher
    model_override = patcher.get_clone_model_override()
    return patcher.clone(disable_dynamic=True, model_override=model_override)


def prepare_distorch_object(value):
    """Return a loader output backed by a non-dynamic patcher.

    DisTorch wrappers call this immediately after the underlying loader returns,
    before that output can be branched elsewhere. Reusing the freshly loaded
    model is therefore safe and also supports loaders without cached reload
    factories (for example CLIP Vision and ControlNet).
    """
    patcher = get_model_patcher(value)
    if patcher is None:
        return value

    legacy_patcher = _legacy_clone_for_loader_output(patcher)
    if legacy_patcher is patcher:
        return value

    if patcher is value:
        return legacy_patcher

    copier = getattr(value, "copy", None)
    result = copier() if callable(copier) and hasattr(value, "control_model_wrapped") else copy.copy(value)

    if getattr(value, "patcher", None) is patcher:
        result.patcher = legacy_patcher
        if hasattr(result, "cond_stage_model"):
            result.cond_stage_model = legacy_patcher.model
        if hasattr(result, "first_stage_model"):
            result.first_stage_model = legacy_patcher.model
        if hasattr(result, "model"):
            result.model = legacy_patcher.model
    elif getattr(value, "control_model_wrapped", None) is patcher:
        result.control_model_wrapped = legacy_patcher
        result.control_model = legacy_patcher.model

    return result


def mark_distorch_object(value, meta):
    patcher = get_model_patcher(value)
    if patcher is None:
        return False
    patcher._distorch_v2_meta = meta
    patcher.model._distorch_v2_meta = meta
    return True


def prepare_and_mark_distorch_outputs(outputs, meta=None):
    """Convert every model-bearing output; annotate only with real allocation metadata."""
    converted = []
    for value in outputs:
        value = prepare_distorch_object(value)
        if meta is not None:
            mark_distorch_object(value, meta)
        converted.append(value)
    if isinstance(outputs, tuple):
        return tuple(converted)
    if isinstance(outputs, list):
        return converted
    return type(outputs)(converted)


def is_distorch_object(value):
    patcher = get_model_patcher(value)
    if patcher is None:
        return False
    for current in (patcher, getattr(patcher, "model", None)):
        if current is not None and any(hasattr(current, marker) for marker in _DISTORCH_MARKERS):
            return True
    return False
