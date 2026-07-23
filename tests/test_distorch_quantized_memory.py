from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "distorch_memory.py"


def load_module():
    spec = importlib.util.spec_from_file_location("distorch_memory_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def __init__(self, state_dict, manual_cast_dtype=None):
        self._state_dict = state_dict
        self.manual_cast_dtype = manual_cast_dtype

    def state_dict(self):
        return self._state_dict


class RecordingModule:
    def __init__(self):
        self.calls = []

    def to(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self


def test_compute_dtype_prefers_explicit_floating_manual_cast_dtype():
    module = load_module()
    model = FakeModel(
        {
            "weight": torch.zeros(64, dtype=torch.int8),
            "bias": torch.zeros(4, dtype=torch.float32),
        },
        manual_cast_dtype=torch.bfloat16,
    )

    assert module.get_compute_dtype(model) is torch.bfloat16


def test_compute_dtype_ignores_integer_storage_and_uses_largest_float_population():
    module = load_module()
    model = FakeModel(
        {
            "weight": torch.zeros(4096, dtype=torch.int8),
            "bias": torch.zeros(32, dtype=torch.float32),
            "scale": torch.zeros(64, dtype=torch.float16),
        }
    )

    assert module.get_compute_dtype(model) is torch.float16


def test_compute_dtype_is_none_for_integer_only_model():
    module = load_module()
    model = FakeModel({"weight": torch.zeros(64, dtype=torch.int8)})

    assert module.get_compute_dtype(model) is None


def test_device_move_preserves_tensor_dtypes_without_global_cast():
    module = load_module()
    target = torch.device("cpu")
    value = RecordingModule()

    module.move_module_to(value, target)

    assert value.calls == [((), {"device": target})]


def test_device_move_does_not_flatten_mixed_parameter_dtypes():
    module = load_module()
    value = torch.nn.Module()
    value.register_parameter(
        "quantized_weight",
        torch.nn.Parameter(torch.tensor([1], dtype=torch.int8), requires_grad=False),
    )
    value.register_parameter(
        "scale",
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float16)),
    )
    value.register_parameter(
        "bias",
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
    )

    module.move_module_to(value, torch.device("cpu"))

    assert [parameter.dtype for parameter in value.parameters()] == [
        torch.int8,
        torch.float16,
        torch.float32,
    ]


def test_device_move_still_moves_integer_only_module_without_dtype_cast():
    module = load_module()
    target = torch.device("cpu")
    value = RecordingModule()

    module.move_module_to(value, target)

    assert value.calls == [((), {"device": target})]


def test_unpack_load_item_uses_core_offload_memory_with_zero_fallback():
    module = load_module()
    block = object()
    params = {"weight": object()}

    assert module.unpack_load_item((240, 100, "block", block, params)) == (
        240,
        "block",
        block,
        params,
    )
    assert module.unpack_load_item((0, 100, "block", block, params)) == (
        100,
        "block",
        block,
        params,
    )
    assert module.unpack_load_item((50, 100, "block", block, params)) == (
        100,
        "block",
        block,
        params,
    )
    assert module.unpack_load_item((100, "block", block, params)) == (
        100,
        "block",
        block,
        params,
    )


def test_total_load_memory_uses_same_effective_values_as_distribution():
    module = load_module()
    block = object()
    params = {}
    loading = [
        (240, 100, "quantized", block, params),
        (0, 50, "fallback", block, params),
        (25, "legacy", block, params),
    ]

    assert module.total_load_memory(loading) == 315


def main():
    tests = [
        test_compute_dtype_prefers_explicit_floating_manual_cast_dtype,
        test_compute_dtype_ignores_integer_storage_and_uses_largest_float_population,
        test_compute_dtype_is_none_for_integer_only_model,
        test_device_move_preserves_tensor_dtypes_without_global_cast,
        test_device_move_does_not_flatten_mixed_parameter_dtypes,
        test_device_move_still_moves_integer_only_module_without_dtype_cast,
        test_unpack_load_item_uses_core_offload_memory_with_zero_fallback,
        test_total_load_memory_uses_same_effective_values_as_distribution,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} DisTorch quantized memory tests")


if __name__ == "__main__":
    main()
