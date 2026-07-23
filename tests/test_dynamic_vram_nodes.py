from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "dynamic_vram_nodes.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dynamic_vram_nodes_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePatcher:
    def __init__(self, dynamic: bool, label: str = "source"):
        self.dynamic = dynamic
        self.label = label
        self.model = object()
        self.clone_calls = []

    def is_dynamic(self):
        return self.dynamic

    def clone(self, disable_dynamic=False):
        self.clone_calls.append(disable_dynamic)
        return FakePatcher(self.dynamic and not disable_dynamic, label=f"clone:{self.label}")


class FakeModel(FakePatcher):
    pass


class FakeClip:
    def __init__(self, patcher):
        self.patcher = patcher
        self.cond_stage_model = patcher.model
        self.clone_calls = []

    def clone(self, disable_dynamic=False):
        self.clone_calls.append(disable_dynamic)
        result = FakeClip(self.patcher.clone(disable_dynamic=disable_dynamic))
        # Match comfy.sd.CLIP.clone(), which preserves the old container model.
        result.cond_stage_model = self.cond_stage_model
        return result


class FakeVAE:
    def __init__(self, patcher, marker="source"):
        self.patcher = patcher
        self.first_stage_model = patcher.model
        self.marker = marker


def test_model_node_disables_dynamic_vram_on_output_branch_only():
    module = load_module()
    source = FakeModel(dynamic=True)

    output, = module.DisableDynamicVRAMModel().convert(source)

    assert source.is_dynamic() is True
    assert output is not source
    assert output.is_dynamic() is False
    assert source.clone_calls == [True]


def test_clip_node_disables_dynamic_vram_and_syncs_output_model_only():
    module = load_module()
    source = FakeClip(FakePatcher(dynamic=True))

    output, = module.DisableDynamicVRAMCLIP().convert(source)

    assert source.patcher.is_dynamic() is True
    assert output is not source
    assert output.patcher.is_dynamic() is False
    assert output.cond_stage_model is output.patcher.model
    assert output.cond_stage_model is not source.cond_stage_model
    assert source.clone_calls == [True]


def test_vae_node_clones_container_disables_patcher_and_syncs_model():
    module = load_module()
    source = FakeVAE(FakePatcher(dynamic=True))

    output, = module.DisableDynamicVRAMVAE().convert(source)

    assert output is not source
    assert output.marker == source.marker
    assert source.patcher.is_dynamic() is True
    assert output.patcher.is_dynamic() is False
    assert output.first_stage_model is output.patcher.model
    assert output.first_stage_model is not source.first_stage_model
    assert source.patcher.clone_calls == [True]


def test_node_contracts_are_typed_passthroughs():
    module = load_module()
    cases = [
        ("DisableDynamicVRAMModel", "MODEL", "model"),
        ("DisableDynamicVRAMCLIP", "CLIP", "clip"),
        ("DisableDynamicVRAMVAE", "VAE", "vae"),
    ]
    for node_name, socket_type, input_name in cases:
        node = getattr(module, node_name)
        assert node.INPUT_TYPES() == {"required": {input_name: (socket_type,)}}
        assert node.RETURN_TYPES == (socket_type,)
        assert node.RETURN_NAMES == (input_name,)
        assert node.FUNCTION == "convert"
        assert node.CATEGORY == "multigpu/dynamic_vram"


def test_node_mapping_exports_all_dynamic_vram_nodes():
    module = load_module()
    expected = {
        "DisableDynamicVRAMModel": module.DisableDynamicVRAMModel,
        "DisableDynamicVRAMCLIP": module.DisableDynamicVRAMCLIP,
        "DisableDynamicVRAMVAE": module.DisableDynamicVRAMVAE,
    }

    assert module.NODE_CLASS_MAPPINGS == expected
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == set(expected)


def main():
    tests = [
        test_model_node_disables_dynamic_vram_on_output_branch_only,
        test_clip_node_disables_dynamic_vram_and_syncs_output_model_only,
        test_vae_node_clones_container_disables_patcher_and_syncs_model,
        test_node_contracts_are_typed_passthroughs,
        test_node_mapping_exports_all_dynamic_vram_nodes,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} dynamic VRAM node tests")


if __name__ == "__main__":
    main()
