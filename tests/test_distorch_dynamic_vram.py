from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "distorch_compat.py"


def load_module():
    spec = importlib.util.spec_from_file_location("distorch_compat_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeInnerModel:
    pass


class FakePatcher:
    def __init__(self, dynamic=True, model=None):
        self.dynamic = dynamic
        self.model = model if model is not None else FakeInnerModel()
        self.clone_calls = []

    def is_dynamic(self):
        return self.dynamic

    def get_clone_model_override(self):
        return self.model, ("state",)

    def clone(self, disable_dynamic=False, model_override=None):
        self.clone_calls.append({
            "disable_dynamic": disable_dynamic,
            "model_override": model_override,
        })
        model = model_override[0] if model_override is not None else object()
        return FakePatcher(dynamic=self.dynamic and not disable_dynamic, model=model)


class FakeCLIP:
    def __init__(self, patcher):
        self.patcher = patcher
        self.cond_stage_model = patcher.model


class FakeVAE:
    def __init__(self, patcher):
        self.patcher = patcher
        self.first_stage_model = patcher.model


class FakeCLIPVision:
    def __init__(self, patcher):
        self.patcher = patcher
        self.model = patcher.model


class FakeControlNet:
    def __init__(self, patcher):
        self.control_model_wrapped = patcher
        self.control_model = patcher.model

    def copy(self):
        result = FakeControlNet(self.control_model_wrapped)
        result.control_model = self.control_model
        return result


def test_direct_model_patcher_becomes_legacy_without_reloading_model():
    module = load_module()
    source = FakePatcher(dynamic=True)

    output = module.prepare_distorch_object(source)

    assert output is not source
    assert output.is_dynamic() is False
    assert output.model is source.model
    assert source.clone_calls == [{
        "disable_dynamic": True,
        "model_override": source.get_clone_model_override(),
    }]


def test_clip_vae_and_clip_vision_keep_container_in_sync():
    module = load_module()
    cases = [
        (FakeCLIP, "cond_stage_model"),
        (FakeVAE, "first_stage_model"),
        (FakeCLIPVision, "model"),
    ]

    for container_type, model_attribute in cases:
        source = container_type(FakePatcher(dynamic=True))
        output = module.prepare_distorch_object(source)

        assert output is not source
        assert output.patcher.is_dynamic() is False
        assert getattr(output, model_attribute) is output.patcher.model
        assert getattr(output, model_attribute) is source.patcher.model
        assert source.patcher.is_dynamic() is True


def test_controlnet_wrapper_and_executable_model_stay_in_sync():
    module = load_module()
    source = FakeControlNet(FakePatcher(dynamic=True))

    output = module.prepare_distorch_object(source)

    assert output is not source
    assert output.control_model_wrapped.is_dynamic() is False
    assert output.control_model is output.control_model_wrapped.model
    assert output.control_model is source.control_model


def test_prepare_and_mark_outputs_handles_every_model_bearing_output():
    module = load_module()
    source_model = FakePatcher(dynamic=True)
    source_clip = FakeCLIP(FakePatcher(dynamic=True))
    source_vae = FakeVAE(FakePatcher(dynamic=True))
    untouched = {"not": "a model"}
    meta = {"full_allocation": "#cuda:0;4.0;cpu"}

    output = module.prepare_and_mark_distorch_outputs(
        (source_model, source_clip, source_vae, untouched), meta
    )

    assert isinstance(output, tuple)
    assert output[3] is untouched
    for value in output[:3]:
        patcher = module.get_model_patcher(value)
        assert patcher.is_dynamic() is False
        assert patcher._distorch_v2_meta == meta
        assert patcher.model._distorch_v2_meta == meta


def test_zero_allocation_converts_outputs_without_marking_distorch():
    module = load_module()
    source_model = FakePatcher(dynamic=True)
    source_clip = FakeCLIP(FakePatcher(dynamic=True))

    output = module.prepare_and_mark_distorch_outputs(
        (source_model, source_clip), None
    )

    for value in output:
        patcher = module.get_model_patcher(value)
        assert patcher.is_dynamic() is False
        assert not hasattr(patcher, "_distorch_v2_meta")
        assert not hasattr(patcher.model, "_distorch_v2_meta")
    assert source_model.is_dynamic() is True
    assert source_clip.patcher.is_dynamic() is True


def test_non_dynamic_patcher_is_not_cloned_but_is_marked():
    module = load_module()
    source = FakePatcher(dynamic=False)
    meta = {"full_allocation": "#cuda:0;4.0;cpu"}

    output, = module.prepare_and_mark_distorch_outputs((source,), meta)

    assert output is source
    assert source.clone_calls == []
    assert source._distorch_v2_meta == meta
    assert source.model._distorch_v2_meta == meta


def test_distorch_detection_finds_patcher_and_inner_model_metadata():
    module = load_module()
    patcher = FakePatcher(dynamic=False)
    clip = FakeCLIP(patcher)
    assert module.is_distorch_object(clip) is False

    patcher.model._distorch_v2_meta = {"full_allocation": "x"}
    assert module.is_distorch_object(clip) is True


def main():
    tests = [
        test_direct_model_patcher_becomes_legacy_without_reloading_model,
        test_clip_vae_and_clip_vision_keep_container_in_sync,
        test_controlnet_wrapper_and_executable_model_stay_in_sync,
        test_prepare_and_mark_outputs_handles_every_model_bearing_output,
        test_zero_allocation_converts_outputs_without_marking_distorch,
        test_non_dynamic_patcher_is_not_cloned_but_is_marked,
        test_distorch_detection_finds_patcher_and_inner_model_metadata,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} DisTorch DynamicVRAM compatibility tests")


if __name__ == "__main__":
    main()
