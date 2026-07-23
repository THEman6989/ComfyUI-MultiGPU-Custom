"""Branch-local opt-out nodes for ComfyUI DynamicVRAM."""

import copy


def _clone_container_without_dynamic_vram(value, model_attribute):
    """Shallow-copy a model container and replace only its ModelPatcher."""
    result = copy.copy(value)
    patcher = getattr(value, "patcher", None)
    if patcher is not None:
        result.patcher = patcher.clone(disable_dynamic=True)
        setattr(result, model_attribute, result.patcher.model)
    return result


class DisableDynamicVRAMModel:
    """Clone a MODEL branch with ComfyUI's legacy ModelPatcher."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "convert"
    CATEGORY = "multigpu/dynamic_vram"

    def convert(self, model):
        return (model.clone(disable_dynamic=True),)


class DisableDynamicVRAMCLIP:
    """Clone a CLIP branch with ComfyUI's legacy ModelPatcher."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",)}}

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "convert"
    CATEGORY = "multigpu/dynamic_vram"

    def convert(self, clip):
        result = clip.clone(disable_dynamic=True)
        result.cond_stage_model = result.patcher.model
        return (result,)


class DisableDynamicVRAMVAE:
    """Copy a VAE branch and replace its dynamic patcher with a legacy clone."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "convert"
    CATEGORY = "multigpu/dynamic_vram"

    def convert(self, vae):
        return (_clone_container_without_dynamic_vram(vae, "first_stage_model"),)


NODE_CLASS_MAPPINGS = {
    "DisableDynamicVRAMModel": DisableDynamicVRAMModel,
    "DisableDynamicVRAMCLIP": DisableDynamicVRAMCLIP,
    "DisableDynamicVRAMVAE": DisableDynamicVRAMVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DisableDynamicVRAMModel": "Disable DynamicVRAM (MODEL)",
    "DisableDynamicVRAMCLIP": "Disable DynamicVRAM (CLIP)",
    "DisableDynamicVRAMVAE": "Disable DynamicVRAM (VAE)",
}
