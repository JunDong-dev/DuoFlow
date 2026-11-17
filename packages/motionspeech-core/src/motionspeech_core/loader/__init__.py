# Modified for DuoFlow; see NOTICE.md.
"""Loader utilities for model weights, LoRAs, and safetensor operations."""

from motionspeech_core.loader.fuse_loras import apply_loras
from motionspeech_core.loader.module_ops import ModuleOps
from motionspeech_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from motionspeech_core.loader.registry import DummyRegistry, Registry, StateDictRegistry
from motionspeech_core.loader.sd_ops import (
    MOTIONSPEECH_LORA_COMFY_RENAMING_MAP,
    ContentMatching,
    ContentReplacement,
    KeyValueOperation,
    KeyValueOperationResult,
    SDKeyValueOperation,
    SDOps,
)
from motionspeech_core.loader.sft_loader import SafetensorsModelStateDictLoader, SafetensorsStateDictLoader
from motionspeech_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

__all__ = [
    "MOTIONSPEECH_LORA_COMFY_RENAMING_MAP",
    "ContentMatching",
    "ContentReplacement",
    "DummyRegistry",
    "KeyValueOperation",
    "KeyValueOperationResult",
    "LoRAAdaptableProtocol",
    "LoraPathStrengthAndSDOps",
    "LoraStateDictWithStrength",
    "ModelBuilderProtocol",
    "ModuleOps",
    "Registry",
    "SDKeyValueOperation",
    "SDOps",
    "SafetensorsModelStateDictLoader",
    "SafetensorsStateDictLoader",
    "SingleGPUModelBuilder",
    "StateDict",
    "StateDictLoader",
    "StateDictRegistry",
    "apply_loras",
]
