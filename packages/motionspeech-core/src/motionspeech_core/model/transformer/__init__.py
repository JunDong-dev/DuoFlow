# Modified for DuoFlow; see NOTICE.md.
"""Transformer model components."""

from motionspeech_core.model.transformer.modality import Modality
from motionspeech_core.model.transformer.model import MotionSpeechModel, X0Model
from motionspeech_core.model.transformer.model_configurator import (
    MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP,
    MOTIONSPEECH_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    MotionSpeechModelConfigurator,
    MotionSpeechVideoOnlyModelConfigurator,
    UpcastWithStochasticRounding,
)

# Public DuoFlow names. The MotionSpeech aliases remain available so released
# checkpoints and existing user scripts continue to load unchanged.
DuoFlowModel = MotionSpeechModel

__all__ = [
    "MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP",
    "MOTIONSPEECH_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP",
    "UPCAST_DURING_INFERENCE",
    "MotionSpeechModel",
    "DuoFlowModel",
    "MotionSpeechModelConfigurator",
    "MotionSpeechVideoOnlyModelConfigurator",
    "Modality",
    "UpcastWithStochasticRounding",
    "X0Model",
]
