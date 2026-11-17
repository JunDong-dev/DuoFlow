# Modified for DuoFlow; see NOTICE.md.
"""Gemma text encoder components."""

from motionspeech_core.text_encoders.gemma.encoders.av_encoder import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaEncoderOutput,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
)
from motionspeech_core.text_encoders.gemma.encoders.base_encoder import (
    GemmaTextEncoderModelBase,
    encode_text,
    module_ops_from_gemma_root,
)
from motionspeech_core.text_encoders.gemma.encoders.video_only_encoder import (
    VideoGemmaEncoderOutput,
    VideoGemmaTextEncoderModel,
    VideoGemmaTextEncoderModelConfigurator,
)

__all__ = [
    "AV_GEMMA_TEXT_ENCODER_KEY_OPS",
    "AVGemmaEncoderOutput",
    "AVGemmaTextEncoderModel",
    "AVGemmaTextEncoderModelConfigurator",
    "GemmaTextEncoderModelBase",
    "VideoGemmaEncoderOutput",
    "VideoGemmaTextEncoderModel",
    "VideoGemmaTextEncoderModelConfigurator",
    "encode_text",
    "module_ops_from_gemma_root",
]
