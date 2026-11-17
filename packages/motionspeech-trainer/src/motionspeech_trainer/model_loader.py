# Modified for DuoFlow; see NOTICE.md.
# ruff: noqa: PLC0415

"""
Model loader for DuoFlow trainer using the new motionspeech-core package.
This module provides a unified interface for loading DuoFlow model components
for training, using SingleGPUModelBuilder from motionspeech-core.
Example usage:
    # Load individual components
    vae_encoder = load_video_vae_encoder("/path/to/checkpoint.safetensors", device="cuda")
    vae_decoder = load_video_vae_decoder("/path/to/checkpoint.safetensors", device="cuda")
    text_encoder = load_text_encoder("/path/to/checkpoint.safetensors", "/path/to/gemma", device="cuda")
    # Load all components at once
    components = load_model("/path/to/checkpoint.safetensors", text_encoder_path="/path/to/gemma")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch

from motionspeech_trainer import logger

# Type alias for device specification
Device = Union[str, torch.device]

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from motionspeech_core.components.schedulers import MotionSpeechScheduler
    from motionspeech_core.model.audio_vae import AudioDecoder, AudioEncoder, Vocoder
    from motionspeech_core.model.transformer import MotionSpeechModel
    from motionspeech_core.model.video_vae import VideoDecoder, VideoEncoder
    from motionspeech_core.text_encoders.gemma import AVGemmaTextEncoderModel


def _to_torch_device(device: Device) -> torch.device:
    """Convert device specification to torch.device."""
    return torch.device(device) if isinstance(device, str) else device


# =============================================================================
# Individual Component Loaders
# =============================================================================


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    transformer_in_channels: int | None = None,
    transformer_out_channels: int | None = None,
    audio_in_channels: int | None = None,
    transformer_num_layers: int | None = None,
    transformer_num_attention_heads: int | None = None,
    transformer_attention_head_dim: int | None = None,
    transformer_cross_attention_dim: int | None = None,
    audio_num_attention_heads: int | None = None,
    audio_attention_head_dim: int | None = None,
    audio_cross_attention_dim: int | None = None,
    audio_caption_projection_use_skip: bool = True,
    audio_caption_projection_act_fn: str = "gelu_tanh",
    audio_phoneme_caption_projection_act_fn: str = "gelu_tanh",
    audio_audcap_caption_projection_act_fn: str = "gelu_tanh",
    video_caption_projection_use_skip: bool = False,
    video_caption_projection_act_fn: str = "gelu_tanh",
    use_text_cross_attention: bool = True,
    use_phoneme_time_anchor: bool = False,
    aux_ctc_tap_block: int | None = None,
    aux_ctc_tap_blocks: list[int] | None = None,
    init_from_scratch: bool = False,
) -> "MotionSpeechModel":
    """Load the DuoFlow transformer model.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded MotionSpeechModel transformer
    """
    from motionspeech_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from motionspeech_core.model.transformer.model_configurator import (
        MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP,
        MotionSpeechModelConfigurator,
    )

    builder = SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=MotionSpeechModelConfigurator,
        model_sd_ops=MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP,
    )

    if (
        init_from_scratch
        or transformer_in_channels is not None
        or transformer_out_channels is not None
        or audio_in_channels is not None
        or transformer_num_layers is not None
        or transformer_num_attention_heads is not None
        or transformer_attention_head_dim is not None
        or transformer_cross_attention_dim is not None
        or audio_num_attention_heads is not None
        or audio_attention_head_dim is not None
        or audio_cross_attention_dim is not None
    ):
        try:
            config = builder.model_config()
        except (TypeError, KeyError) as exc:
            logger.warning(
                "Transformer checkpoint metadata load failed (%s). Using default config.",
                exc,
            )
            config = {"transformer": {}}
        config_transformer = config.get("transformer", {})
        default_transformer_cfg = {
            "dropout": 0.0,
            "attention_bias": True,
            "num_vector_embeds": None,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
        }
        for key, value in default_transformer_cfg.items():
            config_transformer.setdefault(key, value)
        if transformer_in_channels is not None:
            config_transformer["in_channels"] = transformer_in_channels
        if transformer_out_channels is not None:
            config_transformer["out_channels"] = transformer_out_channels
        if audio_in_channels is not None:
            config_transformer["audio_in_channels"] = audio_in_channels
        if transformer_num_layers is not None:
            config_transformer["num_layers"] = transformer_num_layers
        if transformer_num_attention_heads is not None:
            config_transformer["num_attention_heads"] = transformer_num_attention_heads
        if transformer_attention_head_dim is not None:
            config_transformer["attention_head_dim"] = transformer_attention_head_dim
        if transformer_cross_attention_dim is not None:
            config_transformer["cross_attention_dim"] = transformer_cross_attention_dim
        if audio_num_attention_heads is not None:
            config_transformer["audio_num_attention_heads"] = audio_num_attention_heads
        if audio_attention_head_dim is not None:
            config_transformer["audio_attention_head_dim"] = audio_attention_head_dim
        if audio_cross_attention_dim is not None:
            config_transformer["audio_cross_attention_dim"] = audio_cross_attention_dim
        # Derive cross-attention dims from heads x head_dim when not explicitly set.
        if (
            transformer_num_attention_heads is not None
            or transformer_attention_head_dim is not None
        ) and "cross_attention_dim" not in config_transformer:
            heads = config_transformer.get("num_attention_heads", 32)
            d_head = config_transformer.get("attention_head_dim", 32)
            config_transformer["cross_attention_dim"] = heads * d_head

        if (
            audio_num_attention_heads is not None
            or audio_attention_head_dim is not None
        ) and "audio_cross_attention_dim" not in config_transformer:
            a_heads = config_transformer.get("audio_num_attention_heads", 32)
            a_d_head = config_transformer.get("audio_attention_head_dim", 32)
            config_transformer["audio_cross_attention_dim"] = a_heads * a_d_head
        config_transformer["audio_caption_projection_use_skip"] = audio_caption_projection_use_skip
        config_transformer["audio_caption_projection_act_fn"] = audio_caption_projection_act_fn
        config_transformer["audio_phoneme_caption_projection_act_fn"] = audio_phoneme_caption_projection_act_fn
        config_transformer["audio_audcap_caption_projection_act_fn"] = audio_audcap_caption_projection_act_fn
        config_transformer["video_caption_projection_use_skip"] = video_caption_projection_use_skip
        config_transformer["video_caption_projection_act_fn"] = video_caption_projection_act_fn
        config_transformer["use_text_cross_attention"] = use_text_cross_attention
        config_transformer["use_phoneme_time_anchor"] = use_phoneme_time_anchor
        config_transformer["aux_ctc_tap_block"] = aux_ctc_tap_block
        config_transformer["aux_ctc_tap_blocks"] = aux_ctc_tap_blocks
        # Drop None values to satisfy config validators that expect defaults
        config_transformer = {k: v for k, v in config_transformer.items() if v is not None}
        config["transformer"] = config_transformer

        model = MotionSpeechModelConfigurator.from_config(config)
        # Initialize scale/shift tables created with torch.empty
        for name, param in model.named_parameters():
            if "scale_shift_table" in name:
                torch.nn.init.zeros_(param)
        return model.to(device=_to_torch_device(device), dtype=dtype)

    try:
        return builder.build(device=_to_torch_device(device), dtype=dtype)
    except (TypeError, KeyError) as exc:
        logger.warning(
            "Transformer checkpoint metadata load failed (%s). Using fallback loader.",
            exc,
        )

        config = {"transformer": {}}
        config_transformer = config["transformer"]
        default_transformer_cfg = {
            "dropout": 0.0,
            "attention_bias": True,
            "num_vector_embeds": None,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
        }
        for key, value in default_transformer_cfg.items():
            config_transformer.setdefault(key, value)
        if transformer_in_channels is not None:
            config_transformer["in_channels"] = transformer_in_channels
        if transformer_out_channels is not None:
            config_transformer["out_channels"] = transformer_out_channels
        if audio_in_channels is not None:
            config_transformer["audio_in_channels"] = audio_in_channels
        if transformer_num_layers is not None:
            config_transformer["num_layers"] = transformer_num_layers
        if transformer_num_attention_heads is not None:
            config_transformer["num_attention_heads"] = transformer_num_attention_heads
        if transformer_attention_head_dim is not None:
            config_transformer["attention_head_dim"] = transformer_attention_head_dim
        if transformer_cross_attention_dim is not None:
            config_transformer["cross_attention_dim"] = transformer_cross_attention_dim
        if audio_num_attention_heads is not None:
            config_transformer["audio_num_attention_heads"] = audio_num_attention_heads
        if audio_attention_head_dim is not None:
            config_transformer["audio_attention_head_dim"] = audio_attention_head_dim
        if audio_cross_attention_dim is not None:
            config_transformer["audio_cross_attention_dim"] = audio_cross_attention_dim

        # Derive cross-attention dims from heads x head_dim when not explicitly set.
        if (
            transformer_num_attention_heads is not None
            or transformer_attention_head_dim is not None
        ) and "cross_attention_dim" not in config_transformer:
            heads = config_transformer.get("num_attention_heads", 32)
            d_head = config_transformer.get("attention_head_dim", 32)
            config_transformer["cross_attention_dim"] = heads * d_head

        if (
            audio_num_attention_heads is not None
            or audio_attention_head_dim is not None
        ) and "audio_cross_attention_dim" not in config_transformer:
            a_heads = config_transformer.get("audio_num_attention_heads", 32)
            a_d_head = config_transformer.get("audio_attention_head_dim", 32)
            config_transformer["audio_cross_attention_dim"] = a_heads * a_d_head

        config_transformer["use_text_cross_attention"] = use_text_cross_attention
        config_transformer["use_phoneme_time_anchor"] = use_phoneme_time_anchor
        config_transformer["aux_ctc_tap_block"] = aux_ctc_tap_block
        config_transformer["aux_ctc_tap_blocks"] = aux_ctc_tap_blocks

        model = MotionSpeechModelConfigurator.from_config(config)
        for name, param in model.named_parameters():
            if "scale_shift_table" in name:
                torch.nn.init.zeros_(param)

        from motionspeech_core.loader.sft_loader import SafetensorsStateDictLoader

        loader = SafetensorsStateDictLoader()
        state_dict = loader.load(
            str(checkpoint_path),
            sd_ops=MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP,
            device=_to_torch_device(device),
        ).sd

        target_state = model.state_dict()
        filtered_state = {}
        skipped = 0
        for key, value in state_dict.items():
            target = target_state.get(key)
            if target is None or target.shape != value.shape:
                skipped += 1
                continue
            filtered_state[key] = value

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if missing:
            logger.warning("Transformer missing keys: %s", len(missing))
        if unexpected:
            logger.warning("Transformer unexpected keys: %s", len(unexpected))
        if skipped:
            logger.warning("Transformer skipped mismatched keys: %s", skipped)

        return model.to(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoEncoder":
    """Load the video VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoEncoder
    """
    from motionspeech_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from motionspeech_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoDecoder":
    """Load the video VAE decoder (for inference/validation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoDecoder
    """
    from motionspeech_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from motionspeech_core.model.video_vae import VAE_DECODER_COMFY_KEYS_FILTER, VideoDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioEncoder":
    """Load the audio VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights (default bfloat16, but float32 recommended for quality)
    Returns:
        Loaded AudioEncoder
    """
    from motionspeech_core.loader import SingleGPUModelBuilder
    from motionspeech_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioDecoder":
    """Load the audio VAE decoder.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded AudioDecoder
    """
    from motionspeech_core.loader import SingleGPUModelBuilder
    from motionspeech_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "Vocoder":
    """Load the vocoder (for audio waveform generation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded Vocoder
    """
    from motionspeech_core.loader import SingleGPUModelBuilder
    from motionspeech_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    checkpoint_path: str | Path,
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> "AVGemmaTextEncoderModel":
    """Load the Gemma text encoder.
    Args:
        checkpoint_path: Path to the DuoFlow safetensors checkpoint file
        gemma_model_path: Path to Gemma model directory
        device: Device to load model on
        dtype: Data type for model weights
        load_in_8bit: Whether to load the Gemma model in 8-bit precision using bitsandbytes.
            When True, the model is loaded with device_map="auto" and the device argument
            is ignored for the Gemma backbone (feature extractor still uses dtype).
    Returns:
        Loaded AVGemmaTextEncoderModel
    """
    if not Path(gemma_model_path).is_dir():
        raise ValueError(f"Gemma model path is not a directory: {gemma_model_path}")

    # Use 8-bit loading path if requested
    if load_in_8bit:
        from motionspeech_trainer.gemma_8bit import load_8bit_gemma

        return load_8bit_gemma(checkpoint_path, gemma_model_path, dtype)

    # Standard loading path
    from motionspeech_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from motionspeech_core.text_encoders.gemma.encoders.av_encoder import (
        AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        AVGemmaTextEncoderModelConfigurator,
    )
    from motionspeech_core.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root

    torch_device = _to_torch_device(device)
    text_encoder = SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AVGemmaTextEncoderModelConfigurator,
        model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        module_ops=module_ops_from_gemma_root(str(gemma_model_path)),
    ).build(device=torch_device, dtype=dtype)

    # Current DuoFlow checkpoints keep the trained text adapters in the strategy.
    # Legacy side modules can therefore remain on meta, causing the generic
    # builder to return before moving its separately loaded Gemma backbone.
    # Move that initialized backbone explicitly; keep missing side modules visible.
    if text_encoder.model is not None:
        text_encoder.model = text_encoder.model.to(device=torch_device, dtype=dtype)
    return text_encoder


# =============================================================================
# Combined Component Loader
# =============================================================================


@dataclass
class MotionSpeechModelComponents:
    """Container for all DuoFlow model components."""

    transformer: "MotionSpeechModel"
    video_vae_encoder: "VideoEncoder | None" = None
    video_vae_decoder: "VideoDecoder | None" = None
    audio_vae_decoder: "AudioDecoder | None" = None
    vocoder: "Vocoder | None" = None
    text_encoder: "AVGemmaTextEncoderModel | None" = None
    scheduler: "MotionSpeechScheduler | None" = None


def load_model(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    transformer_in_channels: int | None = None,
    transformer_out_channels: int | None = None,
    audio_in_channels: int | None = None,
    transformer_num_layers: int | None = None,
    transformer_num_attention_heads: int | None = None,
    transformer_attention_head_dim: int | None = None,
    transformer_cross_attention_dim: int | None = None,
    audio_num_attention_heads: int | None = None,
    audio_attention_head_dim: int | None = None,
    audio_cross_attention_dim: int | None = None,
    audio_caption_projection_use_skip: bool = True,
    audio_caption_projection_act_fn: str = "gelu_tanh",
    audio_phoneme_caption_projection_act_fn: str = "gelu_tanh",
    audio_audcap_caption_projection_act_fn: str = "gelu_tanh",
    video_caption_projection_use_skip: bool = False,
    video_caption_projection_act_fn: str = "gelu_tanh",
    use_text_cross_attention: bool = True,
    use_phoneme_time_anchor: bool = False,
    aux_ctc_tap_block: int | None = None,
    aux_ctc_tap_blocks: list[int] | None = None,
    init_transformer_from_scratch: bool = False,
    with_video_vae_encoder: bool = False,
    with_video_vae_decoder: bool = True,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
) -> MotionSpeechModelComponents:
    """
    Load DuoFlow model components from a safetensors checkpoint.
    This is a convenience function that loads multiple components at once.
    For loading individual components, use the dedicated functions:
    - load_transformer()
    - load_video_vae_encoder()
    - load_video_vae_decoder()
    - load_audio_vae_decoder()
    - load_vocoder()
    - load_text_encoder()
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        text_encoder_path: Path to Gemma model directory (required if with_text_encoder=True)
        device: Device to load models on ("cuda", "cpu", etc.)
        dtype: Data type for model weights
        transformer_in_channels: Override transformer video input channels
        transformer_out_channels: Override transformer video output channels
        init_transformer_from_scratch: Initialize transformer without loading checkpoint weights
        with_video_vae_encoder: Whether to load the video VAE encoder (for preprocessing)
        with_video_vae_decoder: Whether to load the video VAE decoder (for inference/validation)
        with_audio_vae_decoder: Whether to load the audio VAE decoder
        with_vocoder: Whether to load the vocoder
        with_text_encoder: Whether to load the text encoder
    Returns:
        MotionSpeechModelComponents containing all loaded model components
    """
    from motionspeech_core.components.schedulers import MotionSpeechScheduler

    if checkpoint_path is None:
        if not init_transformer_from_scratch:
            raise ValueError("model_path is required unless init_transformer_from_scratch is true")
        if with_video_vae_encoder or with_video_vae_decoder or with_audio_vae_decoder or with_vocoder:
            raise ValueError("model_path is required to load VAE/vocoder components")
        checkpoint_path = None
        logger.info("No checkpoint provided; initializing transformer from scratch")
    else:
        checkpoint_path = Path(checkpoint_path)

        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading DuoFlow model from {checkpoint_path}")

    torch_device = _to_torch_device(device)

    # Load transformer
    logger.debug("Loading transformer...")
    if checkpoint_path is None:
        from motionspeech_core.model.transformer.model_configurator import MotionSpeechModelConfigurator

        config = {"transformer": {
            "dropout": 0.0,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
        }}
        config_transformer = config["transformer"]
        if transformer_in_channels is not None:
            config_transformer["in_channels"] = transformer_in_channels
        if transformer_out_channels is not None:
            config_transformer["out_channels"] = transformer_out_channels
        if audio_in_channels is not None:
            config_transformer["audio_in_channels"] = audio_in_channels
        if transformer_num_layers is not None:
            config_transformer["num_layers"] = transformer_num_layers
        if transformer_num_attention_heads is not None:
            config_transformer["num_attention_heads"] = transformer_num_attention_heads
        if transformer_attention_head_dim is not None:
            config_transformer["attention_head_dim"] = transformer_attention_head_dim
        if transformer_cross_attention_dim is not None:
            config_transformer["cross_attention_dim"] = transformer_cross_attention_dim
        if audio_num_attention_heads is not None:
            config_transformer["audio_num_attention_heads"] = audio_num_attention_heads
        if audio_attention_head_dim is not None:
            config_transformer["audio_attention_head_dim"] = audio_attention_head_dim
        if audio_cross_attention_dim is not None:
            config_transformer["audio_cross_attention_dim"] = audio_cross_attention_dim
        config_transformer["audio_caption_projection_use_skip"] = audio_caption_projection_use_skip
        config_transformer["audio_caption_projection_act_fn"] = audio_caption_projection_act_fn
        config_transformer["audio_phoneme_caption_projection_act_fn"] = audio_phoneme_caption_projection_act_fn
        config_transformer["audio_audcap_caption_projection_act_fn"] = audio_audcap_caption_projection_act_fn
        config_transformer["video_caption_projection_use_skip"] = video_caption_projection_use_skip
        config_transformer["video_caption_projection_act_fn"] = video_caption_projection_act_fn
        config_transformer["use_text_cross_attention"] = use_text_cross_attention
        config_transformer["use_phoneme_time_anchor"] = use_phoneme_time_anchor
        config_transformer["aux_ctc_tap_block"] = aux_ctc_tap_block
        config_transformer["aux_ctc_tap_blocks"] = aux_ctc_tap_blocks

        transformer = MotionSpeechModelConfigurator.from_config(config)
        for name, param in transformer.named_parameters():
            if "scale_shift_table" in name:
                torch.nn.init.zeros_(param)
        transformer = transformer.to(device=torch_device, dtype=dtype)
    else:
        transformer = load_transformer(
            checkpoint_path,
            torch_device,
            dtype,
            transformer_in_channels=transformer_in_channels,
            transformer_out_channels=transformer_out_channels,
            audio_in_channels=audio_in_channels,
            transformer_num_layers=transformer_num_layers,
            transformer_num_attention_heads=transformer_num_attention_heads,
            transformer_attention_head_dim=transformer_attention_head_dim,
            transformer_cross_attention_dim=transformer_cross_attention_dim,
            audio_num_attention_heads=audio_num_attention_heads,
            audio_attention_head_dim=audio_attention_head_dim,
            audio_cross_attention_dim=audio_cross_attention_dim,
            audio_caption_projection_use_skip=audio_caption_projection_use_skip,
            audio_caption_projection_act_fn=audio_caption_projection_act_fn,
            audio_phoneme_caption_projection_act_fn=audio_phoneme_caption_projection_act_fn,
            audio_audcap_caption_projection_act_fn=audio_audcap_caption_projection_act_fn,
            video_caption_projection_use_skip=video_caption_projection_use_skip,
            video_caption_projection_act_fn=video_caption_projection_act_fn,
            use_text_cross_attention=use_text_cross_attention,
            use_phoneme_time_anchor=use_phoneme_time_anchor,
            aux_ctc_tap_block=aux_ctc_tap_block,
            aux_ctc_tap_blocks=aux_ctc_tap_blocks,
            init_from_scratch=init_transformer_from_scratch,
        )

    # Load video VAE encoder
    video_vae_encoder = None
    if with_video_vae_encoder:
        logger.debug("Loading video VAE encoder...")
        video_vae_encoder = load_video_vae_encoder(checkpoint_path, torch_device, dtype)

    # Load video VAE decoder
    video_vae_decoder = None
    if with_video_vae_decoder:
        logger.debug("Loading video VAE decoder...")
        video_vae_decoder = load_video_vae_decoder(checkpoint_path, torch_device, dtype)

    # Load audio VAE decoder
    audio_vae_decoder = None
    if with_audio_vae_decoder:
        logger.debug("Loading audio VAE decoder...")
        audio_vae_decoder = load_audio_vae_decoder(checkpoint_path, torch_device, dtype)

    # Load vocoder
    vocoder = None
    if with_vocoder:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(checkpoint_path, torch_device, dtype)

    # Load text encoder
    text_encoder = None
    if with_text_encoder:
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided when with_text_encoder=True")
        logger.debug("Loading Gemma text encoder...")
        text_encoder = load_text_encoder(checkpoint_path, text_encoder_path, torch_device, dtype)

    # Create scheduler (stateless, no loading needed)
    scheduler = MotionSpeechScheduler()

    return MotionSpeechModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
