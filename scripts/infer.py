#!/usr/bin/env python3
# Modified for DuoFlow; see NOTICE.md.
"""DuoFlow inference helper."""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from safetensors.torch import load_file
from transformers import VitsTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages/motionspeech-trainer/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/motionspeech-core/src"))

AUDIO_SAMPLE_RATE = 16000

MOTION_FPS = 30.0
MOTION_LATENT_FPS = 10.0
MOTION_VAE_STRIDE = 3
AUDIO_LATENT_FPS = 25.0
AUDIO_TO_MOTION_RATIO = 2.51


class ReusableVitsTokenizer(VitsTokenizer):
    """Keep one espeak backend per tokenizer instead of loading it per utterance."""

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False,
        normalize: bool | None = None, **kwargs,
    ) -> tuple[str, dict]:
        if not self.phonemize:
            return super().prepare_for_tokenization(
                text, is_split_into_words=is_split_into_words, normalize=normalize, **kwargs,
            )

        from transformers.models.vits import tokenization_vits as vits

        normalize = normalize if normalize is not None else self.normalize
        if normalize:
            text = self.normalize_text(text)
        filtered_text = self._preprocess_char(text)
        if vits.has_non_roman_characters(filtered_text) and self.is_uroman:
            if not vits.is_uroman_available():
                vits.logger.warning(
                    "Text contains non-Roman characters; install uroman to apply Romanization."
                )
            else:
                filtered_text = vits.ur.Uroman().romanize_string(filtered_text)

        if not vits.is_phonemizer_available():
            raise ImportError("Please install the `phonemizer` Python package to use this tokenizer.")
        from phonemizer.backend import EspeakBackend
        from phonemizer.phonemize import _phonemize
        from phonemizer.separator import default_separator

        if not hasattr(self, "_reusable_espeak_backend"):
            self._reusable_espeak_backend = EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True,
            )
        filtered_text = _phonemize(
            self._reusable_espeak_backend, filtered_text, default_separator,
            strip=True, njobs=1, prepend_text=False, preserve_empty_lines=False,
        )
        return re.sub(r"\s+", " ", filtered_text), kwargs


def parse_args():
    """DuoFlow inference helper."""
    parser = argparse.ArgumentParser(
        description="Text-to-Motion+Audio  -  motion  audio"
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=False,
        default=None,
        help="Optional base checkpoint. The trained checkpoint metadata is used when omitted."
    )
    parser.add_argument(
        "--trained-model-path",
        type=str,
        required=True,
        help='Trained DuoFlow checkpoint in safetensors format.'
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        required=True,
        help='Local Gemma text encoder directory.'
    )
    parser.add_argument(
        "--motion-vae-path",
        type=str,
        required=True,
        help='Pretrained motion VAE checkpoint.'
    )
    parser.add_argument(
        "--num-transformer-layers",
        type=int,
        default=None,
        help='Number of final transformer blocks to retain; inferred from the checkpoint when omitted.'    )
    parser.add_argument(
        "--transformer-num-attention-heads",
        type=int,
        default=None,
        help='Override the motion transformer attention head count.'    )
    parser.add_argument(
        "--transformer-attention-head-dim",
        type=int,
        default=None,
        help='Override the motion transformer attention head dimension.'    )
    parser.add_argument(
        "--transformer-cross-attention-dim",
        type=int,
        default=None,
        help='Override the motion cross-attention context dimension.'    )
    parser.add_argument(
        "--audio-num-attention-heads",
        type=int,
        default=None,
        help='Override the audio attention head count.'    )
    parser.add_argument(
        "--audio-attention-head-dim",
        type=int,
        default=None,
        help='Override the audio attention head dimension.'    )
    parser.add_argument(
        "--audio-cross-attention-dim",
        type=int,
        default=None,
        help='Override the audio cross-attention context dimension.'    )
    parser.add_argument(
        "--audio-positional-embedding-max-pos",
        type=int,
        nargs="+",
        default=None,
        help=(
            'Audio RoPE frequency denominator; must match the training configuration.'
        ),
    )
    parser.add_argument(
        "--motion-latent-channels",
        type=int,
        default=256,
        help='Motion latent channel count before any motion projection.'    )
    parser.add_argument(
        "--use-motion-proj",
        action="store_true",
        help='Enable the motion latent projection.'    )
    parser.add_argument(
        "--motion-proj-type",
        type=str,
        default="aria",
        choices=["aria", "linear"],
        help='Motion projection architecture: ARIABridge or linear.'
    )
    parser.add_argument(
        "--audioldm2-model-id",
        type=str,
        default="anhnct/audioldm2_gigaspeech",
        help='AudioLDM2 model ID on Hugging Face.'
    )
    parser.add_argument(
        "--audioldm2-cache-dir",
        type=str,
        default=None,
        help='Local Hugging Face cache directory for AudioLDM2.'
    )
    parser.add_argument(
        "--use-phoneme-encoder",
        action="store_true",
        help='Enable phoneme conditioning through separate audio cross-attention.'    )
    parser.add_argument(
        "--use-connectors",
        action="store_true",
        help='Enable the text embedding connectors.'    )
    parser.add_argument(
        "--direct-text-concat",
        action="store_true",
        help=(
            'Concatenate raw motion and speech text features with the noisy modality tokens.'        ),
    )
    parser.add_argument(
        "--use-phoneme-time-anchor",
        type=int,
        nargs="?",
        const=1,
        default=None,
        help=(
            'Enable (1) or disable (0) phoneme time anchors; inferred from checkpoint keys when omitted.'        ),
    )
    parser.add_argument(
        "--text-conditioning-dim",
        type=int,
        default=3840,
        help='Feature dimension for direct text concatenation.'    )
    parser.add_argument(
        "--gemma-feature-selection",
        choices=("auto", "last_n", "embedding_first_last"),
        default="auto",
        help=(
            "Gemma hidden-state selection. 'auto' reads checkpoint metadata; "
            "embedding_first_last uses embedding output + layer 1 + final layer."
        ),
    )
    parser.add_argument(
        "--motion-text-cross-attention",
        action="store_true",
        help=(
            'Use the motion Text Connector for motion cross-attention alongside direct text concatenation.'        ),
    )
    parser.add_argument(
        "--motion-thinking-tokens",
        type=int,
        default=40,
        help='Number of learnable thinking tokens appended by the motion Text Connector.'    )
    parser.add_argument(
        "--connector-residual",
        action="store_true",
        default=True,
        help='Enable the residual branch in the text connectors.'    )
    parser.add_argument(
        "--no-connector-residual",
        dest="connector_residual",
        action="store_false",
        help='Disable the residual branch in the text connectors.'
    )
    parser.add_argument(
        "--video-use-connector",
        action="store_true",
        default=True,
        dest="video_use_connector",
        help='Use the motion connector when connectors are enabled (default).'    )
    parser.add_argument(
        "--no-video-use-connector",
        dest="video_use_connector",
        action="store_false",
        help='Bypass the motion connector.'    )
    parser.add_argument(
        "--audio-use-connector",
        action="store_true",
        default=True,
        dest="audio_use_connector",
        help='Use the audio connector when connectors are enabled (default).'    )
    parser.add_argument(
        "--no-audio-use-connector",
        dest="audio_use_connector",
        action="store_false",
        help='Bypass the audio connector.'    )
    parser.add_argument(
        "--audio-caption-projection-use-skip",
        action="store_true",
        default=None,
        dest="audio_caption_projection_use_skip",
        help='Enable the audio caption projection residual branch; inferred from checkpoint keys when omitted.'
    )
    parser.add_argument(
        "--no-audio-caption-projection-use-skip",
        dest="audio_caption_projection_use_skip",
        action="store_false",
        help='Disable the audio caption projection residual branch.'
    )
    parser.add_argument(
        "--audio-caption-projection-act-fn",
        type=str,
        default="none",
        choices=["none", "gelu_tanh", "silu"],
        help="Activation function inside audio_caption_projection. "
             "Use 'none' to recover a pure Linear(3840->1024) projection; "
             "matches the training config when act_fn='none' was used."
    )
    parser.add_argument(
        "--video-caption-projection-act-fn",
        type=str,
        default="none",
        choices=["none", "gelu_tanh", "silu"],
        help='Motion caption projection activation; must match training.'    )
    parser.add_argument(
        "--audio-phoneme-caption-projection-act-fn",
        type=str,
        default="none",
        choices=["none", "gelu_tanh", "silu"],
        help="Activation function inside audio_phoneme_caption_projection. "
             "Use 'none' to match training config when act_fn='none' was used."
    )
    parser.add_argument(
        "--use-audcap",
        action="store_true",
        help='Condition audio on Gemma features extracted from the <AUDCAP>...</AUDCAP> prompt tag.'    )
    parser.add_argument(
        "--audio-audcap-caption-projection-act-fn",
        type=str,
        default="none",
        choices=["none", "gelu_tanh", "silu"],
        help="Activation function inside audio_audcap_caption_projection. "
             "Use 'none' to match training config when act_fn='none' was used."
    )
    parser.add_argument(
        "--phoneme-encoder-checkpoint",
        type=str,
        default=None,
        help='AudioLDM2 Vits phoneme encoder directory (text_encoder_2).'
    )
    parser.add_argument(
        "--phoneme-tokenizer-path",
        type=str,
        default=None,
        help='AudioLDM2 Vits tokenizer directory (tokenizer_2).'
    )
    parser.add_argument(
        "--phoneme-pad-length",
        type=int,
        default=500,
        help='Maximum padded phoneme sequence length, including blank tokens.'    )
    parser.add_argument(
        "--phoneme-add-blank",
        action="store_true",
        default=False,
        help='Insert blank tokens (ID 0) between phoneme IDs.'    )
    parser.add_argument(
        "--no-phoneme-add-blank",
        dest="phoneme_add_blank",
        action="store_false",
        help='Disable blank-token insertion.'
    )
    parser.add_argument(
        "--phoneme-pad-token-id",
        type=int,
        default=0,
        help='Phoneme padding token ID.'
    )
    parser.add_argument(
        "--phoneme-source",
        type=str,
        default="quoted",
        choices=["quoted", "speech_tags", "speech_tags_then_quoted", "full"],
        help=(
            "Where to read the phoneme input text from the caption. "
            "'quoted' (default): first \"...\" segment — legacy caption format. "
            "'speech_tags': extract and join every <S>...</E> segment — new caption format. "
            "'speech_tags_then_quoted': try <S>...</E> first, fall back to quoted if absent. "
            "'full': use the entire caption as-is (no extraction)."
        ),
    )
    parser.add_argument(
        "--phoneme-joiner",
        type=str,
        default=" ",
        help=("When --phoneme-source extracts multiple segments, the string used "
              "to join them (default: single space)."),
    )
    parser.add_argument(
        "--phoneme-npy",
        type=str,
        default=None,
        help=(
            "Path to a training-distribution phoneme id .npy (int64, pad id 0, "
            "e.g. <preprocessed_root>/phonemes/<sample_id>.npy). When set, the "
            "ids are loaded directly and --phoneme-source/--phoneme-joiner text "
            "extraction is skipped for the conditional branch — use this to "
            "avoid the espeak-phonemization vs training-tokenization mismatch."
        ),
    )
    parser.add_argument(
        "--motion-source",
        type=str,
        default="l2",
        choices=["l1", "l2", "full"],
        help=(
            "Where the MOTION branch text comes from (video_connector path). "
            "'l2' (default): <L2>...</L2> inner text — long motion description. "
            "'l1': <L1>...</L1> inner text — short motion description. "
            "'full': entire caption as-is (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--speak-source",
        type=str,
        default="speech_tags",
        choices=["speech_tags", "full"],
        help=(
            "Where the AUDIO branch text comes from (audio_connector path). "
            "'speech_tags' (default): first <S>...</E> pair with tags kept — "
            "matches speak_conditions/ preprocessing. "
            "'full': entire caption as-is (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--connector-weights",
        type=str,
        default=None,
        help="Path to text_encoder_connector.pt (legacy, saved during preprocessing)"
    )

    parser.add_argument(
        "--text-feature-mode",
        choices=["gemma_hidden", "embedding_lookup"],
        default="gemma_hidden",
        help=(
            'Use selected Gemma hidden states or token embedding lookup; must match preprocessing.'
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Input caption with motion, speech, and optional audio-description tags.'
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="static, frozen, unnatural, distorted",
        help='Negative caption for classifier-free guidance.'
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=129,
        help='Number of motion latent frames; also determines audio duration.'    )
    parser.add_argument(
        "--audio-only-inference",
        action="store_true",
        help='Generate audio with a one-token dummy motion branch; --num-frames determines audio duration.'    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help='Frame rate used to save and render motion.'
    )
    parser.add_argument(
        "--use-correct-motion-time-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Use motion time intervals [i / fps, (i + 1) / fps] instead of legacy linspace coordinates; must match training.'        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help='Number of denoising steps.'
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=5.0,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--disable-cfg",
        action="store_true",
        help='Disable classifier-free guidance and its unconditional forward pass.'    )
    parser.add_argument(
        "--cfg-guidance-scale-motion",
        type=float,
        default=1.0,
        help='Motion CFG scale; defaults to --cfg-guidance-scale.'    )
    parser.add_argument(
        "--cfg-guidance-scale-audio",
        type=float,
        default=5.0,
        help='Audio CFG scale; defaults to --cfg-guidance-scale.'    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help='Random seed.'
    )
    parser.add_argument(
        "--reference-motion",
        type=str,
        default=None,
        help='Optional reference motion latent file used to infer the sample ID.'    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help='Output video path; motion and audio files are saved alongside it.'
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help='Optional basename shared by motion and audio outputs; defaults to the output path stem.'    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help='Save motion and audio without rendering a video.'    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help='Keep intermediate inference and rendering files.'
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='PyTorch device for inference.'
    )
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help='Disable FP8 quantization.'
    )

    return parser.parse_args()


def load_motion_vae(checkpoint_path: str, device: str) -> torch.nn.Module:
    """DuoFlow inference helper."""
    print("Loading Motion VAE...")

    motion_vae_source = REPO_ROOT / "third_party/motion_vae"
    if str(motion_vae_source) not in sys.path:
        sys.path.insert(0, str(motion_vae_source))

    try:
        import models.tae3_parts as tae3_parts
    except ImportError as e:
        print(f"Could not import Motion VAE: {e}")
        print(f"   Bundled Motion VAE source was not found under {motion_vae_source}")
        raise

    clip_range = [-30, 20]
    motion_vae = tae3_parts.Causal_HumanTAE3parts(
        inputdim=333,  # mofea333 format
        hidden_size=2048,
        down_t=1,
        stride_t=3,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        latent_dim=64,  # per-part latent dim, total=256
        clip_range=clip_range
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')['net']
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    motion_vae.load_state_dict(checkpoint, strict=True)

    motion_vae = motion_vae.to(device)
    motion_vae.eval()

    print("Motion VAE loaded.")
    return motion_vae


def load_motionspeech_models(args, device: str):
    """DuoFlow inference helper."""
    print("\nLoading DuoFlow...")

    from motionspeech_core.loader import SingleGPUModelBuilder
    from motionspeech_trainer.model_loader import load_text_encoder
    from motionspeech_trainer.training_strategies.text_to_motion_audio import ARIABridge
    from motionspeech_trainer.phoneme_encoder import AudioLDM2PhonemeEncoder

    print("  Loading transformer checkpoint...")
    trained_state = load_file(args.trained_model_path)

    transformer_state = {}
    motion_proj_state = {}
    phoneme_proj_state = {}
    feature_extractor_state = {}
    video_connector_state = {}
    audio_connector_state = {}
    video_connector_norm_state = {}
    audio_connector_norm_state = {}

    for key, value in trained_state.items():
        if key.startswith('strategy.'):
            strategy_key = key.replace('strategy.', '')
            if strategy_key.startswith('motion_proj.'):
                motion_proj_state[strategy_key.replace('motion_proj.', '')] = value
            elif strategy_key.startswith('phoneme_proj.'):
                phoneme_proj_state[strategy_key.replace('phoneme_proj.', '')] = value
            elif strategy_key.startswith('feature_extractor.'):
                feature_extractor_state[strategy_key.replace('feature_extractor.', '')] = value
            elif strategy_key.startswith('video_connector_norm.'):
                video_connector_norm_state[strategy_key.replace('video_connector_norm.', '')] = value
            elif strategy_key.startswith('audio_connector_norm.'):
                audio_connector_norm_state[strategy_key.replace('audio_connector_norm.', '')] = value
            elif strategy_key.startswith('video_connector.'):
                video_connector_state[strategy_key.replace('video_connector.', '')] = value
            elif strategy_key.startswith('audio_connector.'):
                audio_connector_state[strategy_key.replace('audio_connector.', '')] = value
        elif key.startswith('motion_proj.'):
            motion_proj_state[key.replace('motion_proj.', '')] = value
        elif key.startswith('phoneme_proj.'):
            phoneme_proj_state[key.replace('phoneme_proj.', '')] = value
        else:
            transformer_state[key] = value

    if feature_extractor_state:
        print(f"    Loaded feature extractor from checkpoint: {len(feature_extractor_state)} tensors")
    if video_connector_state:
        print(f"    Loaded motion connector from checkpoint: {len(video_connector_state)} tensors")
    if audio_connector_state:
        print(f"    Loaded audio connector from checkpoint: {len(audio_connector_state)} tensors")
    if video_connector_norm_state:
        print(f"    Loaded motion connector normalization: {len(video_connector_norm_state)} tensors")
    if audio_connector_norm_state:
        print(f"    Loaded audio connector normalization: {len(audio_connector_norm_state)} tensors")

    from motionspeech_core.model.transformer.model_configurator import (
        MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP,
        MotionSpeechModelConfigurator,
    )
    from motionspeech_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

    base_checkpoint_path = args.checkpoint_path or args.trained_model_path
    builder = SingleGPUModelBuilder(
        model_path=base_checkpoint_path,
        model_class_configurator=MotionSpeechModelConfigurator,
        model_sd_ops=MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP,
    )

    inferred_in_channels = None
    inferred_out_channels = None
    inferred_audio_in_channels = None
    if "patchify_proj.weight" in transformer_state:
        inferred_in_channels = transformer_state["patchify_proj.weight"].shape[1]
    if "proj_out.weight" in transformer_state:
        inferred_out_channels = transformer_state["proj_out.weight"].shape[0]
    if "audio_patchify_proj.weight" in transformer_state:
        inferred_audio_in_channels = transformer_state["audio_patchify_proj.weight"].shape[1]

    try:
        config = builder.model_config()
        if config is None:
            raise ValueError("Empty config metadata")
    except Exception as exc:
        print(f"Warning: Could not read checkpoint metadata; using architecture defaults: {exc}")
        config = {"transformer": {}}

    config_transformer = config.get("transformer", {})
    defaults = {
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
    for key, value in defaults.items():
        if key not in config_transformer or config_transformer[key] is None:
            config_transformer[key] = value
    if inferred_in_channels is not None:
        config_transformer["in_channels"] = int(inferred_in_channels)
    if inferred_out_channels is not None:
        config_transformer["out_channels"] = int(inferred_out_channels)
    if inferred_audio_in_channels is not None:
        config_transformer["audio_in_channels"] = int(inferred_audio_in_channels)
    if args.num_transformer_layers is not None:
        config_transformer["num_layers"] = int(args.num_transformer_layers)
    if args.transformer_num_attention_heads is not None:
        config_transformer["num_attention_heads"] = int(args.transformer_num_attention_heads)
    if args.transformer_attention_head_dim is not None:
        config_transformer["attention_head_dim"] = int(args.transformer_attention_head_dim)
    if args.transformer_cross_attention_dim is not None:
        config_transformer["cross_attention_dim"] = int(args.transformer_cross_attention_dim)
    if args.audio_num_attention_heads is not None:
        config_transformer["audio_num_attention_heads"] = int(args.audio_num_attention_heads)
    if args.audio_attention_head_dim is not None:
        config_transformer["audio_attention_head_dim"] = int(args.audio_attention_head_dim)
    if args.audio_cross_attention_dim is not None:
        config_transformer["audio_cross_attention_dim"] = int(args.audio_cross_attention_dim)
    if args.audio_positional_embedding_max_pos is not None:
        config_transformer["audio_positional_embedding_max_pos"] = list(args.audio_positional_embedding_max_pos)

    _use_skip = args.audio_caption_projection_use_skip
    if _use_skip is None:
        _use_skip = "audio_caption_projection.linear_skip.weight" in transformer_state
        print(f"    ✓ audio_caption_projection_use_skip : {_use_skip}")
    else:
        print(f"    ✓ audio_caption_projection_use_skip (CLI): {_use_skip}")
    config_transformer["audio_caption_projection_use_skip"] = _use_skip

    _act_fn_auto = "none" if "audio_caption_projection.linear_1.weight" not in transformer_state else "gelu_tanh"
    config_transformer["audio_caption_projection_act_fn"] = args.audio_caption_projection_act_fn or _act_fn_auto
    config_transformer["audio_phoneme_caption_projection_act_fn"] = (
        args.audio_phoneme_caption_projection_act_fn or _act_fn_auto
    )
    config_transformer["audio_audcap_caption_projection_act_fn"] = (
        args.audio_audcap_caption_projection_act_fn or _act_fn_auto
    )
    config_transformer["video_caption_projection_act_fn"] = args.video_caption_projection_act_fn
    config_transformer["use_text_cross_attention"] = bool(
        not args.direct_text_concat or args.motion_text_cross_attention
    )
    if getattr(args, "use_phoneme_time_anchor", None) is not None:
        config_transformer["use_phoneme_time_anchor"] = bool(args.use_phoneme_time_anchor)
    else:
        config_transformer["use_phoneme_time_anchor"] = any(
            key.startswith("audio_phoneme_time_anchor.") for key in transformer_state
        )
    print(
        f"    ✓ audio_caption_projection_act_fn={config_transformer['audio_caption_projection_act_fn']} "
        f"audio_phoneme_caption_projection_act_fn={config_transformer['audio_phoneme_caption_projection_act_fn']} "
        f"audio_audcap_caption_projection_act_fn={config_transformer['audio_audcap_caption_projection_act_fn']} "
        f"use_phoneme_time_anchor={config_transformer['use_phoneme_time_anchor']}"
    )

    if args.direct_text_concat:
        expected_motion_in = args.motion_latent_channels + args.text_conditioning_dim
        expected_audio_in = 128 + args.text_conditioning_dim
        if inferred_in_channels != expected_motion_in:
            raise ValueError(
                "motion direct-concat width does not match checkpoint: "
                f"checkpoint={inferred_in_channels}, expected={expected_motion_in} "
                f"({args.motion_latent_channels}+{args.text_conditioning_dim})"
            )
        if inferred_audio_in_channels != expected_audio_in:
            raise ValueError(
                "audio direct-concat width does not match checkpoint: "
                f"checkpoint={inferred_audio_in_channels}, expected={expected_audio_in} "
                f"(128+{args.text_conditioning_dim})"
            )

    if (
        (args.transformer_num_attention_heads is not None or args.transformer_attention_head_dim is not None)
        and "cross_attention_dim" not in config_transformer
    ):
        heads = config_transformer.get("num_attention_heads", 32)
        d_head = config_transformer.get("attention_head_dim", 32)
        config_transformer["cross_attention_dim"] = int(heads) * int(d_head)

    if (
        (args.audio_num_attention_heads is not None or args.audio_attention_head_dim is not None)
        and "audio_cross_attention_dim" not in config_transformer
    ):
        a_heads = config_transformer.get("audio_num_attention_heads", 32)
        a_d_head = config_transformer.get("audio_attention_head_dim", 32)
        config_transformer["audio_cross_attention_dim"] = int(a_heads) * int(a_d_head)
    config["transformer"] = config_transformer

    config_transformer["aux_ctc_tap_block"] = None
    config_transformer["aux_ctc_tap_blocks"] = None
    transformer = MotionSpeechModelConfigurator.from_config(config).to(device=device, dtype=torch.float32)

    original_num_layers = len(transformer.transformer_blocks)
    inferred_layers = None
    if args.num_transformer_layers is None:
        block_indices = []
        for key in transformer_state.keys():
            if key.startswith("transformer_blocks."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    block_indices.append(int(parts[1]))
        if block_indices:
            inferred_layers = max(block_indices) + 1

    target_layers = args.num_transformer_layers or inferred_layers
    if target_layers is not None:
        if target_layers >= original_num_layers:
            print(f"    Keeping all {original_num_layers} transformer blocks.")
        else:
            print(f"    Retaining the final {target_layers} of {original_num_layers} transformer blocks.")
            start_layer = original_num_layers - target_layers
            transformer.transformer_blocks = transformer.transformer_blocks[start_layer:]

    _is_audcap_key = lambda k: "audcap" in k  # noqa: E731
    missing, unexpected = transformer.load_state_dict(transformer_state, strict=False)
    audcap_missing = [k for k in missing if _is_audcap_key(k)]
    other_missing = [k for k in missing if not _is_audcap_key(k)]
    if audcap_missing:
        print(f"    Checkpoint has no AUDCAP weights ({len(audcap_missing)} missing keys). " + ("AUDCAP conditioning is disabled." if not args.use_audcap else "Warning: AUDCAP conditioning is enabled with uninitialized weights."))
    if other_missing:
        preview = ", ".join(other_missing[:5])
        suffix = "..." if len(other_missing) > 5 else ""
        print(f"    ⚠️  Transformer missing keys: {len(other_missing)} ({preview}{suffix})")
    if unexpected:
        preview = ", ".join(unexpected[:5])
        suffix = "..." if len(unexpected) > 5 else ""
        print(f"    ⚠️  Transformer unexpected keys: {len(unexpected)} ({preview}{suffix})")
    transformer.eval()

    motion_proj = None
    if args.use_motion_proj:
        if args.motion_proj_type == "aria":
            print("  - ARIABridge (256→128)...")
            motion_proj = ARIABridge(
                input_dim=256,
                output_dim=128,
                num_heads=2,
                hidden_dim=256,
                causal=True
            ).to(device)
        else:
            print("  - Linear motion proj (256→128)...")
            motion_proj = torch.nn.Linear(256, 128).to(device)

        if motion_proj_state:
            motion_proj.load_state_dict(motion_proj_state, strict=True)
            print("    Motion projection weights loaded.")
        else:
            print("Warning: Motion projection weights are absent; using initialized parameters.")

        motion_proj.eval()

    print("  - Text Encoder (Gemma)...")

    from motionspeech_core.text_encoders.gemma.encoders.av_encoder import (
        AVGemmaTextEncoderModelConfigurator,
    )
    from motionspeech_core.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root

    text_feature_mode = getattr(args, "text_feature_mode", "gemma_hidden")
    if args.motion_text_cross_attention:
        if not args.direct_text_concat:
            raise ValueError("motion_text_cross_attention requires --direct-text-concat")
        if not args.use_connectors or not args.video_use_connector or args.audio_use_connector:
            raise ValueError(
                "hybrid conditioning requires video connector ON and audio connector OFF"
            )
        if text_feature_mode == "embedding_lookup":
            raise ValueError(
                "motion thinking-token cross-attention requires gemma_hidden features; "
                "embedding_lookup clears the trained connector"
            )
    if text_feature_mode == "embedding_lookup":
        print("  ⚙️  text_feature_mode = embedding_lookup:  skip_connectors=True / skip_feature_extractor=True")
        print("    Using Gemma token embeddings without the Gemma transformer, feature extractor, or connectors.")

    te_config = {"transformer": {}}
    connector_config_keys = (
        "connector_num_attention_heads",
        "connector_attention_head_dim",
        "connector_num_layers",
        "connector_ff_mult",
        "connector_input_dim",
        "connector_output_dim",
        "connector_zero_init_output",
        "num_learnable_registers",
    )
    for key in connector_config_keys:
        if key in config_transformer:
            te_config["transformer"][key] = config_transformer[key]
    if args.motion_text_cross_attention:
        te_config["transformer"]["num_learnable_registers"] = args.motion_thinking_tokens
    env_override = os.getenv("MOTIONSPEECH_NUM_LEARNABLE_REGISTERS")
    if env_override is not None:
        lowered = env_override.strip().lower()
        if lowered in {"none", "null", "nil"}:
            te_config["transformer"]["num_learnable_registers"] = None
        else:
            try:
                te_config["transformer"]["num_learnable_registers"] = int(lowered)
            except ValueError:
                pass

    text_encoder = AVGemmaTextEncoderModelConfigurator.from_config(te_config)
    for op in module_ops_from_gemma_root(str(args.text_encoder_path)):
        if op.matcher(text_encoder):
            text_encoder = op.mutator(text_encoder)

    _layers_env = os.getenv("MOTIONSPEECH_GEMMA_NUM_LAYERS")
    inferred_gemma_layers = (
        int(_layers_env)
        if _layers_env
        else int(config_transformer.get("gemma_num_layers", 1))
    )
    if feature_extractor_state:
        fe_weight = feature_extractor_state.get("aggregate_embed.weight")
        if fe_weight is not None:
            inferred_gemma_layers = fe_weight.shape[1] // 3840
    text_encoder.gemma_num_layers = inferred_gemma_layers
    checkpoint_feature_selection = config_transformer.get(
        "gemma_feature_selection", "last_n"
    )
    feature_selection = (
        checkpoint_feature_selection
        if args.gemma_feature_selection == "auto"
        else args.gemma_feature_selection
    )
    if feature_selection == "embedding_first_last" and inferred_gemma_layers != 3:
        raise ValueError(
            "embedding_first_last checkpoint requires a 3-state projection, "
            f"but inferred gemma_num_layers={inferred_gemma_layers}"
        )
    text_encoder.gemma_feature_selection = feature_selection
    print(
        f"    ✓ Gemma feature selection: {feature_selection} "
        f"(states={inferred_gemma_layers})"
    )

    if text_feature_mode == "embedding_lookup":
        inferred_gemma_layers = 1
        text_encoder.gemma_num_layers = 1
        text_encoder.skip_feature_extractor = True
        text_encoder.feature_extractor_linear = None
        text_encoder.skip_connectors = True
        text_encoder.embeddings_connector = None
        text_encoder.audio_embeddings_connector = None
        text_encoder.video_connector_norm = None
        text_encoder.audio_connector_norm = None
        print("    Feature extractor and connectors bypassed for embedding lookup.")
        _skip_connector_load = True
    else:
        _skip_connector_load = False

    if not _skip_connector_load and inferred_gemma_layers > 1 and feature_extractor_state:
        from motionspeech_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
        text_encoder.feature_extractor_linear = GemmaFeaturesExtractorProjLinear(
            num_layers=inferred_gemma_layers
        ).to(dtype=torch.float32)
        missing, unexpected = text_encoder.feature_extractor_linear.load_state_dict(
            feature_extractor_state, strict=True
        )
        text_encoder.skip_feature_extractor = False
        print(
            f"    ✓ Feature extractor ENABLED: Linear({3840 * inferred_gemma_layers}→3840), "
            f"{len(feature_extractor_state)} tensors loaded "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        text_encoder.skip_feature_extractor = True
        print(f"    ✓ Feature extractor BYPASSED (gemma_num_layers={inferred_gemma_layers})")

    connector_loaded = True
    if not _skip_connector_load and args.use_connectors:
        text_encoder.skip_connectors = False
        text_encoder.connector_residual = args.connector_residual
        text_encoder.skip_video_connector = not args.video_use_connector
        text_encoder.skip_audio_connector = not args.audio_use_connector
        connector_loaded = False
        if video_connector_state:
            register_weight = video_connector_state.get("learnable_registers")
            if args.motion_text_cross_attention:
                if register_weight is None:
                    raise ValueError(
                        "motion_text_cross_attention requires checkpoint thinking-token weights"
                    )
                if register_weight.shape[0] != args.motion_thinking_tokens:
                    raise ValueError(
                        "motion thinking-token K mismatch: "
                        f"checkpoint={register_weight.shape[0]}, requested={args.motion_thinking_tokens}"
                    )
            missing_v, unexpected_v = text_encoder.embeddings_connector.load_state_dict(
                video_connector_state, strict=True
            )
            print(
                f"    ✓ Video connector : {len(video_connector_state)} tensors "
                f"(missing={len(missing_v)}, unexpected={len(unexpected_v)})"
                f"{' [SKIPPED at forward]' if not args.video_use_connector else ''}"
            )
            connector_loaded = True
        else:
            print("    Motion connector weights are absent; using initialized parameters.")
        if audio_connector_state:
            missing_a, unexpected_a = text_encoder.audio_embeddings_connector.load_state_dict(
                audio_connector_state, strict=True
            )
            print(
                f"    ✓ Audio connector : {len(audio_connector_state)} tensors "
                f"(missing={len(missing_a)}, unexpected={len(unexpected_a)})"
                f"{' [SKIPPED at forward]' if not args.audio_use_connector else ''}"
            )
            connector_loaded = True
        else:
            print("    Audio connector weights are absent; using initialized parameters.")
        if video_connector_norm_state:
            text_encoder.video_connector_norm.load_state_dict(video_connector_norm_state, strict=True)
            print(f"    ✓ Video connector norm : {len(video_connector_norm_state)} tensors")
        if audio_connector_norm_state:
            text_encoder.audio_connector_norm.load_state_dict(audio_connector_norm_state, strict=True)
            print(f"    ✓ Audio connector norm : {len(audio_connector_norm_state)} tensors")
        if not connector_loaded:
            print("Warning: Connectors are enabled but the checkpoint contains no connector weights.")
        _parts = []
        if args.video_use_connector:
            _parts.append("video=ON")
        else:
            _parts.append("video=BYPASS")
        if args.audio_use_connector:
            _parts.append("audio=ON")
        else:
            _parts.append("audio=BYPASS")
        print(f"    ✓ Connectors ENABLED ({', '.join(_parts)}, gemma_num_layers={inferred_gemma_layers})")
    elif not _skip_connector_load:
        text_encoder.skip_connectors = True
        connector_loaded = True  # Skip connector loading when bypassed
        print(f"    ✓ Feature extractor & connectors BYPASSED (gemma_num_layers={inferred_gemma_layers})")
        print(f"      Gemma 3840-dim output → caption_projection (in transformer) → cross-attention")

    if not connector_loaded:
        old_connector_path = None
        if hasattr(args, 'connector_weights') and args.connector_weights:
            old_connector_path = args.connector_weights
        else:
            for candidate in [
                Path(args.trained_model_path).parent.parent / "text_encoder_connector.pt",
                Path(".") / "text_encoder_connector.pt",
            ]:
                if candidate.is_file():
                    old_connector_path = str(candidate)
                    break

        if old_connector_path and Path(old_connector_path).is_file():
            print(f"    Loading legacy connector weights: {old_connector_path}")
            cs = torch.load(old_connector_path, map_location=device, weights_only=True)
            text_encoder.load_state_dict(cs, strict=True)
            connector_loaded = True

    if not connector_loaded:
        print("Warning: No trained connector weights were loaded; verify the selected text feature mode.")

    text_encoder = text_encoder.to(device=device, dtype=torch.float32)
    text_encoder.eval()

    phoneme_encoder = None
    phoneme_tokenizer = None
    phoneme_proj = None
    if args.use_phoneme_encoder:
        if not args.phoneme_encoder_checkpoint or not args.phoneme_tokenizer_path:
            raise ValueError("use-phoneme-encoder requires phoneme-encoder-checkpoint and phoneme-tokenizer-path")
        print("  - Phoneme Encoder (AudioLDM2 Vits)...")
        phoneme_encoder = AudioLDM2PhonemeEncoder(
            args.phoneme_encoder_checkpoint,
            pad_token_id=args.phoneme_pad_token_id,
            add_blank=args.phoneme_add_blank,
        )
        phoneme_encoder = phoneme_encoder.to(device=device, dtype=torch.float32)
        phoneme_encoder.eval()
        phoneme_tokenizer = ReusableVitsTokenizer.from_pretrained(args.phoneme_tokenizer_path)
        if hasattr(phoneme_tokenizer, "phonemize"):
            phoneme_tokenizer.phonemize = True
        phoneme_tokenizer.add_blank = args.phoneme_add_blank
        phoneme_tokenizer.model_max_length = 99999

        if phoneme_proj_state:
            has_aria_keys = any(k.startswith("input_proj.") or k.startswith("self_attn.") for k in phoneme_proj_state)
            if has_aria_keys:
                input_proj_w = phoneme_proj_state["input_proj.weight"]
                input_dim = input_proj_w.shape[1]
                output_dim = input_proj_w.shape[0]
                ffn_w = phoneme_proj_state.get("ffn.0.weight")
                hidden_dim = int(ffn_w.shape[0]) if ffn_w is not None else output_dim // 4
                num_heads = phoneme_proj_state["self_attn.num_heads"] if "self_attn.num_heads" in phoneme_proj_state else 4
                if isinstance(num_heads, torch.Tensor):
                    num_heads = int(num_heads.item())
                phoneme_proj = ARIABridge(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    causal=False,
                )
                tensor_state = {k: v for k, v in phoneme_proj_state.items() if isinstance(v, torch.Tensor)}
                phoneme_proj.load_state_dict(tensor_state, strict=True)
                print(f"    phoneme_proj: ARIABridge(input={input_dim}, output={output_dim}, hidden={hidden_dim})")
            else:
                weight = phoneme_proj_state.get("weight")
                bias = phoneme_proj_state.get("bias")
                if weight is None:
                    raise ValueError("phoneme_proj weights not found in checkpoint")
                phoneme_proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
                phoneme_proj.load_state_dict(phoneme_proj_state, strict=True)
                print(f"    phoneme_proj: Linear({weight.shape[1]}, {weight.shape[0]})")
            phoneme_proj = phoneme_proj.to(device=device, dtype=torch.float32)
            phoneme_proj.eval()
        else:
            print("Warning: No phoneme projection weights were found in the checkpoint.")

    print("  - AudioLDM2 VAE + Vocoder...")
    from diffusers import AutoencoderKL
    from transformers import SpeechT5HifiGan

    from model_resources import local_model_path
    model_path = local_model_path(args.audioldm2_model_id, args.audioldm2_cache_dir)
    options = dict(torch_dtype=torch.float32, cache_dir=args.audioldm2_cache_dir)
    audio_vae_decoder = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", **options
    ).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(
        model_path, subfolder="vocoder", **options
    ).to(device).eval()

    audio_sample_rate = getattr(vocoder.config, "sampling_rate", 16000)
    hop_length = getattr(vocoder.config, "hop_size", None)
    if hop_length is None and hasattr(vocoder.config, "upsample_rates"):
        hop_length = 1
        for rate in vocoder.config.upsample_rates:
            hop_length *= rate
    if hop_length is None:
        hop_length = 160

    mel_bins = getattr(vocoder.config, "num_mel_bins", None)
    if mel_bins is None:
        mel_bins = getattr(vocoder.config, "n_mel_channels", 64)

    print("DuoFlow components loaded.")

    return {
        'transformer': transformer,
        'text_encoder': text_encoder,
        'phoneme_encoder': phoneme_encoder,
        'phoneme_tokenizer': phoneme_tokenizer,
        'phoneme_pad_length': args.phoneme_pad_length,
        'phoneme_pad_token_id': args.phoneme_pad_token_id,
        'phoneme_add_blank': args.phoneme_add_blank,
        'phoneme_proj': phoneme_proj,
        'audio_vae_decoder': audio_vae_decoder,
        'vocoder': vocoder,
        'audio_sample_rate': audio_sample_rate,
        'audio_hop_length': hop_length,
        'audio_mel_bins': mel_bins,
        'motion_proj': motion_proj,
        'text_feature_mode': args.text_feature_mode,
    }


def _extract_quoted_text(text: str) -> str:
    """Return the first double-quoted segment, or the full text if none.

    Legacy caption format: text is wrapped in double quotes, e.g. `[wayne]: "hello"`.
    """
    if not text:
        return text
    parts = text.split('"')
    if len(parts) >= 3:
        return parts[1].strip()
    return text


SPEECH_TAG_OPEN = "<S>"
SPEECH_TAG_CLOSE = "</E>"

SPEECH_PAIR_RE = re.compile(r"<S>(.*?)</E>", flags=re.DOTALL)

AUDCAP_TAG_RE = None  # lazily compiled


def _extract_audcap_text(text: str) -> str | None:
    """Extract the inner text of the first ``<AUDCAP>...</AUDCAP>`` tag.

    Mirrors scripts/preprocess_audcaps.py:extract_audcap. Returns None when
    the tag is absent or the inner text is empty/whitespace.
    """
    global AUDCAP_TAG_RE
    if not text:
        return None
    if AUDCAP_TAG_RE is None:
        import re
        AUDCAP_TAG_RE = re.compile(r"<AUDCAP>(.*?)</AUDCAP>", flags=re.DOTALL)
    match = AUDCAP_TAG_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()
    return inner or None


def _extract_speech_segments(text: str, joiner: str = " ") -> str:
    """Extract and join all `<S>...</E>` segments in `text`.

    New caption format: speech is wrapped in `<S>...</E>` tags, e.g.::

        ... <S>pronounciating something correctly ... and so</E> ... motion ...
        <S>second utterance here</E> ...
        <AUDCAP>Male voice, conversational</AUDCAP>

    If no `<S>` tag is found, returns the input text unchanged (graceful fallback
    so legacy prompts still work — they will be phonemized as-is).
    Unterminated tags take everything after `<S>` rather than dropping the data.
    """
    if not text:
        return text
    if SPEECH_TAG_OPEN not in text:
        return text.strip()
    segments: list[str] = []
    cursor = 0
    while True:
        start = text.find(SPEECH_TAG_OPEN, cursor)
        if start == -1:
            break
        end = text.find(SPEECH_TAG_CLOSE, start)
        if end == -1:
            seg = text[start + len(SPEECH_TAG_OPEN):].strip()
            if seg:
                segments.append(seg)
            break
        seg = text[start + len(SPEECH_TAG_OPEN):end].strip()
        if seg:
            segments.append(seg)
        cursor = end + len(SPEECH_TAG_CLOSE)
    if not segments:
        return text.strip()
    return joiner.join(segments)


def _extract_l_tag(text: str, tag: str) -> str | None:
    """Extract the inner text of the first ``<L1>...</L1>`` / ``<L2>...</L2>`` tag.

    Return None when the tag is absent or its contents are empty.
    """
    if not text:
        return None
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not match:
        return None
    inner = match.group(1).strip()
    return inner or None


def _extract_speech_with_tags(text: str) -> str | None:
    """Return the first ``<S>...</E>`` pair INCLUDING the tags themselves.

    the tags are kept so tokenization sees the same sequence as training
    preprocessing (`speak_conditions/`). Returns None when absent/empty.
    """
    if not text:
        return None
    match = SPEECH_PAIR_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()
    if not inner:
        return None
    return f"{SPEECH_TAG_OPEN} {inner} {SPEECH_TAG_CLOSE}"


def _extract_phoneme_text(
    text: str,
    mode: str = "quoted",
    joiner: str = " ",
) -> str:
    """Dispatch helper: pick the right extractor for the phoneme input.

    Args:
        text: caption to extract from.
        mode:
          - ``"quoted"``: legacy behavior — extract the first ``"..."`` segment.
          - ``"speech_tags"``: extract and join every ``<S>...</E>`` segment.
          - ``"speech_tags_then_quoted"``: try speech_tags first, fall back to quoted.
            Useful when the prompt may mix both formats.
          - ``"full"``: use the whole caption as-is (no extraction).
        joiner: used by ``speech_tags`` to combine multiple segments.
    """
    if mode == "quoted":
        return _extract_quoted_text(text)
    if mode == "speech_tags":
        return _extract_speech_segments(text, joiner=joiner)
    if mode == "speech_tags_then_quoted":
        extracted = _extract_speech_segments(text, joiner=joiner)
        if SPEECH_TAG_OPEN not in text:
            return _extract_quoted_text(text)
        return extracted
    if mode == "full":
        return text.strip()
    raise ValueError(f"Unknown phoneme-source mode: {mode!r}")


def _encode_text_features(
    text: str,
    text_encoder,
    mode: str,
    *,
    apply_connectors: bool = True,
) -> "AVGemmaEncoderOutput":
    """Encode text using the configured text_feature_mode.

    Modes:
      - "gemma_hidden": standard path — run Gemma transformer, take last N layers
        hidden states (or single-layer bypass). Returns AVGemmaEncoderOutput with
        video_encoding/audio_encoding of shape (1, seq, 3840).
      - "embedding_lookup": bypass the transformer entirely — use Gemma embed_tokens
        lookup only, then apply the same per-batch normalization used in preprocessing.
        Returns AVGemmaEncoderOutput with video_encoding == audio_encoding == the
        normalized token embeddings (1, seq, 3840). Connectors are not supported
        in this legacy lookup-only mode.
    """
    from motionspeech_core.text_encoders.gemma.encoders.av_encoder import AVGemmaEncoderOutput

    if mode == "embedding_lookup":
        from motionspeech_core.text_encoders.gemma.encoders.base_encoder import (
            _norm_and_concat_padded_batch,
        )

        token_pairs = text_encoder.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor(
            [[t[0] for t in token_pairs]],
            device=text_encoder.model.device,
        )
        attention_mask = torch.tensor(
            [[w[1] for w in token_pairs]],
            device=text_encoder.model.device,
        )
        embed_layer = text_encoder.model.get_input_embeddings()
        token_embeds = embed_layer(input_ids)
        encoded = _norm_and_concat_padded_batch(
            token_embeds.unsqueeze(-1),
            attention_mask.sum(dim=-1),
            padding_side="left",
        )  # (1, seq, 3840)
        binary_mask = attention_mask.unsqueeze(-1)
        encoded = encoded * binary_mask
        return AVGemmaEncoderOutput(encoded, encoded.clone(), attention_mask)

    if apply_connectors:
        return text_encoder(text)
    encoded, attention_mask = text_encoder._preprocess_text(text)
    return AVGemmaEncoderOutput(encoded, encoded.clone(), attention_mask)


def _prepare_direct_text_condition(
    raw_features: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    target_seq_len: int,
    feature_dim: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Match training: compact valid Gemma tokens, then right-pad/truncate."""
    if raw_features.dim() == 2:
        raw_features = raw_features.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if raw_features.dim() != 3 or attention_mask.dim() != 2:
        raise ValueError(
            "direct text concat expects [B,S,D] features and [B,S] mask, got "
            f"{tuple(raw_features.shape)} and {tuple(attention_mask.shape)}"
        )
    if raw_features.shape[:2] != attention_mask.shape:
        raise ValueError("direct text feature/mask sequence dimensions disagree")
    if raw_features.shape[-1] != feature_dim:
        raise ValueError(
            f"direct text width {raw_features.shape[-1]} does not match expected {feature_dim}"
        )

    raw_features = raw_features.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    batch_size = raw_features.shape[0]
    compacted = torch.zeros_like(raw_features)
    for batch_idx in range(batch_size):
        valid = raw_features[batch_idx, attention_mask[batch_idx]]
        valid_count = valid.shape[0]
        if valid_count:
            compacted[batch_idx, :valid_count] = valid

    if compacted.shape[1] >= target_seq_len:
        return compacted[:, :target_seq_len]
    padding = torch.zeros(
        batch_size,
        target_seq_len - compacted.shape[1],
        feature_dim,
        device=device,
        dtype=dtype,
    )
    return torch.cat((compacted, padding), dim=1)


def _load_phoneme_npy(path: str, pad_length: int, device: str) -> torch.Tensor:
    """Load training-distribution phoneme ids from a dataset .npy.

    The dataset phonemes are CMU-annotation-derived ids (built with
    ``tokenizer.phonemize = False``); loading them directly bypasses the
    espeak text phonemization whose symbol sequence differs from training.
    """
    ids = np.asarray(np.load(path), dtype=np.int64).reshape(1, -1)
    ids = torch.from_numpy(ids)
    if ids.shape[1] > pad_length:
        ids = ids[:, :pad_length]
    elif ids.shape[1] < pad_length:
        ids = torch.cat([ids, torch.zeros((1, pad_length - ids.shape[1]), dtype=ids.dtype)], dim=1)
    return ids.to(device=device)


def build_phoneme_ids(text: str, tokenizer: VitsTokenizer, pad_length: int, device: str) -> torch.Tensor:
    """Tokenize text to phoneme IDs using VitsTokenizer (add_blank handled natively by tokenizer)."""
    raw_encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    raw_ids = raw_encoded["input_ids"]  # (1, T) — already includes blanks if tokenizer.add_blank=True
    effective_len = raw_ids.shape[1]

    truncated = effective_len > pad_length
    print(f"    phoneme_len={effective_len} add_blank={tokenizer.add_blank} pad_length={pad_length} truncated={truncated}")

    if effective_len > pad_length:
        raw_ids = raw_ids[:, :pad_length]
    elif effective_len < pad_length:
        padding = torch.zeros((1, pad_length - effective_len), dtype=raw_ids.dtype, device=raw_ids.device)
        raw_ids = torch.cat([raw_ids, padding], dim=1)

    return raw_ids.to(device=device)


@torch.inference_mode()
def generate_from_text(
    prompt: str,
    negative_prompt: str,
    models: dict,
    args,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """DuoFlow inference helper."""
    print("\nGenerating motion and audio latents...")

    from motionspeech_core.components.diffusion_steps import EulerDiffusionStep
    from motionspeech_core.components.guiders import CFGGuider
    from motionspeech_core.components.noisers import GaussianNoiser
    from motionspeech_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier
    from motionspeech_core.components.schedulers import MotionSpeechScheduler
    from motionspeech_core.model.transformer.modality import Modality
    from motionspeech_core.types import LatentState, VideoLatentShape, AudioLatentShape

    transformer = models['transformer']
    text_encoder = models['text_encoder']
    motion_proj = models['motion_proj']
    phoneme_encoder = models.get('phoneme_encoder')
    phoneme_tokenizer = models.get('phoneme_tokenizer')
    phoneme_pad_length = models.get('phoneme_pad_length')
    phoneme_proj = models.get('phoneme_proj')
    phoneme_pad_token_id = models.get('phoneme_pad_token_id', 0)

    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    print("  Encoding text conditions...")
    text_feature_mode = models.get('text_feature_mode', 'gemma_hidden')
    print(f"    text_feature_mode = {text_feature_mode}")

    motion_source = getattr(args, "motion_source", "full")
    speak_source = getattr(args, "speak_source", "full")
    if motion_source in ("l1", "l2") or speak_source == "speech_tags":
        motion_text_pos = _extract_l_tag(prompt, motion_source.upper()) if motion_source in ("l1", "l2") else prompt.strip()
        speak_text_pos = _extract_speech_with_tags(prompt) if speak_source == "speech_tags" else prompt.strip()
        if motion_text_pos is None:
            print(f"Warning: Prompt has no <{motion_source.upper()}>...</{motion_source.upper()}> tag; using the full caption for motion.")
            motion_text_pos = prompt.strip()
        if speak_text_pos is None:
            print("Warning: Prompt has no <S>...</E> tag; using the full caption for speech.")
            speak_text_pos = prompt.strip()
    else:
        motion_text_pos = prompt.strip()
        speak_text_pos = prompt.strip()
    print(f"    motion_source={motion_source} speak_source={speak_source}")

    positive_output = _encode_text_features(
        motion_text_pos,
        text_encoder,
        text_feature_mode,
        apply_connectors=False,
    )
    motion_concat_raw_positive = positive_output.video_encoding
    motion_concat_raw_mask_pos = positive_output.attention_mask
    same_positive_text = motion_text_pos == speak_text_pos
    audio_output = positive_output if same_positive_text else _encode_text_features(
        speak_text_pos,
        text_encoder,
        text_feature_mode,
        apply_connectors=False,
    )
    audio_concat_raw_positive = audio_output.audio_encoding
    audio_concat_raw_mask_pos = audio_output.attention_mask

    def _apply_modality_connectors(
        motion_raw,
        motion_mask,
        audio_raw,
        audio_mask,
        *,
        same_text: bool,
    ):
        video_context = motion_raw
        video_mask = motion_mask
        audio_context = audio_raw
        audio_context_mask = audio_mask
        if not args.use_connectors:
            return video_context, video_mask, audio_context, audio_context_mask

        if same_text:
            connected_video, connected_audio, connected_mask = text_encoder._run_connectors(
                motion_raw,
                motion_mask,
            )
            if args.video_use_connector:
                video_context, video_mask = connected_video, connected_mask
            if args.audio_use_connector:
                audio_context, audio_context_mask = connected_audio, connected_mask
            return video_context, video_mask, audio_context, audio_context_mask

        if args.video_use_connector:
            connected_video, _, connected_mask = text_encoder._run_connectors(
                motion_raw,
                motion_mask,
            )
            video_context, video_mask = connected_video, connected_mask
        if args.audio_use_connector:
            _, connected_audio, connected_mask = text_encoder._run_connectors(
                audio_raw,
                audio_mask,
            )
            audio_context, audio_context_mask = connected_audio, connected_mask
        return video_context, video_mask, audio_context, audio_context_mask

    (
        video_context_positive,
        prompt_attention_mask_pos,
        audio_context_positive,
        audio_attention_mask_pos,
    ) = _apply_modality_connectors(
        motion_concat_raw_positive,
        motion_concat_raw_mask_pos,
        audio_concat_raw_positive,
        audio_concat_raw_mask_pos,
        same_text=same_positive_text,
    )

    if negative_prompt:
        motion_text_neg = _extract_l_tag(negative_prompt, motion_source.upper()) if motion_source in ("l1", "l2") else negative_prompt.strip()
        speak_text_neg = _extract_speech_with_tags(negative_prompt) if speak_source == "speech_tags" else negative_prompt.strip()
        if motion_text_neg is None:
            motion_text_neg = negative_prompt.strip()
        if speak_text_neg is None:
            speak_text_neg = negative_prompt.strip()
        negative_output = _encode_text_features(
            motion_text_neg,
            text_encoder,
            text_feature_mode,
            apply_connectors=False,
        )
        motion_concat_raw_negative = negative_output.video_encoding
        motion_concat_raw_mask_neg = negative_output.attention_mask
        same_negative_text = motion_text_neg == speak_text_neg
        negative_audio_output = negative_output if same_negative_text else _encode_text_features(
            speak_text_neg,
            text_encoder,
            text_feature_mode,
            apply_connectors=False,
        )
        audio_concat_raw_negative = negative_audio_output.audio_encoding
        audio_concat_raw_mask_neg = negative_audio_output.attention_mask
        (
            video_context_negative,
            prompt_attention_mask_neg,
            audio_context_negative,
            audio_attention_mask_neg,
        ) = _apply_modality_connectors(
            motion_concat_raw_negative,
            motion_concat_raw_mask_neg,
            audio_concat_raw_negative,
            audio_concat_raw_mask_neg,
            same_text=same_negative_text,
        )
    else:
        motion_concat_raw_negative = torch.zeros_like(motion_concat_raw_positive)
        motion_concat_raw_mask_neg = torch.zeros_like(motion_concat_raw_mask_pos)
        audio_concat_raw_negative = torch.zeros_like(audio_concat_raw_positive)
        audio_concat_raw_mask_neg = torch.zeros_like(audio_concat_raw_mask_pos)
        video_context_negative = torch.zeros_like(video_context_positive)
        audio_context_negative = torch.zeros_like(audio_context_positive)
        prompt_attention_mask_neg = torch.zeros_like(prompt_attention_mask_pos) if prompt_attention_mask_pos is not None else None
        audio_attention_mask_neg = torch.zeros_like(audio_attention_mask_pos) if audio_attention_mask_pos is not None else None

    if video_context_negative.dim() == 4 and video_context_negative.size(1) == 1:
        video_context_negative = video_context_negative.squeeze(1)
    if audio_context_negative.dim() == 4 and audio_context_negative.size(1) == 1:
        audio_context_negative = audio_context_negative.squeeze(1)
    if prompt_attention_mask_neg is not None and prompt_attention_mask_neg.dim() == 3 and prompt_attention_mask_neg.size(1) == 1:
        prompt_attention_mask_neg = prompt_attention_mask_neg.squeeze(1)
    if audio_attention_mask_pos is not None and audio_attention_mask_pos.dim() == 3 and audio_attention_mask_pos.size(1) == 1:
        audio_attention_mask_pos = audio_attention_mask_pos.squeeze(1)
    if audio_attention_mask_neg is not None and audio_attention_mask_neg.dim() == 3 and audio_attention_mask_neg.size(1) == 1:
        audio_attention_mask_neg = audio_attention_mask_neg.squeeze(1)

    if prompt_attention_mask_pos is None:
        prompt_attention_mask_pos = torch.ones(
            (1, video_context_positive.shape[1]),
            device=video_context_positive.device,
            dtype=torch.int64,
        )
    if prompt_attention_mask_neg is None:
        prompt_attention_mask_neg = torch.ones(
            (1, video_context_negative.shape[1]),
            device=video_context_negative.device,
            dtype=torch.int64,
        )
    if audio_attention_mask_pos is None:
        audio_attention_mask_pos = torch.ones(
            (1, audio_context_positive.shape[1]),
            device=audio_context_positive.device,
            dtype=torch.int64,
        )
    if audio_attention_mask_neg is None:
        audio_attention_mask_neg = torch.ones(
            (1, audio_context_negative.shape[1]),
            device=audio_context_negative.device,
            dtype=torch.int64,
        )

    if prompt_attention_mask_pos.dtype != torch.int64:
        prompt_attention_mask_pos = prompt_attention_mask_pos.to(dtype=torch.int64)
    if prompt_attention_mask_neg.dtype != torch.int64:
        prompt_attention_mask_neg = prompt_attention_mask_neg.to(dtype=torch.int64)
    if audio_attention_mask_pos.dtype != torch.int64:
        audio_attention_mask_pos = audio_attention_mask_pos.to(dtype=torch.int64)
    if audio_attention_mask_neg.dtype != torch.int64:
        audio_attention_mask_neg = audio_attention_mask_neg.to(dtype=torch.int64)

    motion_context_mask_pos = prompt_attention_mask_pos
    motion_context_mask_neg = prompt_attention_mask_neg
    audio_context_mask_pos = audio_attention_mask_pos
    audio_context_mask_neg = audio_attention_mask_neg

    def _get_token_mask(text: str) -> torch.Tensor | None:
        try:
            tokenizer = text_encoder.tokenizer.tokenizer
            max_len = text_encoder.tokenizer.max_length
            tokenized = tokenizer(
                text.strip(),
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            return tokenized.attention_mask.to(device=device, dtype=torch.bool)
        except Exception as exc:
            print(f"Warning: Could not create the text token mask: {exc}")
            return None

    phoneme_context_pos = None
    phoneme_context_mask_pos = None
    phoneme_context_neg = None
    phoneme_context_mask_neg = None
    if phoneme_encoder is not None and phoneme_tokenizer is not None and phoneme_proj is not None:
        print("  - Phoneme conditioning enabled (separate cross-attention): encoding text -> phoneme IDs")
        print(f"    tokenizer={phoneme_tokenizer.__class__.__name__} phonemize={getattr(phoneme_tokenizer, 'phonemize', None)}")
        print(f"    phoneme_source={args.phoneme_source}"
              + (f" (joiner={args.phoneme_joiner!r})" if args.phoneme_source in ("speech_tags", "speech_tags_then_quoted") else ""))
        if args.phoneme_npy:
            phoneme_text_pos = f"<npy:{args.phoneme_npy}>"
            phoneme_idx_pos = _load_phoneme_npy(args.phoneme_npy, phoneme_pad_length, device)
            print(f"    phoneme source: GT npy (training-distribution ids, text phonemization skipped)")
        else:
            phoneme_text_pos = _extract_phoneme_text(prompt, mode=args.phoneme_source, joiner=args.phoneme_joiner)
            phoneme_idx_pos = build_phoneme_ids(phoneme_text_pos, phoneme_tokenizer, phoneme_pad_length, device)
        phoneme_embeds_pos, phoneme_mask_pos = phoneme_encoder(phoneme_idx_pos, None)
        target_dtype = audio_context_positive.dtype
        phoneme_embeds_pos = phoneme_embeds_pos.to(dtype=target_dtype)
        phoneme_proj = phoneme_proj.to(dtype=target_dtype)
        phoneme_context_pos = phoneme_proj(phoneme_embeds_pos)  # (B, 310, audio_dim)
        phoneme_mask_pos = phoneme_mask_pos.to(device=device, dtype=audio_context_mask_pos.dtype)
        if phoneme_mask_pos.dim() == 1:
            phoneme_mask_pos = phoneme_mask_pos.unsqueeze(0)
        phoneme_context_mask_pos = phoneme_mask_pos.long()
        phoneme_context_pos = phoneme_context_pos * phoneme_context_mask_pos.unsqueeze(-1).to(phoneme_context_pos.dtype)

        neg_text = (negative_prompt or "").strip()
        if neg_text:
            phoneme_text_neg = _extract_phoneme_text(neg_text, mode=args.phoneme_source, joiner=args.phoneme_joiner)
            phoneme_idx_neg = build_phoneme_ids(phoneme_text_neg, phoneme_tokenizer, phoneme_pad_length, device)
        else:
            phoneme_idx_neg = torch.zeros_like(phoneme_idx_pos)
        phoneme_embeds_neg, phoneme_mask_neg = phoneme_encoder(phoneme_idx_neg, None)
        phoneme_embeds_neg = phoneme_embeds_neg.to(dtype=target_dtype)
        phoneme_context_neg = phoneme_proj(phoneme_embeds_neg)
        phoneme_mask_neg = phoneme_mask_neg.to(device=device, dtype=audio_context_mask_neg.dtype)
        if phoneme_mask_neg.dim() == 1:
            phoneme_mask_neg = phoneme_mask_neg.unsqueeze(0)
        phoneme_context_mask_neg = phoneme_mask_neg.long()
        phoneme_context_neg = phoneme_context_neg * phoneme_context_mask_neg.unsqueeze(-1).to(phoneme_context_neg.dtype)
        print(f"    phoneme_context_pos shape={tuple(phoneme_context_pos.shape)}")

    audcap_context_pos = None
    audcap_context_mask_pos = None
    audcap_context_neg = None
    audcap_context_mask_neg = None
    if getattr(args, "use_audcap", False):
        print("  - AUDCAP conditioning enabled (separate cross-attention): extracting <AUDCAP> text")
        audcap_text_pos = _extract_audcap_text(prompt)
        if audcap_text_pos is None:
            print("Warning: Prompt has no <AUDCAP>...</AUDCAP> tag; audio-description conditioning is skipped.")
        else:
            print(f"    AUDCAP text: {audcap_text_pos!r}")
            with torch.inference_mode():
                audcap_encoded, audcap_raw_mask = text_encoder._preprocess_text(audcap_text_pos)
            audcap_features = audcap_encoded.to(device=device, dtype=audio_context_positive.dtype)
            audcap_mask = audcap_raw_mask.to(device=device)
            if audcap_mask.dim() == 1:
                audcap_mask = audcap_mask.unsqueeze(0)
            audcap_context_pos = audcap_features
            audcap_context_mask_pos = audcap_mask.long()
            print(f"    audcap_context_pos shape={tuple(audcap_context_pos.shape)} "
                  f"nonpad={int(audcap_context_mask_pos.sum().item())}")

        neg_text = (negative_prompt or "").strip()
        audcap_text_neg = _extract_audcap_text(neg_text) if neg_text else None
        if audcap_text_neg is not None:
            print(f"    negative AUDCAP text: {audcap_text_neg!r}")
            with torch.inference_mode():
                audcap_encoded_neg, audcap_raw_mask_neg = text_encoder._preprocess_text(audcap_text_neg)
            audcap_features_neg = audcap_encoded_neg.to(device=device, dtype=audio_context_positive.dtype)
            audcap_mask_neg = audcap_raw_mask_neg.to(device=device)
            if audcap_mask_neg.dim() == 1:
                audcap_mask_neg = audcap_mask_neg.unsqueeze(0)
            audcap_context_neg = audcap_features_neg
            audcap_context_mask_neg = audcap_mask_neg.long()
        else:
            if audcap_context_pos is not None:
                audcap_context_neg = torch.zeros_like(audcap_context_pos)
                audcap_context_mask_neg = torch.zeros_like(audcap_context_mask_pos)
                print("    negative AUDCAP: all-zero (matches training CFG dropout)")
            else:
                print("    negative AUDCAP: skipped (no positive AUDCAP either)")

    print("  Preparing initial noise...")

    num_motion_frames_user = args.num_frames
    num_audio_timesteps = round(num_motion_frames_user * AUDIO_TO_MOTION_RATIO)

    audio_only_inference = bool(getattr(args, "audio_only_inference", False))
    if audio_only_inference:
        num_motion_frames = 1
    else:
        num_motion_frames = num_motion_frames_user

    motion_duration = num_motion_frames_user / MOTION_LATENT_FPS
    audio_duration = num_audio_timesteps / AUDIO_LATENT_FPS

    print("\nGeneration dimensions:")
    print(f"  Motion latent: {num_motion_frames_user} frames @ {MOTION_LATENT_FPS}fps"
          + (" (audio-only inference: motion modality forced to 1 zero token)"
             if audio_only_inference else ""))
    print(f"  Audio latent:  {num_audio_timesteps} frames @ {AUDIO_LATENT_FPS}fps")
    print(f"  Duration: {motion_duration:.2f}s (motion) / {audio_duration:.2f}s (audio)")
    print(f"   motion: {num_motion_frames_user * MOTION_VAE_STRIDE} frames @ {MOTION_FPS}fps = {num_motion_frames_user * MOTION_VAE_STRIDE / MOTION_FPS:.2f}s")
    if audio_only_inference:
        print("  ⚠️  audio-only inference: motion modality will be 1-token-zero (matches training).")
    print()

    motion_channels = args.motion_latent_channels
    expected_motion_input = motion_channels + args.text_conditioning_dim
    if args.direct_text_concat and transformer.patchify_proj.in_features != expected_motion_input:
        raise ValueError(
            f"motion input projection expects {transformer.patchify_proj.in_features}, "
            f"but latent+text is {motion_channels}+{args.text_conditioning_dim}={expected_motion_input}"
        )

    if audio_only_inference:
        motion_noise = torch.zeros(
            1, motion_channels, num_motion_frames, 1, 1,
            device=device, dtype=torch.float32,
        )
    else:
        motion_noise = torch.randn(
            1, motion_channels, num_motion_frames, 1, 1,
            device=device, dtype=torch.float32,
            generator=torch.Generator(device=device).manual_seed(args.seed)
        )

    audio_noise = torch.randn(
        1, 8, num_audio_timesteps, 16,
        device=device, dtype=torch.float32,
        generator=torch.Generator(device=device).manual_seed(args.seed + 1)
    )

    motion_concat_pos = None
    motion_concat_neg = None
    audio_concat_pos = None
    audio_concat_neg = None
    if args.direct_text_concat:
        motion_concat_pos = _prepare_direct_text_condition(
            motion_concat_raw_positive,
            motion_concat_raw_mask_pos,
            target_seq_len=num_motion_frames,
            feature_dim=args.text_conditioning_dim,
            device=device,
            dtype=motion_noise.dtype,
        )
        motion_concat_neg = _prepare_direct_text_condition(
            motion_concat_raw_negative,
            motion_concat_raw_mask_neg,
            target_seq_len=num_motion_frames,
            feature_dim=args.text_conditioning_dim,
            device=device,
            dtype=motion_noise.dtype,
        )
        audio_concat_pos = _prepare_direct_text_condition(
            audio_concat_raw_positive,
            audio_concat_raw_mask_pos,
            target_seq_len=num_audio_timesteps,
            feature_dim=args.text_conditioning_dim,
            device=device,
            dtype=audio_noise.dtype,
        )
        audio_concat_neg = _prepare_direct_text_condition(
            audio_concat_raw_negative,
            audio_concat_raw_mask_neg,
            target_seq_len=num_audio_timesteps,
            feature_dim=args.text_conditioning_dim,
            device=device,
            dtype=audio_noise.dtype,
        )

    scheduler = MotionSpeechScheduler()
    if audio_only_inference:
        audio_noise_for_scheduler = audio_patchifier.patchify(audio_noise)  # (1, T_audio, 128)
        scheduler_latent = audio_noise_for_scheduler
    else:
        scheduler_latent = motion_noise
    sigmas = scheduler.execute(
        steps=args.num_inference_steps,
        latent=scheduler_latent,
        max_shift=2.05,
        base_shift=0.95,
        stretch=True,
        terminal=0.1
    )

    print(f"  Denoising for {args.num_inference_steps} steps...")

    motion_latents = motion_noise
    audio_latents = audio_noise

    use_correct_motion_time_coordinates = getattr(
        args, "use_correct_motion_time_coordinates", False
    )
    motion_position_fps = (
        MOTION_LATENT_FPS
        if use_correct_motion_time_coordinates or audio_only_inference
        else args.fps
    )
    motion_positions = create_positions(
        num_motion_frames,
        1,
        1,
        device,
        fps=motion_position_fps,
        use_correct_motion_time_coordinates=use_correct_motion_time_coordinates,
    )

    for step_idx, (sigma_current, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        motion_latents_patched = video_patchifier.patchify(motion_latents)

        audio_latents_patched = audio_patchifier.patchify(audio_latents)

        use_motion_main_context = (
            not args.direct_text_concat or args.motion_text_cross_attention
        )
        video_modality_pos = Modality(
            enabled=not audio_only_inference,
            latent=motion_latents_patched,  # Already (B, T, C)
            timesteps=torch.full((1, num_motion_frames), sigma_current, device=device),
            positions=motion_positions,
            context=video_context_positive if use_motion_main_context else None,
            context_mask=motion_context_mask_pos if use_motion_main_context else None,
            concat_condition=motion_concat_pos,
        )

        audio_modality_pos = Modality(
            enabled=True,
            latent=audio_latents_patched,  # Already (B, T_audio, 128)
            timesteps=torch.full((1, num_audio_timesteps), sigma_current, device=device),
            positions=create_audio_positions(num_audio_timesteps, audio_patchifier, device),
            context=None if args.direct_text_concat else audio_context_positive,
            context_mask=None if args.direct_text_concat else audio_context_mask_pos,
            concat_condition=audio_concat_pos,
            phoneme_context=phoneme_context_pos,
            phoneme_context_mask=phoneme_context_mask_pos,
            audcap_context=audcap_context_pos,
            audcap_context_mask=audcap_context_mask_pos,
        )

        video_pred_pos, audio_pred_pos = transformer(
            video=video_modality_pos,
            audio=audio_modality_pos,
            perturbations=None
        )

        if args.disable_cfg:
            video_pred = video_pred_pos
            audio_pred = audio_pred_pos
        else:
            video_modality_neg = Modality(
                enabled=not audio_only_inference,
                latent=motion_latents_patched,  # Already (B, T, C)
                timesteps=torch.full((1, num_motion_frames), sigma_current, device=device),
                positions=motion_positions,
                context=video_context_negative if use_motion_main_context else None,
                context_mask=motion_context_mask_neg if use_motion_main_context else None,
                concat_condition=motion_concat_neg,
            )

            audio_modality_neg = Modality(
                enabled=True,
                latent=audio_latents_patched,  # Already (B, T_audio, 128)
                timesteps=torch.full((1, num_audio_timesteps), sigma_current, device=device),
                positions=create_audio_positions(num_audio_timesteps, audio_patchifier, device),
                context=None if args.direct_text_concat else audio_context_negative,
                context_mask=None if args.direct_text_concat else audio_context_mask_neg,
                concat_condition=audio_concat_neg,
                phoneme_context=phoneme_context_neg,
                phoneme_context_mask=phoneme_context_mask_neg,
                audcap_context=audcap_context_neg,
                audcap_context_mask=audcap_context_mask_neg,
            )

            video_pred_neg, audio_pred_neg = transformer(
                video=video_modality_neg,
                audio=audio_modality_neg,
                perturbations=None
            )

            _motion_scale = args.cfg_guidance_scale_motion if args.cfg_guidance_scale_motion is not None else args.cfg_guidance_scale
            _audio_scale = args.cfg_guidance_scale_audio if args.cfg_guidance_scale_audio is not None else args.cfg_guidance_scale
            video_pred = video_pred_neg + _motion_scale * (video_pred_pos - video_pred_neg)
            audio_pred = audio_pred_neg + _audio_scale * (audio_pred_pos - audio_pred_neg)

        motion_shape = VideoLatentShape(
            batch=1,
            channels=motion_channels,
            frames=num_motion_frames,
            height=1,
            width=1
        )
        video_pred = video_patchifier.unpatchify(video_pred, motion_shape)

        audio_shape = AudioLatentShape(
            batch=1,
            channels=8,
            frames=num_audio_timesteps,
            mel_bins=16
        )
        audio_pred = audio_patchifier.unpatchify(audio_pred, audio_shape)

        dt = sigma_next - sigma_current
        motion_latents = motion_latents + video_pred * dt
        audio_latents = audio_latents + audio_pred * dt


        if (step_idx + 1) % 5 == 0:
            print(f"    Step {step_idx + 1}/{args.num_inference_steps}")

    print("Denoising complete.")

    motion_latents = motion_latents[0]  # [C, T, 1, 1]
    audio_latents = audio_latents[0]    # [8, T_audio, 16]

    print("\nGenerated latent shapes:")
    print(f"  Motion latents: {motion_latents.shape} (C, T, H, W)")
    print(f"  Audio latents:  {audio_latents.shape} (C, T, F)")
    print(f"  Audio duration: {audio_latents.shape[1]} frames @ {AUDIO_LATENT_FPS}fps = {audio_latents.shape[1] / AUDIO_LATENT_FPS:.2f}s")

    motion_latents = motion_latents.squeeze(-1).squeeze(-1)

    if args.use_motion_proj and motion_proj is not None:
        print("  Projecting motion latents from 128 to 256 channels...")
        motion_latents_128 = motion_latents.T  # [T, 128]

        with torch.no_grad():
            if hasattr(motion_proj, "input_proj"):
                proj_weight = motion_proj.input_proj.weight  # [128, 256]
                proj_bias = motion_proj.input_proj.bias      # [128]
            else:
                proj_weight = motion_proj.weight  # [128, 256]
                proj_bias = motion_proj.bias      # [128]
            motion_no_bias = motion_latents_128 - proj_bias
            proj_weight_pinv = torch.linalg.pinv(proj_weight)  # [256, 128]
            motion_latents = (motion_no_bias @ proj_weight_pinv.T).T  # [256, T]

        print(f"    Motion latents shape: {motion_latents.shape}")

    target_duration = num_motion_frames_user / MOTION_LATENT_FPS

    return motion_latents.cpu(), audio_latents.cpu(), target_duration


def create_positions(
    num_frames: int,
    height: int,
    width: int,
    device: str,
    fps: float = 30.0,
    use_correct_motion_time_coordinates: bool = False,
) -> torch.Tensor:
    """DuoFlow inference helper."""
    if fps <= 0:
        raise ValueError(f"Motion FPS must be positive, got {fps}")

    positions = torch.zeros(1, 3, num_frames, 2, device=device, dtype=torch.float32)

    if use_correct_motion_time_coordinates:
        token_indices = torch.arange(num_frames, device=device, dtype=torch.float32)
        positions[0, 0, :, 0] = token_indices / fps
        positions[0, 0, :, 1] = (token_indices + 1) / fps
    else:
        time_pos = torch.linspace(0, 1, num_frames, device=device, dtype=torch.float32)
        legacy_time = time_pos / fps
        positions[0, 0, :, 0] = legacy_time
        positions[0, 0, :, 1] = legacy_time

    return positions


def create_audio_positions(
    num_timesteps: int,
    audio_patchifier,  # AudioPatchifier type
    device: str
) -> torch.Tensor:
    """DuoFlow inference helper."""
    from motionspeech_core.types import AudioLatentShape

    latent_coords = audio_patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(
            batch=1,
            channels=8,
            frames=num_timesteps,
            mel_bins=16
        ),
        device=device
    )

    return latent_coords.to(torch.float32)


def decode_motion_latents(
    latents: torch.Tensor,
    motion_vae: torch.nn.Module,
    device: str
) -> np.ndarray:
    """DuoFlow inference helper."""
    print("\nDecoding motion latents...")

    stats_dir = REPO_ROOT / "assets/motion_vae_stats"
    mean = np.load(stats_dir / "mean.npy")
    std = np.load(stats_dir / "std.npy")

    latents_tensor = latents.T.unsqueeze(0).float().to(device)

    print(f"  Input latents shape: {latents_tensor.shape} (B, T, C)")

    with torch.no_grad():
        out = motion_vae.forward_decoder(latents_tensor).cpu().numpy()

    motion = out[0] if isinstance(out, (tuple, list)) else out
    motion = motion[0]  # Remove batch dim: [T_motion, 333]
    motion = motion * std + mean

    print(f"✅ Motion shape: {motion.shape} (T_motion should be ~{latents_tensor.shape[1]*3})")
    return motion


def decode_audio_latents(
    latents: torch.Tensor,
    audio_vae_decoder: torch.nn.Module,
    vocoder: torch.nn.Module,
    sample_rate: int,
    hop_length: int,
    mel_bins: int,
    device: str,
    target_duration: float = None
) -> np.ndarray:
    """DuoFlow inference helper."""
    print("\nDecoding audio latents...")
    print(f"  Input audio latents shape: {latents.shape} (C, T, F)")

    latents_tensor = latents.unsqueeze(0).to(device=device, dtype=torch.float32)
    print(f"  Latents tensor shape: {latents_tensor.shape} (B, C, T, F)")

    with torch.no_grad():
        mel_spectrogram = audio_vae_decoder.decode(latents_tensor).sample  # [1, 1, T, F]
        print(f"  Mel spectrogram shape: {mel_spectrogram.shape} (B, 1, T, F)")

        mel_time_steps = mel_spectrogram.shape[2]
        mel_duration = mel_time_steps * (1.0 / (sample_rate / hop_length))
        print(f"  Mel time steps: {mel_time_steps}, duration: {mel_duration:.2f}s")

        mel_for_vocoder = mel_spectrogram.squeeze(1)
        if mel_for_vocoder.shape[-1] != mel_bins and mel_for_vocoder.shape[1] == mel_bins:
            mel_for_vocoder = mel_for_vocoder.transpose(1, 2)
        mel_for_vocoder = mel_for_vocoder.contiguous()

        vocoder_dtype = next(vocoder.parameters()).dtype
        mel_for_vocoder = mel_for_vocoder.to(dtype=vocoder_dtype)

        audio_tensor = vocoder(mel_for_vocoder).squeeze(0)  # [T_samples]
        print(f"  Audio tensor shape: {audio_tensor.shape}")

        num_samples = audio_tensor.shape[-1]
        actual_duration = num_samples / sample_rate
        print(f"  Vocoder output: {num_samples} samples, duration: {actual_duration:.2f}s @ {sample_rate}Hz")

        if target_duration is not None:
            target_samples = int(target_duration * sample_rate)
            if num_samples > target_samples:
                print(f"  Trimming audio: {num_samples} -> {target_samples} samples ({actual_duration:.2f}s -> {target_duration:.2f}s).")
                audio_tensor = audio_tensor[..., :target_samples]
            elif num_samples < target_samples:
                print(f"Warning: Audio has {num_samples} samples ({actual_duration:.2f}s), shorter than the requested {target_duration:.2f}s.")

        audio_waveform = audio_tensor.float().cpu().numpy()

    final_duration = audio_waveform.shape[-1] / sample_rate
    print(f"Audio shape: {audio_waveform.shape}, duration: {final_duration:.2f}s @ {sample_rate}Hz")
    return audio_waveform


def convert_mofea333_to_smplx_format(motion_data: np.ndarray, njoints: int = 55):
    """
    Convert mofea333 format (3 trans + 55 joints × 6D) to SMPL-X axis angle format
    Args:
        motion_data: (T, 333) - mofea333 format
        njoints: number of joints (default 55 for SMPL-X)
    Returns:
        poses_axis_angle: (T, njoints*3) - axis angle representation
        trans: (T, 3) - translation
    """
    T = motion_data.shape[0]
    trans = motion_data[:, :3]  # (T, 3) global translation
    motion_6d = motion_data[:, 3:]  # (T, 330) 6D rotations

    expected_6d_dim = njoints * 6
    if motion_6d.shape[1] < expected_6d_dim:
        print(f"Warning: Expected {expected_6d_dim} dims for 6D rotations, got {motion_6d.shape[1]}")
        padding = np.zeros((T, expected_6d_dim - motion_6d.shape[1]))
        motion_6d = np.concatenate([motion_6d, padding], axis=1)
    elif motion_6d.shape[1] > expected_6d_dim:
        motion_6d = motion_6d[:, :expected_6d_dim]

    from scipy.spatial.transform import Rotation

    rotations = motion_6d.reshape(T, njoints, 6)
    first = rotations[..., :3]
    second = rotations[..., 3:]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    matrices = np.stack((first, second, third), axis=-2)
    poses_axis_angle = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec()
    poses_axis_angle = poses_axis_angle.reshape(T, njoints * 3).astype(np.float32)

    return poses_axis_angle, trans


def save_motion_as_smplx_npz(
    output_path: str,
    motion_data: np.ndarray,
    njoints: int = 55,
    fps: float = 30.0
):
    """DuoFlow inference helper."""
    T = motion_data.shape[0]

    poses_axis_angle, trans = convert_mofea333_to_smplx_format(motion_data, njoints)

    betas = np.zeros(300, dtype=np.float32)  # Body shape parameters (1D array)
    expressions = np.zeros((T, 100), dtype=np.float32)  # Facial expressions

    np.savez(
        output_path,
        poses=poses_axis_angle,  # (T, 165) for 55 joints
        trans=trans,  # (T, 3)
        betas=betas,  # (300,)
        expressions=expressions,  # (T, 100)
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=fps
    )


def render_motion_to_video(
    motion_npz_path: str,
    audio_path: str,
    temp_dir: str,
    fps: float = 30.0,
    audio_sample_rate: int = 16000,
) -> str:
    """DuoFlow inference helper."""
    print("\nRendering motion...")

    pantomatrix_path = os.environ.get(
        "MOTIONSPEECH_EMAGE_ROOT", str(REPO_ROOT / "external/PantoMatrix")
    )
    if pantomatrix_path not in sys.path:
        sys.path.insert(0, pantomatrix_path)

    model_folder_candidates = [
        os.environ.get("MOTIONSPEECH_SMPLX_MODELS"),
        str(Path(pantomatrix_path) / "emage_evaltools/smplx_models"),
    ]

    model_folder = None
    for candidate in model_folder_candidates:
        if candidate and os.path.exists(candidate):
            model_folder = candidate
            print(f"  SMPL-X model directory: {model_folder}")
            break

    if not model_folder:
        print("SMPL-X model files were not found.")
        print("Set MOTIONSPEECH_SMPLX_MODELS or install model files under the PantoMatrix directory.")
        return None

    try:
        from emage_utils import fast_render
    except ImportError as e:
        print(f"Could not import PantoMatrix fast_render: {e}")
        return None

    fast_render.args["render_concurrent_num"] = 1

    try:
        npz_basename = os.path.splitext(os.path.basename(motion_npz_path))[0]

        if not audio_path or not os.path.exists(audio_path):
            if audio_path:
                print(f"Warning: Audio file not found: {audio_path}")
            print("  Creating silent audio for rendering.")
            import scipy.io.wavfile as wavfile
            sample_rate = audio_sample_rate
            motion_data = np.load(motion_npz_path)
            duration = max(1, len(motion_data['poses']) / fps)
            silent_audio = np.zeros(int(sample_rate * duration), dtype=np.int16)
            audio_path = os.path.join(temp_dir, "dummy_audio.wav")
            wavfile.write(audio_path, sample_rate, silent_audio)
            print(f"  Created {duration:.2f}s of silent audio.")

        print(f"  - Motion: {os.path.basename(motion_npz_path)}")
        print(f"  - Audio: {os.path.basename(audio_path)}")
        print(f"  - Model folder: {model_folder}")

        import time
        temp_suffix = str(int(time.time() * 1000))
        render_temp_dir = os.path.join(temp_dir, f"render_temp_{temp_suffix}")
        os.makedirs(render_temp_dir, exist_ok=True)

        print("  Rendering sequence...")
        video_path = fast_render.render_one_sequence_no_gt(
            res_npz_path=motion_npz_path,
            output_dir=render_temp_dir,
            audio_path=audio_path,
            model_folder=model_folder,
            remove_transl=True,
        )

        if not video_path or not os.path.exists(video_path):
            print("Warning: Rendering with audio failed; retrying without audio.")
            render_temp_dir_retry = os.path.join(temp_dir, f"render_temp_{temp_suffix}_no_audio")
            os.makedirs(render_temp_dir_retry, exist_ok=True)
            video_path = fast_render.render_one_sequence_no_gt(
                res_npz_path=motion_npz_path,
                output_dir=render_temp_dir_retry,
                audio_path=None,
                model_folder=model_folder,
                remove_transl=True,
            )

        if video_path and os.path.exists(video_path):
            final_video_path = os.path.join(temp_dir, f"{npz_basename}_video.mp4")
            shutil.move(video_path, final_video_path)

            if os.path.exists(render_temp_dir):
                shutil.rmtree(render_temp_dir)

            print(f"Video rendered: {os.path.basename(final_video_path)}")
            return final_video_path
        else:
            print("Rendering did not produce a video.")
            if video_path:
                print(f"  Renderer output path: {video_path}")
            return None

    except Exception as e:
        print(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_video_audio(video_path: str, audio_path: str, output_path: str):
    """DuoFlow inference helper."""
    import subprocess

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True)


def main():
    """DuoFlow inference helper."""
    args = parse_args()

    if args.device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch or use --device cpu.")
        gpu_id = args.device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        torch.cuda.set_device(0)
        actual_device = "cuda:0"
    else:
        actual_device = args.device

    temp_dir = tempfile.mkdtemp(prefix="text_to_motion_audio_")

    print("=" * 80)
    print("Text-to-motion and audio inference")
    print("=" * 80)
    print()
    print("Prompt:", args.prompt)
    print("Motion latent frames:", args.num_frames)
    print("🔬 CFG Scale:", args.cfg_guidance_scale)
    print("Seed:", args.seed)
    print()

    try:
        motion_vae = load_motion_vae(args.motion_vae_path, actual_device)

        motionspeech_models = load_motionspeech_models(args, actual_device)

        motion_latents, audio_latents, target_duration = generate_from_text(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            models=motionspeech_models,
            args=args,
            device=actual_device,
        )

        audio_only_inference_main = bool(getattr(args, "audio_only_inference", False))
        if audio_only_inference_main:
            print("\nAudio-only inference: skipping motion decoding and rendering.")
            motion_data = None
        else:
            motion_data = decode_motion_latents(motion_latents, motion_vae, actual_device)

        audio_waveform = decode_audio_latents(
            audio_latents,
            motionspeech_models['audio_vae_decoder'],
            motionspeech_models['vocoder'],
            motionspeech_models['audio_sample_rate'],
            motionspeech_models['audio_hop_length'],
            motionspeech_models['audio_mel_bins'],
            actual_device,
            target_duration=target_duration
        )

        output_dir = os.path.dirname(args.output_path)
        os.makedirs(output_dir, exist_ok=True)
        if args.output_name:
            output_name = args.output_name
        else:
            output_name = os.path.splitext(os.path.basename(args.output_path))[0]

        audio_output_path = os.path.join(output_dir, f"{output_name}.wav")

        audio_temp_path = os.path.join(temp_dir, "generated_audio.wav")

        import scipy.io.wavfile as wavfile

        if audio_waveform.ndim == 2:
            audio_waveform = audio_waveform.T

        audio_normalized = np.clip(audio_waveform, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

        wavfile.write(audio_output_path, motionspeech_models['audio_sample_rate'], audio_int16)
        print(f"Audio saved: {audio_output_path} (shape: {audio_int16.shape}, duration: {audio_int16.shape[0]/motionspeech_models['audio_sample_rate']:.2f}s)")

        shutil.copy(audio_output_path, audio_temp_path)
        audio_path = audio_temp_path

        if audio_only_inference_main:
            motion_npz_path = None
            motion_output_path = None
        else:
            motion_npz_path = os.path.join(temp_dir, "motion.npz")
            save_motion_as_smplx_npz(motion_npz_path, motion_data, fps=args.fps)

            motion_output_path = os.path.join(output_dir, f"{output_name}.npz")
            shutil.copy(motion_npz_path, motion_output_path)
            print(f"Motion saved: {motion_output_path}")

        if not args.no_render and not audio_only_inference_main:
            video_path = render_motion_to_video(
                motion_npz_path,
                audio_path,
                temp_dir,
                args.fps,
                motionspeech_models['audio_sample_rate'],
            )

            if video_path and os.path.exists(video_path):
                video_output_path = os.path.join(output_dir, f"{output_name}.mp4")
                merge_video_audio(video_path, audio_path, video_output_path)
                print(f"Video saved: {video_output_path}")
            else:
                print("Warning: Rendering failed; motion and audio outputs remain available.")
                print(f"  Audio: {audio_output_path}")
                if motion_output_path:
                    print(f"   - Motion: {motion_output_path}")

        elif audio_only_inference_main:
            print("Audio-only inference completed.")
            print(f"  Audio: {audio_output_path}")
        else:
            print("Rendering skipped (--no-render).")
            print(f"  Audio: {audio_output_path}")
            if motion_npz_path:
                print(f"   - Motion: {motion_npz_path}")

        print("\nInference complete.")

    except Exception as e:
        print(f"\nInference failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        if not args.keep_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nRemoving temporary directory: {temp_dir}")
        elif args.keep_temp:
            print(f"\nTemporary files retained at: {temp_dir}")

    print("\n" + "=" * 80)
    print("Text-to-motion and audio inference complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
