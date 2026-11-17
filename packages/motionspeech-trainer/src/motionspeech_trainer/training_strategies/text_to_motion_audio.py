# Modified for DuoFlow; see NOTICE.md.
"""Text-to-Motion+Audio training strategy.

This strategy implements text-to-motion-audio generation where:
- Input: Text description (+ optional phoneme sequence)
- Output: Motion latents + Audio latents (jointly generated)
- The model learns to generate synchronized motion and audio from text
- Optional phoneme encoder enhances audio generation quality

This is different from motion_to_audio where motion is an input.
Here, motion is generated alongside audio, creating a fully generative model.
"""

from dataclasses import replace
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from motionspeech_core.model.transformer.modality import Modality
from motionspeech_core.text_encoders.gemma.embeddings_connector import (
    Embeddings1DConnectorConfigurator,
)
from motionspeech_trainer import logger
from motionspeech_trainer.timestep_samplers import TimestepSampler
from motionspeech_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class ARIABridge(nn.Module):
    """ARIABridge: Bridge module with self-attention for feature transformation.

    This module provides a more sophisticated feature transformation compared to
    a simple linear layer, using self-attention and FFN to learn complex mappings.
    """

    def __init__(self, input_dim, output_dim=1024, num_heads=4, hidden_dim=1024, causal=True):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for FFN
            causal: If True, use causal mask (for autoregressive generation)
        """
        super().__init__()
        self.causal = causal
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        """Forward pass with optional causal masking.

        Args:
            x: Input tensor of shape (B, seq_len, input_dim)

        Returns:
            Output tensor of shape (B, seq_len, output_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.input_proj(x)
        orig_shape = x.shape

        x_norm = self.norm1(x)

        # Create attention mask
        attn_mask = None
        if self.causal:
            # Create Causal Mask for autoregressive generation
            # x shape: [Batch, SeqLen, Dim]
            seq_len = x.size(1)
            # attn_mask: [SeqLen, SeqLen], upper triangular is True (masked)
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Self-attention with optional causal mask
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_output

        # FFN with residual connection
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x if orig_shape == x.shape else x.squeeze(0) if orig_shape[0] == 1 else x


class TextToMotionAudioConfig(TrainingStrategyConfigBase):
    """Configuration for text-to-motion-audio training strategy."""

    name: Literal["text_to_motion_audio"] = "text_to_motion_audio"

    with_audio: bool = Field(
        default=True,
        description="Whether to include audio in training (should always be True for text-to-motion-audio)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents",
    )

    motion_latents_dir: str = Field(
        default="motion_latents",
        description="Directory name for motion latents (ground truth targets)",
    )

    motion_conditions_dir: str = Field(
        default="motion_conditions",
        description=(
            "Directory name for the motion branch's text conditions "
            "(Gemma-encoded <L1>/<L2> captions). Injected only into the "
            "video/motion Modality's context. Renamed from the legacy "
            "'conditions' folder so the motion and audio branches no longer "
            "share the same encoded caption."
        ),
    )

    speak_conditions_dir: str = Field(
        default="speak_conditions",
        description=(
            "Directory name for the audio branch's text conditions "
            "(Gemma-encoded <S>...</E> speech content, tags preserved). "
            "Injected only into the audio Modality's context. Distinct "
            "from `motion_conditions_dir` so audio and motion get their "
            "own dedicated captions."
        ),
    )

    joint_conditions_dir: str | None = Field(
        default=None,
        description=(
            "Optional directory containing one Gemma-encoded joint caption per "
            "sample (motion + speech + AUDCAP). When set, the same encoded "
            "context is loaded once and injected into both motion and audio "
            "main-text cross-attention branches. It replaces motion_conditions_dir "
            "and speak_conditions_dir for dataset loading."
        ),
    )

    direct_text_concat: bool = Field(
        default=False,
        description=(
            "Project multi-layer Gemma features to text_conditioning_dim when needed, "
            "pad them to the noisy modality token length, and concatenate on the "
            "feature axis. No connector, caption projection, or main-text "
            "cross-attention is used by the concat branch."
        ),
    )

    text_conditioning_dim: int = Field(
        default=3840,
        description="Raw Gemma feature width used by direct_text_concat.",
        ge=1,
    )

    motion_text_cross_attention: bool = Field(
        default=False,
        description=(
            "In addition to direct_text_concat, send motion_conditions through "
            "one DuoFlow motion Text Connector and inject the result into the "
            "motion branch with its main text cross-attention. The speech main "
            "text path remains concat-only."
        ),
    )

    motion_thinking_tokens: int = Field(
        default=40,
        description=(
            "Number of learnable register/thinking tokens appended exactly once "
            "after each sample's valid motion-caption tokens."
        ),
        ge=0,
    )

    motion_loss_weight: float = Field(
        default=1.0,
        description="Weight for motion generation loss",
        ge=0.0,
    )

    audio_loss_weight: float = Field(
        default=1.0,
        description="Weight for audio generation loss",
        ge=0.0,
    )

    aux_ctc_weight: float = Field(
        default=1.0,
        description=(
            "Weight of the phoneme-CTC auxiliary alignment loss computed on "
            "the tapped audio-branch hidden states (model.aux_ctc_tap_blocks). "
            "0 disables the auxiliary loss entirely. Requires the model "
            "config to set the supervised audio block indices."
        ),
        ge=0.0,
    )

    aux_ctc_num_classes: int = Field(
        default=184,
        description=(
            "Number of CTC output classes: phoneme_encoder_vocab_size (183) "
            "plus one CTC blank token placed at the last index. Pad id 0 is "
            "filtered from targets, not used as blank."
        ),
        ge=2,
    )

    aux_ctc_hidden_dim: int = Field(
        default=1024,
        description=(
            "Feature dimension of the tapped audio-branch hidden states "
            "(audio_num_attention_heads * audio_attention_head_dim)."
        ),
        ge=1,
    )

    audio_only_training: bool = Field(
        default=False,
        description="If true, train with audio loss only and ignore motion loss",
    )

    audio_only_disable_video_branch: bool = Field(
        default=True,
        description=(
            "If audio_only_training=True: when True (default), set "
            "Modality(enabled=False) for the dummy motion so the transformer "
            "skips video entirely. When False (legacy), keep enabled=True with "
            "1-token-zero latent (video branch still runs)."
        ),
    )

    motion_proj_enabled: bool = Field(
        default=True,
        description="Whether to project motion latents from 256 to transformer channels",
    )

    motion_proj_type: Literal["aria", "linear"] = Field(
        default="aria",
        description="Type of motion projection when enabled: 'aria' (ARIABridge) or 'linear'",
    )

    motion_proj_output_dim: int = Field(
        default=256,
        description=(
            "Output dim of motion_proj. MUST equal model.transformer_in_channels "
            "so the projected motion latents feed straight into patchify_proj. "
            "Default 256 matches the BEAT2 motion VAE latent dim."
        ),
    )

    phoneme_proj_type: Literal["aria", "linear"] = Field(
        default="aria",
        description="Type of phoneme projection: 'aria' (ARIABridge) or 'linear'",
    )

    use_phoneme_encoder: bool = Field(
        default=False,
        description="Whether phoneme encoder is enabled. Controls phoneme_proj creation.",
    )

    use_audcap: bool = Field(
        default=False,
        description=(
            "Whether to inject AUDCAP (voice/timbre) conditions into the audio "
            "branch via an independent cross-attention path (mirrors the "
            "phoneme path). When True, the dataset must contain an `audcaps/` "
            "directory with Gemma-encoded AUDCAP text per sample."
        ),
    )

    audcap_dropout_prob: float = Field(
        default=0.0,
        description=(
            "Probability of replacing AUDCAP conditions with unconditional "
            "(zeros) during training (CFG dropout). Mirrors phoneme path."
        ),
        ge=0.0,
        le=1.0,
    )

    use_connectors: bool = Field(
        default=False,
        description=(
            "Whether to use video_connector and audio_connector to transform "
            "Gemma features before caption_projection. When False (bypass mode), "
            "raw Gemma 3840-dim features go directly to caption_projection."
        ),
    )

    connector_num_attention_heads: int = Field(
        default=30,
        description="Number of self-attention heads in each text connector.",
        ge=1,
    )

    connector_attention_head_dim: int = Field(
        default=128,
        description="Dimension of each text-connector attention head.",
        ge=1,
    )

    connector_num_layers: int = Field(
        default=2,
        description="Number of Transformer blocks in each text connector.",
        ge=1,
    )

    connector_ff_mult: int = Field(
        default=4,
        description="Expansion multiplier of each text connector's feed-forward block.",
        ge=1,
    )

    connector_zero_init_output: bool = Field(
        default=False,
        description=(
            "Zero-initialize the connector's output projection. Recommended for "
            "bottleneck connectors used with connector_residual=True."
        ),
    )

    connector_num_learnable_registers: int | None = Field(
        default=128,
        description=(
            "Number of learnable register tokens appended by each connector. "
            "Set to null to disable them. MOTIONSPEECH_NUM_LEARNABLE_REGISTERS can override this."
        ),
        ge=0,
    )

    video_use_connector: bool = Field(
        default=True,
        description=(
            "Per-modality switch: whether to create and run the video connector. "
            "Only effective when use_connectors=True. When False, the video branch "
            "receives raw Gemma features (like bypass mode) while the audio branch "
            "can still use its own connector. Set to False to prevent video connector "
            "norm explosion caused by weak motion_loss gradients."
        ),
    )

    audio_use_connector: bool = Field(
        default=True,
        description=(
            "Per-modality switch: whether to create and run the audio connector. "
            "Only effective when use_connectors=True. When False, the audio branch "
            "receives raw Gemma features while the video branch can still use its "
            "own connector."
        ),
    )

    connector_residual: bool = Field(
        default=True,
        description=(
            "When use_connectors=True, add a residual from raw Gemma features "
            "to the connector output. Prevents token-diversity collapse caused "
            "by connector self-attention homogenising all content tokens."
        ),
    )

    gemma_num_layers: int = Field(
        default=1,
        description=(
            "Number of Gemma hidden-state layers to aggregate. "
            "The preprocessing script must have been run with the matching "
            "MOTIONSPEECH_GEMMA_NUM_LAYERS env var. "
            "When 1 (default): conditions are (seq, 3840) → feature_extractor BYPASSED. "
            "When >1 (e.g. 8): conditions are (seq, 3840*N) → trainable "
            "GemmaFeaturesExtractorProjLinear(Linear(3840*N → 3840)) projects "
            "multi-layer features down to 3840 before connectors/caption_projection."
        ),
        ge=1,
    )

    gemma_feature_selection: str = Field(
        default="last_n",
        description=(
            "Gemma hidden-state selection policy. 'last_n' selects the last "
            "gemma_num_layers states. 'embedding_first_last' selects the embedding "
            "output, Transformer layer 1 output, and final layer output and therefore "
            "requires gemma_num_layers=3."
        ),
        pattern="^(last_n|embedding_first_last)$",
    )

    unconditional_prob: float = Field(
        default=0.1,
        description="Probability of replacing all conditions with unconditional embeddings during training (CFG dropout)",
        ge=0.0,
        le=1.0,
    )

    max_motion_frames: int | None = Field(
        default=129,
        description=(
            "Maximum number of motion frames for training. "
            "Longer sequences will be randomly cropped (data augmentation). "
            "Set to None to disable cropping. "
            "Critical for GPU memory: transformer memory grows O(n²) with sequence length."
        ),
        ge=1,
    )

    use_correct_motion_time_coordinates: bool = Field(
        default=False,
        description=(
            "Use real-second interval bounds [i / fps, (i + 1) / fps] for "
            "motion positional embeddings. When False, retain the legacy "
            "torch.linspace(0, 1, num_frames) / fps behavior for backward "
            "compatibility with existing checkpoints."
        ),
    )

    max_audio_steps: int | None = Field(
        default=None,
        description=(
            "Only used when audio_only_training=True. Maximum audio latent "
            "time steps for training; longer audio is randomly cropped. "
            "If None, no cropping (uses padded-to-batch-max as-is from collate). "
            "Critical for GPU memory: transformer memory grows O(n²) with sequence length."
        ),
        ge=1,
    )

    # ----- STAGE-1 AUDIO REPLAY (anti-forgetting) -----
    # Replay batches (no `motion_latents`, drawn from
    # data.replay_preprocessed_data_root) run through the audio-only prep
    # path inside a joint run. This scales their audio flow-matching loss so
    # the replay gradient magnitude can be tuned independently of the joint
    # audio loss (e.g. when this stage trains at a higher LR than stage 1).
    replay_loss_weight: float = Field(
        default=1.0,
        description=(
            "Multiplier applied to the audio flow-matching loss on replay "
            "batches (batches without `motion_latents`). No effect when "
            "replay is disabled. The CTC auxiliary loss is not scaled."
        ),
        ge=0.0,
    )


class TextToMotionAudioStrategy(TrainingStrategy, nn.Module):
    """Text-to-Motion+Audio training strategy.

    This strategy implements joint text-to-motion-audio generation where:
    - Text is the only input (no motion conditioning)
    - Optional phoneme sequence enhances audio quality
    - Both motion and audio are generated as outputs
    - Motion is treated as "pseudo-video" (no spatial dimensions)
    - Supports synchronized motion-audio generation

    Training objective:
    - Learn to generate motion latents from text
    - Learn to generate audio latents from text (+ optional phonemes)
    - Ensure motion and audio are temporally synchronized
    """

    config: TextToMotionAudioConfig

    def __init__(self, config: TextToMotionAudioConfig):
        """Initialize strategy with configuration.

        Args:
            config: Text-to-motion-audio configuration
        """
        # Initialize both parent classes
        TrainingStrategy.__init__(self, config)
        nn.Module.__init__(self)

        if (
            self.config.gemma_feature_selection == "embedding_first_last"
            and self.config.gemma_num_layers != 3
        ):
            raise ValueError(
                "gemma_feature_selection=embedding_first_last requires gemma_num_layers=3"
            )

        # Motion VAE latents are 256-dim. Project to transformer's expected in_channels.
        # output_dim MUST match model.transformer_in_channels so patchify_proj accepts the result.
        self.motion_proj = None
        if self.config.motion_proj_enabled:
            _out = self.config.motion_proj_output_dim
            if self.config.motion_proj_type == "aria":
                # ARIABridge: self-attention + FFN over projected features.
                self.motion_proj = ARIABridge(
                    input_dim=256,      # Motion VAE latent dim
                    output_dim=_out,    # MUST equal transformer_in_channels
                    num_heads=2,
                    hidden_dim=_out,    # keep FFN inner = output dim
                    causal=False,
                )
            elif self.config.motion_proj_type == "linear":
                # Plain Linear; output dim must equal transformer_in_channels.
                self.motion_proj = nn.Linear(256, _out)
            else:
                raise ValueError(f"Unsupported motion_proj_type: {self.config.motion_proj_type}")

        # Phoneme encoder for enhanced audio generation (initialized externally)
        # This will be set by the trainer if use_phoneme_encoder=True in model config
        self.phoneme_encoder = None
        self.use_phoneme_encoder = False
        # Trainable projection for phoneme embeddings -> audio context dim
        # Only created when phoneme encoder is enabled to avoid DDP unused-param errors
        self.phoneme_proj = None

        # CTC alignment auxiliary head over the tapped audio-branch hidden
        # states. Registered as a strategy module so it is picked up by the
        # optimizer, the `strategy.` checkpoint prefix and DDP grad sync
        # automatically. Loss enabled via aux_ctc_weight > 0; the feature tap
        # itself is model-side (model.aux_ctc_tap_block).
        self.audio_ctc_head = None
        self.last_aux_losses: dict[str, float] = {}
        if self.config.aux_ctc_weight > 0:
            self.audio_ctc_head = nn.Linear(self.config.aux_ctc_hidden_dim, self.config.aux_ctc_num_classes)
        self.audio_ctc_heads: nn.ModuleDict = nn.ModuleDict()

        # AUDCAP (voice/timbre) conditioning. Mirrors the phoneme path:
        # - AUDCAP text is Gemma-encoded offline and stored under `audcaps/`.
        # - At forward time we run those features through `self.feature_extractor`
        #   (shared with the main caption path; projects 3840*N -> 3840) and
        #   hand the resulting 3840-dim features to the audio Modality.
        # - The actual 3840 -> audio_inner_dim projection is performed inside
        #   the model by `audio_audcap_caption_projection` (kept on the model
        #   for symmetry with `audio_phoneme_caption_projection`).
        self.use_audcap = self.config.use_audcap
        self.audcap_dropout_prob = self.config.audcap_dropout_prob
        if self.config.use_phoneme_encoder:
            # Phoneme encoder outputs 192-dim embeddings (AudioLDM2 VITS)
            # Gemma audio connector outputs 3840-dim embeddings
            if self.config.phoneme_proj_type == "aria":
                self.phoneme_proj = ARIABridge(
                    input_dim=192,
                    output_dim=3840,
                    num_heads=4,
                    hidden_dim=768,
                    causal=False,
                )
            else:
                self.phoneme_proj = nn.Linear(192, 3840)
            logger.info(f"Phoneme projection created: {self.config.phoneme_proj_type}")
        else:
            logger.info("Phoneme encoder disabled — phoneme_proj not created")

        # ── Trainable feature_extractor_linear ──
        # When gemma_num_layers > 1, preprocessing saved concatenated hidden states
        # of shape (seq, 3840*N). A trainable Linear(3840*N → 3840) projects them
        # back to 3840-dim before connectors / caption_projection.
        # When gemma_num_layers == 1, input is already 3840-dim → no projection needed.
        import os
        self._gemma_num_layers = self.config.gemma_num_layers
        self.feature_extractor = None
        if self._gemma_num_layers > 1:
            from motionspeech_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
            self.feature_extractor = GemmaFeaturesExtractorProjLinear(num_layers=self._gemma_num_layers)
            logger.info(
                f"Feature extractor: ENABLED (gemma_num_layers={self._gemma_num_layers}, "
                f"selection={self.config.gemma_feature_selection}, "
                f"Linear({3840 * self._gemma_num_layers} -> 3840), trainable)"
            )
        else:
            logger.info(
                f"Feature extractor: BYPASSED (gemma_num_layers={self._gemma_num_layers}, "
                f"raw Gemma 3840-dim output passed directly to caption_projection)"
            )

        # ── Trainable embeddings connectors ──
        # In direct-concat mode, only the optional extra motion cross-attention
        # path owns a connector. Speech remains concat-only.
        self.video_connector = None
        self.audio_connector = None
        self.video_connector_norm = None
        self.audio_connector_norm = None
        enable_video_connector = (
            self.config.motion_text_cross_attention
            or (
                self.config.use_connectors
                and not self.config.direct_text_concat
                and self.config.video_use_connector
            )
        )
        enable_audio_connector = (
            self.config.use_connectors
            and not self.config.direct_text_concat
            and self.config.audio_use_connector
        )
        if enable_video_connector or enable_audio_connector:
            import os as _os
            _connector_inner_dim = (
                self.config.connector_num_attention_heads
                * self.config.connector_attention_head_dim
            )
            _connector_config = {
                "transformer": {
                    "connector_num_attention_heads": self.config.connector_num_attention_heads,
                    "connector_attention_head_dim": self.config.connector_attention_head_dim,
                    "connector_num_layers": self.config.connector_num_layers,
                    "connector_ff_mult": self.config.connector_ff_mult,
                    "connector_input_dim": self.config.text_conditioning_dim,
                    "connector_output_dim": self.config.text_conditioning_dim,
                    "connector_zero_init_output": self.config.connector_zero_init_output,
                    "num_learnable_registers": self.config.connector_num_learnable_registers,
                }
            }
            _env = _os.getenv("MOTIONSPEECH_NUM_LEARNABLE_REGISTERS")
            if _env is not None:
                lowered = _env.strip().lower()
                if lowered in {"none", "null", "nil"}:
                    _connector_config["transformer"]["num_learnable_registers"] = None
                else:
                    try:
                        _connector_config["transformer"]["num_learnable_registers"] = int(lowered)
                    except ValueError:
                        pass

            # The additional motion path always uses the paper's thinking-token
            # mechanism, independent of legacy connector defaults.
            if self.config.motion_text_cross_attention:
                _connector_config["transformer"]["num_learnable_registers"] = (
                    self.config.motion_thinking_tokens
                )

            _enabled_parts = []
            _connector_shape = (
                f"{self.config.text_conditioning_dim}->{_connector_inner_dim}"
                f"->{self.config.text_conditioning_dim}, "
                f"layers={self.config.connector_num_layers}, "
                f"ff_mult={self.config.connector_ff_mult}"
            )
            if enable_video_connector:
                self.video_connector = Embeddings1DConnectorConfigurator.from_config(_connector_config)
                self.video_connector_norm = nn.LayerNorm(self.config.text_conditioning_dim)
                _enabled_parts.append(
                    f"motion connector ({_connector_shape}) + "
                    f"{self.config.motion_thinking_tokens} thinking tokens + LayerNorm"
                    if self.config.motion_text_cross_attention
                    else f"video_connector ({_connector_shape}) + LayerNorm"
                )
            if enable_audio_connector:
                self.audio_connector = Embeddings1DConnectorConfigurator.from_config(_connector_config)
                self.audio_connector_norm = nn.LayerNorm(self.config.text_conditioning_dim)
                _enabled_parts.append(f"audio_connector ({_connector_shape}) + LayerNorm")

            if _enabled_parts:
                logger.info(f"Embeddings connectors: ENABLED ({' + '.join(_enabled_parts)})")
            else:
                logger.info("Embeddings connectors: use_connectors=True but both per-modality switches are False → BYPASSED")
        elif self.config.direct_text_concat:
            logger.info("Embeddings connectors: BYPASSED (raw Gemma features concatenate with noisy modality tokens)")
        else:
            logger.info("Embeddings connectors: BYPASSED (raw Gemma features → caption_projection)")


    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return True

    @property
    def requires_motion(self) -> bool:
        """Whether this training strategy requires motion data.

        When audio_only_training=True, motion data is not loaded from disk and
        a dummy 1-token motion modality is fed to the transformer (with
        loss=0). This lets the audio-only path run on datasets that do not
        contain a motion_latents directory.
        """
        return not self.config.audio_only_training

    def train(self, mode: bool = True) -> "TextToMotionAudioStrategy":
        """Keep a frozen phoneme encoder deterministic across train/eval cycles."""
        super().train(mode)
        encoder = getattr(self, "phoneme_encoder", None)
        if encoder is not None and not any(p.requires_grad for p in encoder.parameters()):
            encoder.eval()
        return self

    def set_phoneme_encoder(self, phoneme_encoder: nn.Module) -> None:
        """Set the phoneme encoder (called by trainer if use_phoneme_encoder=True).

        Args:
            phoneme_encoder: PhonemeEncoder instance
        """
        self.phoneme_encoder = phoneme_encoder
        self.use_phoneme_encoder = True
        phoneme_encoder.train(self.training and any(p.requires_grad for p in phoneme_encoder.parameters()))
        logger.info("Phoneme encoder enabled for audio conditioning")

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Text-to-motion-audio training requires audio latents (GT) and
        per-modality text conditions, plus optional motion latents and phonemes.

        Motion and audio now consume their own Gemma-encoded conditions:

          - `motion_conditions/` → injected into the video/motion Modality
          - `speak_conditions/`   → injected into the audio Modality

        AUDCAP (`audcaps/`) and phonemes (`phonemes/`) continue to feed only
        the audio branch.

        When audio_only_training=True, motion latents are NOT loaded (the
        strategy feeds a 1-token dummy to the motion branch with loss=0);
        motion_conditions is still required so the dummy Modality has a valid
        context tensor.
        """
        sources = {}

        if not self.config.audio_only_training:
            sources[self.config.motion_latents_dir] = "motion_latents"

        sources[self.config.audio_latents_dir] = "audio_latents"
        if self.config.joint_conditions_dir:
            # Load a single merged motion+speech+AUDCAP caption once. The
            # preparation path aliases it to both modality contexts without
            # duplicating files or dataloader I/O.
            sources[self.config.joint_conditions_dir] = "joint_conditions"
        else:
            sources[self.config.motion_conditions_dir] = "motion_conditions"
            sources[self.config.speak_conditions_dir] = "speak_conditions"

        # Add phoneme data source if phoneme encoder is enabled
        if self.use_phoneme_encoder:
            sources["phonemes"] = "phonemes"

        # Add AUDCAP data source if voice/timbre conditioning is enabled.
        # Samples without an `audcaps/<sid>.pt` file are automatically dropped
        # by PrecomputedDataset._all_source_files_exist (no extra filtering needed).
        if self.use_audcap:
            sources["audcaps"] = "audcaps"

        return sources

    def _run_feature_extractor_and_connectors(
        self,
        raw_features: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Backwards-compatible wrapper that feeds the SAME encoded features
        into both motion and audio paths. Used by legacy callers / tests
        that don't yet distinguish ``motion_conditions`` from
        ``speak_conditions``. New code should call ``_process_single_modality``
        directly with two separate feature sources.
        """
        motion_emb, motion_mask = self._process_single_modality(
            raw_features=raw_features,
            attention_mask=attention_mask,
            connector=self.video_connector,
            connector_norm=self.video_connector_norm,
        )
        audio_emb, audio_mask = self._process_single_modality(
            raw_features=raw_features,
            attention_mask=attention_mask,
            connector=self.audio_connector,
            connector_norm=self.audio_connector_norm,
        )
        return motion_emb, audio_emb, audio_mask

    def _process_single_modality(
        self,
        raw_features: Tensor,
        attention_mask: Tensor,
        connector: nn.Module | None,
        connector_norm: nn.Module | None,
        *,
        features_are_projected: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Process Gemma features for ONE modality (motion OR audio) through
        the shared feature_extractor + the modality-specific connector.

        Pipeline (mirrors the original per-modality branch of
        ``_run_feature_extractor_and_connectors``):

         1. feature_extractor (if gemma_num_layers > 1): Linear(3840*N → 3840)
         2. connector (if provided for this modality): 1-D transformer over
            the projected features
         3. optional residual from raw Gemma features (when
            ``connector_residual=True``)
         4. connector_norm LayerNorm on the connector output

        Args:
            raw_features: (B, seq, 3840*N) — Gemma features for this modality
                          When gemma_num_layers=1: (B, seq, 3840)
                          For embedding_first_last: (B, seq, 11520)
            attention_mask: (B, seq) int64 — tokenizer attention mask (0=pad, 1=valid)
            connector: ``self.video_connector`` or ``self.audio_connector`` (or None)
            connector_norm: matching ``video_connector_norm`` / ``audio_connector_norm``
                            (or None)

        Returns:
            encoded: (B, seq, 3840) — zero-masked at padded positions
            binary_attention_mask: (B, seq) int64 — 1=valid, 0=pad
        """
        encoded_features = (
            raw_features if features_are_projected else self._project_text_features(raw_features)
        )
        dtype = encoded_features.dtype

        # Binary mask for bypass branches (projected Gemma features zero-masked)
        binary_mask_3d = attention_mask.unsqueeze(-1).to(dtype)  # (B, seq, 1)

        if connector is None:
            encoded = encoded_features * binary_mask_3d
            return encoded, attention_mask

        # ── Run this modality's connector ──
        # Build additive attention mask
        additive_mask = (attention_mask - 1).to(dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(dtype).max
        additive_mask = torch.where(
            additive_mask == 0,
            torch.tensor(-1e-5, dtype=dtype, device=additive_mask.device),
            additive_mask,
        )

        if next(connector.parameters()).dtype != dtype:
            connector = connector.to(dtype=dtype)

        encoded, conn_mask = connector(encoded_features, additive_mask)
        # Connector returns additive mask — threshold to binary.
        binary = (conn_mask >= -9000.0).to(torch.int64).reshape(
            encoded.shape[0], encoded.shape[1], 1
        )
        encoded = encoded * binary

        if self.config.connector_residual:
            if getattr(connector, "num_learnable_registers", None):
                # The connector compacts valid text and appends K registers, so
                # compact the residual in the same layout. Registers and batch
                # alignment padding deliberately receive no raw-text residual.
                raw_residual = torch.zeros_like(encoded)
                source_binary = attention_mask.to(torch.bool)
                for batch_idx in range(encoded_features.shape[0]):
                    valid = encoded_features[batch_idx, source_binary[batch_idx]]
                    raw_residual[batch_idx, : valid.shape[0]] = valid
            else:
                raw_residual = encoded_features * binary_mask_3d
            encoded = encoded + raw_residual
        if connector_norm is not None:
            encoded = connector_norm(encoded)

        binary_attention_mask = binary.squeeze(-1)
        return encoded, binary_attention_mask

    def _project_text_features(self, raw_features: Tensor) -> Tensor:
        """Project cached last-N Gemma features to the shared 3840-D text width."""
        expected_raw_dim = 3840 * self._gemma_num_layers
        if raw_features.shape[-1] != expected_raw_dim:
            raise ValueError(
                "cached Gemma feature width does not match gemma_num_layers: "
                f"got {raw_features.shape[-1]}, expected {expected_raw_dim} "
                f"for N={self._gemma_num_layers}"
            )
        if self.feature_extractor is not None:
            dtype = raw_features.dtype
            if next(self.feature_extractor.parameters()).dtype != dtype:
                self.feature_extractor = self.feature_extractor.to(dtype=dtype)
            return self.feature_extractor(raw_features)  # (B, seq, 3840)
        return raw_features  # N=1 is already 3840-dim

    def _prepare_direct_text_condition(
        self,
        raw_features: Tensor,
        attention_mask: Tensor,
        *,
        target_batch: int,
        target_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Zero-mask and right-pad projected Gemma features to a modality length.

        Projection (for N>1) happens once before this method. The
        returned tensor is concatenated with noisy latent tokens immediately
        before the model's existing modality input projection, matching the
        conditioning layout used by F5-TTS.
        """
        if raw_features.dim() == 2:
            raw_features = raw_features.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if raw_features.dim() != 3 or attention_mask.dim() != 2:
            raise ValueError(
                "direct text concat expects features [B,S,D] and mask [B,S], "
                f"got {tuple(raw_features.shape)} and {tuple(attention_mask.shape)}"
            )
        if raw_features.shape[:2] != attention_mask.shape:
            raise ValueError(
                "text feature/mask dimensions disagree: "
                f"features={tuple(raw_features.shape)}, mask={tuple(attention_mask.shape)}"
            )
        if raw_features.shape[-1] != self.config.text_conditioning_dim:
            raise ValueError(
                "raw text width does not match text_conditioning_dim: "
                f"got {raw_features.shape[-1]}, expected {self.config.text_conditioning_dim}"
            )
        if raw_features.shape[0] != target_batch:
            if raw_features.shape[0] == 1:
                raw_features = raw_features.expand(target_batch, -1, -1)
                attention_mask = attention_mask.expand(target_batch, -1)
            else:
                raise ValueError(
                    f"text batch {raw_features.shape[0]} does not match latent batch {target_batch}"
                )

        raw_features = raw_features.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        # Gemma uses left padding, while F5-style feature concatenation expects
        # effective text tokens at the start of the modality timeline. Compact
        # every sample independently before right-padding/truncating; otherwise
        # slicing a 320-token caption to a 129-token motion sequence selects only
        # the left-padding zeros for ordinary short captions.
        condition = torch.zeros_like(raw_features)
        for batch_idx in range(target_batch):
            valid_features = raw_features[batch_idx, attention_mask[batch_idx]]
            valid_count = valid_features.shape[0]
            if valid_count:
                condition[batch_idx, :valid_count] = valid_features

        source_seq_len = condition.shape[1]
        if source_seq_len > target_seq_len:
            condition = condition[:, :target_seq_len]
        elif source_seq_len < target_seq_len:
            padding = torch.zeros(
                target_batch,
                target_seq_len - source_seq_len,
                self.config.text_conditioning_dim,
                device=device,
                dtype=dtype,
            )
            condition = torch.cat((condition, padding), dim=1)
        return condition

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for text-to-motion-audio training with optional random cropping."""
        # Stage-1 audio replay batches (drawn inside a joint run) carry no
        # `motion_latents` source; route them through the audio-only path with
        # the video branch force-disabled so their conditioning
        # distribution matches stage-1 training exactly.
        is_replay = "motion_latents" not in batch
        if self.config.audio_only_training or is_replay:
            model_inputs = self._prepare_audio_only_inputs(
                batch,
                timestep_sampler,
                disable_video_branch=True if is_replay else None,
            )
            if is_replay:
                model_inputs = replace(model_inputs, is_replay=True)
            return model_inputs
        # Fall through to joint motion+audio path below.
        # Get pre-encoded motion latents (ground truth targets)
        motion_data = batch["motion_latents"]
        motion_latents = motion_data["latents"]  # After dataset normalization: (B, C=256, T, H=1, W=1)

        # Get audio latents (ground truth targets)
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]  # (B, C, T, F)

        # Get motion dimensions
        num_frames = motion_data["num_frames"][0].item()
        audio_timesteps = audio_data["num_time_steps"][0].item()

        # Motion latents are already in the correct format (B, C, F, H, W) from dataset
        # If they're 3D (B, C, T), add spatial dimensions
        if motion_latents.dim() == 3:
            motion_latents = motion_latents.unsqueeze(-1).unsqueeze(-1)  # (B, 256, T, 1, 1)
        # else: already (B, C, F, H, W) from _normalize_video_latents

        # === DATA AUGMENTATION: Random crop for memory efficiency ===
        # Audio feature rate ≈ 2.51x Motion frame rate (empirically observed)
        # If we crop motion, we must crop audio proportionally to maintain sync
        if self.config.max_motion_frames is not None and num_frames > self.config.max_motion_frames:
            # Random start frame for motion
            max_start = num_frames - self.config.max_motion_frames
            start_frame = torch.randint(0, max_start + 1, (1,)).item()
            end_frame = start_frame + self.config.max_motion_frames

            # Crop motion: (B, C, T, H, W) -> (B, C, max_frames, H, W)
            motion_latents = motion_latents[:, :, start_frame:end_frame, :, :]

            # Calculate corresponding audio crop using the observed ratio
            # Audio timesteps ≈ 2.51 * Motion frames
            audio_motion_ratio = audio_timesteps / num_frames
            audio_start = int(start_frame * audio_motion_ratio)
            audio_end = int(end_frame * audio_motion_ratio)

            # Ensure audio_end doesn't exceed bounds
            audio_end = min(audio_end, audio_timesteps)

            # Crop audio: (B, C, T, F) -> (B, C, cropped_T, F)
            audio_latents = audio_latents[:, :, audio_start:audio_end, :]

            # Update dimensions
            num_frames = self.config.max_motion_frames
            audio_timesteps = audio_end - audio_start

            logger.debug(
                f"Random crop: motion [{start_frame}:{end_frame}] ({num_frames} frames), "
                f"audio [{audio_start}:{audio_end}] ({audio_timesteps} timesteps), "
                f"ratio: {audio_timesteps / num_frames:.2f}"
            )

        # Patchify motion latents: (B, 256, T, 1, 1) -> (B, seq_len, 256)
        motion_latents = self._video_patchifier.patchify(motion_latents)

        # Project motion latents if enabled
        # Shape: (B, seq_len, 256) -> (B, seq_len, 128)
        if self.motion_proj is not None:
            motion_latents = self.motion_proj(motion_latents)

        # Handle FPS
        fps = motion_data.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else 10.0  # Motion latent FPS (30/3)

        # Batch size from motion latents (training batch)
        batch_size = motion_latents.shape[0]

        # ── CFG Unconditional Dropout ──
        # With probability unconditional_prob, replace ALL conditions with unconditional embeddings
        # This teaches the model the unconditional distribution needed for classifier-free guidance
        cfg_dropout = False
        if self.training and self.config.unconditional_prob > 0:
            if torch.rand(1).item() < self.config.unconditional_prob:
                cfg_dropout = True

        # ── Per-modality text conditions ──
        # Motion and audio now consume SEPARATE Gemma-encoded captions:
        #   - `batch["motion_conditions"]` → injected into the video Modality
        #   - `batch["speak_conditions"]`  → injected into the audio Modality
        # Each is Gemma-encoded offline (same schema as the legacy `conditions/`)
        # and is processed independently through the shared feature_extractor
        # and its per-modality connector.
        joint_conditions = batch.get("joint_conditions")
        if joint_conditions is not None:
            motion_conditions = joint_conditions
            audio_conditions = joint_conditions
        else:
            motion_conditions = batch["motion_conditions"]
            audio_conditions = batch["speak_conditions"]
        # Direct-concat consumes the untouched raw features.  The optional
        # motion cross-attention path consumes a second, connector-processed
        # view of the same motion_conditions tensor.
        motion_concat_features = None
        motion_concat_attention_mask = None

        # ── New format: pre-connector Gemma features ──
        # feature_extractor + connectors in *this* strategy are trainable,
        # so we run them every forward pass.
        if "encoded_features" in motion_conditions and "encoded_features" in audio_conditions:
            motion_encoded = motion_conditions["encoded_features"]
            motion_raw_mask = motion_conditions["raw_attention_mask"]
            motion_token_mask = motion_conditions.get("prompt_token_mask")

            audio_encoded = audio_conditions["encoded_features"]
            audio_raw_mask = audio_conditions["raw_attention_mask"]
            audio_token_mask = audio_conditions.get("prompt_token_mask")

            # Ensure 2-D batch dimension
            for tens, name in ((motion_encoded, "motion_encoded"),
                               (audio_encoded, "audio_encoded")):
                if tens.dim() == 2:
                    if name == "motion_encoded":
                        motion_encoded = tens.unsqueeze(0)
                    else:
                        audio_encoded = tens.unsqueeze(0)
            for tens, name in ((motion_raw_mask, "motion_raw_mask"),
                               (audio_raw_mask, "audio_raw_mask")):
                if tens.dim() == 1:
                    if name == "motion_raw_mask":
                        motion_raw_mask = tens.unsqueeze(0)
                    else:
                        audio_raw_mask = tens.unsqueeze(0)

            # Cast to model dtype / device (shared by both modalities)
            target_device = motion_latents.device
            target_dtype = torch.float32
            motion_encoded = motion_encoded.to(device=target_device, dtype=target_dtype)
            audio_encoded = audio_encoded.to(device=target_device, dtype=target_dtype)
            motion_raw_mask = motion_raw_mask.to(device=target_device)
            audio_raw_mask = audio_raw_mask.to(device=target_device)

            if self.config.direct_text_concat:
                # Project cached N-layer features once; both direct concat and
                # the optional connector consume this shared 3840-D view.
                motion_projected = self._project_text_features(motion_encoded)
                audio_projected = self._project_text_features(audio_encoded)
                motion_concat_features = motion_projected
                motion_concat_attention_mask = motion_raw_mask
                # Optionally create a second DuoFlow-style motion context.  The
                # connector replaces padding with learnable thinking tokens;
                # this processed sequence is used only by motion cross-attn.
                if self.config.motion_text_cross_attention:
                    if self.video_connector is None:
                        raise RuntimeError(
                            "motion_text_cross_attention=true requires a motion Text Connector"
                        )
                    motion_prompt_embeds, motion_attention_mask = self._process_single_modality(
                        raw_features=motion_projected,
                        attention_mask=motion_raw_mask,
                        connector=self.video_connector,
                        connector_norm=self.video_connector_norm,
                        features_are_projected=True,
                    )
                else:
                    motion_prompt_embeds = motion_projected
                    motion_attention_mask = motion_raw_mask
                audio_prompt_embeds = audio_projected
                audio_attention_mask = audio_raw_mask
            else:
                # Legacy main-text cross-attention path.
                motion_prompt_embeds, motion_attention_mask = self._process_single_modality(
                    raw_features=motion_encoded,
                    attention_mask=motion_raw_mask,
                    connector=self.video_connector,
                    connector_norm=self.video_connector_norm,
                )
                audio_prompt_embeds, audio_attention_mask = self._process_single_modality(
                    raw_features=audio_encoded,
                    attention_mask=audio_raw_mask,
                    connector=self.audio_connector,
                    connector_norm=self.audio_connector_norm,
                )

        else:
            # ── Legacy format: post-connector embeddings (frozen) ──
            # Both modalities pull from the SAME legacy `conditions` entry;
            # this branch only triggers if the preprocessed data still uses
            # the old single-`conditions/` schema (kept for back-compat tests).
            conditions = motion_conditions if "video_prompt_embeds" in motion_conditions else audio_conditions
            motion_prompt_embeds = conditions["video_prompt_embeds"]
            audio_prompt_embeds = conditions["audio_prompt_embeds"]
            motion_attention_mask = conditions["prompt_attention_mask"]
            audio_attention_mask = conditions["prompt_attention_mask"]
            prompt_token_mask = conditions.get("prompt_token_mask")
            motion_token_mask = prompt_token_mask
            audio_token_mask = prompt_token_mask
            if self.config.direct_text_concat:
                motion_concat_features = motion_prompt_embeds
                motion_concat_attention_mask = motion_attention_mask

        # Normalize shapes: some connectors return [B, 1, T, D]
        if motion_prompt_embeds.dim() == 4 and motion_prompt_embeds.size(1) == 1:
            motion_prompt_embeds = motion_prompt_embeds.squeeze(1)
        if audio_prompt_embeds.dim() == 4 and audio_prompt_embeds.size(1) == 1:
            audio_prompt_embeds = audio_prompt_embeds.squeeze(1)
        if motion_attention_mask.dim() == 3 and motion_attention_mask.size(1) == 1:
            motion_attention_mask = motion_attention_mask.squeeze(1)
        if audio_attention_mask.dim() == 3 and audio_attention_mask.size(1) == 1:
            audio_attention_mask = audio_attention_mask.squeeze(1)
        if motion_token_mask is not None and motion_token_mask.dim() == 3 and motion_token_mask.size(1) == 1:
            motion_token_mask = motion_token_mask.squeeze(1)
        if audio_token_mask is not None and audio_token_mask.dim() == 3 and audio_token_mask.size(1) == 1:
            audio_token_mask = audio_token_mask.squeeze(1)

        # Preserve per-sample masks. Caption lengths differ within a batch after
        # packing as [valid text] + [K registers] + [masked alignment padding].
        def _ensure_mask_format(mask: torch.Tensor) -> torch.Tensor:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            return mask

        def _align_context(ctx: torch.Tensor, target_batch: int) -> torch.Tensor:
            if ctx.size(0) != target_batch:
                if ctx.size(0) == 1:
                    ctx = ctx.expand(target_batch, -1, -1)
                else:
                    ctx = ctx[:target_batch]
            return ctx

        motion_prompt_embeds = _align_context(motion_prompt_embeds, batch_size)
        audio_prompt_embeds = _align_context(audio_prompt_embeds, batch_size)
        motion_context_mask = _ensure_mask_format(motion_attention_mask)
        audio_context_mask = _ensure_mask_format(audio_attention_mask)

        # ── CFG Unconditional Dropout: zero out all text conditions ──
        if cfg_dropout:
            motion_prompt_embeds = torch.zeros_like(motion_prompt_embeds)
            audio_prompt_embeds = torch.zeros_like(audio_prompt_embeds)
            if motion_concat_features is not None:
                motion_concat_features = torch.zeros_like(motion_concat_features)
            if motion_concat_attention_mask is not None:
                motion_concat_attention_mask = torch.zeros_like(motion_concat_attention_mask)
            # Zero out masks so model sees "no valid text tokens"
            motion_context_mask = torch.zeros_like(motion_context_mask)
            audio_context_mask = torch.zeros_like(audio_context_mask)

        # Process phoneme embeddings as SEPARATE context (not concatenated)
        phoneme_context = None
        phoneme_context_mask = None
        phoneme_ids_raw = None
        if self.use_phoneme_encoder and "phonemes" in batch:
            phoneme_idx = batch["phonemes"]  # (B, pad_length), int64 tensor
            # Retain raw ids for diagnostics; CTC is disabled for CFG-dropout batches.
            phoneme_ids_raw = phoneme_idx

            if cfg_dropout:
                # CFG dropout: replace phoneme IDs with all-pad (unconditional)
                phoneme_idx = torch.zeros_like(phoneme_idx)

            # Encode phonemes: returns (phoneme_embeds, phoneme_mask)
            # phoneme_embeds: (B, pad_length, out_channels=192)
            # phoneme_mask: (B, pad_length), bool tensor
            phoneme_embeds, phoneme_mask = self.phoneme_encoder(phoneme_idx, None)

            # Project phoneme embeddings to audio dimension
            target_dtype = audio_prompt_embeds.dtype
            phoneme_embeds = phoneme_embeds.to(dtype=target_dtype)
            # Ensure phoneme_proj matches dtype (handles both Linear and ARIABridge)
            if hasattr(self.phoneme_proj, 'weight'):
                if self.phoneme_proj.weight.dtype != target_dtype:
                    self.phoneme_proj = self.phoneme_proj.to(dtype=target_dtype)
            else:
                self.phoneme_proj = self.phoneme_proj.to(dtype=target_dtype)
            phoneme_embeds = self.phoneme_proj(phoneme_embeds)  # (B, 500, 3840)

            # Keep the full-batch phoneme mask (B, pad_length).
            # Different samples have different phoneme lengths — collapsing
            # to batch=1 via _ensure_mask_format made all samples share the
            # first sample's mask, causing incorrect masking for the rest.
            phoneme_mask = phoneme_mask.long()  # already (B, pad_length) from PhonemeEncoder

            # Re-zero padding positions after phoneme projection.
            # ARIABridge self-attention (no internal padding mask) plus Linear
            # bias can transform originally-zero padding positions into non-zero
            # embeddings. Without re-zeroing, the cross-attention would attend
            # to garbage values at those positions.
            phoneme_embeds = phoneme_embeds * phoneme_mask.unsqueeze(-1).to(phoneme_embeds.dtype)

            # Store as separate context for the audio Modality's phoneme cross-attention
            phoneme_context = phoneme_embeds
            phoneme_context_mask = phoneme_mask

        # Process AUDCAP (voice/timbre) features as SEPARATE context.
        # AUDCAP was Gemma-encoded offline and stored under `audcaps/<sid>.pt`
        # with the same schema as `conditions/<sid>.pt` (encoded_features,
        # raw_attention_mask, prompt_token_mask). We run them through the
        # shared `feature_extractor` (when gemma_num_layers > 1) to project
        # multi-layer Gemma features back to 3840-dim, then hand the 3840-dim
        # features to the audio Modality — the model's
        # `audio_audcap_caption_projection` then projects to audio_inner_dim
        # inside the preprocessor, exactly mirroring the phoneme path.
        audcap_context = None
        audcap_context_mask = None
        if self.use_audcap and "audcaps" in batch:
            audcap_batch = batch["audcaps"]
            audcap_encoded = audcap_batch.get("encoded_features")
            audcap_mask = audcap_batch.get("raw_attention_mask")

            if audcap_encoded is not None and audcap_mask is not None:
                # Ensure 2-D batch dim (single-sample on disk => loader may give 1-D)
                if audcap_encoded.dim() == 2:
                    audcap_encoded = audcap_encoded.unsqueeze(0)
                if audcap_mask.dim() == 1:
                    audcap_mask = audcap_mask.unsqueeze(0)

                # Move to the audio device/dtype used for the other conditions
                audcap_target_device = audio_prompt_embeds.device
                audcap_target_dtype = torch.float32
                audcap_encoded = audcap_encoded.to(device=audcap_target_device, dtype=audcap_target_dtype)
                audcap_mask = audcap_mask.to(device=audcap_target_device)

                # Reuse the shared feature_extractor (3840*N -> 3840 when
                # gemma_num_layers > 1; bypassed when == 1).
                if self.feature_extractor is not None:
                    if next(self.feature_extractor.parameters()).dtype != audcap_target_dtype:
                        self.feature_extractor = self.feature_extractor.to(dtype=audcap_target_dtype)
                    audcap_features = self.feature_extractor(audcap_encoded)  # (B, S_a, 3840)
                else:
                    audcap_features = audcap_encoded  # already (B, S_a, 3840)

                # Align batch dim with the rest of the conditions
                audcap_features = _align_context(audcap_features, batch_size)
                audcap_mask_aligned = _ensure_mask_format(audcap_mask)  # (1, S_a) int64

                # CFG dropout for AUDCAP (mirrors phoneme behavior).
                if cfg_dropout:
                    audcap_features = torch.zeros_like(audcap_features)
                    audcap_mask_aligned = torch.zeros_like(audcap_mask_aligned)

                audcap_context = audcap_features
                audcap_context_mask = audcap_mask_aligned.long()
            else:
                logger.warning(
                    "[audcap] batch['audcaps'] missing encoded_features/raw_attention_mask; "
                    "skipping AUDCAP conditioning for this step."
                )

        motion_seq_len = motion_latents.shape[1]
        device = motion_latents.device
        dtype = motion_latents.dtype

        # Sample noise and sigmas (same for both motion and audio)
        sigmas = timestep_sampler.sample_for(motion_latents)
        motion_noise = torch.randn_like(motion_latents)

        # Apply noise to motion: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_motion = (1 - sigmas_expanded) * motion_latents + sigmas_expanded * motion_noise

        # Compute motion targets (velocity prediction)
        motion_targets = motion_noise - motion_latents

        # Create per-token timesteps for motion (all tokens use the same sigma)
        motion_timesteps = sigmas.view(-1, 1).expand(-1, motion_seq_len)

        # Generate motion positions
        motion_positions = self._get_motion_positions(
            num_frames=num_frames,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        motion_concat_condition = None
        motion_cross_context = motion_prompt_embeds
        motion_cross_context_mask = motion_context_mask
        if self.config.direct_text_concat:
            if motion_concat_features is None or motion_concat_attention_mask is None:
                raise RuntimeError("direct_text_concat is missing raw motion condition features")
            motion_concat_condition = self._prepare_direct_text_condition(
                motion_concat_features,
                motion_concat_attention_mask,
                target_batch=batch_size,
                target_seq_len=motion_seq_len,
                device=noisy_motion.device,
                dtype=noisy_motion.dtype,
            )
            if not self.config.motion_text_cross_attention:
                motion_cross_context = None
                motion_cross_context_mask = None

        # Create motion Modality (this is what we're generating)
        motion_modality = Modality(
            enabled=True,
            latent=noisy_motion,
            timesteps=motion_timesteps,
            positions=motion_positions,
            context=motion_cross_context,
            context_mask=motion_cross_context_mask,
            concat_condition=motion_concat_condition,
        )

        # Motion loss mask: all tokens contribute to loss
        motion_loss_mask = torch.ones(batch_size, motion_seq_len, dtype=torch.bool, device=device)

        # Handle audio (also being generated)
        audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
            audio_latents=audio_latents,  # Already cropped in sync with motion
            sigmas=sigmas,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=(audio_attention_mask if self.config.direct_text_concat else audio_context_mask),
            phoneme_context=phoneme_context,
            phoneme_context_mask=phoneme_context_mask,
            audcap_context=audcap_context,
            audcap_context_mask=audcap_context_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        return ModelInputs(
            video=motion_modality,  # Motion (being generated from text)
            audio=audio_modality,    # Audio (being generated from text)
            video_targets=motion_targets,
            audio_targets=audio_targets,
            video_loss_mask=motion_loss_mask,
            audio_loss_mask=audio_loss_mask,
            phoneme_ids=phoneme_ids_raw,
            ctc_enabled=not cfg_dropout,
        )

    def _prepare_audio_only_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
        *,
        disable_video_branch: bool | None = None,
    ) -> ModelInputs:
        """Prepare inputs for audio-only training (audio_only_training=True).

        No motion data is loaded from disk. A 1-token zero latent is fed to the
        motion branch so the transformer forward path stays the same, but
        motion loss is forced to 0 by ``compute_loss_components``.

        ``disable_video_branch`` overrides ``audio_only_disable_video_branch``
        for this call. Stage-1 replay batches pass True so the video branch is
        skipped regardless of the run-level flag (joint runs keep it False for
        the legacy path, but replay must match stage-1 conditioning).

        Audio latency:
            - Read audio_latents (already padded-to-batch-max by collate_fn).
            - If ``max_audio_steps`` is set and any sample exceeds it, apply
              a per-sample random crop on the time dim. ``audio_loss_mask``
              (B, T) from collate marks which positions are real vs padding.
        """
        # ── Audio latents (already padded in collate_fn) ──
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]  # (B, C=8, T, F=16), all T equal
        # Optional per-sample loss mask from collate_fn (B, T), True=valid
        # May be absent (older collate path / motion-only batch) — default to all-True
        external_audio_mask = audio_data.get("audio_loss_mask")
        B, _, audio_T, _ = audio_latents.shape

        device = audio_latents.device
        dtype = audio_latents.dtype

        # ── Optional audio cropping (max_audio_steps) ──
        if (
            self.config.max_audio_steps is not None
            and self.config.max_audio_steps > 0
            and audio_T > self.config.max_audio_steps
        ):
            max_start = audio_T - self.config.max_audio_steps
            start = int(torch.randint(0, max_start + 1, (1,)).item())
            end = start + self.config.max_audio_steps
            audio_latents = audio_latents[:, :, start:end, :]
            audio_T = self.config.max_audio_steps
            if external_audio_mask is not None:
                external_audio_mask = external_audio_mask[:, start:end]
            logger.debug(
                f"[audio_only] Random audio crop [{start}:{end}] ({audio_T} steps)"
            )

        # ── Sample sigmas ──
        # sampler.sample_for expects a 3-D tensor (B, seq, ...); patchify the
        # audio latents to (B, T, 128) so we can sample sigmas of shape (B,).
        sigmas = timestep_sampler.sample_for(self._audio_patchifier.patchify(audio_latents[:, :, :1, :]))  # (B,)

        # ── Dummy motion branch (1 token, zero latent) ──
        # Patchifier expects (B, C, F, H, W); with patch_size=1 this is the
        # same shape as input. We use F=1, H=1, W=1 → patchify gives (B, 1, C*1*1*1).
        motion_latents_dummy = torch.zeros((B, 256, 1, 1, 1), device=device, dtype=dtype)
        motion_latents_patched = self._video_patchifier.patchify(motion_latents_dummy)  # (B, 1, 256)
        # Projection (if enabled) — keeps tensor shapes consistent
        if self.motion_proj is not None:
            motion_latents_patched = self.motion_proj.to(dtype=dtype)(motion_latents_patched)
        motion_targets = torch.zeros_like(motion_latents_patched)
        motion_timesteps = sigmas.view(-1, 1).expand(-1, 1)
        motion_positions = self._get_motion_positions(
            num_frames=1, batch_size=B, fps=10.0, device=device, dtype=dtype,
        )

        # ── Text / phoneme conditions (same path as joint training) ──
        # Per-modality conditions: motion from `motion_conditions/`, audio
        # from `speak_conditions/`.
        # In audio-only mode the motion Modality is disabled downstream
        # (see `audio_only_disable_video_branch`), so motion_conditions is
        # allowed to be absent from the batch — fall back to audio_conditions
        # as a benign placeholder so the forward pass doesn't crash. The video
        # branch's `context` is unused when enabled=False.
        joint_conditions = batch.get("joint_conditions")
        if joint_conditions is not None:
            audio_conditions = joint_conditions
            motion_conditions = joint_conditions
        else:
            audio_conditions = batch["speak_conditions"]
            motion_conditions = batch.get("motion_conditions", audio_conditions)
        if "encoded_features" in motion_conditions and "encoded_features" in audio_conditions:
            motion_encoded = motion_conditions["encoded_features"]
            motion_raw_mask = motion_conditions["raw_attention_mask"]
            motion_token_mask = motion_conditions.get("prompt_token_mask")

            audio_encoded = audio_conditions["encoded_features"]
            audio_raw_mask = audio_conditions["raw_attention_mask"]
            audio_token_mask = audio_conditions.get("prompt_token_mask")

            if motion_encoded.dim() == 2:
                motion_encoded = motion_encoded.unsqueeze(0)
            if motion_raw_mask.dim() == 1:
                motion_raw_mask = motion_raw_mask.unsqueeze(0)
            if audio_encoded.dim() == 2:
                audio_encoded = audio_encoded.unsqueeze(0)
            if audio_raw_mask.dim() == 1:
                audio_raw_mask = audio_raw_mask.unsqueeze(0)

            target_device = motion_latents_patched.device
            target_dtype = torch.float32
            motion_encoded = motion_encoded.to(device=target_device, dtype=target_dtype)
            motion_raw_mask = motion_raw_mask.to(device=target_device)
            audio_encoded = audio_encoded.to(device=target_device, dtype=target_dtype)
            audio_raw_mask = audio_raw_mask.to(device=target_device)

            if self.config.direct_text_concat:
                motion_prompt_embeds = self._project_text_features(motion_encoded)
                motion_attention_mask = motion_raw_mask
                audio_prompt_embeds = self._project_text_features(audio_encoded)
                audio_attention_mask = audio_raw_mask
            else:
                motion_prompt_embeds, motion_attention_mask = self._process_single_modality(
                    raw_features=motion_encoded,
                    attention_mask=motion_raw_mask,
                    connector=self.video_connector,
                    connector_norm=self.video_connector_norm,
                )
                audio_prompt_embeds, audio_attention_mask = self._process_single_modality(
                    raw_features=audio_encoded,
                    attention_mask=audio_raw_mask,
                    connector=self.audio_connector,
                    connector_norm=self.audio_connector_norm,
                )
        else:
            # ── Legacy fallback: shared `conditions/` entry ──
            conditions = motion_conditions if "video_prompt_embeds" in motion_conditions else audio_conditions
            motion_prompt_embeds = conditions["video_prompt_embeds"]
            audio_prompt_embeds = conditions["audio_prompt_embeds"]
            motion_attention_mask = conditions["prompt_attention_mask"]
            audio_attention_mask = conditions["prompt_attention_mask"]
            motion_token_mask = conditions.get("prompt_token_mask")
            audio_token_mask = conditions.get("prompt_token_mask")

        if motion_prompt_embeds.dim() == 4 and motion_prompt_embeds.size(1) == 1:
            motion_prompt_embeds = motion_prompt_embeds.squeeze(1)
        if audio_prompt_embeds.dim() == 4 and audio_prompt_embeds.size(1) == 1:
            audio_prompt_embeds = audio_prompt_embeds.squeeze(1)
        if motion_attention_mask.dim() == 3 and motion_attention_mask.size(1) == 1:
            motion_attention_mask = motion_attention_mask.squeeze(1)
        if audio_attention_mask.dim() == 3 and audio_attention_mask.size(1) == 1:
            audio_attention_mask = audio_attention_mask.squeeze(1)
        if motion_token_mask is not None and motion_token_mask.dim() == 3 and motion_token_mask.size(1) == 1:
            motion_token_mask = motion_token_mask.squeeze(1)
        if audio_token_mask is not None and audio_token_mask.dim() == 3 and audio_token_mask.size(1) == 1:
            audio_token_mask = audio_token_mask.squeeze(1)

        def _ensure_mask_format(mask: torch.Tensor) -> torch.Tensor:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            return mask

        def _align_context(ctx: torch.Tensor, target_batch: int) -> torch.Tensor:
            if ctx.size(0) != target_batch:
                if ctx.size(0) == 1:
                    ctx = ctx.expand(target_batch, -1, -1)
                else:
                    ctx = ctx[:target_batch]
            return ctx

        motion_prompt_embeds = _align_context(motion_prompt_embeds, B)
        audio_prompt_embeds = _align_context(audio_prompt_embeds, B)
        motion_context_mask = _ensure_mask_format(motion_attention_mask)
        audio_context_mask = _ensure_mask_format(audio_attention_mask)

        # CFG dropout
        cfg_dropout = False
        if self.training and self.config.unconditional_prob > 0:
            if torch.rand(1).item() < self.config.unconditional_prob:
                cfg_dropout = True
        if cfg_dropout:
            motion_prompt_embeds = torch.zeros_like(motion_prompt_embeds)
            audio_prompt_embeds = torch.zeros_like(audio_prompt_embeds)
            motion_context_mask = torch.zeros_like(motion_context_mask)
            audio_context_mask = torch.zeros_like(audio_context_mask)

        # ── Phoneme context (optional) ──
        phoneme_context = None
        phoneme_context_mask = None
        phoneme_ids_raw = None
        if self.use_phoneme_encoder and "phonemes" in batch:
            phoneme_idx = batch["phonemes"]
            # Retain raw ids for diagnostics; CTC is disabled for CFG-dropout batches.
            phoneme_ids_raw = phoneme_idx
            if cfg_dropout:
                phoneme_idx = torch.zeros_like(phoneme_idx)
            phoneme_embeds, phoneme_mask = self.phoneme_encoder(phoneme_idx, None)
            target_dtype = audio_prompt_embeds.dtype
            phoneme_embeds = phoneme_embeds.to(dtype=target_dtype)
            if hasattr(self.phoneme_proj, "weight"):
                if self.phoneme_proj.weight.dtype != target_dtype:
                    self.phoneme_proj = self.phoneme_proj.to(dtype=target_dtype)
            else:
                self.phoneme_proj = self.phoneme_proj.to(dtype=target_dtype)
            phoneme_embeds = self.phoneme_proj(phoneme_embeds)
            phoneme_mask = phoneme_mask.long()
            phoneme_embeds = phoneme_embeds * phoneme_mask.unsqueeze(-1).to(phoneme_embeds.dtype)
            phoneme_context = phoneme_embeds
            phoneme_context_mask = phoneme_context_mask if phoneme_context_mask is not None else phoneme_mask

        # ── AUDCAP context (optional; mirrors phoneme path) ──
        audcap_context = None
        audcap_context_mask = None
        if self.use_audcap and "audcaps" in batch:
            audcap_batch = batch["audcaps"]
            audcap_encoded = audcap_batch.get("encoded_features")
            audcap_mask = audcap_batch.get("raw_attention_mask")

            if audcap_encoded is not None and audcap_mask is not None:
                if audcap_encoded.dim() == 2:
                    audcap_encoded = audcap_encoded.unsqueeze(0)
                if audcap_mask.dim() == 1:
                    audcap_mask = audcap_mask.unsqueeze(0)

                audcap_target_device = audio_prompt_embeds.device
                audcap_target_dtype = torch.float32
                audcap_encoded = audcap_encoded.to(device=audcap_target_device, dtype=audcap_target_dtype)
                audcap_mask = audcap_mask.to(device=audcap_target_device)

                if self.feature_extractor is not None:
                    if next(self.feature_extractor.parameters()).dtype != audcap_target_dtype:
                        self.feature_extractor = self.feature_extractor.to(dtype=audcap_target_dtype)
                    audcap_features = self.feature_extractor(audcap_encoded)
                else:
                    audcap_features = audcap_encoded

                audcap_features = _align_context(audcap_features, B)
                audcap_mask_aligned = _ensure_mask_format(audcap_mask)

                if cfg_dropout:
                    audcap_features = torch.zeros_like(audcap_features)
                    audcap_mask_aligned = torch.zeros_like(audcap_mask_aligned)

                audcap_context = audcap_features
                audcap_context_mask = audcap_mask_aligned.long()
            else:
                logger.warning(
                    "[audcap] batch['audcaps'] missing encoded_features/raw_attention_mask; "
                    "skipping AUDCAP conditioning for this step."
                )

        # Keep a valid dummy motion input for the preprocessor. When disabled,
        # the transformer skips motion attention and cross-modal attention.
        # The legacy enabled mode retains a single zero motion token.
        if disable_video_branch is None:
            disable_video_branch = self.config.audio_only_disable_video_branch
        motion_modality_enabled = not disable_video_branch
        motion_concat_condition = None
        motion_cross_context = motion_prompt_embeds
        motion_cross_context_mask = motion_context_mask
        if self.config.direct_text_concat:
            motion_concat_condition = self._prepare_direct_text_condition(
                motion_prompt_embeds,
                motion_attention_mask,
                target_batch=B,
                target_seq_len=1,
                device=motion_latents_patched.device,
                dtype=motion_latents_patched.dtype,
            )
            motion_cross_context = None
            motion_cross_context_mask = None
        motion_modality = Modality(
            enabled=motion_modality_enabled,
            latent=motion_latents_patched,
            timesteps=motion_timesteps,
            positions=motion_positions,
            context=motion_cross_context,
            context_mask=motion_cross_context_mask,
            concat_condition=motion_concat_condition,
        )
        motion_loss_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)

        # ── Build audio Modality (real) ──
        audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
            audio_latents=audio_latents,
            sigmas=sigmas,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=(audio_attention_mask if self.config.direct_text_concat else audio_context_mask),
            phoneme_context=phoneme_context,
            phoneme_context_mask=phoneme_context_mask,
            audcap_context=audcap_context,
            audcap_context_mask=audcap_context_mask,
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        # Combine: collate-supplied per-sample audio validity mask (B, max_T)
        # must be intersected with the per-sample seq_len mask that
        # _prepare_audio_inputs produced (in case the seq_len was cropped to
        # ``max_audio_steps`` within this function).
        if external_audio_mask is not None:
            if external_audio_mask.shape[1] != audio_loss_mask.shape[1]:
                # After cropping the length differs; trim or pad the external mask
                T_now = audio_loss_mask.shape[1]
                if external_audio_mask.shape[1] >= T_now:
                    external_audio_mask = external_audio_mask[:, :T_now]
                else:
                    pad = torch.zeros(
                        (external_audio_mask.shape[0], T_now - external_audio_mask.shape[1]),
                        dtype=external_audio_mask.dtype,
                        device=external_audio_mask.device,
                    )
                    external_audio_mask = torch.cat([external_audio_mask, pad], dim=1)
            audio_loss_mask = audio_loss_mask & external_audio_mask.to(audio_loss_mask.device)

        # Inject the same pad mask into audio self-attn so padded tokens (zero
        # latents from collate padding) are explicitly ignored, preventing them
        # from polluting real tokens via self-attention. Inference has no pad,
        # so leaving self_attn_mask=None there is equivalent.
        # Shape (B, 1, T, T): query axis all-valid, key axis gated by pad mask.
        # This additive key-mask shape is handled identically by Pytorch and
        # XFormers backends (avoids the ndim==2 path that drops the batch dim).
        self_attn_mask = audio_loss_mask[:, None, None, :].to(dtype)
        audio_modality = replace(audio_modality, self_attn_mask=self_attn_mask)

        return ModelInputs(
            video=motion_modality,
            audio=audio_modality,
            video_targets=motion_targets,
            audio_targets=audio_targets,
            video_loss_mask=motion_loss_mask,
            audio_loss_mask=audio_loss_mask,
            phoneme_ids=phoneme_ids_raw,
            ctc_enabled=not cfg_dropout,
        )

    def _get_motion_positions(
        self,
        num_frames: int,
        batch_size: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Create motion position encodings in legacy or real-second mode.

        Motion is 1D temporal sequence, so we create simplified position encodings.
        Format: (B, 3, seq_len, 2) to match video position format.
        With ``use_correct_motion_time_coordinates=True``, each token receives
        its real temporal interval ``[i / fps, (i + 1) / fps]`` in seconds.

        Args:
            num_frames: Number of motion frames
            batch_size: Batch size
            fps: Frames per second
            device: Device
            dtype: Data type

        Returns:
            Position tensor (B, 3, seq_len, 2)
        """
        # For motion, we only have temporal dimension (no spatial H/W)
        # Position format: (B, 3, seq_len, 2)
        # - dim 0: temporal position
        # - dim 1, 2: spatial (set to 0 for motion)

        if fps <= 0:
            raise ValueError(f"Motion FPS must be positive, got {fps}")

        positions = torch.zeros(
            (batch_size, 3, num_frames, 2),
            device=device,
            dtype=dtype,
        )

        if self.config.use_correct_motion_time_coordinates:
            # Real seconds: token i spans one latent-frame interval.
            token_indices = torch.arange(num_frames, device=device, dtype=dtype)
            positions[:, 0, :, 0] = token_indices / fps
            positions[:, 0, :, 1] = (token_indices + 1) / fps
        else:
            # Legacy behavior retained so old checkpoints remain reproducible.
            time_pos = torch.linspace(0, 1, num_frames, device=device, dtype=dtype)
            legacy_time = time_pos / fps
            positions[:, 0, :, 0] = legacy_time
            positions[:, 0, :, 1] = legacy_time

        return positions

    def _prepare_audio_inputs(
        self,
        audio_latents: Tensor,
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        phoneme_context: Tensor | None,
        phoneme_context_mask: Tensor | None,
        audcap_context: Tensor | None,
        audcap_context_mask: Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint text-to-motion-audio training.

        Args:
            audio_latents: Pre-cropped audio latents (B, C=8, T, Freq=16), already synchronized with motion
            sigmas: Sampled sigma values (same as motion for synchronization)
            audio_prompt_embeds: Audio context embeddings from text
            prompt_attention_mask: Attention mask for context
            phoneme_context: Projected phoneme embeddings for separate cross-attention, or None
            phoneme_context_mask: Attention mask for phoneme context, or None
            batch_size: Batch size
            device: Target device
            dtype: Target dtype

        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Audio latents already cropped to match motion length (maintaining 2.51x ratio)
        # Format: (B, C=8, T, Freq=16)

        # Patchify audio latents: (B, 8, T, 16) -> (B, T, 128)
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to audio (same sigma as motion for synchronization)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        audio_concat_condition = None
        audio_cross_context = audio_prompt_embeds
        audio_cross_context_mask = prompt_attention_mask
        if self.config.direct_text_concat:
            audio_concat_condition = self._prepare_direct_text_condition(
                audio_prompt_embeds,
                prompt_attention_mask,
                target_batch=batch_size,
                target_seq_len=audio_seq_len,
                device=noisy_audio.device,
                dtype=noisy_audio.dtype,
            )
            audio_cross_context = None
            audio_cross_context_mask = None

        # Compute audio targets (velocity prediction)
        audio_targets = audio_noise - audio_latents

        # Audio timesteps: all tokens use the sampled sigma
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        # Generate audio positions
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_cross_context,
            context_mask=audio_cross_context_mask,
            concat_condition=audio_concat_condition,
            phoneme_context=phoneme_context,  # Separate phoneme cross-attention
            phoneme_context_mask=phoneme_context_mask,
            audcap_context=audcap_context,  # Separate AUDCAP cross-attention
            audcap_context_mask=audcap_context_mask,
        )

        # Audio loss mask: all tokens contribute to loss
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss_components(
        self,
        video_pred: Tensor,  # Actually motion_pred
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Compute motion/audio losses and combined total loss."""
        # Motion generation loss (optional). Zeroed for audio-only runs, for
        # stage-1 replay batches (is_replay), and for any batch whose motion
        # loss mask is empty — an all-zero mask would make the masked-mean
        # normalization below divide by zero.
        if (
            self.config.audio_only_training
            or inputs.is_replay
            or not bool(inputs.video_loss_mask.any())
        ):
            motion_loss = torch.zeros((), device=video_pred.device, dtype=video_pred.dtype)
        else:
            motion_loss = (video_pred - inputs.video_targets).pow(2)
            motion_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
            motion_loss = motion_loss.mul(motion_loss_mask).div(motion_loss_mask.mean())
            motion_loss = motion_loss.mean() * self.config.motion_loss_weight

        # Audio generation loss
        if audio_pred is None or inputs.audio_targets is None:
            logger.warning("Audio prediction or targets is None, using motion loss only")
            return motion_loss, None, motion_loss

        audio_loss = (audio_pred - inputs.audio_targets).pow(2)
        audio_loss_mask = inputs.audio_loss_mask.unsqueeze(-1).float()
        audio_loss = audio_loss.mul(audio_loss_mask).div(audio_loss_mask.mean())
        audio_loss_weight = self.config.audio_loss_weight
        if inputs.is_replay:
            audio_loss_weight = audio_loss_weight * self.config.replay_loss_weight
        audio_loss = audio_loss.mean() * audio_loss_weight

        # Combined weighted loss
        total_loss = audio_loss if self.config.audio_only_training else (motion_loss + audio_loss)

        # Phoneme-CTC auxiliary alignment loss. Monotonically binds audio
        # frames to the phoneme sequence — the alignment signal the
        # flow-matching MSE itself provides almost no gradient for.
        ctc_loss = self._compute_ctc_aux_loss(inputs)
        if ctc_loss is not None:
            total_loss = total_loss + self.config.aux_ctc_weight * ctc_loss
            self.last_aux_losses = {"ctc_loss": float(ctc_loss.detach().item())}
        else:
            self.last_aux_losses = {}

        return motion_loss, audio_loss, total_loss

    def build_per_tap_ctc_heads(self, tap_blocks: list[int]) -> None:
        """Create one independent CTC head per tapped block.

        Called by the trainer right after strategy construction (before
        trainable parameters are collected) when the model taps more than
        one block. The shared audio_ctc_head is dropped in this mode.
        """
        tap_blocks = sorted(set(tap_blocks))
        if self.config.aux_ctc_weight <= 0 or not tap_blocks or len(tap_blocks) <= 1:
            return
        for block in tap_blocks:
            self.audio_ctc_heads[str(block)] = nn.Linear(
                self.config.aux_ctc_hidden_dim, self.config.aux_ctc_num_classes
            )
        self.audio_ctc_head = None
        logger.info(
            "[ctc_aux] independent per-tap CTC heads for blocks %s",
            list(self.audio_ctc_heads.keys()),
        )

    def _compute_ctc_aux_loss(self, inputs: ModelInputs) -> Tensor | None:
        """Phoneme-CTC loss over the tapped audio-branch hidden states.

        Returns None when the auxiliary loss is disabled or the feature tap /
        phoneme ids are unavailable (e.g. during validation, where the trainer
        deliberately leaves ``audio_hidden_aux`` unset so the eval loss stays
        a pure flow-matching quantity).
        """
        # Report skipped auxiliary loss periodically so disabled supervision is visible.
        reasons = []
        if self.config.aux_ctc_weight <= 0:
            reasons.append("aux_ctc_weight<=0")
        if not inputs.ctc_enabled:
            reasons.append("cfg_dropout")
        if self.audio_ctc_head is None and not getattr(self, "audio_ctc_heads", None):
            reasons.append("no_ctc_head")
        if inputs.audio_hidden_aux is None:
            reasons.append("no_tapped_hidden")
        if inputs.phoneme_ids is None:
            reasons.append("no_phoneme_ids")
        if inputs.audio_loss_mask is None:
            reasons.append("no_audio_loss_mask")
        if inputs.audio is None:
            reasons.append("no_audio_modality")
        elif inputs.audio.phoneme_context_mask is None:
            reasons.append("no_phoneme_context_mask")
        if reasons:
            self._ctc_skip_count = getattr(self, "_ctc_skip_count", 0) + 1
            if self._ctc_skip_count == 1 or self._ctc_skip_count % 1000 == 0:
                logger.info(
                    "[ctc_aux] skipped (%s) [occurrence #%d]",
                    ",".join(reasons),
                    self._ctc_skip_count,
                )
            return None

        # Multi-tap: stack taps on the batch dim. F.ctc_loss(reduction='mean')
        # over B*n_tap rows equals the mean of per-tap CTC losses.
        multi = getattr(inputs, "audio_hidden_aux_multi", None)
        if multi:
            hidden = torch.cat(
                [h.to(dtype=torch.float32) for h in multi], dim=0
            )
        else:
            hidden = inputs.audio_hidden_aux.to(dtype=torch.float32)
        if getattr(self, "audio_ctc_heads", None) and multi:
            # Independent heads: apply each tap's head to its B-row slice
            # of the stacked batch, then re-concatenate the logits.
            n_taps = len(multi)
            per_b = hidden.shape[0] // n_taps
            blocks = list(
                getattr(inputs, "audio_hidden_aux_tap_blocks", None) or range(n_taps)
            )
            chunks = []
            for k, block in enumerate(blocks):
                key = str(block)
                if key not in self.audio_ctc_heads:
                    raise ValueError(f"Missing CTC head for tapped block {block}")
                chunks.append(self.audio_ctc_heads[key](hidden[k * per_b : (k + 1) * per_b]))
            if not chunks:
                return None
            logits = torch.cat(chunks, dim=0)  # (B*n_tap, T, num_classes)
        else:
            logits = self.audio_ctc_head(hidden)  # (B, T, num_classes)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, num_classes)

        input_lengths = inputs.audio_loss_mask.sum(dim=-1).long().clamp(min=1)
        input_lengths = input_lengths.to(device=log_probs.device)

        batch = inputs.phoneme_ids.shape[0]
        n_tap = hidden.shape[0] // batch
        blank_id = self.config.aux_ctc_num_classes - 1
        targets: list[Tensor] = []
        for b in range(batch):
            # Slice the original ids; pad id 0 is filtered. CFG-dropout
            # batches have already returned above.
            ids = inputs.phoneme_ids[b]
            targets.append(ids[ids > 0].to(device=log_probs.device))  # drop pad id 0
        target_lengths = torch.tensor(
            [t.numel() for t in targets], device=log_probs.device, dtype=torch.long
        )
        if int(target_lengths.max().item()) == 0:
            self._ctc_empty_target_count = getattr(self, "_ctc_empty_target_count", 0) + 1
            if self._ctc_empty_target_count == 1 or self._ctc_empty_target_count % 1000 == 0:
                logger.info(
                    "[ctc_aux] skipped (empty_phoneme_targets) [occurrence #%d]",
                    self._ctc_empty_target_count,
                )
            return None
        max_target_len = int(target_lengths.max().item())
        target_batch = torch.zeros(
            (batch, max_target_len), device=log_probs.device, dtype=torch.long
        )
        for b, t in enumerate(targets):
            if t.numel() > 0:
                target_batch[b, : t.numel()] = t

        if n_tap > 1:
            # Hidden is stacked [tap1_all_B, tap2_all_B, ...], so targets must be
            # tiled (whole-batch copies), NOT repeat_interleave'd per sample.
            input_lengths = input_lengths.repeat(n_tap)
            target_lengths = target_lengths.repeat(n_tap)
            target_batch = target_batch.repeat(n_tap, 1)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=target_batch,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=blank_id,
            zero_infinity=True,
        )
        return ctc_loss

    def compute_loss(
        self,
        video_pred: Tensor,  # Actually motion_pred
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute weighted joint loss for motion and audio generation."""
        motion_loss, audio_loss, total_loss = self.compute_loss_components(video_pred, audio_pred, inputs)

        # Log individual losses for monitoring
        if audio_loss is not None and hasattr(self, '_log_individual_losses'):
            logger.debug(
                f"Loss breakdown - Motion: {motion_loss.item():.4f}, "
                f"Audio: {audio_loss.item():.4f}, Total: {total_loss.item():.4f}"
            )

        return total_loss
