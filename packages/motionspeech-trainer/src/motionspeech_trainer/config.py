# Modified for DuoFlow; see NOTICE.md.
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationInfo, field_validator, model_validator

from motionspeech_trainer.quantization import QuantizationOptions
from motionspeech_trainer.training_strategies.base_strategy import TrainingStrategyConfigBase
from motionspeech_trainer.training_strategies.motion_to_audio import MotionToAudioConfig
from motionspeech_trainer.training_strategies.text_to_motion_audio import TextToMotionAudioConfig
from motionspeech_trainer.training_strategies.text_to_video import TextToVideoConfig
from motionspeech_trainer.training_strategies.video_to_video import VideoToVideoConfig


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode"""

    model_path: str | Path | None = Field(
        default=None,
        description="Model path - local path to safetensors checkpoint file",
    )

    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Path to text encoder (required for DuoFlow/Gemma models, optional for DuoFlow/T5 models)",
    )

    use_phoneme_encoder: bool = Field(
        default=False,
        description="Whether to use phoneme encoder for enhanced audio generation",
    )

    phoneme_encoder_checkpoint: str | Path | None = Field(
        default=None,
        description="Optional checkpoint path to load pretrained phoneme encoder weights",
    )

    phoneme_encoder_checkpoint_prefix: str | None = Field(
        default=None,
        description="Optional key prefix for phoneme encoder weights inside the checkpoint",
    )

    phoneme_encoder_trainable: bool = Field(
        default=False,
        description="Whether to train the phoneme encoder (unfreeze its weights)",
    )

    phoneme_encoder_vocab_size: int = Field(
        default=183,
        description="Vocabulary size for phoneme encoder (number of unique phoneme symbols)",
        ge=1,
    )

    phoneme_encoder_pad_length: int = Field(
        default=310,
        description="Maximum phoneme sequence length (sequences will be padded to this length)",
        ge=1,
    )

    phoneme_encoder_out_channels: int = Field(
        default=192,
        description="Output dimension of phoneme encoder embeddings",
        ge=1,
    )

    phoneme_encoder_hidden_channels: int = Field(
        default=192,
        description="Hidden dimension for phoneme encoder internal layers",
        ge=1,
    )

    phoneme_encoder_filter_channels: int = Field(
        default=768,
        description="Filter channels for FFN in phoneme encoder",
        ge=1,
    )

    phoneme_encoder_n_heads: int = Field(
        default=2,
        description="Number of attention heads in phoneme encoder",
        ge=1,
    )

    phoneme_encoder_n_layers: int = Field(
        default=6,
        description="Number of transformer layers in phoneme encoder",
        ge=1,
    )

    phoneme_encoder_kernel_size: int = Field(
        default=3,
        description="Kernel size for convolutional layers in phoneme encoder",
        ge=1,
    )

    phoneme_encoder_p_dropout: float = Field(
        default=0.1,
        description="Dropout probability for phoneme encoder",
        ge=0.0,
        le=1.0,
    )

    phoneme_encoder_add_blank: bool = Field(
        default=True,
        description=(
            "Whether phoneme data uses add_blank format (id=0 interleaved between phonemes). "
            "When True, the encoder mask correctly treats interleaved blanks as valid "
            "content and only masks trailing padding."
        ),
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    num_transformer_layers: int | None = Field(
        default=8,
        description=(
            "Number of transformer layers to use. If less than the original model, "
            "only the last N layers will be loaded (layer pruning for memory/parameter reduction). "
            "Set to None to use all layers from the original model. "
            "Example: Original model has 32 layers, set to 16 to use only last 16 layers (50% params)."
        ),
        ge=1,
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    allow_missing_checkpoint_keys: bool = Field(
        default=False,
        description=(
            "When True, load_checkpoint tolerates parameters that exist in the "
            "model but are missing from the checkpoint (they keep their random "
            "initialization). Use this when resuming from a checkpoint trained "
            "before a new module was added — e.g. stage-2 resuming from a "
            "stage-1 checkpoint that has no audio_audcap_* weights after the "
            "AUDCAP cross-attention was introduced. Keys present in the "
            "checkpoint but not in the model are still reported but do not "
            "block loading. Default False (strict load, any mismatch raises)."
        ),
    )

    transformer_in_channels: int | None = Field(
        default=None,
        description="Override transformer video input channels (default from checkpoint metadata)",
        ge=1,
    )

    transformer_out_channels: int | None = Field(
        default=None,
        description="Override transformer video output channels (default from checkpoint metadata)",
        ge=1,
    )

    audio_in_channels: int | None = Field(
        default=None,
        description=(
            "Override transformer audio input channels. For direct raw-text "
            "concatenation this is audio latent width plus text feature width."
        ),
        ge=1,
    )

    transformer_num_attention_heads: int | None = Field(
        default=None,
        description="Override transformer video attention heads (requires init from scratch)",
        ge=1,
    )

    transformer_attention_head_dim: int | None = Field(
        default=None,
        description="Override transformer video attention head dim (requires init from scratch)",
        ge=1,
    )

    transformer_cross_attention_dim: int | None = Field(
        default=None,
        description="Override transformer video cross-attention dim (requires init from scratch)",
        ge=1,
    )

    audio_num_attention_heads: int | None = Field(
        default=None,
        description="Override transformer audio attention heads (requires init from scratch)",
        ge=1,
    )

    audio_attention_head_dim: int | None = Field(
        default=None,
        description="Override transformer audio attention head dim (requires init from scratch)",
        ge=1,
    )

    audio_cross_attention_dim: int | None = Field(
        default=None,
        description="Override transformer audio cross-attention dim (requires init from scratch)",
        ge=1,
    )

    audio_caption_projection_use_skip: bool = Field(
        default=True,
        description=(
            'Enable a parallel linear residual branch in the audio caption projection. Match the checkpoint architecture.'
        ),
    )

    audio_caption_projection_act_fn: str = Field(
        default="gelu_tanh",
        description=(
            "Activation in the audio caption projection. Use 'none' for a single linear layer, or 'gelu_tanh' or 'silu' for a two-layer projection. Match the checkpoint architecture."
        ),
    )

    audio_phoneme_caption_projection_act_fn: str = Field(
        default="gelu_tanh",
        description=(
            "Activation function used inside the audio phoneme caption projection. "
            "Same semantics as audio_caption_projection_act_fn. Strongly recommended "
            "'none' for from-scratch training so that phoneme information actually "
            "reaches the audio branch."
        ),
    )

    audio_audcap_caption_projection_act_fn: str = Field(
        default="gelu_tanh",
        description=(
            "Activation function used inside the audio AUDCAP caption projection. "
            "Mirrors audio_phoneme_caption_projection_act_fn. Strongly recommended "
            "'none' for from-scratch training so that voice/timbre information "
            "actually reaches the audio branch."
        ),
    )

    video_caption_projection_use_skip: bool = Field(
        default=False,
        description=(
            'Enable a parallel linear residual branch in the motion caption projection. Match the checkpoint architecture.'
        ),
    )

    video_caption_projection_act_fn: str = Field(
        default="gelu_tanh",
        description=(
            "Activation in the motion caption projection. Use 'none' for a single linear layer, or 'gelu_tanh' or 'silu' for a two-layer projection. Match the checkpoint architecture."
        ),
    )

    use_text_cross_attention: bool = Field(
        default=True,
        description=(
            "Whether motion/speech text uses the legacy per-layer cross-attention. "
            "Set false when raw text features are concatenated with noisy latent "
            "tokens before the modality input projection. Phoneme and AUDCAP "
            "audio cross-attention are unaffected."
        ),
    )

    init_transformer_from_scratch: bool = Field(
        default=False,
        description="Initialize transformer weights from scratch (do not load checkpoint weights)",
    )

    use_phoneme_time_anchor: bool = Field(
        default=False,
        description=(
            "Add a monotonic position anchor (sinusoidal embedding of the "
            "normalized phoneme index) to the phoneme cross-attention context "
            "of the audio branch. Zero-initialized output projection: a no-op "
            "at resume time. Breaks the order-invariance of phoneme tokens so "
            "the audio branch can learn word-order alignment."
        ),
    )

    aux_ctc_tap_block: int | None = Field(
        default=None,
        description=(
            "Index of the audio transformer block whose output is tapped as "
            "the feature source for the CTC alignment auxiliary loss. None "
            "disables the tap. The CTC loss itself is enabled via the "
            "training_strategy.aux_ctc_weight key."
        ),
    )

    aux_ctc_tap_blocks: list[int] | None = Field(
        default_factory=lambda: list(range(8)),
        description=(
            "Audio block indices for independent per-layer CTC heads; layer losses are averaged. "
            "When omitted, all configured blocks are supervised (0-7 by default). Explicit null/[] retains single-tap behavior."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def default_ctc_taps(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        if "aux_ctc_tap_blocks" not in values:
            if "aux_ctc_tap_block" in values:
                values["aux_ctc_tap_blocks"] = None
            else:
                values["aux_ctc_tap_blocks"] = list(range(values.get("num_transformer_layers") or 8))
        return values

    @field_validator("aux_ctc_tap_blocks")
    @classmethod
    def normalize_ctc_taps(cls, taps: list[int] | None) -> list[int] | None:
        if not taps:
            return None
        if any(block < 0 for block in taps):
            raise ValueError("CTC tap indices must be nonnegative")
        return sorted(set(taps))

    @model_validator(mode="after")
    def validate_ctc_tap_range(self) -> "ModelConfig":
        if self.aux_ctc_tap_blocks and self.num_transformer_layers is not None:
            if max(self.aux_ctc_tap_blocks) >= self.num_transformer_layers:
                raise ValueError("CTC tap index must be smaller than num_transformer_layers")
        return self

    audioldm2_model_id: str | None = Field(
        default=None,
        description="AudioLDM2 model ID or local path for audio VAE/Vocoder during validation",
    )

    audioldm2_cache_dir: str | None = Field(
        default=None,
        description="Optional cache directory for AudioLDM2 model downloads",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str | Path | None) -> str | Path | None:
        """Validate that model_path is either a valid URL or an existing local path."""
        if v is None:
            return None
        is_url = str(v).startswith(("http://", "https://"))

        if is_url:
            raise ValueError(f"Model path cannot be a URL: {v}")

        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")

        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning"""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="List of modules to target with LoRA",
    )


def _get_strategy_discriminator(v: dict | TrainingStrategyConfigBase) -> str:
    """Discriminator function for strategy config union."""
    if isinstance(v, dict):
        return v.get("name", "text_to_video")
    return v.name


# Union type for all strategy configs with discriminator
TrainingStrategyConfig = Annotated[
    Annotated[TextToVideoConfig, Tag("text_to_video")]
    | Annotated[VideoToVideoConfig, Tag("video_to_video")]
    | Annotated[MotionToAudioConfig, Tag("motion_to_audio")]
    | Annotated[TextToMotionAudioConfig, Tag("text_to_motion_audio")],
    Discriminator(_get_strategy_discriminator),
]


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters"""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=30000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for AdamW optimizer. Set to 0 for no weight decay.",
    )

    audio_learning_rate: float | None = Field(
        default=None,
        description=(
            "Optional per-parameter-group learning rate for audio-branch parameters "
            "(parameters whose name matches 'audio_*' or 'av_ca_audio_*' or "
            "'av_ca_v2a_*', or whose name contains 'scale_shift_table_a2v_ca_audio'). "
            "When null, all trainable parameters share `learning_rate`. "
            "Useful for stage-2 fine-tuning when the audio branch was already "
            "trained on a different (e.g. larger) dataset and you want it to "
            "adapt more slowly than the motion branch."
        ),
    )

    audio_weight_decay: float | None = Field(
        default=None,
        description=(
            "Optional weight decay override for the audio parameter group. "
            "Falls back to `weight_decay` when null."
        ),
    )

    freeze_modules: list[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns (matched against fully-qualified parameter names via "
            "fnmatch). Parameters whose name matches any pattern are frozen "
            "(requires_grad=False) and excluded from the optimizer entirely. "
            "Examples: ['audio_*', 'av_ca_audio_*', 'av_ca_v2a_*', "
            "'*scale_shift_table_a2v_ca_audio*']. Use this to fully lock the "
            "audio branch during stage-2 fine-tuning so stage-1 weights are "
            "not disturbed."
        ),
    )

    warmup_steps: int = Field(
        default=0,
        description="Number of linear warmup steps before the main scheduler takes over.",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )

    use_ema: bool = Field(
        default=False,
        description="Use Exponential Moving Average of model weights for improved generation quality",
    )

    ema_decay: float = Field(
        default=0.9999,
        description="EMA decay rate. Higher values (e.g., 0.9999) give slower, more stable averaging.",
    )


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization"""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )

    process_group_timeout_seconds: int = Field(
        default=3600,
        description=(
            "Distributed process-group collective timeout in seconds. A longer "
            "timeout prevents slow checkpoint I/O from tripping the NCCL watchdog."
        ),
        gt=0,
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing"""

    preprocessed_data_root: str = Field(
        description="Path to folder containing preprocessed training data",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )

    # ----- STAGE-1 AUDIO REPLAY (anti-forgetting) -----
    # Optionally draw a fraction of training batches from a stage-1 audio-only
    # dataset (e.g. Emilia) so the audio branch keeps its speech capability
    # while the current stage trains the joint motion+audio task.
    replay_preprocessed_data_root: str | None = Field(
        default=None,
        description=(
            "Stage-1 (audio-only) preprocessed data root replayed during "
            "training. Must contain audio_latents/, speak_conditions/ and "
            "(when the phoneme encoder is enabled) phonemes/. Replay batches "
            "load no motion_latents source, which is how the trainer and the "
            "training strategy detect them. null disables replay."
        ),
    )

    replay_batch_prob: float = Field(
        default=0.0,
        description=(
            "Target fraction of training micro-batches drawn from the replay "
            "dataset (0 disables replay). Implemented as a deterministic "
            "schedule — every round(1/prob)-th micro-step — so all DDP ranks "
            "switch in lockstep. 0.2 = every 5th batch."
        ),
        ge=0.0,
        le=1.0,
    )

    replay_batch_size: int | None = Field(
        default=None,
        description=(
            "Batch size for the replay dataloader. null = use "
            "optimization.batch_size. Audio-only replay batches are cheaper "
            "than joint batches, so a larger value is affordable."
        ),
        ge=1,
    )


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training"""

    val_preprocessed_data_root: str | None = Field(
        default=None,
        description=(
            "Path to preprocessed data for latent-loss evaluation. "
            "Set to a train/val subset directory to choose the eval set."
        ),
    )

    fixed_noise_eval: bool = Field(
        default=False,
        description="Use fixed RNG seeds for deterministic latent-loss evaluation",
    )

    fixed_noise_seed: int = Field(
        default=1234,
        description="Seed used when fixed_noise_eval is enabled",
    )

    eval_batch_size: int | None = Field(
        default=None,
        description="Override batch size for latent-loss evaluation (None = use training batch size)",
        ge=1,
    )

    eval_progress_bar: bool = Field(
        default=False,
        description="Show a progress bar during latent-loss evaluation",
    )

    latent_eval_subset_size: int | None = Field(
        default=None,
        description=(
            "Evaluate latent loss on a fixed deterministic subset at regular "
            "checkpoint intervals. None evaluates the full validation set."
        ),
        ge=1,
    )

    latent_eval_subset_seed: int = Field(
        default=1234,
        description="Seed used to choose the fixed latent-loss evaluation subset",
    )

    latent_eval_full_interval: int | None = Field(
        default=None,
        description=(
            "Additionally evaluate latent loss on the full validation set at "
            "this optimization-step interval. The value should be a multiple "
            "of checkpoints.interval. None disables milestone full evaluation."
        ),
        ge=1,
    )

    prompts: list[str] = Field(
        default_factory=list,
        description="List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    video_dims: tuple[int, int, int] = Field(
        default=(960, 544, 97),
        description="Dimensions of validation videos (width, height, frames). "
        "Width and height must be divisible by 32. Frames must satisfy frames % 8 == 1 for DuoFlow.",
    )

    @field_validator("video_dims")
    @classmethod
    def validate_video_dims(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate video dimensions for DuoFlow compatibility."""
        width, height, frames = v

        # Special case: [1, 1, N] is allowed for motion-to-audio (no spatial dimensions)
        if width == 1 and height == 1:
            # Only validate frames for motion case
            if frames % 8 != 1:
                raise ValueError(f"Frames ({frames}) must satisfy frames % 8 == 1 for DuoFlow (e.g., 1, 9, 17, 25, ...)")
            return v

        # Standard video case: width and height must be divisible by 32
        if width % 32 != 0:
            raise ValueError(f"Width ({width}) must be divisible by 32")
        if height % 32 != 0:
            raise ValueError(f"Height ({height}) must be divisible by 32")
        if frames % 8 != 1:
            raise ValueError(f"Frames ({frames}) must satisfy frames % 8 == 1 for DuoFlow (e.g., 1, 9, 17, 25, ...)")

        return v

    frame_rate: float = Field(
        default=25.0,
        description="Frame rate for validation videos",
        gt=0,
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=50,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    videos_per_prompt: int = Field(
        default=1,
        description="Number of videos to generate per validation prompt",
        gt=0,
    )

    guidance_scale: float = Field(
        default=4.0,
        description="CFG guidance scale to use during validation",
        ge=1.0,
    )

    motion_guidance_scale: float | None = Field(default=None, ge=1.0)
    audio_guidance_scale: float | None = Field(default=None, ge=1.0)
    stg_scale: float = Field(
        default=1.0,
        description="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. "
        "Recommended value is 1.0. STG is combined with CFG for improved video quality.",
        ge=0.0,
    )

    stg_blocks: list[int] | None = Field(
        default=[29],
        description="Which transformer blocks to perturb for STG. "
        "None means all blocks are perturbed. Recommended for DuoFlow: [29].",
    )

    stg_mode: Literal["stg_av", "stg_v"] = Field(
        default="stg_av",
        description="STG mode: 'stg_av' skips both audio and video self-attention, "
        "'stg_v' skips only video self-attention.",
    )

    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio in validation samples. "
        "Independent of training strategy setting - you can generate audio "
        "in validation even when not training the audio branch.",
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    include_reference_in_output: bool = Field(
        default=False,
        description="For video-to-video training: concatenate the original reference video side-by-side "
        "with the generated output. The reference comes from the input video, not from the model's output.",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of images (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of images ({len(v)}) must match number of prompts ({num_prompts})")

        for image_path in v:
            if not Path(image_path).exists():
                raise ValueError(f"Image path '{image_path}' does not exist")

        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_reference_videos(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of reference videos (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})")

        for video_path in v:
            if not Path(video_path).exists():
                raise ValueError(f"Reference video path '{video_path}' does not exist")

        return v


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training"""

    resume_from_checkpoint: str | Path | None = Field(
        default=None,
        description=(
            "Resume a training run from an output directory, checkpoints directory, "
            "model weights file, or training-state file. The latest complete "
            "weights/state pair is selected automatically. Legacy weights-only "
            "runs restore model weights, global step, and scheduler position."
        ),
    )

    save_training_state: bool = Field(
        default=True,
        description=(
            "Save optimizer, scheduler, gradient-scaler, global-step, EMA, and "
            "RNG state beside each model checkpoint for a full-state resume."
        ),
    )

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )

    precision: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision to use when saving checkpoint weights. Options: 'bfloat16' or 'float32'.",
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration"""

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging"""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="motionspeech",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )


class TensorBoardConfig(ConfigBaseModel):
    """Configuration for TensorBoard logging.

    When enabled, training/eval/checkpoint metrics are written to a
    ``SummaryWriter`` located at ``log_dir`` (defaults to
    ``output_dir/tensorboard``). When TensorBoard is enabled, W&B is
    automatically disabled to avoid duplicate logging.
    """

    enabled: bool = Field(
        default=False,
        description="Whether to enable TensorBoard logging",
    )

    log_dir: str | None = Field(
        default=None,
        description=(
            "Directory to write TensorBoard event files to. "
            "If None, defaults to '<output_dir>/tensorboard'."
        ),
    )

    flush_secs: int = Field(
        default=30,
        description="How often (in seconds) to flush the TensorWriter buffers to disk.",
        gt=0,
    )

    log_every_n_steps: int = Field(
        default=1,
        description=(
            "Frequency (in optimization steps) at which train/* metrics are written "
            "to TensorBoard. Validation/eval/checkpoint events are always written "
            "when they happen."
        ),
        gt=0,
    )

    comment: str = Field(
        default="",
        description="Comment suffix appended to the TensorBoard log_dir.",
    )


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training"""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class MotionSpeechTrainerConfig(ConfigBaseModel):
    """Unified configuration for DuoFlow training"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    training_strategy: TrainingStrategyConfig = Field(
        default_factory=TextToVideoConfig,
        description="Training strategy configuration. Determines the training mode and its parameters.",
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    tensorboard: TensorBoardConfig = Field(default_factory=TensorBoardConfig)

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def validate_strategy_compatibility(self) -> "MotionSpeechTrainerConfig":
        """Validate that training strategy and other configurations are compatible."""

        if self.checkpoints.resume_from_checkpoint and self.model.load_checkpoint:
            raise ValueError(
                "Set either checkpoints.resume_from_checkpoint for a training resume "
                "or model.load_checkpoint for a weights-only warm start, not both"
            )

        # Check that reference videos are provided when using video_to_video strategy
        if (
            self.training_strategy.name == "video_to_video"
            and self.validation.interval
            and not self.validation.reference_videos
        ):
            raise ValueError(
                "reference_videos must be provided in validation config when using video_to_video strategy"
            )

        # Check that LoRA config is provided when training mode is lora
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")

        # Check that LoRA config is provided when using video_to_video strategy
        if self.training_strategy.name == "video_to_video" and self.model.training_mode != "lora":
            raise ValueError("Training mode must be 'lora' when using video_to_video strategy")

        # For motion-based strategies, allow [1, 1, N] dimensions (motion has no spatial dimensions)
        if self.training_strategy.name in ("motion_to_audio", "text_to_motion_audio"):
            width, height, frames = self.validation.video_dims
            if width != 1 or height != 1:
                raise ValueError(
                    f"For {self.training_strategy.name} strategy, video_dims must be [1, 1, frames]. "
                    f"Got [{width}, {height}, {frames}]"
                )

        return self
