# Modified for DuoFlow; see NOTICE.md.
"""Motion-to-Audio training strategy.

This strategy implements motion-to-audio generation training where:
- Motion latents are used instead of video latents
- Standard noise application and loss computation
- Supports joint motion-audio training
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from motionspeech_core.model.transformer.modality import Modality
from motionspeech_trainer import logger
from motionspeech_trainer.timestep_samplers import TimestepSampler
from motionspeech_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class MotionToAudioConfig(TrainingStrategyConfigBase):
    """Configuration for motion-to-audio training strategy."""

    name: Literal["motion_to_audio"] = "motion_to_audio"

    motion_conditioning_p: float = Field(
        default=0.0,
        description="Probability of conditioning on motion (similar to first frame conditioning)",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=True,
        description="Whether to include audio in training (should always be True for motion-to-audio)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents",
    )

    motion_latents_dir: str = Field(
        default="motion_latents",
        description="Directory name for motion latents",
    )


class MotionToAudioStrategy(TrainingStrategy):
    """Motion-to-audio training strategy.

    This strategy implements motion-to-audio generation where:
    - Motion latents replace video latents
    - Motion is treated as pseudo-video (no spatial dimensions)
    - Supports joint motion-audio training
    """

    config: MotionToAudioConfig

    def __init__(self, config: MotionToAudioConfig):
        """Initialize strategy with configuration.

        Args:
            config: Motion-to-audio configuration
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return True

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Motion-to-audio training requires motion latents, audio latents, and text conditions."""
        sources = {
            self.config.motion_latents_dir: "motion_latents",
            self.config.audio_latents_dir: "audio_latents",
            "conditions": "conditions",
        }
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for motion-to-audio training."""
        # Get pre-encoded motion latents - format: (B, C=256, T)
        motion_data = batch["motion_latents"]
        motion_latents = motion_data["latents"]

        # Get motion dimensions
        num_frames = motion_data["num_frames"][0].item()

        # Motion latents are (B, C=256, T)
        # We need to reshape to pseudo-video format: (B, C, F, H=1, W=1)
        motion_latents = motion_latents.unsqueeze(-1).unsqueeze(-1)  # (B, 256, T, 1, 1)

        # Patchify motion latents: (B, 256, T, 1, 1) -> (B, seq_len, 256)
        motion_latents = self._video_patchifier.patchify(motion_latents)

        # Handle FPS
        fps = motion_data.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else 30.0  # BEAT2 default FPS

        # Get text embeddings
        conditions = batch["conditions"]
        motion_prompt_embeds = conditions["video_prompt_embeds"]  # Reuse video embeddings for motion
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = motion_latents.shape[0]
        motion_seq_len = motion_latents.shape[1]
        device = motion_latents.device
        dtype = motion_latents.dtype

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(motion_latents)
        motion_noise = torch.randn_like(motion_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_motion = (1 - sigmas_expanded) * motion_latents + sigmas_expanded * motion_noise

        # Compute motion targets (velocity prediction)
        motion_targets = motion_noise - motion_latents

        # Create per-token timesteps (no conditioning for now)
        motion_timesteps = sigmas.view(-1, 1).expand(-1, motion_seq_len)

        # Generate motion positions
        motion_positions = self._get_motion_positions(
            num_frames=num_frames,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Create motion Modality
        motion_modality = Modality(
            enabled=True,
            latent=noisy_motion,
            timesteps=motion_timesteps,
            positions=motion_positions,
            context=motion_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Motion loss mask: all tokens contribute to loss
        motion_loss_mask = torch.ones(batch_size, motion_seq_len, dtype=torch.bool, device=device)

        # Handle audio
        audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
            batch=batch,
            sigmas=sigmas,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        return ModelInputs(
            video=motion_modality,  # Motion replaces video
            audio=audio_modality,
            video_targets=motion_targets,
            audio_targets=audio_targets,
            video_loss_mask=motion_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _get_motion_positions(
        self,
        num_frames: int,
        batch_size: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Create motion position encodings.

        Motion is 1D temporal sequence, so we create simplified position encodings.
        Format: (B, 3, seq_len, 2) to match video position format.

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

        positions = torch.zeros(
            (batch_size, 3, num_frames, 2),
            device=device,
            dtype=dtype,
        )

        # Temporal positions (normalized to [0, 1] and scaled by FPS)
        time_pos = torch.linspace(0, 1, num_frames, device=device, dtype=dtype)
        positions[:, 0, :, 0] = time_pos / fps  # Temporal start
        positions[:, 0, :, 1] = time_pos / fps  # Temporal end

        return positions

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint motion-audio training.

        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as motion)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype

        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Get audio latents - format: (B, C=8, T, Freq=16)
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]

        # Patchify audio latents: (B, 8, T, 16) -> (B, T, 128)
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to audio (same sigma as motion)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Compute audio targets
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
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Audio loss mask: all tokens contribute to loss
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,  # Actually motion_pred
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for motion and audio."""
        # Motion loss
        motion_loss = (video_pred - inputs.video_targets).pow(2)
        motion_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        motion_loss = motion_loss.mul(motion_loss_mask).div(motion_loss_mask.mean())
        motion_loss = motion_loss.mean()

        # Audio loss
        if audio_pred is None or inputs.audio_targets is None:
            logger.warning("Audio prediction or targets is None, using motion loss only")
            return motion_loss

        audio_loss = (audio_pred - inputs.audio_targets).pow(2)
        audio_loss_mask = inputs.audio_loss_mask.unsqueeze(-1).float()
        audio_loss = audio_loss.mul(audio_loss_mask).div(audio_loss_mask.mean())
        audio_loss = audio_loss.mean()

        # Combined loss (equal weighting)
        return motion_loss + audio_loss
