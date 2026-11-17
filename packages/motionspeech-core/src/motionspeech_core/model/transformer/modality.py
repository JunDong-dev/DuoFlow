# Modified for DuoFlow; see NOTICE.md.
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor | None
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    # Optional raw per-token condition concatenated with ``latent`` immediately
    # before the modality input projection.  This is the F5-TTS-style path:
    # the condition is only length-aligned by the caller and receives no
    # standalone projection or cross-attention.
    concat_condition: torch.Tensor | None = None
    phoneme_context: torch.Tensor | None = None
    phoneme_context_mask: torch.Tensor | None = None
    audcap_context: torch.Tensor | None = None
    audcap_context_mask: torch.Tensor | None = None
    self_attn_mask: torch.Tensor | None = None
