# Modified for DuoFlow; see NOTICE.md.
import math

import torch


class PhonemeTimeAnchor(torch.nn.Module):
    """Temporal anchor for phoneme conditioning tokens.

    Phoneme context tokens are produced by a text encoder that adds no
    positional information, so the phoneme cross-attention sees an
    order-invariant bag of phonemes. This module gives every valid phoneme
    token a monotonic "position in utterance" prior: its normalized index
    p = (j+1)/N mapped through a multi-frequency sinusoidal basis and a
    zero-initialized linear projection. The result is added to the phoneme
    context (the K/V side of the phoneme cross-attention), which lets audio
    queries at time t discover which phoneme keys are near their expected
    position without presuming uniform phoneme durations.

    The output projection is zero-initialized so that a model resumed from a
    checkpoint trained without this module computes an identical function at
    step 0 (the anchor starts as a no-op and learns from there).
    """

    def __init__(self, dim: int, num_freqs: int = 32, max_cycle: float = 512.0) -> None:
        super().__init__()
        # Geometric frequency grid: phoneme index fractions p in (0, 1] are
        # encoded at cycles from 1 (coarse order) to max_cycle (fine local
        # order), giving the cross-attention both utterance-scale and
        # phoneme-scale positional resolution.
        freqs = 2.0 ** torch.linspace(0.0, math.log2(max_cycle), num_freqs)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = torch.nn.Linear(num_freqs * 2, dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, phoneme_context_mask: torch.Tensor) -> torch.Tensor:
        """Compute additive anchor embeddings for phoneme context tokens.

        Args:
            phoneme_context_mask: raw 0/1 mask of shape (B, N); 1 = valid
                phoneme token, 0 = padding.

        Returns:
            Tensor of shape (B, N, dim); zeros at padded positions and at
            initialization.
        """
        mask = phoneme_context_mask.to(torch.float32)
        lengths = mask.sum(dim=-1).clamp(min=1.0)  # (B,)
        idx = torch.arange(mask.shape[1], device=mask.device, dtype=torch.float32).unsqueeze(0)
        # Normalized position in (0, 1]: j+1 over the per-sample valid length.
        positions = (idx + 1.0) / lengths.unsqueeze(1)
        positions = positions.clamp(max=1.0) * mask
        angles = positions.unsqueeze(-1) * self.freqs.to(dtype=torch.float32) * (2.0 * math.pi)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        out = self.proj(embedding.to(dtype=self.proj.weight.dtype))
        return out * mask.unsqueeze(-1).to(dtype=out.dtype)
