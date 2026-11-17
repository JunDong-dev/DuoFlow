# Modified for DuoFlow; see NOTICE.md.
from dataclasses import dataclass, replace

import torch

from motionspeech_core.model.transformer.adaln import AdaLayerNormSingle
from motionspeech_core.model.transformer.modality import Modality
from motionspeech_core.model.transformer.rope import (
    MotionSpeechRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)
from motionspeech_core.model.transformer.text_projection import PixArtAlphaTextProjection


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor | None
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    cross_positional_embeddings: torch.Tensor | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool
    phoneme_context: torch.Tensor | None = None
    phoneme_context_mask: torch.Tensor | None = None
    audcap_context: torch.Tensor | None = None
    audcap_context_mask: torch.Tensor | None = None
    self_attn_mask: torch.Tensor | None = None


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: MotionSpeechRopeType,
        phoneme_caption_projection: PixArtAlphaTextProjection | None = None,
        audcap_caption_projection: PixArtAlphaTextProjection | None = None,
        phoneme_time_anchor: torch.nn.Module | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.phoneme_caption_projection = phoneme_caption_projection
        self.audcap_caption_projection = audcap_caption_projection
        self.phoneme_time_anchor = phoneme_time_anchor
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

    def _prepare_timestep(
        self, timestep: torch.Tensor, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""

        timestep = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = self.adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )

        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor | None,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare context for transformer blocks."""
        if context is None:
            return None, None
        batch_size = x.shape[0]
        context = self.caption_projection(context)
        context = context.view(batch_size, -1, x.shape[-1])

        return context, attention_mask

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask.

        Converts a 0/1 integer mask (1 = valid, 0 = pad) into an additive
        attention mask where valid positions get a near-zero bias and padded
        positions get -inf.

        IMPORTANT: Valid positions are set to ``-1e-5`` (not ``0.0``) so that
        downstream ``_normalize_attention_mask`` does not re-interpret the mask
        as a binary [0, 1] tensor and accidentally mask out all tokens.
        """
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        additive_mask = (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

        # Replace exact 0.0 (valid tokens) with -1e-5 to prevent
        # _normalize_attention_mask from treating them as "pad" in [0,1] range.
        additive_mask = torch.where(
            additive_mask == 0,
            torch.tensor(-1e-5, dtype=x_dtype, device=additive_mask.device),
            additive_mask,
        )
        return additive_mask

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        patchify_input = modality.latent
        if modality.concat_condition is not None:
            condition = modality.concat_condition
            if condition.shape[:2] != patchify_input.shape[:2]:
                raise ValueError(
                    "concat_condition must match latent batch/sequence dimensions: "
                    f"latent={tuple(patchify_input.shape)}, condition={tuple(condition.shape)}"
                )
            condition = condition.to(device=patchify_input.device, dtype=patchify_input.dtype)
            patchify_input = torch.cat((patchify_input, condition), dim=-1)
        if patchify_input.shape[-1] != self.patchify_proj.in_features:
            raise ValueError(
                "modality input width does not match patchify projection: "
                f"input={patchify_input.shape[-1]}, expected={self.patchify_proj.in_features}"
            )
        x = self.patchify_proj(patchify_input)
        timestep, embedded_timestep = self._prepare_timestep(modality.timesteps, x.shape[0], modality.latent.dtype)
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        # Phoneme context (separate cross-attention path)
        phoneme_context = None
        phoneme_context_mask = None
        if modality.phoneme_context is not None and self.phoneme_caption_projection is not None:
            batch_size = x.shape[0]
            phoneme_context = self.phoneme_caption_projection(modality.phoneme_context)
            phoneme_context = phoneme_context.view(batch_size, -1, x.shape[-1])
            if self.phoneme_time_anchor is not None:
                # Add the monotonic position anchor using the RAW 0/1 phoneme
                # mask (before it is converted to an additive attention bias
                # below), so padded tokens contribute nothing.
                phoneme_context = phoneme_context + self.phoneme_time_anchor(
                    modality.phoneme_context_mask
                )
            phoneme_context_mask = self._prepare_attention_mask(
                modality.phoneme_context_mask, modality.latent.dtype
            )

        # AUDCAP context (separate cross-attention path; mirrors phoneme path)
        audcap_context = None
        audcap_context_mask = None
        if modality.audcap_context is not None and self.audcap_caption_projection is not None:
            batch_size = x.shape[0]
            audcap_context = self.audcap_caption_projection(modality.audcap_context)
            audcap_context = audcap_context.view(batch_size, -1, x.shape[-1])
            audcap_context_mask = self._prepare_attention_mask(
                modality.audcap_context_mask, modality.latent.dtype
            )

        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            phoneme_context=phoneme_context,
            phoneme_context_mask=phoneme_context_mask,
            audcap_context=audcap_context,
            audcap_context_mask=audcap_context_mask,
            self_attn_mask=modality.self_attn_mask,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: MotionSpeechRopeType,
        av_ca_timestep_scale_multiplier: int,
        phoneme_caption_projection: PixArtAlphaTextProjection | None = None,
        audcap_caption_projection: PixArtAlphaTextProjection | None = None,
        phoneme_time_anchor: torch.nn.Module | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            phoneme_caption_projection=phoneme_caption_projection,
            audcap_caption_projection=audcap_caption_projection,
            phoneme_time_anchor=phoneme_time_anchor,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=modality.timesteps,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep
