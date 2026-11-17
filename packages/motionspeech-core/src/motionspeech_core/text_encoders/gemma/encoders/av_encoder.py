# Modified for DuoFlow; see NOTICE.md.
from typing import NamedTuple

import torch
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from motionspeech_core.loader.sd_ops import SDOps
from motionspeech_core.model.model_protocol import ModelConfigurator
from motionspeech_core.text_encoders.gemma.embeddings_connector import (
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from motionspeech_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoderModelBase
from motionspeech_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from motionspeech_core.text_encoders.gemma.tokenizer import MotionSpeechGemmaTokenizer


class AVGemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor
    attention_mask: torch.Tensor


class AVGemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    AVGemma Text Encoder Model.
    This class combines the tokenizer, Gemma model, feature extractor from base class and a
    video and audio embeddings connectors to provide a preprocessing for audio-visual pipeline.

    When ``skip_connectors`` is True, the feature_extractor and connectors are bypassed:
    raw Gemma hidden states (3840-dim for num_layers=1) are passed directly as both
    video and audio encodings. This is used when only caption_projection inside the
    transformer performs the channel transformation.

    Per-modality control is available via ``skip_video_connector`` and
    ``skip_audio_connector``.  When a modality's connector is skipped, that branch
    receives raw Gemma features (like bypass mode) while the other branch can still
    use its connector.
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector,
        tokenizer: MotionSpeechGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        dtype: torch.dtype = torch.float32,
        skip_connectors: bool = False,
        connector_residual: bool = True,
        skip_video_connector: bool = False,
        skip_audio_connector: bool = False,
    ) -> None:
        super().__init__(
            feature_extractor_linear=feature_extractor_linear,
            tokenizer=tokenizer,
            model=model,
            dtype=dtype,
        )
        self.embeddings_connector = embeddings_connector.to(dtype=dtype)
        self.audio_embeddings_connector = audio_embeddings_connector.to(dtype=dtype)
        # LayerNorm after connectors to stabilize norm before caption_projection
        self.video_connector_norm = torch.nn.LayerNorm(3840).to(dtype=dtype)
        self.audio_connector_norm = torch.nn.LayerNorm(3840).to(dtype=dtype)
        self.skip_connectors = skip_connectors
        self.connector_residual = connector_residual
        self.skip_video_connector = skip_video_connector
        self.skip_audio_connector = skip_audio_connector

    def _run_connectors(
        self, encoded_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        # Binary mask for bypass branches (raw Gemma features zero-masked)
        binary_mask_3d = attention_mask.unsqueeze(-1).to(encoded_input.dtype)  # (B, seq, 1)

        def _aligned_residual(connector: Embeddings1DConnector, output: torch.Tensor) -> torch.Tensor:
            """Match residual layout to [valid text] + [K registers] packing."""
            if not self.connector_residual:
                return torch.zeros_like(output)
            if connector.num_learnable_registers:
                residual = torch.zeros_like(output)
                source_binary = attention_mask.to(torch.bool)
                for batch_idx in range(encoded_input.shape[0]):
                    valid = encoded_input[batch_idx, source_binary[batch_idx]]
                    residual[batch_idx, : valid.shape[0]] = valid
                return residual
            return encoded_input * binary_mask_3d

        # Default attention_mask output
        attention_mask_out = attention_mask

        # ── Video branch ──
        if not self.skip_video_connector:
            encoded, encoded_connector_attention_mask = self.embeddings_connector(
                encoded_input,
                connector_attention_mask,
            )
            # Convert additive mask back to binary int64.
            attention_mask_out = (encoded_connector_attention_mask >= -9000.0).to(torch.int64)
            attention_mask_out = attention_mask_out.reshape([encoded.shape[0], encoded.shape[1], 1])
            encoded = encoded * attention_mask_out
            attention_mask_out = attention_mask_out.squeeze(-1)

            if self.connector_residual:
                encoded = encoded + _aligned_residual(self.embeddings_connector, encoded)
            encoded = self.video_connector_norm(encoded)
        else:
            # Video BYPASSED: raw Gemma features
            encoded = encoded_input * binary_mask_3d

        # ── Audio branch ──
        if not self.skip_audio_connector:
            encoded_for_audio, audio_connector_mask = self.audio_embeddings_connector(
                encoded_input, connector_attention_mask
            )
            audio_binary_mask = (audio_connector_mask >= -9000.0).to(torch.int64)
            audio_binary_mask = audio_binary_mask.reshape([encoded_for_audio.shape[0], encoded_for_audio.shape[1], 1])
            encoded_for_audio = encoded_for_audio * audio_binary_mask

            if self.connector_residual:
                encoded_for_audio = encoded_for_audio + _aligned_residual(
                    self.audio_embeddings_connector, encoded_for_audio
                )
            encoded_for_audio = self.audio_connector_norm(encoded_for_audio)
        else:
            # Audio BYPASSED: raw Gemma features
            encoded_for_audio = encoded_input * binary_mask_3d

        return encoded, encoded_for_audio, attention_mask_out

    def forward(self, text: str, padding_side: str = "left") -> AVGemmaEncoderOutput:
        encoded_inputs, attention_mask = self._preprocess_text(text, padding_side)
        if self.skip_connectors or (self.skip_video_connector and self.skip_audio_connector):
            # BYPASS: pass raw Gemma features directly as video & audio context.
            # Zero-mask pad positions for consistency.
            # encoded_inputs: (1, seq, 3840), attention_mask: (1, seq)
            binary_mask = attention_mask.unsqueeze(-1)  # (1, seq, 1)
            encoded_inputs = encoded_inputs * binary_mask
            # Return same format as _run_connectors: keep batch dim
            return AVGemmaEncoderOutput(encoded_inputs, encoded_inputs.clone(), attention_mask)
        video_encoding, audio_encoding, attention_mask = self._run_connectors(encoded_inputs, attention_mask)
        return AVGemmaEncoderOutput(video_encoding, audio_encoding, attention_mask)


class AVGemmaTextEncoderModelConfigurator(ModelConfigurator[AVGemmaTextEncoderModel]):
    @classmethod
    def from_config(cls: type["AVGemmaTextEncoderModel"], config: dict) -> "AVGemmaTextEncoderModel":
        feature_extractor_linear = GemmaFeaturesExtractorProjLinear.from_config(config)
        embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        audio_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        return AVGemmaTextEncoderModel(
            feature_extractor_linear=feature_extractor_linear,
            embeddings_connector=embeddings_connector,
            audio_embeddings_connector=audio_embeddings_connector,
        )


AV_GEMMA_TEXT_ENCODER_KEY_OPS = (
    SDOps("AV_GEMMA_TEXT_ENCODER_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
)
