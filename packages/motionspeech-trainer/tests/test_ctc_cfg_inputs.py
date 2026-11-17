# Modified for DuoFlow; see NOTICE.md.
"""CPU regressions for CFG/CTC metadata through the actual trainer path."""

import unittest
from dataclasses import replace
from types import MethodType, SimpleNamespace

import torch
from torch import nn

from motionspeech_core.model.transformer.modality import Modality
from motionspeech_trainer.trainer import MotionSpeechTrainer
from motionspeech_trainer.training_strategies.base_strategy import ModelInputs
from motionspeech_trainer.training_strategies.text_to_motion_audio import TextToMotionAudioStrategy


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.last_aux_audio_hidden: torch.Tensor | None = None

    def forward(
        self, video: Modality, audio: Modality, perturbations: object = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_aux_audio_hidden = self.projection(audio.latent)
        return self.projection(video.latent), self.last_aux_audio_hidden


class CtcCfgInputsTest(unittest.TestCase):
    def make_trainer(self, ctc_enabled: bool) -> tuple[MotionSpeechTrainer, ModelInputs]:
        torch.manual_seed(17)
        latent = torch.randn(1, 6, 4, dtype=torch.bfloat16)
        mask = torch.ones(1, 6, dtype=torch.bool)
        context_mask = mask if ctc_enabled else torch.zeros_like(mask)
        context = latent if ctc_enabled else torch.zeros_like(latent)
        modality = Modality(
            latent=latent,
            timesteps=torch.ones(1, 6, dtype=torch.bfloat16),
            positions=torch.zeros(1, 1, 6, 2, dtype=torch.bfloat16),
            context=context,
            context_mask=context_mask,
            phoneme_context=context,
            phoneme_context_mask=context_mask,
        )
        prepared = ModelInputs(
            video=modality,
            audio=modality,
            video_targets=torch.zeros_like(latent),
            audio_targets=torch.zeros_like(latent),
            video_loss_mask=mask,
            audio_loss_mask=mask,
            ref_seq_len=2,
            phoneme_ids=torch.tensor([[1, 2, 0]]),
            ctc_enabled=ctc_enabled,
        )
        # Avoid loading real models; execute the real input adapter, training
        # step, feature-tap attachment, and strategy loss on tiny CPU tensors.
        strategy = TextToMotionAudioStrategy.__new__(TextToMotionAudioStrategy)
        nn.Module.__init__(strategy)
        strategy.config = SimpleNamespace(
            audio_only_training=False,
            motion_loss_weight=2.0,
            audio_loss_weight=1.0,
            aux_ctc_weight=0.01,
            aux_ctc_num_classes=4,
        )
        strategy.audio_ctc_head = nn.Linear(4, 4)
        strategy.prepare_training_inputs = MethodType(lambda _self, _batch, _sampler: prepared, strategy)
        trainer = MotionSpeechTrainer.__new__(MotionSpeechTrainer)
        trainer._training_strategy = strategy
        trainer._transformer = TinyTransformer()
        trainer._timestep_sampler = None
        trainer._accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        return trainer, prepared

    @staticmethod
    def batch() -> dict[str, dict[str, torch.Tensor]]:
        return {"joint_conditions": {"encoded_features": torch.zeros(1)}}

    def test_dtype_alignment_preserves_cfg_flag_and_metadata(self) -> None:
        for enabled in (False, True):
            with self.subTest(ctc_enabled=enabled):
                trainer, original = self.make_trainer(enabled)
                aligned = trainer._prepare_model_inputs(self.batch())
                self.assertEqual(aligned.ctc_enabled, enabled)
                self.assertIs(aligned.phoneme_ids, original.phoneme_ids)
                self.assertIs(aligned.video_loss_mask, original.video_loss_mask)
                self.assertIs(aligned.audio_loss_mask, original.audio_loss_mask)
                self.assertEqual(aligned.ref_seq_len, original.ref_seq_len)
                self.assertEqual(aligned.video.latent.dtype, torch.float32)
                self.assertEqual(aligned.audio_targets.dtype, torch.float32)
                self.assertEqual(aligned.audio.phoneme_context_mask.dtype, torch.bool)
                self.assertEqual(original.video.latent.dtype, torch.bfloat16)

    def test_existing_aux_hidden_is_preserved(self) -> None:
        trainer, original = self.make_trainer(True)
        hidden = torch.randn(1, 6, 4)
        prepared = replace(original, audio_hidden_aux=hidden)
        trainer._training_strategy.prepare_training_inputs = MethodType(
            lambda _self, _batch, _sampler: prepared, trainer._training_strategy
        )
        aligned = trainer._prepare_model_inputs(self.batch())
        self.assertIs(aligned.audio_hidden_aux, hidden)

    def test_cfg_dropout_skips_ctc_in_training_step(self) -> None:
        trainer, _ = self.make_trainer(False)
        head = trainer._training_strategy.audio_ctc_head

        def unexpected_ctc_head(_module: nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
            self.fail("CFG-dropout batch reached the CTC head")

        handle = head.register_forward_pre_hook(unexpected_ctc_head)
        try:
            loss, motion_loss, audio_loss = trainer._training_step(self.batch())
            torch.testing.assert_close(loss, motion_loss + audio_loss)
            self.assertEqual(trainer._training_strategy.last_aux_losses, {})
            loss.backward()
            self.assertTrue(all(p.grad is None for p in head.parameters()))
            self.assertIsNotNone(trainer._transformer.projection.weight.grad)
        finally:
            handle.remove()

    def test_conditional_batch_keeps_ctc_and_gradients(self) -> None:
        trainer, _ = self.make_trainer(True)
        loss, motion_loss, audio_loss = trainer._training_step(self.batch())
        raw_ctc = trainer._training_strategy.last_aux_losses["ctc_loss"]
        self.assertGreater(raw_ctc, 0.0)
        torch.testing.assert_close(loss, motion_loss + audio_loss + 0.01 * raw_ctc)
        loss.backward()
        grad = trainer._training_strategy.audio_ctc_head.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
