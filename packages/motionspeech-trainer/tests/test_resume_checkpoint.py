# Modified for DuoFlow; see NOTICE.md.
"""CPU-only regression tests for full-state and legacy training resume."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from motionspeech_trainer.config import (
    AccelerationConfig,
    CheckpointsConfig,
    MotionSpeechTrainerConfig,
    ModelConfig,
)
from motionspeech_trainer.trainer import (
    MotionSpeechTrainer,
    ResumeCheckpoint,
    _atomic_link_or_copy,
    _training_state_path,
    resolve_resume_checkpoint,
)


def minimal_trainer_config(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(output_dir),
        checkpoints=SimpleNamespace(save_training_state=True),
        optimization=SimpleNamespace(
            optimizer_type="adamw",
            gradient_accumulation_steps=1,
        ),
    )


class DummyAccelerator:
    device = torch.device("cpu")
    num_processes = 1


class ResumeCheckpointTests(unittest.TestCase):
    def test_atomic_archive_uses_hard_link_and_survives_source_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "checkpoint.safetensors"
            destination = root / "permanent/checkpoint.safetensors"
            source.write_bytes(b"checkpoint")
            destination.parent.mkdir()
            destination.write_bytes(b"old")

            method = _atomic_link_or_copy(source, destination)

            self.assertEqual(method, "hard link")
            self.assertEqual(source.stat().st_ino, destination.stat().st_ino)
            source.unlink()
            self.assertEqual(destination.read_bytes(), b"checkpoint")

    def test_atomic_archive_falls_back_to_copy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "checkpoint.safetensors"
            destination = root / "best/checkpoint.safetensors"
            source.write_bytes(b"checkpoint")

            with patch("motionspeech_trainer.trainer.os.link", side_effect=OSError("unsupported")):
                method = _atomic_link_or_copy(source, destination)

            self.assertEqual(method, "copy")
            self.assertEqual(destination.read_bytes(), b"checkpoint")
            self.assertFalse(destination.with_name(destination.name + ".tmp").exists())

    def test_best_checkpoint_is_archived_without_large_copy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            source = checkpoints / "model_weights_step_00003.safetensors"
            source.write_bytes(b"checkpoint")

            trainer = MotionSpeechTrainer.__new__(MotionSpeechTrainer)
            trainer._config = SimpleNamespace(
                output_dir=str(run),
                model=SimpleNamespace(training_mode="full"),
            )
            trainer._global_step = 3
            trainer._ema = None
            trainer._save_best_checkpoint(1.25)

            best = checkpoints / "best/model_weights_best.safetensors"
            self.assertEqual(source.stat().st_ino, best.stat().st_ino)
            self.assertEqual(
                (checkpoints / "best/best_info.txt").read_text(),
                "step=3\neval_loss=1.250000\n",
            )

    def test_distributed_timeout_default_is_one_hour(self) -> None:
        self.assertEqual(AccelerationConfig().process_group_timeout_seconds, 3600)

    def test_resolver_prefers_latest_complete_pair(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "training_config.yaml").write_text("seed: 1\n")
            checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            step_100 = checkpoints / "model_weights_step_00100.safetensors"
            step_200 = checkpoints / "model_weights_step_00200.safetensors"
            step_100.touch()
            step_200.touch()
            _training_state_path(step_100).touch()

            resolved = resolve_resume_checkpoint(run)
            self.assertEqual(resolved.weights_path, step_100)
            self.assertEqual(resolved.global_step, 100)
            self.assertEqual(resolved.run_dir, run)

            _training_state_path(step_200).touch()
            resolved = resolve_resume_checkpoint(checkpoints)
            self.assertEqual(resolved.weights_path, step_200)
            self.assertEqual(resolved.training_state_path, _training_state_path(step_200))

            permanent = checkpoints / "permanent"
            permanent.mkdir()
            step_300 = permanent / "model_weights_step_00300.safetensors"
            step_300.touch()
            _training_state_path(step_300).touch()
            resolved = resolve_resume_checkpoint(run)
            self.assertEqual(resolved.weights_path, step_300)
            self.assertEqual(resolved.training_state_path, _training_state_path(step_300))

    def test_resolver_supports_legacy_weights_and_direct_state(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root / "model_weights_step_25000.safetensors"
            weights.touch()
            legacy = resolve_resume_checkpoint(root)
            self.assertEqual(legacy.global_step, 25000)
            self.assertIsNone(legacy.training_state_path)

            state = _training_state_path(weights)
            state.touch()
            direct = resolve_resume_checkpoint(state)
            self.assertEqual(direct.weights_path, weights)
            self.assertEqual(direct.training_state_path, state)

    def test_config_rejects_warm_start_and_resume_together(self) -> None:
        self.assertTrue(CheckpointsConfig().save_training_state)
        with self.assertRaisesRegex(ValueError, "either checkpoints.resume_from_checkpoint"):
            MotionSpeechTrainerConfig.model_validate(
                {
                    "model": ModelConfig(load_checkpoint="warm.safetensors").model_dump(),
                    "data": {"preprocessed_data_root": "."},
                    "checkpoints": {"resume_from_checkpoint": "."},
                }
            )

    def test_full_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            weights = checkpoints / "model_weights_step_00003.safetensors"
            weights.touch()

            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            for _ in range(3):
                optimizer.zero_grad()
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
                scheduler.step()

            original = MotionSpeechTrainer.__new__(MotionSpeechTrainer)
            original._config = minimal_trainer_config(run)
            original._accelerator = DummyAccelerator()
            original._optimizer = optimizer
            original._lr_scheduler = scheduler
            original._ema = None
            original._global_step = 3
            original._best_eval_loss = 1.25
            original._save_training_state(weights)

            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=10
            )
            restored = MotionSpeechTrainer.__new__(MotionSpeechTrainer)
            restored._config = minimal_trainer_config(run)
            restored._accelerator = DummyAccelerator()
            restored._optimizer = restored_optimizer
            restored._lr_scheduler = restored_scheduler
            restored._ema = None
            restored._global_step = -1
            restored._best_eval_loss = float("inf")
            restored._checkpoint_paths = []
            restored._resume_checkpoint = ResumeCheckpoint(
                weights_path=weights,
                training_state_path=_training_state_path(weights),
                global_step=3,
                run_dir=run,
            )
            restored._restore_training_state()

            self.assertEqual(restored._global_step, 3)
            self.assertEqual(restored._lr_scheduler.last_epoch, 3)
            self.assertEqual(restored._best_eval_loss, 1.25)
            self.assertAlmostEqual(
                restored._optimizer.param_groups[0]["lr"],
                optimizer.param_groups[0]["lr"],
            )
            self.assertTrue(restored._optimizer.state_dict()["state"])

    def test_legacy_resume_fast_forwards_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoints = run / "checkpoints"
            (checkpoints / "best").mkdir(parents=True)
            (checkpoints / "best/best_info.txt").write_text(
                "step=2\neval_loss=0.75\n"
            )
            weights = checkpoints / "model_weights_step_00002.safetensors"
            weights.touch()
            model = torch.nn.Linear(1, 1)

            trainer = MotionSpeechTrainer.__new__(MotionSpeechTrainer)
            trainer._config = minimal_trainer_config(run)
            trainer._accelerator = DummyAccelerator()
            trainer._optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            trainer._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer._optimizer, T_max=10
            )
            trainer._ema = None
            trainer._checkpoint_paths = []
            trainer._best_eval_loss = float("inf")
            trainer._resume_checkpoint = ResumeCheckpoint(
                weights_path=weights,
                training_state_path=None,
                global_step=2,
                run_dir=run,
            )
            trainer._restore_training_state()

            self.assertEqual(trainer._global_step, 2)
            self.assertEqual(trainer._lr_scheduler.last_epoch, 2)
            self.assertEqual(trainer._best_eval_loss, 0.75)


if __name__ == "__main__":
    unittest.main()
