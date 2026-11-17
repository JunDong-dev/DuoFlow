# Modified for DuoFlow; see NOTICE.md.
import copy
import gc
import json
import os
import random
import re
import shutil
import time
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import wandb
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs, set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as F  # noqa: N812

from motionspeech_trainer import add_file_handler, logger
from motionspeech_trainer.config import MotionSpeechTrainerConfig
from motionspeech_trainer.config_display import print_config
from motionspeech_trainer.datasets import PrecomputedDataset
from motionspeech_trainer.gpu_utils import free_gpu_memory, free_gpu_memory_context, get_gpu_memory_gb
from motionspeech_trainer.hf_hub_utils import push_to_hub
from motionspeech_trainer.model_loader import load_model as load_motionspeech_model
from motionspeech_trainer.model_loader import load_text_encoder
from motionspeech_trainer.progress import TrainingProgress
from motionspeech_trainer.quantization import quantize_model
from motionspeech_trainer.timestep_samplers import SAMPLERS
from motionspeech_trainer.training_strategies import get_training_strategy
from motionspeech_trainer.utils import open_image_as_srgb, save_image
from motionspeech_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from motionspeech_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200

_WEIGHTS_CHECKPOINT_RE = re.compile(r"^(model|lora)_weights_step_(\d+)\.safetensors$")
_TRAINING_STATE_RE = re.compile(r"^training_state_step_(\d+)\.pt$")


@dataclass(frozen=True)
class ResumeCheckpoint:
    """Resolved model weights and optional full optimization state for a resume."""

    weights_path: Path
    training_state_path: Path | None
    global_step: int
    run_dir: Path | None


def _checkpoint_step(path: Path) -> int:
    match = _WEIGHTS_CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(2))


def _training_state_path(weights_path: Path) -> Path:
    return weights_path.with_name(f"training_state_step_{_checkpoint_step(weights_path):05d}.pt")


def _atomic_link_or_copy(source: Path, destination: Path) -> str:
    """Atomically archive a checkpoint, preferring a zero-copy hard link.

    Checkpoint archives live on the same filesystem in normal runs, so a hard
    link is effectively instantaneous and remains valid after the rotating
    source name is removed. Copying to a temporary path is the portable
    fallback for filesystems that do not support hard links.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    method = "hard link"
    try:
        try:
            os.link(source, temporary)
        except OSError:
            method = "copy"
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return method


def _find_run_dir(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / "training_config.yaml").is_file():
            return candidate
    return None


def resolve_resume_checkpoint(resume_from: str | Path) -> ResumeCheckpoint:
    """Resolve a run/checkpoints directory or checkpoint file for resume.

    Directories prefer the latest weights file that has an atomically completed
    training-state sidecar. If the directory predates full-state checkpointing,
    the latest weights-only checkpoint is returned as a compatibility fallback.
    """
    source = Path(resume_from).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Resume path does not exist: {source}")

    if source.is_file():
        state_match = _TRAINING_STATE_RE.match(source.name)
        if state_match is not None:
            step = int(state_match.group(1))
            weights = [
                path
                for path in source.parent.glob(f"*_weights_step_{step:05d}.safetensors")
                if _WEIGHTS_CHECKPOINT_RE.match(path.name)
            ]
            if not weights:
                raise ValueError(f"No model weights found beside training state {source}")
            weights_path = sorted(weights, key=lambda path: path.name)[0]
            state_path: Path | None = source
        elif _WEIGHTS_CHECKPOINT_RE.match(source.name):
            weights_path = source
            step = _checkpoint_step(source)
            candidate_state = _training_state_path(source)
            state_path = candidate_state if candidate_state.is_file() else None
        else:
            raise ValueError(
                "Resume file must be model/lora weights or training_state_step_*.pt: "
                f"{source}"
            )
    else:
        weights = [
            path
            for path in source.rglob("*_weights_step_*.safetensors")
            if _WEIGHTS_CHECKPOINT_RE.match(path.name)
        ]
        if not weights:
            raise ValueError(f"No step checkpoints found below resume directory: {source}")
        complete_pairs = [path for path in weights if _training_state_path(path).is_file()]
        candidates = complete_pairs or weights
        weights_path = max(
            candidates,
            key=lambda path: (
                _checkpoint_step(path),
                "permanent" not in path.parts,
                -len(path.parts),
            ),
        )
        step = _checkpoint_step(weights_path)
        candidate_state = _training_state_path(weights_path)
        state_path = candidate_state if candidate_state.is_file() else None

    return ResumeCheckpoint(
        weights_path=weights_path,
        training_state_path=state_path,
        global_step=step,
        run_dir=_find_run_dir(weights_path),
    )


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model parameters and updates them
    with an exponential moving average after each training step.
    Follows the AudioLDM2 pattern: EMA decay = 0.9999.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Store shadow parameters on CPU to save GPU memory;
        # move to device only when needed.
        self.shadow: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update shadow parameters with EMA."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> dict[str, Tensor]:
        """Replace model parameters with EMA shadow; return the original params for restore."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, Tensor]) -> None:
        """Restore original model parameters from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


class MotionSpeechTrainer:
    def __init__(self, trainer_config: MotionSpeechTrainerConfig) -> None:
        self._config = trainer_config
        resume_from = self._config.checkpoints.resume_from_checkpoint
        self._resume_checkpoint = (
            resolve_resume_checkpoint(resume_from) if resume_from is not None else None
        )
        if self._resume_checkpoint is not None and self._resume_checkpoint.run_dir is not None:
            configured_output = Path(self._config.output_dir).resolve()
            if configured_output != self._resume_checkpoint.run_dir:
                logger.warning(
                    "Resume output directory changed from %s to original run directory %s",
                    configured_output,
                    self._resume_checkpoint.run_dir,
                )
                self._config.output_dir = str(self._resume_checkpoint.run_dir)
        self._text_encoder_state_for_ckpt = None
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        # Multi-tap CTC: build one independent head per tapped block
        # BEFORE parameters are collected so they join the optimizer.
        _taps = getattr(self._config.model, "aux_ctc_tap_blocks", None)
        if _taps and len(_taps) > 1 and hasattr(
            self._training_strategy, "build_per_tap_ctc_heads"
        ):
            self._training_strategy.build_per_tap_ctc_heads(_taps)
        self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()
        self._load_models()
        self._setup_accelerator()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._val_dataset = None
        self._replay_dataset: PrecomputedDataset | None = None
        self._replay_dataloader = None
        self._replay_every = 0
        self._global_step = -1
        self._checkpoint_paths = []
        self._ema: EMAModel | None = None
        self._best_eval_loss = float('inf')
        self._tb_writer: SummaryWriter | None = None
        self._init_wandb()
        self._init_tensorboard()

    def _effective_model_checkpoint(self) -> str | Path | None:
        if self._resume_checkpoint is not None:
            return self._resume_checkpoint.weights_path
        return self._config.model.load_checkpoint

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        # Use the same seed for all processes and ensure deterministic operations
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()
        self._restore_training_state()
        if self._global_step >= cfg.optimization.steps:
            raise ValueError(
                f"Resume checkpoint step {self._global_step} has already reached "
                f"optimization.steps={cfg.optimization.steps}"
            )
        resume_step = self._global_step
        remaining_steps = cfg.optimization.steps - resume_step
        self._init_dataloader()
        data_iter = iter(self._dataloader)
        replay_iter = iter(self._replay_dataloader) if self._replay_dataloader is not None else None
        self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        if IS_MAIN_PROCESS:
            add_file_handler(Path(cfg.output_dir) / "train.log")

        # Save the training configuration as YAML
        self._save_config()

        logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=cfg.optimization.steps,
            initial_step=resume_step,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._transformer.train()

        peak_mem_during_training = start_mem

        sampled_videos_paths = None
        logged_batch_shapes = False

        # Interval average loss tracking
        _interval_loss_sum = 0.0
        _interval_loss_count = 0
        # Replay-batch audio loss tracking (logged separately from joint loss
        # so the anti-forgetting signal is visible in train.log / TensorBoard)
        _interval_replay_sum = 0.0
        _interval_replay_count = 0
        # Interval average of the strategy's auxiliary losses (e.g. CTC) so
        # the checkpoint-metrics line in train.log records them too.
        _interval_ctc_sum = 0.0
        _interval_ctc_count = 0

        def _log_batch_shapes(batch_data: dict) -> None:
            def _shape_str(t: Tensor) -> str:
                return "x".join(str(d) for d in t.shape)

            batch_size = None
            for value in batch_data.values():
                if isinstance(value, dict):
                    for sub_val in value.values():
                        if isinstance(sub_val, Tensor):
                            batch_size = sub_val.shape[0]
                            break
                elif isinstance(value, Tensor):
                    batch_size = value.shape[0]
                if batch_size is not None:
                    break

            logger.info("[batch] size=%s", batch_size if batch_size is not None else "unknown")
            for key, value in batch_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, Tensor):
                            logger.info("[batch] %s.%s=%s", key, sub_key, _shape_str(sub_val))
                elif isinstance(value, Tensor):
                    logger.info("[batch] %s=%s", key, _shape_str(value))

        with progress:
            # Initial validation before training starts
            if (
                resume_step == 0
                and cfg.validation.interval
                and not cfg.validation.skip_initial_validation
            ):
                sampled_videos_paths = self._sample_videos(progress)
                if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                    self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)
            elif resume_step > 0 and IS_MAIN_PROCESS:
                logger.info("Skipping initial validation for resumed run at step %d", resume_step)

            self._accelerator.wait_for_everyone()

            for step in range(remaining_steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed. Replay
                # micro-steps (every self._replay_every-th step) draw the
                # stage-1 audio-only dataset instead; the resulting batch has
                # no `motion_latents` key, which routes it through the
                # strategy's audio-only path.
                is_replay = False
                absolute_micro_step = (
                    resume_step * cfg.optimization.gradient_accumulation_steps + step
                )
                if replay_iter is not None and absolute_micro_step % self._replay_every == 0:
                    try:
                        batch = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(self._replay_dataloader)
                        batch = next(replay_iter)
                    is_replay = True
                else:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self._dataloader)
                        batch = next(data_iter)

                if IS_MAIN_PROCESS and not logged_batch_shapes:
                    _log_batch_shapes(batch)
                    logged_batch_shapes = True

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    loss, motion_loss, audio_loss = self._training_step(batch)
                    self._accelerator.backward(loss)
                    if self._accelerator.sync_gradients:
                        self._sync_training_strategy_gradients()

                    # Accumulate loss for interval average
                    _interval_loss_sum += loss.item()
                    _interval_loss_count += 1
                    if is_replay and audio_loss is not None:
                        _interval_replay_sum += float(audio_loss.item())
                        _interval_replay_count += 1
                    _aux_losses_now = getattr(self._training_strategy, "last_aux_losses", None) or {}
                    if "ctc_loss" in _aux_losses_now:
                        _interval_ctc_sum += float(_aux_losses_now["ctc_loss"])
                        _interval_ctc_count += 1

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    # Update EMA after each optimization step
                    if self._ema is not None and is_optimization_step:
                        ema_modules = torch.nn.ModuleDict()
                        unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
                        ema_modules["transformer"] = unwrapped
                        if isinstance(self._training_strategy, torch.nn.Module):
                            ema_modules["strategy"] = self._training_strategy
                        self._ema.update(ema_modules)

                    if is_optimization_step and self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        if self._accelerator.distributed_type == DistributedType.FSDP:
                            # FSDP: All processes must participate in validation
                            sampled_videos_paths = self._sample_videos(progress)
                            if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)
                        # DDP: Only main process runs validation
                        elif IS_MAIN_PROCESS:
                            sampled_videos_paths = self._sample_videos(progress)
                            if sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        # Keep routine checkpoint metrics cheap and comparable by
                        # evaluating the same deterministic subset every time.
                        # Milestones additionally run the complete validation set;
                        # the subset result remains the best-checkpoint criterion.
                        eval_result = self._evaluate_latent_loss()
                        full_eval_result = None
                        full_eval_interval = cfg.validation.latent_eval_full_interval
                        if (
                            cfg.validation.latent_eval_subset_size is not None
                            and full_eval_interval
                            and self._global_step % full_eval_interval == 0
                        ):
                            full_eval_result = self._evaluate_latent_loss(full_dataset=True)
                        checkpoint_lr = self._optimizer.param_groups[0]["lr"]
                        checkpoint_loss = loss.item()
                        # Compute interval average loss
                        interval_avg_loss = (
                            _interval_loss_sum / _interval_loss_count
                            if _interval_loss_count > 0
                            else checkpoint_loss
                        )
                        interval_avg_ctc = (
                            _interval_ctc_sum / _interval_ctc_count
                            if _interval_ctc_count > 0
                            else None
                        )
                        ctc_log_label = (
                            f" ctc={interval_avg_ctc:.6f}" if interval_avg_ctc is not None else ""
                        )
                        if IS_MAIN_PROCESS and eval_result is not None:
                            eval_label, eval_loss = eval_result
                            self._log_metrics(
                                {
                                    f"{eval_label}/latent_loss": eval_loss["total"],
                                    f"{eval_label}/motion_latent_loss": eval_loss["motion"],
                                    f"{eval_label}/audio_latent_loss": eval_loss["audio"],
                                    "train/interval_avg_loss": interval_avg_loss,
                                }
                            )
                            if interval_avg_ctc is not None:
                                self._log_metrics({"train/interval_avg_ctc_loss": interval_avg_ctc})
                            logger.info(
                                "📌 Checkpoint metrics: eval=%s total=%.6f motion=%.6f audio=%.6f | loss=%.6f avg_loss=%.6f%s lr=%.2e (interval=%d steps)",
                                eval_label,
                                eval_loss["total"],
                                eval_loss["motion"],
                                eval_loss["audio"],
                                checkpoint_loss,
                                interval_avg_loss,
                                ctc_log_label,
                                checkpoint_lr,
                                _interval_loss_count,
                            )
                        elif IS_MAIN_PROCESS:
                            self._log_metrics(
                                {"train/interval_avg_loss": interval_avg_loss}
                            )
                            if interval_avg_ctc is not None:
                                self._log_metrics({"train/interval_avg_ctc_loss": interval_avg_ctc})
                            logger.info(
                                "📌 Checkpoint metrics: eval=none | loss=%.6f avg_loss=%.6f%s lr=%.2e (interval=%d steps)",
                                checkpoint_loss,
                                interval_avg_loss,
                                ctc_log_label,
                                checkpoint_lr,
                                _interval_loss_count,
                            )
                        if IS_MAIN_PROCESS and full_eval_result is not None:
                            full_eval_label, full_eval_loss = full_eval_result
                            self._log_metrics(
                                {
                                    f"{full_eval_label}/latent_loss": full_eval_loss["total"],
                                    f"{full_eval_label}/motion_latent_loss": full_eval_loss["motion"],
                                    f"{full_eval_label}/audio_latent_loss": full_eval_loss["audio"],
                                }
                            )
                            logger.info(
                                "Milestone latent loss: eval=%s total=%.6f motion=%.6f audio=%.6f",
                                full_eval_label,
                                full_eval_loss["total"],
                                full_eval_loss["motion"],
                                full_eval_loss["audio"],
                            )
                        # Replay-batch average audio loss over the interval
                        if IS_MAIN_PROCESS and _interval_replay_count > 0:
                            interval_avg_replay = _interval_replay_sum / _interval_replay_count
                            logger.info(
                                "📌 Replay metrics: replay_audio=%.6f (replay batches in interval=%d)",
                                interval_avg_replay,
                                _interval_replay_count,
                            )
                            self._log_metrics({"train/interval_avg_replay_audio_loss": interval_avg_replay})
                        # Reset interval accumulators
                        _interval_loss_sum = 0.0
                        _interval_loss_count = 0
                        _interval_ctc_sum = 0.0
                        _interval_ctc_count = 0
                        _interval_replay_sum = 0.0
                        _interval_replay_count = 0
                        best_improved = False
                        if IS_MAIN_PROCESS and eval_result is not None:
                            current_eval_loss = eval_loss["total"]
                            best_improved = current_eval_loss < self._best_eval_loss
                            if best_improved:
                                self._best_eval_loss = current_eval_loss
                        self._save_checkpoint()

                        # Save best checkpoint if eval loss improved
                        if IS_MAIN_PROCESS and best_improved:
                            self._save_best_checkpoint(current_eval_loss)

                        # TensorBoard: record per-checkpoint event metadata
                        # (interval in steps, save time, best-eval improvement flag).
                        # This is independent of metric scalars and makes it easy to
                        # spot checkpointing cadence vs. loss progression in TB.
                        if IS_MAIN_PROCESS and self._tb_writer is not None:
                            try:
                                self._tb_writer.add_scalar(
                                    "checkpoint/steps_since_last_save",
                                    float(_interval_loss_count),
                                    global_step=self._global_step,
                                )
                                self._tb_writer.add_scalar(
                                    "checkpoint/checkpoint_loss",
                                    float(checkpoint_loss),
                                    global_step=self._global_step,
                                )
                                self._tb_writer.add_scalar(
                                    "checkpoint/interval_avg_loss",
                                    float(interval_avg_loss),
                                    global_step=self._global_step,
                                )
                                if eval_result is not None:
                                    self._tb_writer.add_scalar(
                                        "checkpoint/eval_total_loss",
                                        float(eval_loss["total"]),
                                        global_step=self._global_step,
                                    )
                                    self._tb_writer.add_scalar(
                                        "checkpoint/eval_motion_loss",
                                        float(eval_loss["motion"]),
                                        global_step=self._global_step,
                                    )
                                    self._tb_writer.add_scalar(
                                        "checkpoint/eval_audio_loss",
                                        float(eval_loss["audio"]),
                                        global_step=self._global_step,
                                    )
                                    self._tb_writer.add_scalar(
                                        "checkpoint/best_eval_loss",
                                        float(self._best_eval_loss),
                                        global_step=self._global_step,
                                    )
                                self._tb_writer.add_text(
                                    "checkpoint/events",
                                    f"step={self._global_step} interval_steps={cfg.checkpoints.interval} "
                                    f"lr={checkpoint_lr:.3e} loss={checkpoint_loss:.6f} "
                                    f"avg={interval_avg_loss:.6f}",
                                    global_step=self._global_step,
                                )
                            except Exception as exc:  # pragma: no cover - non-fatal
                                logger.warning("TensorBoard checkpoint event log failed: %s", exc)

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress and log metrics
                    current_lr = self._optimizer.param_groups[0]["lr"]
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps

                    progress.update_training(
                        loss=loss.item(),
                        lr=current_lr,
                        step_time=step_time,
                        advance=is_optimization_step,
                    )

                    # Log metrics to W&B / TensorBoard (only on main process and optimization steps)
                    if IS_MAIN_PROCESS and is_optimization_step:
                        train_metrics: dict[str, float] = {
                            "train/loss": loss.item(),
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/global_step": float(self._global_step),
                        }
                        # When the strategy exposes compute_loss_components we
                        # also have per-modality losses available for logging.
                        if motion_loss is not None:
                            train_metrics["train/motion_loss"] = float(motion_loss.item())
                        if audio_loss is not None:
                            train_metrics["train/audio_loss"] = float(audio_loss.item())
                        # Auxiliary alignment losses (e.g. CTC) recorded by the
                        # strategy during compute_loss_components.
                        aux_losses = getattr(self._training_strategy, "last_aux_losses", None)
                        if aux_losses:
                            for aux_name, aux_value in aux_losses.items():
                                train_metrics[f"train/{aux_name}"] = float(aux_value)

                        # Rate-limit TensorBoard writes (per-step is too granular for some
                        # setups — controlled by tensorboard.log_every_n_steps). W&B
                        # writes are kept at every step when enabled.
                        tb_step_skip = max(1, self._config.tensorboard.log_every_n_steps)
                        if self._tb_writer is None or self._global_step % tb_step_skip == 0:
                            self._log_metrics(train_metrics)
                        elif self._wandb_run is not None:
                            self._wandb_run.log(train_metrics)

                    # Fallback logging when progress bars are disabled
                    if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:
                        elapsed = time.time() - train_start_time
                        progress_percentage = (
                            (self._global_step - resume_step) / remaining_steps
                        )
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."
                        logger.info(
                            f"Step {self._global_step}/{cfg.optimization.steps} - "
                            f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                            f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                        )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = remaining_steps / total_time_seconds

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

            if self._tb_writer is not None:
                try:
                    self._tb_writer.flush()
                    self._tb_writer.close()
                except Exception as exc:  # pragma: no cover - non-fatal
                    logger.warning("TensorBoard close failed: %s", exc)

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats

    def _prepare_model_inputs(self, batch: dict[str, dict[str, Tensor]]) -> "ModelInputs":
        """Prepare and dtype-align ModelInputs for the transformer."""
        # Apply embedding connectors to transform pre-computed text embeddings.
        # Joint-caption schemas carry joint_conditions; per-modality schemas
        # carry motion_conditions/speak_conditions (and optionally conditions);
        # legacy schema carries only conditions.
        conditions = (
            batch.get("joint_conditions")
            or batch.get("conditions")
            or batch.get("motion_conditions")
            or batch.get("speak_conditions")
        )
        if conditions is None:
            raise KeyError(
                "Batch has no text conditions: expected one of "
                "'joint_conditions' / 'conditions' / 'motion_conditions' / "
                "'speak_conditions', got "
                f"{sorted(batch.keys())}"
            )

        # New format: pre-connector Gemma features → strategy owns the trainable
        # connectors so we just pass through and let strategy handle it.
        if "encoded_features" in conditions:
            pass  # strategy.prepare_training_inputs will run connectors
        elif "video_prompt_embeds" not in conditions:
            # Legacy format: need to run connectors via text_encoder
            if not hasattr(self, "_text_encoder") or self._text_encoder is None:
                raise RuntimeError(
                    "Legacy conditions format detected (no 'encoded_features' / "
                    "'video_prompt_embeds') but _text_encoder was not loaded. "
                    "Set model.model_path or model.load_checkpoint, or re-run "
                    "preprocessing with --text-feature-mode embedding_lookup "
                    "to produce 'encoded_features'."
                )
            video_embeds, audio_embeds, attention_mask = self._text_encoder._run_connectors(
                conditions["prompt_embeds"], conditions["prompt_attention_mask"]
            )
            conditions["video_prompt_embeds"] = video_embeds
            conditions["audio_prompt_embeds"] = audio_embeds
            conditions["prompt_attention_mask"] = attention_mask
        # else: already have video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)

        # Align model input dtypes with transformer weights to avoid rms_norm dtype mismatch
        target_parameter = next(self._transformer.parameters())
        target_dtype = target_parameter.dtype
        target_device = target_parameter.device

        def _cast_modality(modality: "Modality | None") -> "Modality | None":
            if modality is None:
                return None

            def _cast_value(value: Tensor | None, *, floating: bool = True) -> Tensor | None:
                if value is None:
                    return None
                if floating and torch.is_floating_point(value):
                    return value.to(device=target_device, dtype=target_dtype)
                return value.to(device=target_device)

            from motionspeech_core.model.transformer.modality import Modality

            return Modality(
                enabled=modality.enabled,
                latent=_cast_value(modality.latent),
                timesteps=_cast_value(modality.timesteps),
                positions=_cast_value(modality.positions),
                context=_cast_value(modality.context),
                context_mask=_cast_value(modality.context_mask, floating=False),
                concat_condition=_cast_value(modality.concat_condition),
                phoneme_context=_cast_value(modality.phoneme_context),
                phoneme_context_mask=_cast_value(modality.phoneme_context_mask, floating=False),
                audcap_context=_cast_value(modality.audcap_context),
                audcap_context_mask=_cast_value(modality.audcap_context_mask, floating=False),
                self_attn_mask=_cast_value(modality.self_attn_mask, floating=False),
            )

        video_targets = model_inputs.video_targets
        if torch.is_floating_point(video_targets) and video_targets.dtype != target_dtype:
            video_targets = video_targets.to(dtype=target_dtype)

        audio_targets = model_inputs.audio_targets
        if audio_targets is not None and torch.is_floating_point(audio_targets) and audio_targets.dtype != target_dtype:
            audio_targets = audio_targets.to(dtype=target_dtype)

        from dataclasses import replace as _dc_replace

        # Preserve strategy metadata, especially ctc_enabled=False on CFG
        # dropout batches. Rebuilding ModelInputs resets omitted fields.
        return _dc_replace(
            model_inputs,
            video=_cast_modality(model_inputs.video),
            audio=_cast_modality(model_inputs.audio),
            video_targets=video_targets,
            audio_targets=audio_targets,
        )

    def _training_step(
        self,
        batch: dict[str, dict[str, Tensor]],
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Perform a single training step using the configured strategy.

        Returns:
            ``(total_loss, motion_loss | None, audio_loss | None)``. When the
            strategy exposes ``compute_loss_components``, motion/audio
            components are returned alongside the total so they can be logged
            at every optimization step.
        """
        model_inputs = self._prepare_model_inputs(batch)

        def _check_tensor(name: str, tensor: Tensor | None) -> None:
            if tensor is None or not torch.is_floating_point(tensor):
                return
            if not torch.isfinite(tensor).all():
                stats = (
                    float(tensor.min().item()),
                    float(tensor.max().item()),
                    float(tensor.mean().item()),
                )
                raise ValueError(f"NaN/Inf detected in {name}: min={stats[0]:.6g} max={stats[1]:.6g} mean={stats[2]:.6g}")

        _check_tensor("video.latent", model_inputs.video.latent)
        _check_tensor("video.timesteps", model_inputs.video.timesteps)
        _check_tensor("video.positions", model_inputs.video.positions)
        _check_tensor("video.context", model_inputs.video.context)
        _check_tensor("video.targets", model_inputs.video_targets)
        if model_inputs.audio is not None:
            _check_tensor("audio.latent", model_inputs.audio.latent)
            _check_tensor("audio.timesteps", model_inputs.audio.timesteps)
            _check_tensor("audio.positions", model_inputs.audio.positions)
            _check_tensor("audio.context", model_inputs.audio.context)
        if model_inputs.audio_targets is not None:
            _check_tensor("audio.targets", model_inputs.audio_targets)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        _check_tensor("video_pred", video_pred)
        if audio_pred is not None:
            _check_tensor("audio_pred", audio_pred)

        # Attach the CTC feature tap captured during this forward pass
        # (None when the model has no aux_ctc_tap_block, or during validation
        # so the eval loss stays pure flow-matching).
        if getattr(model_inputs, "audio_hidden_aux", None) is None:
            from dataclasses import replace as _dc_replace

            # IMPORTANT: read the tap from the UNWRAPPED model. In DDP the
            # prepared `self._transformer` is a DistributedDataParallel
            # wrapper; `last_aux_audio_hidden` is a plain instance attribute
            # of MotionSpeechModel, so attribute lookup on the wrapper silently
            # returns None and the CTC loss would be skipped every step.
            tapped = None
            try:
                tap_model = self._accelerator.unwrap_model(self._transformer)
            except Exception:  # noqa: BLE001 — accelerator may be missing in some tests
                tap_model = getattr(self._transformer, "module", self._transformer)
            if tap_model is not None:
                tapped = getattr(tap_model, "last_aux_audio_hidden", None)
            if tapped is not None:
                model_inputs = _dc_replace(model_inputs, audio_hidden_aux=tapped)
                taps = getattr(tap_model, "aux_audio_hiddens", None) or {}
                if len(taps) > 1:
                    ordered = tuple(taps[k] for k in sorted(taps))
                    model_inputs = _dc_replace(
                        model_inputs,
                        audio_hidden_aux_multi=ordered,
                        audio_hidden_aux_tap_blocks=tuple(sorted(taps)),
                    )

        # Use strategy to compute loss (and split when possible for logging)
        motion_loss: Tensor | None = None
        audio_loss: Tensor | None = None
        if hasattr(self._training_strategy, "compute_loss_components"):
            motion_loss, audio_loss, loss = self._training_strategy.compute_loss_components(
                video_pred, audio_pred, model_inputs
            )
        else:
            loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)

        _check_tensor("loss", loss)

        return loss, motion_loss, audio_loss

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        """Load text encoder, computes and returns validation embeddings."""
        model_path = self._config.model.model_path
        effective_checkpoint = self._effective_model_checkpoint()
        if model_path is None and effective_checkpoint:
            resolved = self._find_checkpoint(effective_checkpoint)
            if resolved is not None:
                model_path = resolved
                logger.info("Using load_checkpoint for text encoder load: %s", resolved)
            else:
                logger.warning(
                    "load_checkpoint set but no checkpoint found at %s; skipping text encoder load.",
                    effective_checkpoint,
                )

        if model_path is None:
            logger.info("No model_path provided; skipping text encoder load and validation embedding cache.")
            return None

        # This method:
        #   1. Loads the text encoder on GPU
        #   2. If validation prompts are configured, computes and caches their embeddings
        #   3. Unloads the heavy Gemma model while keeping the lightweight embedding connectors
        #   The text encoder is kept (as self._text_encoder) but with model/tokenizer/feature_extractor
        #   set to None. Only the embedding connectors remain for use during training.

        # Load text encoder on GPU
        logger.debug("Loading text encoder...")

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            text_device = f"cuda:{local_rank}"
        else:
            text_device = "cpu"

        try:
            self._text_encoder = load_text_encoder(
                checkpoint_path=model_path,
                gemma_model_path=self._config.model.text_encoder_path,
                device=text_device,
                dtype=torch.float32,
                load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
            )
        except (TypeError, KeyError, RuntimeError) as exc:
            logger.warning(
                "Text encoder checkpoint metadata load failed (%s). Using fallback loader.",
                exc,
            )
            self._text_encoder = self._load_text_encoder_fallback(
                checkpoint_path=model_path,
                gemma_model_path=self._config.model.text_encoder_path,
                device=text_device,
            )

        # Cache validation embeddings if prompts are configured
        cached_embeddings = None
        if self._config.validation.prompts:
            # NOTE: The text encoder's feature_extractor_linear has random weights
            # (the trained version is in the strategy).  Validation embeddings will
            # NOT match training quality, but this is acceptable when
            # skip_initial_validation=True and validation.interval=null.
            logger.info(
                "Pre-computing embeddings for %d validation prompts "
                "(feature_extractor_linear uses random init — for reference only)...",
                len(self._config.validation.prompts),
            )

            # Bypass connectors (they may be uninitialized/meta when loaded from
            # a from-scratch checkpoint that never trained them).
            if hasattr(self._text_encoder, 'skip_connectors'):
                self._text_encoder.skip_connectors = True
            if hasattr(self._text_encoder, 'skip_feature_extractor'):
                self._text_encoder.skip_feature_extractor = True

            cached_embeddings = []
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    v_ctx_pos, a_ctx_pos, _ = self._text_encoder(prompt)
                    v_ctx_neg, a_ctx_neg, _ = self._text_encoder(self._config.validation.negative_prompt)

                    cached_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=v_ctx_pos.cpu(),
                            audio_context_positive=a_ctx_pos.cpu(),
                            video_context_negative=v_ctx_neg.cpu() if v_ctx_neg is not None else None,
                            audio_context_negative=a_ctx_neg.cpu() if a_ctx_neg is not None else None,
                        )
                    )

        # Unload heavy components to free VRAM, keeping only the embedding connectors
        if IS_MAIN_PROCESS:
            self._text_encoder_state_for_ckpt = {
                key: value.detach().cpu()
                for key, value in self._text_encoder.state_dict().items()
                if not value.is_meta  # skip uninitialized (meta device) tensors
            }
        self._text_encoder.model = None
        self._text_encoder.tokenizer = None
        self._text_encoder.feature_extractor_linear = None

        logger.debug("Validation prompt embeddings cached. Gemma model unloaded")
        return cached_embeddings

    def _prune_transformer_layers(self, num_layers: int) -> None:
        """Prune transformer to use only the last N layers.

        This reduces model parameters and memory usage by keeping only the
        specified number of layers from the end of the model.

        Args:
            num_layers: Number of layers to keep (from the end)
        """
        # Access transformer blocks
        if not hasattr(self._transformer, 'transformer_blocks'):
            logger.warning("Transformer doesn't have 'transformer_blocks' attribute. Skipping layer pruning.")
            return

        original_num_layers = len(self._transformer.transformer_blocks)

        if num_layers >= original_num_layers:
            logger.info(f"Requested {num_layers} layers >= original {original_num_layers} layers. No pruning needed.")
            return

        if num_layers < 1:
            raise ValueError(f"num_transformer_layers must be >= 1, got {num_layers}")

        # Keep only the last num_layers
        # For example: original has 32 layers [0-31], keep last 16 means layers [16-31]
        start_layer = original_num_layers - num_layers

        logger.info(f"🔧 Pruning transformer: keeping layers [{start_layer}:{original_num_layers}] "
                   f"({num_layers}/{original_num_layers} layers, {num_layers/original_num_layers*100:.1f}%)")

        # Create new ModuleList with only the last N layers
        from torch import nn
        pruned_blocks = nn.ModuleList([
            self._transformer.transformer_blocks[i]
            for i in range(start_layer, original_num_layers)
        ])

        # Replace the transformer_blocks
        self._transformer.transformer_blocks = pruned_blocks

        # Also update depth in config if it exists
        if hasattr(self._transformer, 'config') and hasattr(self._transformer.config, 'depth'):
            self._transformer.config.depth = num_layers

        logger.info(f"✅ Model pruned: {num_layers} layers retained, "
                   f"estimated {100 * (1 - num_layers/original_num_layers):.1f}% parameter reduction")

    @staticmethod
    def _load_text_encoder_fallback(
        checkpoint_path: str | Path,
        gemma_model_path: str | Path,
        device: str,
    ) -> "AVGemmaTextEncoderModel":
        """Load text encoder without relying on checkpoint metadata."""
        from motionspeech_core.loader.sft_loader import SafetensorsStateDictLoader
        from motionspeech_core.text_encoders.gemma.encoders.av_encoder import (
            AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            AVGemmaTextEncoderModelConfigurator,
        )
        from motionspeech_core.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root

        config = {"transformer": {}}
        env_override = os.getenv("MOTIONSPEECH_NUM_LEARNABLE_REGISTERS")
        if env_override is not None:
            lowered = env_override.strip().lower()
            if lowered in {"none", "null", "nil"}:
                config["transformer"]["num_learnable_registers"] = None
            else:
                try:
                    config["transformer"]["num_learnable_registers"] = int(lowered)
                except ValueError:
                    pass

        text_encoder = AVGemmaTextEncoderModelConfigurator.from_config(config)
        for op in module_ops_from_gemma_root(str(gemma_model_path)):
            if op.matcher(text_encoder):
                text_encoder = op.mutator(text_encoder)

        loader = SafetensorsStateDictLoader()
        state_dict = loader.load(
            str(checkpoint_path),
            sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            device=torch.device(device),
        ).sd

        target_state = text_encoder.state_dict()
        for key in list(state_dict.keys()):
            if "learnable_registers" in key:
                target = target_state.get(key)
                if target is None or state_dict[key].shape != target.shape:
                    state_dict.pop(key, None)

        text_encoder.load_state_dict(state_dict, strict=False)
        return text_encoder.to(device=device, dtype=torch.float32)

    def _load_models(self) -> None:
        """Load the DuoFlow model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = self._training_strategy.requires_audio or self._config.validation.generate_audio
        use_audioldm2 = load_audio and self._config.model.audioldm2_model_id is not None

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        # Check if we need video VAE decoder
        # For motion-based training, we don't need video VAE decoder since we use motion latents
        need_video_vae_decoder = self._config.training_strategy.name not in ("motion_to_audio", "text_to_motion_audio")

        # Load all model components (except text encoder - already handled)
        model_path = self._config.model.model_path
        effective_checkpoint = self._effective_model_checkpoint()
        if model_path is None and effective_checkpoint:
            resolved = self._find_checkpoint(effective_checkpoint)
            if resolved is not None:
                model_path = resolved
                logger.info("Using load_checkpoint for base model load: %s", resolved)
            else:
                logger.warning(
                    "load_checkpoint set but no checkpoint found at %s; model_path remains unset.",
                    effective_checkpoint,
                )

        components = load_motionspeech_model(
            checkpoint_path=model_path,
            device="cpu",
            dtype=torch.float32,
            transformer_in_channels=self._config.model.transformer_in_channels,
            transformer_out_channels=self._config.model.transformer_out_channels,
            audio_in_channels=self._config.model.audio_in_channels,
            transformer_num_layers=self._config.model.num_transformer_layers,
            transformer_num_attention_heads=self._config.model.transformer_num_attention_heads,
            transformer_attention_head_dim=self._config.model.transformer_attention_head_dim,
            transformer_cross_attention_dim=self._config.model.transformer_cross_attention_dim,
            audio_num_attention_heads=self._config.model.audio_num_attention_heads,
            audio_attention_head_dim=self._config.model.audio_attention_head_dim,
            audio_cross_attention_dim=self._config.model.audio_cross_attention_dim,
            audio_caption_projection_use_skip=self._config.model.audio_caption_projection_use_skip,
            audio_caption_projection_act_fn=self._config.model.audio_caption_projection_act_fn,
            audio_phoneme_caption_projection_act_fn=self._config.model.audio_phoneme_caption_projection_act_fn,
            audio_audcap_caption_projection_act_fn=self._config.model.audio_audcap_caption_projection_act_fn,
            video_caption_projection_use_skip=self._config.model.video_caption_projection_use_skip,
            video_caption_projection_act_fn=self._config.model.video_caption_projection_act_fn,
            use_text_cross_attention=self._config.model.use_text_cross_attention,
            use_phoneme_time_anchor=self._config.model.use_phoneme_time_anchor,
            aux_ctc_tap_block=self._config.model.aux_ctc_tap_block,
            aux_ctc_tap_blocks=self._config.model.aux_ctc_tap_blocks,
            init_transformer_from_scratch=self._config.model.init_transformer_from_scratch,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=need_video_vae_decoder,  # Not needed for motion-to-audio
            with_audio_vae_decoder=load_audio and not use_audioldm2,
            with_vocoder=load_audio and not use_audioldm2,
            with_text_encoder=False,  # Text encoder handled separately
        )

        # Extract components
        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder
        if self._vae_decoder is not None:
            self._vae_decoder = self._vae_decoder.to(dtype=torch.float32)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.float32)
        self._scheduler = components.scheduler
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder

        if load_audio and use_audioldm2:
            from diffusers import AudioLDM2Pipeline

            logger.info("Loading AudioLDM2 VAE/Vocoder for training validation...")
            audio_pipe = AudioLDM2Pipeline.from_pretrained(
                self._config.model.audioldm2_model_id,
                torch_dtype=torch.float32,
                cache_dir=self._config.model.audioldm2_cache_dir,
            )
            audio_pipe = audio_pipe.to("cpu")
            self._audio_vae = audio_pipe.vae
            self._vocoder = audio_pipe.vocoder
        elif load_audio and not use_audioldm2:
            logger.info("Using DuoFlow audio VAE/Vocoder from checkpoint")
        elif load_audio and self._config.model.audioldm2_model_id is None:
            raise ValueError("audioldm2_model_id must be set when generate_audio is enabled")
        # Note: self._text_encoder was set in _load_text_encoder_and_cache_embeddings

        # Apply layer pruning if configured
        if self._config.model.num_transformer_layers is not None:
            self._prune_transformer_layers(self._config.model.num_transformer_layers)

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        if self._vae_decoder is not None:
            self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

        # Initialize phoneme encoder if enabled
        if self._config.model.use_phoneme_encoder:
            from motionspeech_trainer.phoneme_encoder import AudioLDM2PhonemeEncoder, PhonemeEncoder

            logger.info("Initializing PhonemeEncoder for enhanced audio generation...")

            ckpt_path = self._config.model.phoneme_encoder_checkpoint
            if ckpt_path and Path(ckpt_path).is_dir():
                self._phoneme_encoder = AudioLDM2PhonemeEncoder(
                    pretrained_dir=str(ckpt_path),
                    pad_token_id=0,
                    add_blank=self._config.model.phoneme_encoder_add_blank,
                )
            else:
                self._phoneme_encoder = PhonemeEncoder(
                    vocabs_size=self._config.model.phoneme_encoder_vocab_size,
                    out_channels=self._config.model.phoneme_encoder_out_channels,
                    hidden_channels=self._config.model.phoneme_encoder_hidden_channels,
                    filter_channels=self._config.model.phoneme_encoder_filter_channels,
                    n_heads=self._config.model.phoneme_encoder_n_heads,
                    n_layers=self._config.model.phoneme_encoder_n_layers,
                    kernel_size=self._config.model.phoneme_encoder_kernel_size,
                    p_dropout=self._config.model.phoneme_encoder_p_dropout,
                    pad_length=self._config.model.phoneme_encoder_pad_length,
                    add_blank=self._config.model.phoneme_encoder_add_blank,
                )

                if self._config.model.phoneme_encoder_checkpoint:
                    self._load_phoneme_encoder_checkpoint(
                        self._phoneme_encoder,
                        self._config.model.phoneme_encoder_checkpoint,
                        self._config.model.phoneme_encoder_checkpoint_prefix,
                    )

            self._phoneme_encoder = self._phoneme_encoder.to(dtype=torch.float32)
            # Freeze phoneme encoder unless explicitly marked trainable
            self._phoneme_encoder.requires_grad_(self._config.model.phoneme_encoder_trainable)

            # Set phoneme encoder in training strategy
            if hasattr(self._training_strategy, 'set_phoneme_encoder'):
                self._training_strategy.set_phoneme_encoder(self._phoneme_encoder)
                logger.info("✅ PhonemeEncoder initialized and set in training strategy")
        else:
            self._phoneme_encoder = None

    def _load_phoneme_encoder_checkpoint(
        self,
        phoneme_encoder: torch.nn.Module,
        checkpoint_path: str | Path,
        key_prefix: str | None,
    ) -> None:
        """Load pretrained phoneme encoder weights from an AudioLDM2 checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Phoneme encoder checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading phoneme encoder weights from: {checkpoint_path}")

        if checkpoint_path.suffix == ".safetensors":
            state = load_file(str(checkpoint_path))
        else:
            state = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if isinstance(state, dict) and "phoneme_encoder" in state and isinstance(state["phoneme_encoder"], dict):
            state = state["phoneme_encoder"]
        elif isinstance(state, dict) and "crossattn_vits_phoneme" in state and isinstance(state["crossattn_vits_phoneme"], dict):
            state = state["crossattn_vits_phoneme"]

        if not isinstance(state, dict):
            raise ValueError("Unsupported checkpoint format for phoneme encoder")

        target_keys = set(phoneme_encoder.state_dict().keys())
        candidate_prefixes = [
            key_prefix,
            "cond_stage_model.crossattn_vits_phoneme.",
            "cond_stage_model.phoneme_encoder.",
            "crossattn_vits_phoneme.",
            "phoneme_encoder.",
        ]

        best_state = {}
        for prefix in [p for p in candidate_prefixes if p]:
            subset = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if len(subset) > len(best_state):
                best_state = subset

        if not best_state:
            best_state = {k: v for k, v in state.items() if k in target_keys}

        if not best_state:
            raise ValueError(
                "Could not locate phoneme encoder weights in checkpoint. "
                "Set phoneme_encoder_checkpoint_prefix to the correct key prefix."
            )

        try:
            missing, unexpected = phoneme_encoder.load_state_dict(best_state, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load phoneme encoder weights. "
                "Check that phoneme encoder config matches AudioLDM2 (hidden_channels=192, "
                "out_channels=192, n_layers=6, n_heads=2, kernel_size=3, p_dropout=0.1)."
            ) from exc

        if missing:
            logger.warning(f"Phoneme encoder missing keys: {missing}")
        if unexpected:
            logger.warning(f"Phoneme encoder unexpected keys: {unexpected}")

        logger.info("✅ Phoneme encoder weights loaded")

    def _get_audio_sample_rate(self) -> int | None:
        if self._vocoder is None:
            return None
        if hasattr(self._vocoder, "output_sample_rate"):
            return self._vocoder.output_sample_rate
        config = getattr(self._vocoder, "config", None)
        if config is not None:
            sampling_rate = getattr(config, "sampling_rate", None)
            if sampling_rate is not None:
                return sampling_rate
        return None

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode.

        Honors `optimization.freeze_modules` (fnmatch glob against fully-qualified
        parameter names) — matched parameters have requires_grad set to False
        and are excluded from the optimizer.

        Also splits the remaining trainable parameters into two groups:
        `audio_params` (audio-branch / audio-side cross-modal parameters) and
        `other_params` (motion / video / connectors / everything else). The split
        is consumed by `_init_optimizer` to honor
        `optimization.audio_learning_rate` and `optimization.audio_weight_decay`.
        """
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
            # ``requires_grad_(True)`` also re-enables the main-caption
            # projections and text cross-attention modules that MotionSpeechModel froze
            # for direct-text-concat mode during construction.  Freeze those
            # unused modules again so DDP does not wait for gradients that can
            # never be produced.  Phoneme and AUDCAP audio cross-attention are
            # intentionally not part of this freeze helper.
            if not self._config.model.use_text_cross_attention:
                motion_text_ca = getattr(
                    self._config.training_strategy,
                    "motion_text_cross_attention",
                    False,
                )
                if motion_text_ca:
                    # Hybrid mode: motion text is injected twice (raw concat +
                    # connector cross-attn), while speech main text remains
                    # concat-only. Keep video caption_projection/attn2 trainable
                    # and freeze only the unused audio main-text path.
                    if hasattr(self._transformer, "audio_caption_projection"):
                        self._transformer.audio_caption_projection.requires_grad_(False)
                    for block in self._transformer.transformer_blocks:
                        if hasattr(block, "audio_attn2"):
                            block.audio_attn2.requires_grad_(False)
                    logger.info(
                        "[hybrid_text_conditioning] Motion main-text cross-attention "
                        "enabled; unused audio main-text projection/attention frozen"
                    )
                else:
                    freeze_main_text = getattr(self._transformer, "_freeze_main_text_cross_attention", None)
                    if freeze_main_text is None:
                        raise RuntimeError(
                            "use_text_cross_attention=false requires the transformer "
                            "to implement _freeze_main_text_cross_attention()"
                        )
                    freeze_main_text()
                    logger.info(
                        "[direct_text_concat] Re-froze unused main-text caption "
                        "projections and cross-attention after enabling full fine-tuning"
                    )

            # In joint-caption mode AUDCAP is already part of the shared main
            # text context. Freeze the now-unused dedicated AUDCAP projection
            # and per-block attention instead of paying DDP unused-parameter
            # discovery overhead on every step.
            if not getattr(self._config.training_strategy, "use_audcap", False):
                if hasattr(self._transformer, "audio_audcap_caption_projection"):
                    self._transformer.audio_audcap_caption_projection.requires_grad_(False)
                for block in self._transformer.transformer_blocks:
                    if hasattr(block, "audio_audcap_attn"):
                        block.audio_audcap_attn.requires_grad_(False)
                logger.info(
                    "[joint_text_conditioning] Dedicated audio AUDCAP cross-attention frozen; "
                    "AUDCAP is carried by the shared joint text context"
                )
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        # ── Apply freeze_modules BEFORE collecting trainable params ──
        # Glob match against the fully-qualified parameter name. Use the
        # bare parameter name (not "transformer_blocks.0.audio_...") since
        # LoRA / hook layers sometimes inspect names directly.
        freeze_patterns = list(getattr(self._config.optimization, 'freeze_modules', []) or [])
        if freeze_patterns:
            import fnmatch
            frozen_count = 0
            for name, p in self._transformer.named_parameters():
                if not p.requires_grad:
                    continue
                if any(fnmatch.fnmatch(name, pat) for pat in freeze_patterns):
                    p.requires_grad_(False)
                    frozen_count += 1
            if frozen_count:
                logger.info(
                    f"[freeze_modules] {frozen_count} transformer parameter tensors frozen "
                    f"(matched patterns: {freeze_patterns})"
                )

        # ── Collect trainable params ──
        all_trainable: list[tuple[str, torch.nn.Parameter]] = (
            [(n, p) for n, p in self._transformer.named_parameters() if p.requires_grad]
        )

        # Add strategy parameters (e.g., motion projection layer, connectors)
        if hasattr(self._training_strategy, 'named_parameters'):
            strat_named = list(self._training_strategy.named_parameters())
            strat_trainable = [(n, p) for n, p in strat_named if p.requires_grad]
            all_trainable.extend(strat_trainable)
            if strat_trainable:
                logger.debug(f"Added {len(strat_trainable)} strategy parameters")
                module_param_counts = {}
                for name, param in strat_trainable:
                    prefix = name.split('.')[0]  # e.g. video_connector, audio_connector, phoneme_proj
                    module_param_counts[prefix] = module_param_counts.get(prefix, 0) + param.numel()
                for mod_name, count in sorted(module_param_counts.items()):
                    logger.info(
                        f"Trainable module '{mod_name}': {count:,} params"
                    )

        # ── Split into audio / other groups ──
        # Convention: a parameter belongs to the audio group if its fully-qualified
        # name starts with or contains one of the audio markers. Patterns cover
        # both top-level model attributes (audio_patchify_proj) and nested
        # transformer-block attributes (transformer_blocks.0.audio_*), plus the
        # cross-modal AdaLN params whose audio-side names don't begin with "audio_".
        def _is_audio_param(name: str) -> bool:
            return (
                name.startswith("audio_")
                or ".audio_" in name
                or name.startswith("phoneme_proj.")
                or name.startswith("av_ca_audio_")
                or name.startswith("av_ca_v2a_")
                or "scale_shift_table_a2v_ca_audio" in name
            )

        audio_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, p in all_trainable:
            if _is_audio_param(name):
                audio_params.append(p)
            else:
                other_params.append(p)

        self._trainable_params = [p for _, p in all_trainable]
        self._audio_trainable_params = audio_params
        self._other_trainable_params = other_params

        # Summarize trainable parameter groups.
        audio_total = sum(p.numel() for p in audio_params)
        other_total = sum(p.numel() for p in other_params)
        logger.info(
            f"[param_groups] audio={audio_total:,} ({(audio_total/max(audio_total+other_total,1))*100:.1f}%) "
            f"other={other_total:,} ({(other_total/max(audio_total+other_total,1))*100:.1f}%) "
            f"frozen_patterns={freeze_patterns}"
        )
        if freeze_patterns:
            self._summarize_freeze(freeze_patterns)
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _sync_training_strategy_gradients(self) -> None:
        """All-reduce trainable strategy-module gradients across DDP ranks.

        The motion Text Connector, motion projection and phoneme projection live
        on the training strategy rather than inside the DDP-wrapped transformer.
        Without this explicit reduction, every rank would silently optimize a
        different connector even though the transformer itself stays in sync.
        """
        if self._accelerator.num_processes <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("distributed strategy-gradient sync requested without an initialized process group")

        world_size = torch.distributed.get_world_size()
        for parameter in self._training_strategy.parameters():
            if not parameter.requires_grad:
                continue

            has_grad = torch.tensor(
                1 if parameter.grad is not None else 0,
                device=parameter.device,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(has_grad, op=torch.distributed.ReduceOp.MAX)
            if has_grad.item() == 0:
                continue

            reduced_grad = parameter.grad
            if reduced_grad is None:
                reduced_grad = torch.zeros_like(parameter)
            torch.distributed.all_reduce(reduced_grad, op=torch.distributed.ReduceOp.SUM)
            reduced_grad.div_(world_size)
            if parameter.grad is None:
                parameter.grad = reduced_grad

    def _summarize_freeze(self, patterns: list[str]) -> None:
        """Log how many params each freeze pattern matched (informational)."""
        import fnmatch
        per_pattern: dict[str, int] = {pat: 0 for pat in patterns}
        for name, p in self._transformer.named_parameters():
            for pat in patterns:
                if fnmatch.fnmatch(name, pat):
                    per_pattern[pat] += 1
                    break
        for pat, count in per_pattern.items():
            logger.info(f"[freeze_modules]   pattern {pat!r}: {count} tensors frozen")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config."""
        effective_checkpoint = self._effective_model_checkpoint()
        if not effective_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(effective_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {effective_checkpoint}")
            return

        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

    def _initialize_checkpoint_history(self) -> None:
        """Include existing routine checkpoints in retention after a resume."""
        checkpoint_dir = Path(self._config.output_dir) / "checkpoints"
        if not checkpoint_dir.is_dir():
            return
        self._checkpoint_paths = sorted(
            (
                path
                for path in checkpoint_dir.glob("*_weights_step_*.safetensors")
                if _WEIGHTS_CHECKPOINT_RE.match(path.name)
                and _checkpoint_step(path) <= self._global_step
            ),
            key=_checkpoint_step,
        )

    def _load_legacy_best_eval_loss(self) -> None:
        """Restore best-loss bookkeeping from weights-only legacy runs."""
        best_info = Path(self._config.output_dir) / "checkpoints/best/best_info.txt"
        if not best_info.is_file():
            return
        for line in best_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("eval_loss="):
                self._best_eval_loss = float(line.split("=", 1)[1])
                return

    def _fast_forward_scheduler(self, completed_steps: int) -> None:
        """Place a fresh scheduler at a legacy checkpoint's inferred step."""
        if self._lr_scheduler is None:
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
            )
            warnings.filterwarnings(
                "ignore", message=r"The epoch parameter in `scheduler.step\(\)`"
            )
            for _ in range(completed_steps):
                self._lr_scheduler.step()

    def _restore_training_state(self) -> None:
        """Restore optimizer/scheduler/step state, with legacy fallback."""
        resume = self._resume_checkpoint
        if resume is None:
            self._global_step = 0
            return

        self._global_step = resume.global_step
        self._initialize_checkpoint_history()
        state_path = resume.training_state_path
        if state_path is None:
            self._fast_forward_scheduler(self._global_step)
            self._load_legacy_best_eval_loss()
            logger.warning(
                "Legacy weights-only resume at step %d: model and scheduler position "
                "restored, but optimizer momentum and RNG state were unavailable",
                self._global_step,
            )
            logger.info(
                "Resume LR after scheduler fast-forward: %.12g",
                self._optimizer.param_groups[0]["lr"],
            )
            return

        try:
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was added
            payload = torch.load(state_path, map_location="cpu")

        state_step = int(payload.get("global_step", -1))
        if state_step != resume.global_step:
            raise RuntimeError(
                f"Training-state step {state_step} does not match weights step "
                f"{resume.global_step}: {state_path}"
            )
        saved_optimizer_type = payload.get("optimizer_type")
        if saved_optimizer_type not in (None, self._config.optimization.optimizer_type):
            raise RuntimeError(
                f"Optimizer type changed across resume: {saved_optimizer_type} -> "
                f"{self._config.optimization.optimizer_type}"
            )
        saved_accumulation = payload.get("gradient_accumulation_steps")
        if saved_accumulation not in (
            None,
            self._config.optimization.gradient_accumulation_steps,
        ):
            raise RuntimeError(
                "gradient_accumulation_steps changed across a full-state resume: "
                f"{saved_accumulation} -> "
                f"{self._config.optimization.gradient_accumulation_steps}"
            )

        self._optimizer.load_state_dict(payload["optimizer"])
        scheduler_state = payload.get("lr_scheduler")
        if self._lr_scheduler is None:
            if scheduler_state is not None:
                raise RuntimeError("Checkpoint has scheduler state but current config disables it")
        elif scheduler_state is None:
            raise RuntimeError("Checkpoint is missing scheduler state required by current config")
        else:
            self._lr_scheduler.load_state_dict(scheduler_state)

        self._best_eval_loss = float(payload.get("best_eval_loss", float("inf")))
        if self._ema is not None and payload.get("ema") is not None:
            self._ema.load_state_dict(payload["ema"])
        scaler = getattr(self._accelerator, "scaler", None)
        if scaler is not None and payload.get("grad_scaler") is not None:
            scaler.load_state_dict(payload["grad_scaler"])

        if payload.get("python_rng_state") is not None:
            random.setstate(payload["python_rng_state"])
        if payload.get("numpy_rng_state") is not None:
            np.random.set_state(payload["numpy_rng_state"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng_state"], self._accelerator.device)

        saved_world_size = payload.get("world_size")
        if saved_world_size not in (None, self._accelerator.num_processes):
            logger.warning(
                "World size changed across resume: %s -> %s; optimization state is "
                "restored, but data ordering/global batch semantics differ",
                saved_world_size,
                self._accelerator.num_processes,
            )
        logger.info(
            "✅ Full training state restored from %s at step %d (lr=%.12g)",
            state_path,
            self._global_step,
            self._optimizer.param_groups[0]["lr"],
        )
        del payload
        gc.collect()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint including strategy parameters."""
        state_dict = load_file(checkpoint_path)

        # Separate transformer and strategy parameters
        transformer_state = {}
        strategy_state = {}
        connector_prefixes = (
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
            "text_embedding_projection.",
        )
        skipped_connector = 0

        for key, value in state_dict.items():
            if key.startswith('strategy.'):
                # Strategy parameters (e.g., ARIABridge for motion projection)
                strategy_key = key.replace('strategy.', '', 1)
                strategy_state[strategy_key] = value
            else:
                if key.startswith(connector_prefixes):
                    skipped_connector += 1
                    continue
                # Transformer parameters
                transformer_state[key] = value

        # F5-TTS-style raw text concatenation widens only the two modality
        # input projections.  Preserve the checkpoint's latent columns and
        # zero-initialize the newly added text columns so resuming starts from
        # the previous latent behavior without injecting a random 3840-D jump.
        model_state = self._transformer.state_dict()
        for projection_key in ("patchify_proj.weight", "audio_patchify_proj.weight"):
            source = transformer_state.get(projection_key)
            target = model_state.get(projection_key)
            if source is None or target is None or source.shape == target.shape:
                continue
            if source.ndim == 2 and target.ndim == 2 and source.shape[0] == target.shape[0] and source.shape[1] < target.shape[1]:
                expanded = torch.zeros_like(target)
                expanded[:, : source.shape[1]].copy_(source.to(device=expanded.device, dtype=expanded.dtype))
                transformer_state[projection_key] = expanded
                logger.info(
                    "Expanded %s for direct text concat: %s -> %s; copied %d latent columns and zero-initialized %d text columns",
                    projection_key,
                    tuple(source.shape),
                    tuple(target.shape),
                    source.shape[1],
                    target.shape[1] - source.shape[1],
                )

        # Load transformer weights.
        # allow_missing_checkpoint_keys=True permits parameters that exist in the
        # model but not in the checkpoint (e.g. resuming a pre-AUDCAP stage-1
        # checkpoint after the audio_audcap_* modules were added) — those keep
        # their random initialization. Unexpected checkpoint keys (in the
        # checkpoint but not the model) are dropped with a warning in that mode.
        allow_missing = getattr(self._config.model, 'allow_missing_checkpoint_keys', False)
        if allow_missing:
            model_keys = set(self._transformer.state_dict().keys())
            missing = [k for k in model_keys if k not in transformer_state]
            unexpected = [k for k in transformer_state if k not in model_keys]
            filtered_state = {k: v for k, v in transformer_state.items() if k in model_keys}
            self._transformer.load_state_dict(filtered_state, strict=False)
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "..." if len(missing) > 5 else ""
                logger.warning(
                    f"⚠️  allow_missing_checkpoint_keys=True: {len(missing)} model parameters "
                    f"not in checkpoint, keeping random init ({preview}{suffix})"
                )
            if unexpected:
                preview = ", ".join(unexpected[:5])
                suffix = "..." if len(unexpected) > 5 else ""
                logger.warning(
                    f"⚠️  allow_missing_checkpoint_keys=True: {len(unexpected)} checkpoint keys "
                    f"not in model, skipped ({preview}{suffix})"
                )
        else:
            self._transformer.load_state_dict(transformer_state, strict=True)
        logger.info("✅ Transformer checkpoint loaded successfully")
        if skipped_connector:
            logger.info("Skipped %s text encoder connector keys for transformer load", skipped_connector)

        # Load strategy weights if present
        if strategy_state:
            if hasattr(self._training_strategy, 'load_state_dict'):
                # Filter out connector weights that don't match current per-modality config.
                # E.g. checkpoint has video_connector.* but config sets video_use_connector=False,
                # so the strategy no longer has that module.
                strategy_modules = {name for name, _ in self._training_strategy.named_parameters()}
                strategy_buffers = {name for name, _ in self._training_strategy.named_buffers()}
                known_keys = strategy_modules | strategy_buffers
                filtered_state = {}
                skipped_keys = []
                for k, v in strategy_state.items():
                    if k in known_keys:
                        filtered_state[k] = v
                    else:
                        skipped_keys.append(k)
                if skipped_keys:
                    logger.info(
                        f"⚠️  Skipped {len(skipped_keys)} strategy keys not in current config "
                        f"(e.g. {skipped_keys[:3]})"
                    )
                missing_now = known_keys - filtered_state.keys()
                heads_missing = {
                    k for k in missing_now if k.startswith("audio_ctc_heads.")
                }
                relax_heads = (
                    not allow_missing and bool(missing_now) and missing_now == heads_missing
                )
                if relax_heads:
                    logger.info(
                        "⚠️  %d per-tap CTC head params absent from checkpoint "
                        "(checkpoint resumed into a new multi-tap layout); "
                        "keeping fresh initialization for them",
                        len(heads_missing),
                    )
                strategy_load_result = self._training_strategy.load_state_dict(
                    filtered_state,
                    strict=not allow_missing and not relax_heads,
                )
                if allow_missing and strategy_load_result.missing_keys:
                    preview = ", ".join(strategy_load_result.missing_keys[:5])
                    suffix = "..." if len(strategy_load_result.missing_keys) > 5 else ""
                    logger.warning(
                        "⚠️  allow_missing_checkpoint_keys=True: %d strategy parameters "
                        "not in checkpoint, keeping random init (%s%s)",
                        len(strategy_load_result.missing_keys),
                        preview,
                        suffix,
                    )
                logger.info(f"✅ Strategy checkpoint loaded successfully ({len(filtered_state)} parameters)")
            else:
                logger.warning(f"⚠️  Found {len(strategy_state)} strategy parameters in checkpoint but strategy doesn't support loading")
        else:
            # No strategy parameters in checkpoint (old format or strategy without trainable params)
            if hasattr(self._training_strategy, 'parameters') and sum(1 for _ in self._training_strategy.parameters()) > 0:
                logger.warning(f"⚠️  Checkpoint doesn't contain strategy parameters - strategy will use random initialization")

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)

        logger.info("✅ LoRA checkpoint loaded successfully")

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        use_grad_ckpt = self._config.optimization.enable_gradient_checkpointing
        if getattr(self._config.training_strategy, "audio_only_training", False):
            # Avoid checkpoint recompute mismatches when motion branch is unused.
            use_grad_ckpt = False
        transformer.set_gradient_checkpointing(use_grad_ckpt)

        # Keep frozen models on CPU for memory efficiency
        if self._vae_decoder is not None:
            self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")

        # Embedding connectors are already on GPU from _load_text_encoder_and_cache_embeddings

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)

        # Prepare strategy if it has parameters (e.g., motion projection layer)
        if hasattr(self._training_strategy, 'parameters'):
            # Move strategy to device without wrapping with DDP/FSDP
            # (strategy is lightweight, no need for distributed wrapping)
            if hasattr(self._training_strategy, 'to'):
                self._training_strategy = self._training_strategy.to(self._accelerator.device)

        # Initialize EMA if configured
        if self._config.optimization.use_ema:
            # Build EMA over both transformer and strategy trainable params
            # We combine them into a single pseudo-module for simplicity
            ema_modules = torch.nn.ModuleDict()
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            ema_modules["transformer"] = unwrapped
            if hasattr(self._training_strategy, 'parameters') and isinstance(self._training_strategy, torch.nn.Module):
                ema_modules["strategy"] = self._training_strategy
            self._ema = EMAModel(ema_modules, decay=self._config.optimization.ema_decay)
            logger.info(
                "EMA initialized with decay=%.6f, tracking %d parameter tensors",
                self._config.optimization.ema_decay,
                len(self._ema.shadow),
            )

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _get_optional_sources(self) -> set[str]:
        """Sources whose absence on disk is tolerated.

        - Audio-only training never touches the video/motion branch, so
          `motion_conditions` is allowed to be missing — the strategy uses
          `batch.get("motion_conditions", audio_conditions)` as a fallback.
        - `audcaps/` is treated as optional unconditionally: legacy Emilia
          preprocessed data does not contain it, and the strategy guards on
          `if "audcaps" in batch` before consuming it (with a warning if the
          schema is incomplete). When the directory is absent, AUDCAP
          conditioning is simply disabled for the run.
        """
        optional: set[str] = {"audcaps"}
        if getattr(self._config.training_strategy, "audio_only_training", False):
            optional.add("motion_conditions")
        return optional

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()
            optional_sources = self._get_optional_sources()

            # A collective barrier cannot surround a slow disk scan: NCCL's
            # watchdog would time it out. Non-main ranks poll the atomic
            # manifest on the shared filesystem without enqueueing a collective.
            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root,
                data_sources=data_sources,
                optional_sources=optional_sources,
                wait_for_manifest=not self._accelerator.is_main_process,
            )
            if self._accelerator.num_processes > 1 and not self._dataset.manifest_available:
                raise RuntimeError(
                    "Rank 0 could not publish the dataset manifest required by the other distributed ranks: "
                    f"{self._dataset.manifest_path}"
                )
            logger.debug(
                f"Loaded dataset with {len(self._dataset):,} samples from sources: "
                f"{list(data_sources)} (optional={sorted(optional_sources)})"
            )

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
            collate_fn=PrecomputedDataset.collate_precomputed_batch,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

        self._init_replay_dataloader(num_workers)

    def _init_replay_dataloader(self, num_workers: int) -> None:
        """Build the stage-1 audio-only replay dataloader (anti-forgetting).

        Enabled when ``data.replay_preprocessed_data_root`` is set together
        with ``data.replay_batch_prob > 0``. The replay dataset deliberately
        loads no ``motion_latents`` source: a replay batch therefore carries no
        ``motion_latents`` key, which is the signal the training strategy uses
        to route it through the audio-only preparation path.
        """
        replay_root = self._config.data.replay_preprocessed_data_root
        replay_prob = self._config.data.replay_batch_prob
        if not replay_root or replay_prob <= 0:
            self._replay_dataloader = None
            self._replay_every = 0
            return

        replay_sources = {
            self._training_strategy.config.audio_latents_dir: "audio_latents",
            self._training_strategy.config.speak_conditions_dir: "speak_conditions",
        }
        if getattr(self._training_strategy, "use_phoneme_encoder", False):
            # Older strategy configs lack the `phonemes_dir` field (they
            # hardcode the "phonemes" source name) — fall back to it.
            replay_sources[getattr(self._training_strategy.config, "phonemes_dir", "phonemes")] = "phonemes"

        # Same rank-0-manifest protocol as the main dataset: the shared
        # filesystem scan is too slow to hold an NCCL collective open, so
        # non-main ranks poll the atomic manifest written by rank 0.
        self._replay_dataset = PrecomputedDataset(
            replay_root,
            data_sources=replay_sources,
            wait_for_manifest=not self._accelerator.is_main_process,
        )
        if self._accelerator.num_processes > 1 and not self._replay_dataset.manifest_available:
            raise RuntimeError(
                "Rank 0 could not publish the replay dataset manifest required by the other distributed ranks: "
                f"{self._replay_dataset.manifest_path}"
            )
        self._replay_every = max(1, int(round(1.0 / replay_prob)))
        replay_batch_size = self._config.data.replay_batch_size or self._config.optimization.batch_size
        replay_loader = DataLoader(
            self._replay_dataset,
            batch_size=replay_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
            collate_fn=PrecomputedDataset.collate_precomputed_batch,
        )
        self._replay_dataloader = self._accelerator.prepare(replay_loader)
        logger.info(
            "Replay dataset ready: %d samples from %s | schedule: 1-in-%d micro-steps | "
            "batch_size=%d | replay_loss_weight=%s",
            len(self._replay_dataset),
            replay_root,
            self._replay_every,
            replay_batch_size,
            getattr(self._training_strategy.config, "replay_loss_weight", 1.0),
        )

    @torch.no_grad()
    def _get_eval_dataset(self) -> tuple[PrecomputedDataset, str] | None:
        """Select evaluation dataset based on config."""
        if self._dataset is None:
            return None

        val_root = self._config.validation.val_preprocessed_data_root
        if not val_root:
            return self._dataset, "train"

        if self._val_dataset is None:
            data_sources = self._training_strategy.get_data_sources()
            optional_sources = self._get_optional_sources()
            self._val_dataset = PrecomputedDataset(
                val_root,
                data_sources=data_sources,
                optional_sources=optional_sources,
                wait_for_manifest=not self._accelerator.is_main_process,
            )
            if self._accelerator.num_processes > 1 and not self._val_dataset.manifest_available:
                raise RuntimeError(
                    "Rank 0 could not publish the evaluation dataset manifest required by the other distributed ranks: "
                    f"{self._val_dataset.manifest_path}"
                )
            logger.debug(
                f"Loaded evaluation dataset with {len(self._val_dataset):,} samples from: {val_root}"
            )
        return self._val_dataset, "eval"

    def _evaluate_latent_loss(
        self,
        *,
        full_dataset: bool = False,
    ) -> tuple[str, dict[str, float]] | None:
        """Evaluate average latent loss over the selected dataset."""
        eval_dataset_info = self._get_eval_dataset()
        if eval_dataset_info is None:
            return None

        eval_dataset, base_eval_label = eval_dataset_info
        subset_size = self._config.validation.latent_eval_subset_size
        if not full_dataset and subset_size is not None and subset_size < len(eval_dataset):
            generator = torch.Generator().manual_seed(
                self._config.validation.latent_eval_subset_seed
            )
            indices = torch.randperm(len(eval_dataset), generator=generator)[:subset_size].tolist()
            eval_dataset = Subset(eval_dataset, indices)
            eval_label = f"{base_eval_label}_subset"
            logger.info(
                "Using fixed latent-loss subset: %d/%d samples (seed=%d, first_indices=%s)",
                subset_size,
                len(eval_dataset_info[0]),
                self._config.validation.latent_eval_subset_seed,
                indices[:8],
            )
        else:
            eval_label = (
                f"{base_eval_label}_full"
                if subset_size is not None
                else base_eval_label
            )
            if full_dataset:
                logger.info(
                    "Running milestone latent-loss evaluation on all %d samples",
                    len(eval_dataset),
                )

        was_training = self._transformer.training
        self._transformer.eval()
        if hasattr(self._training_strategy, "eval"):
            self._training_strategy.eval()

        num_workers = self._config.data.num_dataloader_workers
        eval_batch_size = self._config.validation.eval_batch_size or self._config.optimization.batch_size
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            # Validation loaders are recreated at every evaluation interval.
            # Persistent workers would survive the function call, inherit the
            # forked model address space, and accumulate one worker group per
            # validation. Training keeps persistent workers because its loader
            # is created once and reused for the whole run.
            persistent_workers=False,
            collate_fn=PrecomputedDataset.collate_precomputed_batch,
        )
        eval_loader = self._accelerator.prepare(eval_loader)

        device = self._accelerator.device
        total_loss = torch.tensor(0.0, device=device)
        motion_loss_total = torch.tensor(0.0, device=device)
        audio_loss_total = torch.tensor(0.0, device=device)
        total_count = torch.tensor(0, device=device)

        def _run_eval() -> None:
            nonlocal total_loss, motion_loss_total, audio_loss_total, total_count
            eval_iter = eval_loader
            use_tqdm = False
            if IS_MAIN_PROCESS and self._config.validation.eval_progress_bar:
                try:
                    import sys  # noqa: PLC0415
                    from tqdm import tqdm  # noqa: PLC0415

                    eval_iter = tqdm(
                        eval_loader,
                        desc="Eval",
                        leave=True,
                        dynamic_ncols=True,
                        file=sys.stdout,
                    )
                    use_tqdm = True
                except Exception:
                    eval_iter = eval_loader

            for batch in eval_iter:
                model_inputs = self._prepare_model_inputs(batch)
                video_pred, audio_pred = self._transformer(
                    video=model_inputs.video,
                    audio=model_inputs.audio,
                    perturbations=None,
                )

                if hasattr(self._training_strategy, "compute_loss_components"):
                    motion_loss, audio_loss, loss = self._training_strategy.compute_loss_components(
                        video_pred, audio_pred, model_inputs
                    )
                else:
                    loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)
                    motion_loss, audio_loss = loss, None

                batch_size = None
                for value in batch.values():
                    if isinstance(value, dict) and "latents" in value and torch.is_tensor(value["latents"]):
                        batch_size = value["latents"].shape[0]
                        break
                    if torch.is_tensor(value):
                        batch_size = value.shape[0]
                        break
                if batch_size is None:
                    batch_size = eval_batch_size

                total_loss += loss.detach() * batch_size
                motion_loss_total += motion_loss.detach() * batch_size
                if audio_loss is not None:
                    audio_loss_total += audio_loss.detach() * batch_size
                total_count += batch_size

            if use_tqdm:
                try:
                    import sys  # noqa: PLC0415

                    sys.stdout.write("\n")
                    sys.stdout.flush()
                except Exception:
                    pass

        if self._config.validation.fixed_noise_eval:
            seed = self._config.validation.fixed_noise_seed
            if torch.cuda.is_available():
                device_index = device.index if device.index is not None else 0
                rng_ctx = torch.random.fork_rng(devices=[device_index])
            else:
                rng_ctx = torch.random.fork_rng()

            with rng_ctx:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                _run_eval()
        else:
            _run_eval()

        total_loss = self._accelerator.gather(total_loss)
        motion_loss_total = self._accelerator.gather(motion_loss_total)
        audio_loss_total = self._accelerator.gather(audio_loss_total)
        total_count = self._accelerator.gather(total_count)

        if IS_MAIN_PROCESS:
            count = max(total_count.sum().item(), 1)
            avg_loss = total_loss.sum().item() / count
            avg_motion_loss = motion_loss_total.sum().item() / count
            avg_audio_loss = audio_loss_total.sum().item() / count
            logger.info(
                f"📊 {eval_label.capitalize()}-set latent loss: "
                f"total={avg_loss:.6f} motion={avg_motion_loss:.6f} audio={avg_audio_loss:.6f}"
            )
        else:
            avg_loss = None
            avg_motion_loss = None
            avg_audio_loss = None

        if was_training:
            self._transformer.train()
            if hasattr(self._training_strategy, "train"):
                self._training_strategy.train()

        if IS_MAIN_PROCESS:
            return eval_label, {
                "total": avg_loss,
                "motion": avg_motion_loss,
                "audio": avg_audio_loss,
            }
        return None

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler.

        If `optimization.audio_learning_rate` is set (non-null), audio-branch
        parameters are placed in a separate param group with that LR (and
        optionally `optimization.audio_weight_decay`). All other trainable
        parameters share the main `learning_rate` / `weight_decay`. The
        scheduler (Cosine / Linear / Polynomial / …) operates on all groups
        uniformly (groups inherit the same schedule by default — pass
        `scheduler_params['eta_min_audio']` etc. as needed in the future).
        """
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        weight_decay = getattr(opt_cfg, 'weight_decay', 0.01)
        audio_lr = getattr(opt_cfg, 'audio_learning_rate', None)
        audio_wd = getattr(opt_cfg, 'audio_weight_decay', None)
        if audio_wd is None:
            audio_wd = weight_decay

        if self._accelerator.is_main_process:
            logger.info(
                f"⚙️ Optimizer: {opt_cfg.optimizer_type}, lr={lr}, weight_decay={weight_decay}"
                + (f", audio_lr={audio_lr}, audio_weight_decay={audio_wd}" if audio_lr is not None else "")
            )

        # Build param_groups honoring audio_learning_rate.
        audio_params = getattr(self, '_audio_trainable_params', [])
        other_params = getattr(self, '_other_trainable_params', [])

        param_groups: list[dict] = []
        if audio_lr is not None and audio_params:
            # When audio_lr is set, give audio params their own group.
            # Other params form a separate group so they keep the default LR
            # (otherwise a single shared `lr` would apply to the whole list).
            param_groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
            param_groups.append({"params": audio_params, "lr": float(audio_lr), "weight_decay": float(audio_wd)})
        else:
            # Single group: backward-compatible behavior.
            param_groups.append({"params": self._trainable_params, "lr": lr, "weight_decay": weight_decay})

        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(param_groups)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(param_groups)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        # Add scheduler initialization
        lr_scheduler = self._create_scheduler(optimizer)

        # Only prepare the optimizer through Accelerate, NOT the scheduler.
        # Accelerate's prepare() wraps the scheduler in AcceleratedScheduler which
        # auto-steps on optimizer.step() AND our manual .step() call, causing the
        # scheduler to advance N×faster (N = num_processes). Keep the raw scheduler
        # so it is stepped exactly once per optimization step.
        self._optimizer = self._accelerator.prepare(optimizer)
        self._lr_scheduler = lr_scheduler

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config, with optional warmup."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = dict(self._config.optimization.scheduler_params or {})
        # Ensure numeric params are not accidentally parsed as strings by YAML
        for k, v in params.items():
            if isinstance(v, str):
                try:
                    params[k] = float(v)
                except ValueError:
                    pass
        warmup_steps = getattr(self._config.optimization, 'warmup_steps', 0)

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            main_scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps - warmup_steps,
                **params,
            )
        elif scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps - warmup_steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            main_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", (steps - warmup_steps) // 4),
                T_mult=params.pop("T_mult", 1),
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            main_scheduler = PolynomialLR(
                optimizer,
                total_iters=steps - warmup_steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            main_scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", (steps - warmup_steps) // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            # ConstantLR instead of a bare None so the warmup wrapper below
            # still applies (None made constant+warmup silently skip warmup).
            params.pop("eta_min", None)
            main_scheduler = ConstantLR(
                optimizer,
                factor=params.pop("factor", 1.0),
                total_iters=steps - warmup_steps,
                **params,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        # Wrap with warmup if configured
        if warmup_steps > 0 and main_scheduler is not None:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-6 / max(self._config.optimization.learning_rate, 1e-10),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
            if self._accelerator.is_main_process:
                logger.info(
                    f"📈 LR schedule: warmup {warmup_steps} steps → {scheduler_type} "
                    f"for {steps - warmup_steps} steps"
                )
            return scheduler

        return main_scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        ddp_kwargs = None
        _needs_find_unused = False
        if getattr(self._config.training_strategy, "audio_only_training", False):
            # Audio-only training can leave motion-related params unused.
            _needs_find_unused = True
        if self._config.data.replay_preprocessed_data_root and self._config.data.replay_batch_prob > 0:
            # Replay batches run the audio-only path inside a joint run, so
            # motion-branch parameters get no gradient on those steps.
            _needs_find_unused = True
        if not getattr(self._config.model, "use_phoneme_encoder", True):
            # Phoneme encoder disabled — transformer still has audio_phoneme_attn
            # and audio_phoneme_caption_projection which won't receive gradients.
            _needs_find_unused = True
        if _needs_find_unused:
            from accelerate import DistributedDataParallelKwargs  # noqa: PLC0415

            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        process_group_timeout = self._config.acceleration.process_group_timeout_seconds
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=process_group_timeout)
        )
        kwargs_handlers = [process_group_kwargs]
        if ddp_kwargs is not None:
            kwargs_handlers.append(ddp_kwargs)

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            kwargs_handlers=kwargs_handlers,
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )
            logger.info(
                "Distributed collective timeout: %d seconds",
                process_group_timeout,
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            grad_accum = self._config.optimization.gradient_accumulation_steps
            effective_batch = global_batch * grad_accum
            logger.info(
                "Local batch size: %s, global batch size: %s, grad accumulation: %s, effective global batch size: %s",
                local_batch,
                global_batch,
                grad_accum,
                effective_batch,
            )

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _sample_videos(self, progress: TrainingProgress) -> list[Path] | None:
        """Run validation by generating videos from validation prompts."""
        # Skip validation if video VAE decoder is not loaded (e.g., motion-to-audio training)
        if self._vae_decoder is None:
            logger.info("⏭️  Skipping validation sampling (no video VAE decoder loaded)")
            return None

        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        inference_steps = self._config.validation.inference_steps

        # Zero gradients and free GPU memory to reclaim memory before validation sampling
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        # Start sampling progress tracking
        sampling_ctx = progress.start_sampling(
            num_prompts=len(self._config.validation.prompts),
            num_steps=inference_steps,
        )

        # Create validation sampler with loaded models and progress tracking
        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        width, height, num_frames = self._config.validation.video_dims

        for prompt_idx, prompt in enumerate(self._config.validation.prompts):
            # Update progress to show current video
            sampling_ctx.start_video(prompt_idx)

            # Load conditioning image if provided
            condition_image = None
            if use_images:
                image_path = self._config.validation.images[prompt_idx]
                image = open_image_as_srgb(image_path)
                # Convert PIL image to tensor [C, H, W] in [0, 1]
                condition_image = F.to_tensor(image)

            # Load reference video if provided (for IC-LoRA)
            reference_video = None
            if use_reference_videos:
                ref_video_path = self._config.validation.reference_videos[prompt_idx]
                # read_video returns [F, C, H, W] in [0, 1]
                reference_video, _ = read_video(ref_video_path, max_frames=num_frames)

            # Get cached embeddings for this prompt if available
            cached_embeddings = (
                self._cached_validation_embeddings[prompt_idx]
                if self._cached_validation_embeddings is not None
                else None
            )

            # Create generation config
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                motion_guidance_scale=self._config.validation.motion_guidance_scale,
                audio_guidance_scale=self._config.validation.audio_guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
            )

            # Generate sample
            video, audio = sampler.generate(
                config=gen_config,
                device=self._accelerator.device,
            )

            # Save output (image for single frame, video otherwise)
            if IS_MAIN_PROCESS:
                ext = "png" if num_frames == 1 else "mp4"
                output_path = output_dir / f"step_{self._global_step:06d}_{prompt_idx + 1}.{ext}"
                if num_frames == 1:
                    save_image(video, output_path)
                else:
                    save_video(
                        video_tensor=video,
                        output_path=output_path,
                        fps=self._config.validation.frame_rate,
                        audio=audio,
                        audio_sample_rate=self._get_audio_sample_rate() if audio is not None else None,
                    )
                video_paths.append(output_path)

        # Clean up progress tasks
        sampling_ctx.cleanup()

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_training_state(self, weights_path: Path) -> Path | None:
        """Atomically save optimizer, scheduler, scaler, and progress state."""
        if not IS_MAIN_PROCESS or not self._config.checkpoints.save_training_state:
            return None
        state_path = _training_state_path(weights_path)
        temporary = state_path.with_name(state_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        payload = {
            "format_version": 1,
            "weights_filename": weights_path.name,
            "global_step": self._global_step,
            "optimizer": self._optimizer.state_dict(),
            "lr_scheduler": (
                self._lr_scheduler.state_dict() if self._lr_scheduler is not None else None
            ),
            "best_eval_loss": self._best_eval_loss,
            "ema": self._ema.state_dict() if self._ema is not None else None,
            "grad_scaler": (
                self._accelerator.scaler.state_dict()
                if getattr(self._accelerator, "scaler", None) is not None
                else None
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(self._accelerator.device)
                if torch.cuda.is_available()
                else None
            ),
            "optimizer_type": self._config.optimization.optimizer_type,
            "gradient_accumulation_steps": (
                self._config.optimization.gradient_accumulation_steps
            ),
            "world_size": self._accelerator.num_processes,
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, state_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        logger.info(
            "💾 Training state for step %d saved in %s",
            self._global_step,
            state_path.relative_to(self._config.output_dir),
        )
        return state_path

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        # Also collect strategy parameters (e.g., ARIABridge for motion projection, connectors)
        strategy_state = {}
        if hasattr(self._training_strategy, 'state_dict'):
            strategy_state = self._accelerator.get_state_dict(self._training_strategy)
            # Add strategy weights with prefix to avoid conflicts
            for key, value in strategy_state.items():
                full_state_dict[f"strategy.{key}"] = value
            logger.debug(f"Added {len(strategy_state)} strategy parameters to checkpoint")

        if IS_MAIN_PROCESS and strategy_state:
            connector_state = {}
            compatibility_prefixes = {
                "video_connector.": "model.diffusion_model.video_embeddings_connector.",
                "audio_connector.": "model.diffusion_model.audio_embeddings_connector.",
            }
            for key, value in strategy_state.items():
                for strategy_prefix, checkpoint_prefix in compatibility_prefixes.items():
                    if key.startswith(strategy_prefix):
                        suffix = key[len(strategy_prefix):]
                        # Safetensors rejects two keys that share storage. Keep a
                        # compatibility copy while strategy.* remains canonical.
                        connector_state[checkpoint_prefix + suffix] = value.detach().cpu().clone()
                        break

            if connector_state:
                full_state_dict.update(connector_state)
                logger.debug(
                    "Added %s trained connector tensors in text-encoder compatibility format",
                    len(connector_state),
                )

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        # Determine save precision
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

        def build_checkpoint_config() -> dict:
            model_cfg = self._config.model
            strategy_cfg = self._config.training_strategy
            transformer_cfg = {
                "in_channels": model_cfg.transformer_in_channels,
                "out_channels": model_cfg.transformer_out_channels,
                "num_layers": model_cfg.num_transformer_layers,
                "num_attention_heads": model_cfg.transformer_num_attention_heads,
                "attention_head_dim": model_cfg.transformer_attention_head_dim,
                "cross_attention_dim": model_cfg.transformer_cross_attention_dim,
                "audio_num_attention_heads": model_cfg.audio_num_attention_heads,
                "audio_attention_head_dim": model_cfg.audio_attention_head_dim,
                "audio_cross_attention_dim": model_cfg.audio_cross_attention_dim,
                "audio_caption_projection_use_skip": model_cfg.audio_caption_projection_use_skip,
                "video_caption_projection_use_skip": model_cfg.video_caption_projection_use_skip,
                "video_caption_projection_act_fn": model_cfg.video_caption_projection_act_fn,
                "video_use_connector": getattr(strategy_cfg, "video_use_connector", True),
                "audio_use_connector": getattr(strategy_cfg, "audio_use_connector", True),
                "connector_num_attention_heads": getattr(
                    strategy_cfg, "connector_num_attention_heads", 30
                ),
                "connector_attention_head_dim": getattr(
                    strategy_cfg, "connector_attention_head_dim", 128
                ),
                "connector_num_layers": getattr(strategy_cfg, "connector_num_layers", 2),
                "connector_ff_mult": getattr(strategy_cfg, "connector_ff_mult", 4),
                "connector_input_dim": getattr(strategy_cfg, "text_conditioning_dim", 3840),
                "connector_output_dim": getattr(strategy_cfg, "text_conditioning_dim", 3840),
                "connector_zero_init_output": getattr(
                    strategy_cfg, "connector_zero_init_output", False
                ),
                "num_learnable_registers": next(
                    (
                        connector.num_learnable_registers
                        for connector in (
                            getattr(self._training_strategy, "video_connector", None),
                            getattr(self._training_strategy, "audio_connector", None),
                        )
                        if connector is not None
                    ),
                    getattr(strategy_cfg, "connector_num_learnable_registers", 128),
                ),
                "gemma_num_layers": getattr(strategy_cfg, "gemma_num_layers", 1),
                "gemma_feature_selection": getattr(
                    strategy_cfg, "gemma_feature_selection", "last_n"
                ),
            }
            transformer_cfg = {
                k: v
                for k, v in transformer_cfg.items()
                if v is not None or k == "num_learnable_registers"
            }
            return {"transformer": transformer_cfg}

        metadata = {"config": json.dumps(build_checkpoint_config())}

        # For LoRA: extract only adapter weights; for full: use as-is
        if is_lora:
            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            # For FSDP, pass full_state_dict since model params aren't directly accessible
            state_dict = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict if is_fsdp else None)

            # Remove "base_model.model." prefix added by PEFT
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

            # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

            # Cast to configured precision
            state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}

            # Save atomically so resume discovery never selects a partial file.
            temporary_weights = saved_weights_path.with_name(saved_weights_path.name + ".tmp")
            if temporary_weights.exists():
                temporary_weights.unlink()
            save_file(state_dict, temporary_weights, metadata=metadata)
            os.replace(temporary_weights, saved_weights_path)
        else:
            # Cast to configured precision
            full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}

            # Save atomically so resume discovery never selects a partial file.
            temporary_weights = saved_weights_path.with_name(saved_weights_path.name + ".tmp")
            if temporary_weights.exists():
                temporary_weights.unlink()
            save_file(full_state_dict, temporary_weights, metadata=metadata)
            os.replace(temporary_weights, saved_weights_path)

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}")
        saved_state_path = self._save_training_state(saved_weights_path)

        # Save EMA weights as a separate file
        if self._ema is not None:
            ema_filename = f"{prefix}_ema_weights_step_{self._global_step:05d}.safetensors"
            ema_path = save_dir / ema_filename
            ema_state = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in self._ema.state_dict().items()}
            save_file(ema_state, ema_path, metadata=metadata)
            ema_rel = ema_path.relative_to(self._config.output_dir)
            logger.info(f"💾 EMA weights for step {self._global_step} saved in {ema_rel}")

        # Keep track of checkpoint paths, and cleanup old checkpoints if needed
        self._checkpoint_paths.append(saved_weights_path)
        # Permanent save every 5K steps — these are never cleaned up.
        if self._global_step % 5000 == 0:
            permanent_dir = save_dir / "permanent"
            permanent_dir.mkdir(parents=True, exist_ok=True)
            perm_path = permanent_dir / f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
            archive_method = _atomic_link_or_copy(saved_weights_path, perm_path)
            perm_rel = perm_path.relative_to(self._config.output_dir)
            logger.info(
                "📌 Permanent checkpoint saved via %s: %s",
                archive_method,
                perm_rel,
            )
            if saved_state_path is not None:
                permanent_state = permanent_dir / saved_state_path.name
                state_archive_method = _atomic_link_or_copy(
                    saved_state_path, permanent_state
                )
                logger.info(
                    "📌 Permanent training state saved via %s: %s",
                    state_archive_method,
                    permanent_state.relative_to(self._config.output_dir),
                )
            if self._ema is not None:
                ema_filename = f"{prefix}_ema_weights_step_{self._global_step:05d}.safetensors"
                ema_src = save_dir / ema_filename
                if ema_src.exists():
                    ema_method = _atomic_link_or_copy(
                        ema_src, permanent_dir / ema_filename
                    )
                    logger.info(
                        "📌 Permanent EMA checkpoint saved via %s: %s",
                        ema_method,
                        permanent_dir.relative_to(self._config.output_dir) / ema_filename,
                    )
        self._cleanup_checkpoints()
        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoints: {old_checkpoint}")
                old_state = _training_state_path(old_checkpoint)
                if old_state.exists():
                    old_state.unlink()
                    logger.info(f"Removed old training state: {old_state}")
            # Update the list to only contain kept checkpoints
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_best_checkpoint(self, eval_loss: float) -> None:
        """Save the current checkpoint as the best model weights (overwrites previous best)."""
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if self._config.model.training_mode == "lora" else "model"

        # Source: the checkpoint just saved for the current step
        src_path = save_dir / f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        if not src_path.exists():
            return

        best_dir = save_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        # Save best weights (overwrite previous best)
        best_path = best_dir / f"{prefix}_weights_best.safetensors"
        archive_method = _atomic_link_or_copy(src_path, best_path)

        # Write a small metadata file with step and loss info
        meta_path = best_dir / "best_info.txt"
        with open(meta_path, "w") as f:
            f.write(f"step={self._global_step}\neval_loss={eval_loss:.6f}\n")

        best_rel = best_path.relative_to(self._config.output_dir)
        logger.info(
            "🏆 Best checkpoint updated via %s: %s (eval_loss=%.6f, step=%d)",
            archive_method,
            best_rel,
            eval_loss,
            self._global_step,
        )

        # Also save best EMA if available
        if self._ema is not None:
            ema_src = save_dir / f"{prefix}_ema_weights_step_{self._global_step:05d}.safetensors"
            if ema_src.exists():
                ema_method = _atomic_link_or_copy(
                    ema_src, best_dir / f"{prefix}_ema_weights_best.safetensors"
                )
                logger.info("🏆 Best EMA checkpoint updated via %s", ema_method)

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(
                self._config.model_dump(mode="json"),
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
            )

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        # Skip W&B when TensorBoard is enabled — per user preference, TB replaces W&B.
        if (
            not self._config.wandb.enabled
            or not IS_MAIN_PROCESS
            or self._config.tensorboard.enabled
        ):
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard SummaryWriter (main process only)."""
        if not self._config.tensorboard.enabled or not IS_MAIN_PROCESS:
            return

        tb_cfg = self._config.tensorboard
        log_dir = tb_cfg.log_dir or str(Path(self._config.output_dir) / "tensorboard")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._tb_writer = SummaryWriter(
            log_dir=log_dir,
            max_queue=10,
            flush_secs=tb_cfg.flush_secs,
        )
        # Log hyperparameters as text so they're visible in the HParams / Text tab.
        try:
            cfg_text = yaml.safe_dump(self._config.model_dump(), default_flow_style=False, indent=2)
            self._tb_writer.add_text("config/training_config", cfg_text, global_step=0)
        except Exception as exc:  # pragma: no cover - non-fatal
            logger.warning("Failed to write config to TensorBoard: %s", exc)

        logger.info(
            "📊 TensorBoard logging enabled → %s (flush every %ds, log_every_n_steps=%d)",
            log_dir, tb_cfg.flush_secs, tb_cfg.log_every_n_steps,
        )

    def _log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to Weights & Biases and TensorBoard.

        TensorBoard uses ``self._global_step`` (or the supplied ``step``) as the
        global x-axis so train/, eval/ and stats/ curves line up. Each metric
        is written directly under its full key (e.g. ``train/loss``) so all
        data for one run lives under a single SummaryWriter — no nested
        ``metrics/<key>`` prefix.
        """
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
        if self._tb_writer is not None:
            tb_step = self._global_step if step is None else step
            try:
                for name, value in metrics.items():
                    self._tb_writer.add_scalar(
                        name,
                        float(value),
                        global_step=tb_step,
                    )
            except Exception as exc:  # pragma: no cover - non-fatal
                logger.warning("TensorBoard add_scalar failed: %s", exc)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples (videos or images) to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and sample_paths[0].suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".webp")
        media_cls = wandb.Image if is_image else wandb.Video

        samples = [media_cls(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)]
        self._wandb_run.log({"validation_samples": samples}, step=self._global_step)
