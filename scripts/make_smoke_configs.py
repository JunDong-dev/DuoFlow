#!/usr/bin/env python3
"""Create reduced-model training configs for the ten-clip smoke test."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def configure(base: dict, data_root: Path, validation_root: Path, output_dir: Path) -> dict:
    config = deepcopy(base)
    model = config["model"]
    model.update(
        num_transformer_layers=1,
        aux_ctc_tap_block=0,
        aux_ctc_tap_blocks=[0],
        transformer_num_attention_heads=4,
        transformer_attention_head_dim=32,
        transformer_cross_attention_dim=128,
        audio_num_attention_heads=4,
        audio_attention_head_dim=32,
        audio_cross_attention_dim=128,
    )
    strategy = config["training_strategy"]
    strategy.update(
        aux_ctc_hidden_dim=128,
        connector_num_attention_heads=4,
        connector_attention_head_dim=64,
        connector_num_layers=1,
        connector_ff_mult=2,
    )
    config["optimization"].update(
        steps=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        optimizer_type="adamw",
        warmup_steps=0,
        enable_gradient_checkpointing=False,
    )
    config["data"].update(
        preprocessed_data_root=str(data_root), num_dataloader_workers=0,
        replay_preprocessed_data_root=None, replay_batch_prob=0.0,
    )
    config["validation"].update(
        val_preprocessed_data_root=str(validation_root),
        interval=None,
        skip_initial_validation=True,
        eval_batch_size=1,
        inference_steps=2,
    )
    config["checkpoints"].update(interval=1, keep_last_n=1, precision="bfloat16")
    config["wandb"]["enabled"] = False
    config["tensorboard"].update(enabled=False, log_dir=str(output_dir / "tensorboard"))
    config["output_dir"] = str(output_dir)
    return config


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (REPO_ROOT / "configs/stage1.yaml").open() as handle:
        stage1 = configure(yaml.safe_load(handle), args.preprocessed_root, args.validation_root, args.output_root / "stage1")
    stage1["optimization"]["steps"] = args.steps
    stage1["model"].update(load_checkpoint=None, init_transformer_from_scratch=True)
    stage1_path = args.output_root / "stage1_smoke.yaml"
    stage1_path.write_text(yaml.safe_dump(stage1, sort_keys=False))

    with (REPO_ROOT / "configs/stage2.yaml").open() as handle:
        stage2 = configure(yaml.safe_load(handle), args.preprocessed_root, args.validation_root, args.output_root / "stage2")
    stage1_checkpoint = args.output_root / "stage1/checkpoints/best/model_weights_best.safetensors"
    stage2["optimization"]["steps"] = args.steps
    stage2["model"].update(load_checkpoint=str(stage1_checkpoint), init_transformer_from_scratch=False)
    stage2_path = args.output_root / "stage2_smoke.yaml"
    stage2_path.write_text(yaml.safe_dump(stage2, sort_keys=False))
    print(stage1_path)
    print(stage2_path)


if __name__ == "__main__":
    main()
