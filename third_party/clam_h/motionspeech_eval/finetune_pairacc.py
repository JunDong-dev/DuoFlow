from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motionspeech_eval.common import (
    build_models,
    discover_stats,
    encode_motions,
    encode_texts,
    global_pair_accuracy,
    load_prepared,
)


def pairwise_ranking_loss(logits: torch.Tensor, captions: list[str]) -> torch.Tensor:
    """Train the same motion-to-text ordering used by Global PairAcc."""
    normalized = [" ".join(caption.lower().split()) for caption in captions]
    batch = logits.shape[0]
    valid = torch.ones((batch, batch), dtype=torch.bool, device=logits.device)
    valid.fill_diagonal_(False)
    for row, caption in enumerate(normalized):
        duplicates = [column for column, other in enumerate(normalized) if column != row and other == caption]
        if duplicates:
            valid[row, torch.tensor(duplicates, device=logits.device)] = False
    positive = logits.diagonal().unsqueeze(1).expand_as(logits)
    return F.softplus(logits[valid] - positive[valid]).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLaM-H with MotionSpeech L1/L2 multi-positive supervision.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--l1-weight", type=float, default=0.5)
    parser.add_argument("--loss", choices=("infonce", "pairwise"), default="infonce")
    parser.add_argument("--target-pairacc", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    motions: np.ndarray,
    l1: list[str],
    l2: list[str],
    text_encoder: torch.nn.Module,
    motion_encoder: torch.nn.Module,
    movement_encoder: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    motion_batch_size: int,
    text_batch_size: int,
) -> dict[str, float]:
    text_encoder.eval()
    motion_encoder.eval()
    movement_encoder.eval()
    parts: list[torch.Tensor] = []
    for start in range(0, len(motions), motion_batch_size):
        batch = torch.from_numpy(np.asarray(motions[start : start + motion_batch_size], dtype=np.float32)).to(device)
        parts.append(encode_motions(motion_encoder, movement_encoder, (batch - mean) / std.clamp_min(1e-8)))
    motion_embeddings = torch.cat(parts)
    text_l1 = encode_texts(text_encoder, l1, device, text_batch_size)
    text_l2 = encode_texts(text_encoder, l2, device, text_batch_size)
    return {
        "l1_global_pairacc": global_pair_accuracy(motion_embeddings, text_l1, l1),
        "l2_global_pairacc": global_pair_accuracy(motion_embeddings, text_l2, l2),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    metrics: dict[str, float],
    text_encoder: torch.nn.Module,
    motion_encoder: torch.nn.Module,
    movement_encoder: torch.nn.Module,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "text_encoder": text_encoder.state_dict(),
            "motion_encoder": motion_encoder.state_dict(),
            "movement_encoder": movement_encoder.state_dict(),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.l1_weight <= 1.0:
        raise ValueError("--l1-weight must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_motions, _, train_l1, train_l2 = load_prepared(args.train)
    val_motions, _, val_l1, val_l2 = load_prepared(args.val)
    text_encoder, motion_encoder, movement_encoder, source_checkpoint = build_models(device, args.checkpoint)
    movement_encoder.eval()
    for parameter in movement_encoder.parameters():
        parameter.requires_grad_(False)

    mean_path, std_path = discover_stats()
    mean = torch.from_numpy(np.load(mean_path)).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(np.load(std_path)).to(device=device, dtype=torch.float32)
    parameters = list(text_encoder.parameters()) + list(motion_encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    best_min = -1.0
    for epoch in range(1, args.epochs + 1):
        text_encoder.train()
        motion_encoder.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_motions))
        running = 0.0
        steps = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            if len(indices) < 2:
                continue
            motion_np = np.asarray(train_motions[indices], dtype=np.float32)
            motion_batch = torch.from_numpy(motion_np).to(device)
            motion_embeddings = encode_motions(
                motion_encoder,
                movement_encoder,
                (motion_batch - mean) / std.clamp_min(1e-8),
            )
            captions_l1 = [train_l1[i] for i in indices]
            captions_l2 = [train_l2[i] for i in indices]
            captions = captions_l1 + captions_l2
            text_embeddings = encode_texts(text_encoder, captions, device, args.text_batch_size)
            batch = len(indices)
            targets = torch.arange(batch, device=device)
            logits_l1 = motion_embeddings @ text_embeddings[:batch].T / args.temperature
            logits_l2 = motion_embeddings @ text_embeddings[batch:].T / args.temperature
            # Optimize the two hierarchy levels separately and with equal weight.
            # Treating L1 and L2 as interchangeable positives lets the more specific
            # L2 description dominate the log-sum-exp, which can leave coarse L1
            # retrieval under-trained even when L2 is already well separated.
            if args.loss == "pairwise":
                loss_l1 = pairwise_ranking_loss(logits_l1, captions_l1)
                loss_l2 = pairwise_ranking_loss(logits_l2, captions_l2)
            else:
                loss_l1 = 0.5 * (
                    F.cross_entropy(logits_l1, targets) + F.cross_entropy(logits_l1.T, targets)
                )
                loss_l2 = 0.5 * (
                    F.cross_entropy(logits_l2, targets) + F.cross_entropy(logits_l2.T, targets)
                )
            loss = args.l1_weight * loss_l1 + (1.0 - args.l1_weight) * loss_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            running += loss.item()
            steps += 1

        metrics = evaluate(
            val_motions,
            val_l1,
            val_l2,
            text_encoder,
            motion_encoder,
            movement_encoder,
            mean,
            std,
            device,
            args.eval_batch_size,
            args.text_batch_size,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(steps, 1)
        history.append(metrics)
        print(json.dumps(metrics))
        save_checkpoint(args.output_dir / "latest.tar", epoch, metrics, text_encoder, motion_encoder, movement_encoder)
        current_min = min(metrics["l1_global_pairacc"], metrics["l2_global_pairacc"])
        if current_min > best_min:
            best_min = current_min
            save_checkpoint(args.output_dir / "best.tar", epoch, metrics, text_encoder, motion_encoder, movement_encoder)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if current_min >= args.target_pairacc:
            break

    summary = {
        "source_checkpoint": str(source_checkpoint),
        "train": str(args.train),
        "val": str(args.val),
        "target_pairacc": args.target_pairacc,
        "best_min_pairacc": best_min,
        "epochs_completed": len(history),
        "seed": args.seed,
        "l1_weight": args.l1_weight,
        "loss": args.loss,
        "test_used_for_training_or_selection": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
