from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "CLIP_CLaM"))
sys.path.insert(0, str(REPO))

from clip import MotionTransformer, TextTransformer  # noqa: E402
from networks.modules import MovementConvEncoder  # noqa: E402


def discover_checkpoint(repo: Path = REPO) -> Path:
    preferred = sorted(repo.glob("checkpoints/motionx/**/text_mot_match_aug/**/finest.tar"))
    if not preferred:
        preferred = sorted(repo.glob("checkpoints/motionx/**/text_mot_match*/**/finest.tar"))
    if not preferred:
        preferred = sorted(repo.glob("checkpoints/**/finest.tar"))
    if not preferred:
        raise FileNotFoundError("No CLaM-H finest.tar found under checkpoints/")
    return preferred[0]


def discover_stats(repo: Path = REPO) -> tuple[Path, Path]:
    roots = [
        repo / "checkpoints" / "motionx" / "Decomp_SP001_SM001_H512" / "meta",
        repo / "dataset" / "motionx",
    ]
    for root in roots:
        mean_candidates = [root / "mean.npy", root / "Mean.npy"]
        std_candidates = [root / "std.npy", root / "Std.npy"]
        mean = next((p for p in mean_candidates if p.is_file()), None)
        std = next((p for p in std_candidates if p.is_file()), None)
        if mean and std:
            return mean, std
    means = sorted(repo.glob("checkpoints/motionx/**/meta/mean.npy"))
    stds = sorted(repo.glob("checkpoints/motionx/**/meta/std.npy"))
    if means and stds:
        return means[0], stds[0]
    raise FileNotFoundError("Motion-X mean/std files were not found")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_models(
    device: torch.device,
    checkpoint: Path | None = None,
) -> tuple[TextTransformer, MotionTransformer, MovementConvEncoder, Path]:
    checkpoint = checkpoint or discover_checkpoint()
    text_encoder = TextTransformer(
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        device=str(device),
    )
    motion_encoder = MotionTransformer(
        input_resolution=49,
        patch_size=1,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        device=str(device),
    )
    movement_encoder = MovementConvEncoder(623 - 4, 512, 512)
    state = load_checkpoint(checkpoint, device)
    text_encoder.load_state_dict(state["text_encoder"])
    motion_encoder.load_state_dict(state["motion_encoder"])
    movement_encoder.load_state_dict(state["movement_encoder"])
    return (
        text_encoder.to(device),
        motion_encoder.to(device),
        movement_encoder.to(device),
        checkpoint,
    )


def split_text(text: str, max_words: int = 32) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return ["a person moves"]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences or [text]:
        words = sentence.split()
        while len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(current) + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def encode_texts(
    encoder: TextTransformer,
    texts: Iterable[str],
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    texts = list(texts)
    flat: list[str] = []
    owners: list[int] = []
    for owner, caption in enumerate(texts):
        parts = split_text(caption)
        flat.extend(parts)
        owners.extend([owner] * len(parts))

    encoded: list[torch.Tensor] = []
    for start in range(0, len(flat), batch_size):
        encoded.append(encoder(flat[start : start + batch_size]))
    chunk_embeddings = F.normalize(torch.cat(encoded, dim=0), dim=-1)
    owner_tensor = torch.tensor(owners, device=device, dtype=torch.long)
    merged = torch.zeros(len(texts), chunk_embeddings.shape[-1], device=device, dtype=chunk_embeddings.dtype)
    merged.index_add_(0, owner_tensor, chunk_embeddings)
    counts = torch.bincount(owner_tensor, minlength=len(texts)).to(merged.dtype).unsqueeze(1)
    return F.normalize(merged / counts.clamp_min(1), dim=-1)


def encode_motions(
    motion_encoder: MotionTransformer,
    movement_encoder: MovementConvEncoder,
    motions: torch.Tensor,
) -> torch.Tensor:
    movement = movement_encoder(motions[..., :-4])
    lengths = torch.full(
        (motions.shape[0],),
        motions.shape[1] // 4,
        dtype=torch.long,
        device=motions.device,
    )
    return F.normalize(motion_encoder(movement, lengths), dim=-1)


def load_prepared(directory: Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    motions = np.load(directory / "motions.npy", mmap_mode="r")
    ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    l1 = json.loads((directory / "captions_l1.json").read_text(encoding="utf-8"))
    l2 = json.loads((directory / "captions_l2.json").read_text(encoding="utf-8"))
    if not (len(motions) == len(ids) == len(l1) == len(l2)):
        raise ValueError("Prepared motions, IDs, L1, and L2 have inconsistent lengths")
    return motions, ids, l1, l2


def global_pair_accuracy(motion: torch.Tensor, text: torch.Tensor, captions: list[str], block: int = 512) -> float:
    if motion.shape != text.shape:
        raise ValueError(f"embedding shape mismatch: {motion.shape} vs {text.shape}")
    total = 0.0
    count = 0
    normalized = [" ".join(c.lower().split()) for c in captions]
    for start in range(0, len(motion), block):
        stop = min(start + block, len(motion))
        scores = motion[start:stop] @ text.T
        correct = scores[torch.arange(stop - start, device=scores.device), torch.arange(start, stop, device=scores.device)].unsqueeze(1)
        valid = torch.ones_like(scores, dtype=torch.bool)
        for local, absolute in enumerate(range(start, stop)):
            valid[local, absolute] = False
            duplicate = [j for j, caption in enumerate(normalized) if j != absolute and caption == normalized[absolute]]
            if duplicate:
                valid[local, torch.tensor(duplicate, device=valid.device)] = False
        wins = (correct > scores) & valid
        ties = (correct == scores) & valid
        total += wins.sum().item() + 0.5 * ties.sum().item()
        count += valid.sum().item()
    if count == 0:
        raise ValueError("No non-equivalent caption pairs remain")
    return 100.0 * total / count
