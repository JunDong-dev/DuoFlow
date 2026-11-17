from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionspeech_eval.common import (
    build_models,
    discover_stats,
    encode_motions,
    encode_texts,
    global_pair_accuracy,
    load_prepared,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MotionSpeech L1/L2 Global PairAcc with CLaM-H.")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    motions, ids, captions_l1, captions_l2 = load_prepared(args.prepared)
    text_encoder, motion_encoder, movement_encoder, checkpoint = build_models(device, args.checkpoint)
    text_encoder.eval()
    motion_encoder.eval()
    movement_encoder.eval()
    mean_path, std_path = discover_stats()
    mean = torch.from_numpy(np.load(mean_path)).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(np.load(std_path)).to(device=device, dtype=torch.float32)

    motion_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(motions), args.batch_size):
            batch = torch.from_numpy(np.asarray(motions[start : start + args.batch_size], dtype=np.float32)).to(device)
            batch = (batch - mean) / std.clamp_min(1e-8)
            motion_parts.append(encode_motions(motion_encoder, movement_encoder, batch).cpu())
        motion_embeddings = torch.cat(motion_parts).to(device)
        text_l1 = encode_texts(text_encoder, captions_l1, device, args.text_batch_size)
        text_l2 = encode_texts(text_encoder, captions_l2, device, args.text_batch_size)

    results = {
        "evaluator": "CLaM-H",
        "checkpoint": str(checkpoint),
        "prepared": str(args.prepared),
        "count": len(ids),
        "l1_global_pairacc": global_pair_accuracy(motion_embeddings, text_l1, captions_l1),
        "l2_global_pairacc": global_pair_accuracy(motion_embeddings, text_l2, captions_l2),
        "chance": 50.0,
        "long_text_handling": "sentence/word chunks of at most 32 words; normalized chunk embeddings are mean-pooled",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
