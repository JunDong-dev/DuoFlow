#!/usr/bin/env python3
"""Encode SMPL-X motion, decode cached latents, or reconstruct motion with the VAE."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party/motion_vae"))
from motion_vae_runtime import DEFAULT_CONFIG, DEFAULT_STATS, decode, encode, load_config, load_model, load_stats, read_motion, sample_ids, save_motion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mode", choices=("encode", "decode", "reconstruct"), default="reconstruct")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--latent-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.mode == "decode" and a.latent_dir is None:
        p.error("--latent-dir is required for decode")
    if a.mode != "decode" and a.data_root is None:
        p.error("--data-root is required for encode/reconstruct")
    torch.manual_seed(a.seed)
    model = load_model(load_config(a.config), a.checkpoint, a.device).eval()
    mean, std = load_stats(a.stats_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    latent_paths = sorted([*a.latent_dir.glob("*.pt"), *a.latent_dir.glob("*.npy")]) if a.mode == "decode" else []
    ids = [path.stem for path in latent_paths] if a.mode == "decode" else sample_ids(a.data_root, a.split, a.limit)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate latent sample names")
    path_by_id = dict(zip(ids, latent_paths))
    if a.limit is not None:
        ids = ids[:a.limit]
    if not ids:
        raise ValueError("No input samples found")
    with torch.inference_mode():
        for sample in ids:
            original = None
            if a.mode == "decode":
                source = path_by_id[sample]
                if source.suffix == ".pt":
                    payload = torch.load(source, map_location="cpu", weights_only=True)
                    array, frames = payload["latents"].float(), payload.get("motion_frames")
                else:
                    array, frames = torch.from_numpy(np.load(source)).float(), None
                if array.ndim != 2 or 256 not in array.shape:
                    raise ValueError("Latents must have shape [256,T] or [T,256]")
                z = (array.T if array.shape[0] == 256 else array).unsqueeze(0).to(a.device)
            else:
                original = read_motion(a.data_root / "motion_smplx" / (sample + ".npz"))
                z, frames = encode(model, (original - mean) / std, a.device)
                latent_dir = a.output_dir / "latents"
                latent_dir.mkdir(exist_ok=True)
                torch.save({"latents": z.squeeze(0).T.cpu(), "num_frames": z.shape[1], "motion_frames": frames,
                            "height": 1, "width": 1, "fps": 10.0}, latent_dir / (sample + ".pt"))
            if z.shape[-1] != 256 or not torch.isfinite(z).all():
                raise ValueError("Expected finite 256-channel motion latents")
            row = {"sample_id": sample, "latent_shape": list(z.shape)}
            if a.mode != "encode":
                motion = decode(model, z, mean, std, frames)
                save_motion(a.output_dir / sample, motion)
                row["motion_shape"] = list(motion.shape)
                if original is not None:
                    row["mse"] = float(np.mean((motion - original) ** 2))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (a.output_dir / "inference_report.json").write_text(json.dumps({"mode": a.mode, "samples": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
