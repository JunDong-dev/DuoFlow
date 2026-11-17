#!/usr/bin/env python3
"""Run a ten-sample training, reconstruction, and checkpoint-reload smoke test."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use an empty output directory for a smoke run")
    train = a.output_dir / "train"
    inference = a.output_dir / "reconstruct"
    decoded = a.output_dir / "decode_trained"

    def run(script, args):
        with (a.output_dir / (Path(script).stem + ".log")).open("a") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    run("train_motion_vae.py", ["--data-root", a.data_root, "--checkpoint", a.checkpoint,
        "--limit", 10, "--batch-size", 2, "--max-steps", 5, "--device", a.device, "--output-dir", train])
    training = json.loads((train / "training_report.json").read_text())
    assert len(training["selected_samples"]) == len(training["seen_samples"]) == 10
    assert training["steps"] == 5 and training["parameter_max_change"] > 0
    run("infer_motion_vae.py", ["--data-root", a.data_root, "--checkpoint", a.checkpoint,
        "--limit", 10, "--mode", "reconstruct", "--device", a.device, "--output-dir", inference])
    run("infer_motion_vae.py", ["--latent-dir", inference / "latents", "--checkpoint", train / "checkpoint_last.pth",
        "--limit", 10, "--mode", "decode", "--device", a.device, "--output-dir", decoded])
    first = json.loads((inference / "inference_report.json").read_text())["samples"]
    second = json.loads((decoded / "inference_report.json").read_text())["samples"]
    assert {r["sample_id"] for r in first} == set(training["selected_samples"]) == {r["sample_id"] for r in second}
    for folder, rows in ((inference, first), (decoded, second)):
        for row in rows:
            assert row["latent_shape"][-1] == 256 and row["motion_shape"][1] == 333
            assert 0 <= row["latent_shape"][1] * 3 - row["motion_shape"][0] < 3
            features = np.load(folder / (row["sample_id"] + ".npy"))
            assert list(features.shape) == row["motion_shape"] and np.isfinite(features).all()
    result = {"passed": True, "training_steps": 5, "training_samples": 10, "reconstructed_samples": 10,
              "decoded_with_trained_checkpoint": 10, "parameter_max_change": training["parameter_max_change"],
              "mean_reconstruction_mse": float(np.mean([r["mse"] for r in first])), "checkpoint": str(a.checkpoint)}
    (a.output_dir / "smoke_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
