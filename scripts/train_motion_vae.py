#!/usr/bin/env python3
"""Train the 333D Motion VAE without text, audio, or evaluation-model dependencies."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party/motion_vae"))
from motion_vae_runtime import DEFAULT_CONFIG, DEFAULT_STATS, load_config, load_model, load_stats, read_motion, sample_ids, vae_loss


class MotionDataset(Dataset):
    def __init__(self, root, ids, stats, window):
        self.root, self.ids, self.stats, self.window = Path(root), ids, stats, window

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample = self.ids[index]
        motion = read_motion(self.root / "motion_smplx" / (sample + ".npz"))
        if len(motion) < self.window:
            motion = np.pad(motion, ((0, self.window - len(motion)), (0, 0)), mode="edge")
        start = int(torch.randint(len(motion) - self.window + 1, (1,)).item())
        mean, std = self.stats
        return torch.from_numpy((motion[start:start + self.window] - mean) / std), sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="Optional model weights for a warm start; optimizer starts fresh")
    p.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    config = load_config(a.config)
    cfg = config["training"]
    for key, value in (("max_steps", a.max_steps), ("batch_size", a.batch_size), ("learning_rate", a.learning_rate)):
        if value is not None:
            cfg[key] = value
    if cfg["window_size"] % 3 or min(cfg["batch_size"], cfg["max_steps"], cfg["window_size"]) < 1:
        raise ValueError("Positive steps/batch and a window divisible by three are required")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    ids = sample_ids(a.data_root, a.split, a.limit)
    dataset = MotionDataset(a.data_root, ids, load_stats(a.stats_dir), cfg["window_size"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=a.num_workers, drop_last=False)
    model = load_model(config, a.checkpoint, a.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    monitor = next(model.parameters())
    before = monitor.detach().clone()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "config.yaml").write_text(__import__("yaml").safe_dump(config, sort_keys=False))
    iterator = iter(loader)
    seen, last_metrics = set(), None

    def checkpoint(step):
        path = a.output_dir / "checkpoint_last.pth"
        temporary = path.with_suffix(".tmp")
        torch.save({"net": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "model_config": config["model"], "step": step}, temporary)
        os.replace(temporary, path)

    for step in range(1, cfg["max_steps"] + 1):
        try:
            motion, names = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            motion, names = next(iterator)
        seen.update(names)
        lr = cfg["learning_rate"] * min(1.0, step / max(1, cfg["warmup_steps"]))
        lr *= cfg["gamma"] ** sum(step >= milestone for milestone in cfg["milestones"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        motion = motion.to(a.device)
        predicted, mu, logvar = model(motion)
        loss, terms = vae_loss(predicted, motion, mu, logvar, cfg["root_weight"], cfg["kl_weight"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(norm), "lr": lr, **{k: float(v.detach()) for k, v in terms.items()}}
        last_metrics = row
        print(json.dumps(row), flush=True)
        with (a.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step % cfg["save_every"] == 0 or step == cfg["max_steps"]:
            checkpoint(step)
    delta = float((monitor.detach() - before).abs().max())
    if not delta > 0:
        raise RuntimeError("The optimizer did not update the monitored model parameter")
    report = {"steps": cfg["max_steps"], "selected_samples": ids, "seen_samples": sorted(seen), "parameter_max_change": delta,
              "last_metrics": last_metrics, "checkpoint": str(a.output_dir / "checkpoint_last.pth")}
    (a.output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
