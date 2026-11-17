from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motionspeech_eval.common import (
    build_models,
    discover_stats,
    encode_motions,
    encode_texts,
    global_pair_accuracy,
)
from motionspeech_eval.prepare_dataset import configure_motionx_process, to_joints

import smplx
from scripts import motion_process as process


@dataclass(frozen=True)
class PairAccJob:
    sample_id: str
    level: str
    seed: int
    caption: str
    motion: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated MotionSpeech motions with L1/L2 Global PairAcc."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-format", choices=("auto", "metadata", "cosy"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Parent directory containing smplx/SMPLX_NEUTRAL.npz",
    )
    parser.add_argument("--model-gender", default="neutral")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conversion-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Maximum jobs per level and seed; 0 uses all")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=("l1", "l2"), default=("l1", "l2"))
    parser.add_argument("--stats-dir", type=Path, default=None)
    return parser.parse_args()


def normalized_text(text: str) -> str:
    return " ".join(str(text).split())


def tagged_caption(prompt: str, level: str) -> str | None:
    match = re.search(rf"<{level}>(.*?)</{level}>", prompt, flags=re.IGNORECASE | re.DOTALL)
    return normalized_text(match.group(1)) if match else None


def resolve_recorded_path(metadata_path: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_file():
        return path.resolve()
    local = metadata_path.parent / path.name
    return local.resolve() if local.is_file() else None


def discover_metadata_jobs(root: Path) -> list[PairAccJob]:
    jobs: list[PairAccJob] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete":
                continue
            level = str(metadata.get("variant") or metadata.get("caption_level") or "").lower()
            if level not in {"l1", "l2"}:
                continue
            outputs = metadata.get("outputs") or {}
            motion = None
            for key in ("smplx", "motion", "mofea333"):
                motion = resolve_recorded_path(metadata_path, outputs.get(key))
                if motion is not None:
                    break
            if motion is None:
                candidates = sorted(metadata_path.parent.glob("*.motion.npz"))
                candidates += sorted(metadata_path.parent.glob("*.mofea333.npy"))
                candidates += [
                    candidate
                    for candidate in sorted(metadata_path.parent.glob("*.npz"))
                    if not candidate.name.endswith("motion_codes.npz")
                ]
                if len(candidates) != 1:
                    raise FileNotFoundError(f"cannot resolve generated motion ({len(candidates)} candidates)")
                motion = candidates[0].resolve()
            prompt = str(metadata.get("prompt", ""))
            caption = tagged_caption(prompt, level) or normalized_text(metadata.get("motion_caption", ""))
            if not caption:
                raise ValueError("missing motion caption")
            jobs.append(
                PairAccJob(
                    sample_id=str(metadata.get("base_id") or metadata.get("sample_id")),
                    level=level,
                    seed=int(metadata.get("seed", 0)),
                    caption=caption,
                    motion=motion,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"invalid metadata {metadata_path}: {type(exc).__name__}: {exc}") from exc
    return jobs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_cosy_jobs(root: Path) -> list[PairAccJob]:
    run_path = root / "batch_run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"missing CosyVoice batch manifest: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    generated_path = Path(run.get("generated_manifest") or root / "cosyvoice" / "manifest.jsonl")
    source_path = Path(run.get("source_manifest") or "")
    result_dir = Path(
        run.get("result_dir")
        or root / "mambatalk_output" / "custom" / "mambatalk_motionspeech_batch" / "999"
    )
    generated = read_jsonl(generated_path)
    source = {str(row["id"]): row for row in read_jsonl(source_path)}
    run_seed = int(run.get("seed", 0))
    jobs: list[PairAccJob] = []
    for generated_row in generated:
        sample_id = str(generated_row["id"])
        original = source[sample_id]
        motion = (result_dir / f"res_{sample_id}.npz").resolve()
        if not motion.is_file():
            # The cascade can drop utterances whose MFA alignment fails. Its
            # legacy evaluator treats these as rejected/partial records, so
            # PairAcc follows the same completed-output policy.
            continue
        for level in ("l1", "l2"):
            caption = normalized_text(original.get(f"motion_{level}", ""))
            if not caption:
                raise ValueError(f"missing {level} caption for {sample_id}")
            jobs.append(PairAccJob(sample_id, level, run_seed, caption, motion))
    return jobs


def choose_input_format(root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (root / "batch_run.json").is_file():
        return "cosy"
    return "metadata"


def filter_jobs(jobs: list[PairAccJob], seeds: list[int] | None, limit: int) -> list[PairAccJob]:
    selected = [job for job in jobs if seeds is None or job.seed in seeds]
    selected.sort(key=lambda job: (job.seed, job.level, job.sample_id, str(job.motion)))
    deduplicated: list[PairAccJob] = []
    seen: set[tuple[str, str, int]] = set()
    counts: dict[tuple[int, str], int] = {}
    for job in selected:
        key = (job.sample_id, job.level, job.seed)
        if key in seen:
            raise ValueError(f"duplicate generated job: sample={job.sample_id} level={job.level} seed={job.seed}")
        group = (job.seed, job.level)
        if limit and counts.get(group, 0) >= limit:
            continue
        seen.add(key)
        counts[group] = counts.get(group, 0) + 1
        deduplicated.append(job)
    if not deduplicated:
        raise ValueError("no generated motions found after filtering")
    for group, count in counts.items():
        if count < 2:
            raise ValueError(f"Global PairAcc needs at least two jobs in group {group}, found {count}")
    return deduplicated


def rotation_6d_to_axis_angle(value: np.ndarray) -> np.ndarray:
    rows = value.reshape(-1, 55, 2, 3).astype(np.float64)
    first = rows[..., 0, :]
    first /= np.linalg.norm(first, axis=-1, keepdims=True).clip(1e-8)
    second = rows[..., 1, :] - np.sum(first * rows[..., 1, :], axis=-1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=-1, keepdims=True).clip(1e-8)
    third = np.cross(first, second)
    matrices = np.stack([first, second, third], axis=-2)
    return Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(len(value), 165).astype(np.float32)


def load_motion_parameters(path: Path, frames: int = 197) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npy":
        motion = np.asarray(np.load(path), dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] < 333:
            raise ValueError(f"expected [T,333+] motion at {path}, got {motion.shape}")
        indices = np.rint(np.linspace(0, len(motion) - 1, frames)).astype(np.int64)
        sampled = motion[indices, :333]
        poses = rotation_6d_to_axis_angle(sampled[:, 3:333])
        trans = sampled[:, :3]
        betas = np.zeros(10, dtype=np.float32)
    else:
        with np.load(path, allow_pickle=True) as value:
            poses_all = np.asarray(value["poses"], dtype=np.float32)
            trans_all = np.asarray(value["trans"], dtype=np.float32)
            raw_betas = np.asarray(value["betas"], dtype=np.float32) if "betas" in value else np.zeros(10)
        if poses_all.ndim != 2 or poses_all.shape[1] != 165 or trans_all.shape != (len(poses_all), 3):
            raise ValueError(f"invalid SMPL-X arrays at {path}: poses={poses_all.shape}, trans={trans_all.shape}")
        indices = np.rint(np.linspace(0, len(poses_all) - 1, frames)).astype(np.int64)
        poses, trans = poses_all[indices], trans_all[indices]
        betas = raw_betas.reshape(-1, raw_betas.shape[-1] if raw_betas.ndim > 1 else len(raw_betas))[0, :10]
        if len(betas) < 10:
            betas = np.pad(betas, (0, 10 - len(betas)))
    if not all(np.isfinite(array).all() for array in (poses, trans, betas)):
        raise ValueError(f"non-finite SMPL-X parameters at {path}")
    return poses, trans, betas.astype(np.float32)


def create_smplx_model(model_root: Path, gender: str, batch_size: int, device: torch.device) -> torch.nn.Module:
    return smplx.create(
        str(model_root),
        model_type="smplx",
        gender=gender,
        ext="npz",
        use_pca=False,
        num_betas=10,
        batch_size=batch_size,
    ).to(device).eval()


def configure_canonical_skeleton(model_root: Path, gender: str, device: torch.device) -> None:
    model = create_smplx_model(model_root, gender, 1, device)
    zero_pose = np.zeros((1, 1, 165), dtype=np.float32)
    zero_trans = np.zeros((1, 1, 3), dtype=np.float32)
    zero_betas = np.zeros((1, 10), dtype=np.float32)
    target = to_joints(model, zero_pose, zero_trans, zero_betas, device)[0]
    configure_motionx_process(target)
    del model


def embed_unique_motions(
    paths: list[Path],
    smplx_model: torch.nn.Module,
    motion_encoder: torch.nn.Module,
    movement_encoder: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[Path, torch.Tensor]:
    embeddings: dict[Path, torch.Tensor] = {}
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        padded_paths = batch_paths + [batch_paths[-1]] * (batch_size - len(batch_paths))
        loaded = [load_motion_parameters(path) for path in padded_paths]
        poses = np.stack([item[0] for item in loaded])
        trans = np.stack([item[1] for item in loaded])
        betas = np.stack([item[2] for item in loaded])
        joints = to_joints(smplx_model, poses, trans, betas, device)[: len(batch_paths)]
        features: list[np.ndarray] = []
        for path, joint_sequence in zip(batch_paths, joints):
            representation, _, _, _ = process.process_file(joint_sequence, 0.002)
            if representation.shape != (196, 623) or not np.isfinite(representation).all():
                raise ValueError(f"invalid Motion-X representation for {path}: {representation.shape}")
            features.append(representation.astype(np.float32))
        motion = torch.from_numpy(np.stack(features)).to(device)
        with torch.inference_mode():
            encoded = encode_motions(motion_encoder, movement_encoder, (motion - mean) / std.clamp_min(1e-8))
        for path, embedding in zip(batch_paths, encoded.cpu()):
            embeddings[path] = embedding
        print(f"encoded motions {min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)
    return embeddings


def main() -> int:
    args = parse_args()
    if args.conversion_batch_size < 1:
        raise ValueError("--conversion-batch-size must be >= 1")
    input_format = choose_input_format(args.input_root, args.input_format)
    discovered = (
        discover_cosy_jobs(args.input_root)
        if input_format == "cosy"
        else discover_metadata_jobs(args.input_root)
    )
    jobs = filter_jobs([job for job in discovered if job.level in args.variants], args.seeds, args.limit)
    missing_levels = set(args.variants) - {job.level for job in jobs}
    if missing_levels:
        raise ValueError(f"No completed jobs for requested levels: {sorted(missing_levels)}")
    device = torch.device(args.device)
    text_encoder, motion_encoder, movement_encoder, checkpoint = build_models(device, args.checkpoint)
    text_encoder.eval()
    motion_encoder.eval()
    movement_encoder.eval()
    configure_canonical_skeleton(args.model_root, args.model_gender, device)
    smplx_model = create_smplx_model(
        args.model_root,
        args.model_gender,
        args.conversion_batch_size * 197,
        device,
    )
    mean_path, std_path = (args.stats_dir / "mean.npy", args.stats_dir / "std.npy") if args.stats_dir else discover_stats()
    mean = torch.from_numpy(np.load(mean_path)).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(np.load(std_path)).to(device=device, dtype=torch.float32)
    paths = sorted({job.motion.resolve() for job in jobs}, key=str)
    motion_embeddings = embed_unique_motions(
        paths,
        smplx_model,
        motion_encoder,
        movement_encoder,
        mean,
        std,
        device,
        args.conversion_batch_size,
    )

    per_group: list[dict[str, Any]] = []
    by_level: dict[str, list[float]] = {"l1": [], "l2": []}
    for seed in sorted({job.seed for job in jobs}):
        for level in ("l1", "l2"):
            group = [job for job in jobs if job.seed == seed and job.level == level]
            if not group:
                continue
            captions = [job.caption for job in group]
            motion = torch.stack([motion_embeddings[job.motion.resolve()] for job in group]).to(device)
            with torch.inference_mode():
                text = encode_texts(text_encoder, captions, device, args.text_batch_size)
            pairacc = global_pair_accuracy(motion, text, captions)
            by_level[level].append(pairacc)
            per_group.append({"seed": seed, "level": level, "count": len(group), "global_pairacc": pairacc})

    level_results: dict[str, Any] = {}
    for level, values in by_level.items():
        if values:
            level_results[level] = {
                "global_pairacc": float(np.mean(values)),
                "seed_values": values,
                "num_seeds": len(values),
            }
    result = {
        "evaluator": "CLaM-H adapted for MotionSpeech",
        "checkpoint": str(Path(checkpoint).resolve()),
        "input_root": str(args.input_root.resolve()),
        "input_format": input_format,
        "discovered_jobs": len(discovered),
        "evaluated_jobs": len(jobs),
        "unique_motions": len(paths),
        "chance": 50.0,
        "levels": level_results,
        "groups": per_group,
        "conversion": "SMPL-X/333D -> 52 body-hand joints -> 196x623 Motion-X representation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
