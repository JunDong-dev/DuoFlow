#!/usr/bin/env python3
# Modified for DuoFlow; see NOTICE.md.
"""Resumable, single-model-load held-out inference for DuoFlow.

The script expands validation items into ``(sample, seed)`` jobs, keeps the
large DuoFlow/Motion VAE models resident across jobs, and writes metric-ready
motion/audio artifacts plus append-only audit manifests.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from data_splits import validate_preprocessed_split
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infer as engine  # noqa: E402

DEFAULT_VAL_ROOT = REPO_ROOT / "outputs/preprocessed/motionspeech/test"
DEFAULT_L1_JSON = None
DEFAULT_L2_JSON = None
DEFAULT_CHECKPOINT = None
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/inference/test"


MODEL_ARCHITECTURE = {
    "aux_ctc_hidden_dim": 1024,
    "num_transformer_layers": 8,
    "transformer_num_attention_heads": 32,
    "transformer_attention_head_dim": 32,
    "transformer_cross_attention_dim": 1024,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 32,
    "audio_cross_attention_dim": 1024,
    "connector_num_attention_heads": 15,
    "connector_attention_head_dim": 128,
    "connector_num_layers": 1,
    "connector_ff_mult": 2,
    "connector_num_learnable_registers": None,
}


def read_model_architecture(path: Path) -> dict[str, Any]:
    """Read all inference dimensions from checkpoint metadata."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        config = json.loads(metadata.get("config", "{}")).get("transformer", {})
        architecture = dict(MODEL_ARCHITECTURE)
        mapping = {
            "num_transformer_layers": "num_layers",
            "transformer_num_attention_heads": "num_attention_heads",
            "transformer_attention_head_dim": "attention_head_dim",
            "transformer_cross_attention_dim": "cross_attention_dim",
            "audio_num_attention_heads": "audio_num_attention_heads",
            "audio_attention_head_dim": "audio_attention_head_dim",
            "audio_cross_attention_dim": "audio_cross_attention_dim",
            "connector_num_attention_heads": "connector_num_attention_heads",
            "connector_attention_head_dim": "connector_attention_head_dim",
            "connector_num_layers": "connector_num_layers",
            "connector_ff_mult": "connector_ff_mult",
            "connector_num_learnable_registers": "num_learnable_registers",
        }
        for target, source in mapping.items():
            if source in config:
                architecture[target] = config[source]
        architecture["aux_ctc_hidden_dim"] = (
            architecture["audio_num_attention_heads"]
            * architecture["audio_attention_head_dim"]
        )
        required = [source for source in mapping.values() if source != "num_learnable_registers"]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                "Checkpoint metadata is incomplete; missing transformer keys: "
                + ", ".join(missing)
            )
        return architecture


@dataclass(frozen=True)
class Job:
    sample_id: str
    base_id: str
    variant: str
    seed: int
    prompt: str
    reference_motion: Path
    gt_audio: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate DuoFlow held-out outputs while loading the model only once."
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--val-root", "--data-root", dest="val_root", type=Path, default=None)
    parser.add_argument("--l1-json", type=Path, default=DEFAULT_L1_JSON)
    parser.add_argument("--l2-json", type=Path, default=DEFAULT_L2_JSON)
    parser.add_argument("--trained-model", type=Path, default=DEFAULT_CHECKPOINT, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--variants", nargs="+", choices=("l1", "l2"), default=("l1", "l2")
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(0,),
        help="Fixed seeds. Keep these identical for all compared methods.",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited base IDs or suffixed IDs such as sample_l1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Limit the expanded (sample, seed) jobs; useful for smoke tests.",
    )
    parser.add_argument(
        "--all-samples", dest="limit", action="store_const", const=None,
        help="Select all valid jobs instead of the default --limit 500.",
    )
    parser.add_argument(
        "--val-manifest-cache", type=Path, default=None,
        help="Inventory cache file (default: repository .cache/val_inference_inventory/).",
    )
    parser.add_argument(
        "--no-val-manifest-cache", action="store_true",
        help="Disable reading and writing the validation inventory cache.",
    )
    parser.add_argument(
        "--refresh-val-manifest", action="store_true",
        help="Recheck assets instead of reusing cached inventory; combine with --all-samples for a full rebuild.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing job only when its signature differs.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render", action="store_true", help="Also render MP4 files (slow).")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--motion-guidance-scale", type=float, default=1.0,
                        help="Per-modality CFG for the motion branch (engine "
                             "--cfg-guidance-scale-motion); None falls back to "
                             "--guidance-scale")
    parser.add_argument("--audio-guidance-scale", type=float, default=5.0,
                        help="Per-modality CFG for the audio branch (engine "
                             "--cfg-guidance-scale-audio); None falls back to "
                             "--guidance-scale")
    # ---- CTC aligner guidance (optional; requires an align-trained checkpoint
    # that contains strategy.audio_ctc_head.*) ----
    parser.add_argument(
        "--ctc-guidance-scale",
        type=float,
        default=0.0,
        help=(
            "Enable CTC prefix guidance when greater than zero. Values from "
            "0.05 to 0.5 are useful starting points."
        ),
    )
    parser.add_argument("--ctc-guidance-start-sigma", type=float, default=0.6)
    parser.add_argument("--ctc-guidance-every", type=int, default=1)
    parser.add_argument("--ctc-guidance-tap-block", type=int, default=3)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--duration-mode", choices=["reference", "phoneme"], default="reference",
                        help="reference: length from the reference motion latent (fixed on "
                             "segment datasets); phoneme: content-matched length from the fitted "
                             "phoneme-count regression")
    parser.add_argument("--duration-regression", type=Path, default=None,
                        help="JSON from scripts/fit_duration_regression.py")
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Latent FPS used by transformer positions; decoded motion is saved at 30 FPS.",
    )
    parser.add_argument(
        "--use-correct-motion-time-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use real-second motion token intervals. Enable this for checkpoints "
            "trained with use_correct_motion_time_coordinates=true."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--phoneme-add-blank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Insert phoneme blank tokens (ID 0); match the training setting.",
    )
    parser.add_argument(
        "--phoneme-pad-length", type=int, default=500,
        help="Phoneme sequence length after blank insertion; use 768 for blank768 data.",
    )

    # Defaults below mirror infer_newcaption.sh.
    parser.add_argument("--text-encoder-path", type=str,
                        default=os.environ.get("MOTIONSPEECH_TEXT_ENCODER"), required=False)
    parser.add_argument(
        "--motion-vae-path",
        type=str,
        default=os.environ.get("DUOFLOW_MOTION_VAE", str(REPO_ROOT / "weights/motion_vae.pth")),
    )
    parser.add_argument("--audioldm2-model-id", type=str, default="anhnct/audioldm2_gigaspeech")
    parser.add_argument(
        "--audioldm2-cache-dir", type=str,
        default=os.environ.get("MOTIONSPEECH_AUDIOLDM2_CACHE")
    )
    parser.add_argument(
        "--phoneme-encoder-path",
        type=str,
        default=os.environ.get("MOTIONSPEECH_PHONEME_ENCODER"),
    )
    parser.add_argument(
        "--phoneme-tokenizer-path",
        type=str,
        default=os.environ.get("MOTIONSPEECH_PHONEME_TOKENIZER"),
    )
    args = parser.parse_args()
    if args.val_root is None:
        env_name = "MOTIONSPEECH_TEST_ROOT" if args.split == "test" else "MOTIONSPEECH_VAL_ROOT"
        args.val_root = Path(os.environ.get(env_name, DEFAULT_VAL_ROOT.parent / args.split))
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR.parent / args.split
    validate_preprocessed_split(args.val_root, args.split)
    if args.l1_json is None:
        args.l1_json = args.val_root / f"manifests/{args.split}_l1.json"
    if args.l2_json is None:
        args.l2_json = args.val_root / f"manifests/{args.split}_l2.json"
    required_paths = {
        "--text-encoder-path": args.text_encoder_path,
        "--audioldm2-cache-dir": args.audioldm2_cache_dir,
        "--phoneme-encoder-path": args.phoneme_encoder_path,
        "--phoneme-tokenizer-path": args.phoneme_tokenizer_path,
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        parser.error("missing required model paths: " + ", ".join(missing))
    if args.phoneme_pad_length < 1:
        parser.error("--phoneme-pad-length must be a positive integer")
    return args


def json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def load_json_index(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("sample_id"):
            continue
        sample_id = str(row["sample_id"])
        if sample_id in index:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
        index[sample_id] = row
    return index


def parse_variant(sample_id: str) -> tuple[str, str]:
    for variant in ("l1", "l2"):
        suffix = f"_{variant}"
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)], variant
    raise ValueError(f"Validation item lacks _l1/_l2 suffix: {sample_id}")


def read_requested_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    requested = {
        raw.strip()
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.strip().startswith("#")
    }
    if not requested:
        raise ValueError(f"No sample IDs found in {path}")
    return requested


def required_asset_paths(val_root: Path, sample_id: str) -> list[Path]:
    return [
        val_root / "motion_latents" / f"{sample_id}.pt",
        val_root / "audio_latents" / f"{sample_id}.pt",
        val_root / "joint_conditions" / f"{sample_id}.pt",
        val_root / "phonemes" / f"{sample_id}.npy",
    ]


def inventory_fingerprint(
    args: argparse.Namespace, indexes: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    # Directory timestamps track asset additions/removals without stat-ing each
    # asset on every run. Tensor contents are not validated by this inventory.
    directories = {args.val_root / name for name in (
        "motion_latents", "audio_latents", "joint_conditions", "phonemes"
    )}
    directories.update(
        Path(str(entry.get("audio_path", ""))).parent
        for variant in set(args.variants) for entry in indexes[variant].values()
    )
    stamps = {}
    for directory in sorted(directories, key=str):
        try:
            stat = directory.stat()
            stamps[str(directory)] = [stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns]
        except FileNotFoundError:
            stamps[str(directory)] = None
    captions = json.dumps(indexes, sort_keys=True, ensure_ascii=False).encode("utf-8")
    json_files = {}
    for path in (args.l1_json, args.l2_json):
        stat = path.stat()
        json_files[str(path)] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return {"directories": stamps, "json_files": json_files,
            "json_sha256": hashlib.sha256(captions).hexdigest()}


def write_inventory_cache(path: Path, payload: dict[str, Any]) -> None:
    # Unique temporary files allow concurrent launches to write valid snapshots.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_jobs(args: argparse.Namespace) -> tuple[list[Job], dict[str, Any]]:
    started = time.monotonic()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    print(f"[inventory] Preparing validation jobs (limit={args.limit}) ...", flush=True)
    indexes = {
        "l1": load_json_index(args.l1_json),
        "l2": load_json_index(args.l2_json),
    }
    requested = read_requested_ids(args.sample_ids_file)
    motion_dir = args.val_root / "motion_latents"
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Validation motion_latents directory not found: {motion_dir}")

    selection = {
        "version": 1, "val_root": str(args.val_root),
        "l1_json": str(args.l1_json), "l2_json": str(args.l2_json),
        "variants": sorted(set(args.variants)),
        "requested_ids": sorted(requested) if requested is not None else None,
    }
    cache_enabled = not getattr(args, "no_val_manifest_cache", False)
    cache_path = getattr(args, "val_manifest_cache", None)
    if cache_path is None:
        key = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:24]
        cache_path = REPO_ROOT / ".cache" / "val_inference_inventory" / f"{key}.json"
    fingerprint = inventory_fingerprint(args, indexes) if cache_enabled else None
    valid_ids: list[str] = []
    rejected: list[dict[str, Any]] = []
    scanned = candidate_count = 0
    complete = cache_hit = False
    if cache_enabled and not getattr(args, "refresh_val_manifest", False):
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached["selection"] != selection or cached["fingerprint"] != fingerprint:
                print("[inventory] Cache invalidated: source or selection changed.", flush=True)
            else:
                ids = cached["valid_ids"]
                # Validate the cache structure; damaged caches fall back to a scan.
                if not isinstance(ids, list) or any(not isinstance(s, str) for s in ids):
                    raise ValueError("invalid cached sample IDs")
                for sample_id in ids:
                    base_id, variant = parse_variant(sample_id)
                    if variant not in args.variants or base_id not in indexes[variant]:
                        raise ValueError("cached sample missing from JSON")
                scanned, candidate_count = cached["scanned"], cached["candidate_count"]
                complete, rejected = cached["complete"], cached["rejected"]
                if (type(scanned) is not int or type(candidate_count) is not int
                        or not 0 <= len(ids) <= scanned <= candidate_count
                        or type(complete) is not bool or not isinstance(rejected, list)
                        or (complete and scanned != candidate_count)):
                    raise ValueError("invalid cached inventory counts")
                valid_ids, cache_hit = ids, True
                print(f"[inventory] Cache hit: {len(ids)} valid samples, "
                      f"checked {scanned}/{candidate_count}, complete={complete}", flush=True)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[inventory] Cache unavailable ({exc}); checking assets.", flush=True)
            valid_ids, rejected = [], []
            scanned = candidate_count = 0
            complete = cache_hit = False

    resolved: set[str] = set()
    for sample_id in valid_ids:
        resolved.update((sample_id, parse_variant(sample_id)[0]))

    def enough_jobs() -> bool:
        return (args.limit is not None and len(valid_ids) * len(args.seeds) >= args.limit
                and (requested is None or requested <= resolved))

    candidates = []
    scan_required = not complete and not enough_jobs()
    if scan_required:
        candidates = sorted(path.stem for path in motion_dir.glob("*.pt"))
        candidate_count = len(candidates)
        print(f"[inventory] Checking assets from candidate {scanned}/{candidate_count} ...", flush=True)
    last_progress = time.monotonic()
    for position in range(scanned, len(candidates)):
        if enough_jobs():
            break
        sample_id = candidates[position]
        scanned = position + 1
        if scanned % 500 == 0 or time.monotonic() - last_progress >= 5:
            print(f"[inventory] Checked {scanned}/{candidate_count}, "
                  f"valid={len(valid_ids)}, elapsed={time.monotonic() - started:.1f}s", flush=True)
            last_progress = time.monotonic()
        try:
            base_id, variant = parse_variant(sample_id)
        except ValueError as exc:
            rejected.append({"sample_id": sample_id, "reason": str(exc)})
            continue
        if variant not in args.variants:
            continue
        if requested is not None and sample_id not in requested and base_id not in requested:
            continue

        entry = indexes[variant].get(base_id)
        if entry is None:
            rejected.append({"sample_id": sample_id, "reason": f"missing from {variant} JSON"})
            continue
        prompt = str(entry.get("caption", "")).strip()
        gt_audio = Path(str(entry.get("audio_path", "")))
        assets = required_asset_paths(args.val_root, sample_id)
        missing = [str(path) for path in assets if not path.is_file()]
        if not prompt:
            missing.append("caption")
        if not gt_audio.is_file():
            missing.append(str(gt_audio))
        if missing:
            rejected.append({"sample_id": sample_id, "reason": "missing assets", "missing": missing})
            continue
        expected_tag = f"<{variant.upper()}>"
        if expected_tag not in prompt or "<S>" not in prompt:
            rejected.append(
                {
                    "sample_id": sample_id,
                    "reason": f"prompt missing {expected_tag} or <S>",
                }
            )
            continue
        valid_ids.append(sample_id)
        resolved.update((sample_id, base_id))

    if scan_required:
        complete = scanned == candidate_count
    if cache_enabled and (scan_required or not cache_hit):
        # A scan that overlaps a dataset update must not bless a stale snapshot.
        if inventory_fingerprint(args, indexes) == fingerprint:
            try:
                write_inventory_cache(cache_path, {
                    "selection": selection, "fingerprint": fingerprint,
                    "valid_ids": valid_ids, "rejected": rejected,
                    "candidate_count": candidate_count, "scanned": scanned,
                    "complete": complete,
                })
                print(f"[inventory] Saved {'full' if complete else 'partial'} cache: {cache_path}", flush=True)
            except OSError as exc:
                print(f"[inventory] Could not save cache: {exc}", flush=True)
        else:
            print("[inventory] Sources changed during scan; cache not saved.", flush=True)

    if requested is not None:
        unresolved = sorted(requested - resolved)
        if unresolved:
            raise ValueError(f"Requested IDs not found in valid val inventory: {unresolved[:20]}")

    # Limit before constructing Job objects as well as before filesystem checks.
    selected_ids = valid_ids
    if args.limit is not None:
        selected_ids = valid_ids[: (args.limit + len(args.seeds) - 1) // len(args.seeds)]
    jobs = [
        Job(
            sample_id=sample_id,
            base_id=base_id,
            variant=variant,
            seed=int(seed),
            prompt=str(indexes[variant][base_id]["caption"]).strip(),
            reference_motion=motion_dir / f"{sample_id}.pt",
            gt_audio=Path(str(indexes[variant][base_id]["audio_path"])),
        )
        for sample_id in selected_ids
        for base_id, variant in [parse_variant(sample_id)]
        for seed in args.seeds
    ]
    expanded_count = len(valid_ids) * len(args.seeds)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    inventory = {
        "candidate_motion_files": candidate_count,
        "valid_condition_samples": len(valid_ids),
        "expanded_jobs_before_limit": expanded_count,
        "selected_jobs": len(jobs),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "inventory_complete": complete,
        "scanned_candidates": scanned,
        "counts_scope": "full" if complete else "checked_prefix",
        "manifest_cache_hit": cache_hit,
        "manifest_cache_path": str(cache_path) if cache_enabled else None,
        "inventory_seconds": round(time.monotonic() - started, 3),
    }
    return jobs, inventory


def read_num_frames(reference_motion: Path) -> int:
    payload = torch.load(reference_motion, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "num_frames" in payload:
        frames = int(payload["num_frames"])
    elif isinstance(payload, dict) and "latents" in payload:
        latent = payload["latents"]
        if latent.ndim < 2:
            raise ValueError(f"Unexpected latent shape in {reference_motion}: {tuple(latent.shape)}")
        frames = int(latent.shape[1])
    else:
        raise ValueError(f"Cannot determine num_frames from {reference_motion}")
    if frames < 1:
        raise ValueError(f"Invalid num_frames={frames} in {reference_motion}")
    return frames


AUDIO_TO_MOTION_RATIO = 2.51  # engine constant in generate_text_to_motion_audio.py


def read_phoneme_count(val_root: Path, sample_id: str) -> int:
    """Valid (non-pad) phoneme tokens; pad id is 0."""
    ids = np.load(val_root / "phonemes" / f"{sample_id}.npy")
    return int((ids > 0).sum())


def derive_num_frames_from_phonemes(coeff: dict, n_phonemes: int) -> tuple[int, int]:
    """Duration regression T = a*N + b (clamped to GT token range);
    motion frames = T / AUDIO_TO_MOTION_RATIO."""
    t = coeff["a_tokens_per_phoneme"] * n_phonemes + coeff["b_tokens"]
    t = max(coeff["t_min_clamp"], min(coeff["t_max_clamp"], t))
    t = int(round(t))
    frames = max(1, int(round(t / AUDIO_TO_MOTION_RATIO)))
    return frames, t


_DURATION_COEFF_CACHE: dict[str, dict] = {}


def load_duration_coeff(path) -> dict:
    """Load (and cache) the duration-regression JSON. Safe to call per job."""
    key = str(path)
    if key not in _DURATION_COEFF_CACHE:
        _DURATION_COEFF_CACHE[key] = json.loads(Path(path).read_text())
    return _DURATION_COEFF_CACHE[key]


def make_engine_args(args: argparse.Namespace) -> argparse.Namespace:
    synthetic_argv = [
        "generate_text_to_motion_audio.py",
        "--trained-model-path", str(args.trained_model),
        "--text-encoder-path", args.text_encoder_path,
        "--motion-vae-path", args.motion_vae_path,
        "--audioldm2-model-id", args.audioldm2_model_id,
        "--audioldm2-cache-dir", args.audioldm2_cache_dir,
        "--num-transformer-layers", str(MODEL_ARCHITECTURE["num_transformer_layers"]),
        "--transformer-num-attention-heads", str(MODEL_ARCHITECTURE["transformer_num_attention_heads"]),
        "--transformer-attention-head-dim", str(MODEL_ARCHITECTURE["transformer_attention_head_dim"]),
        "--transformer-cross-attention-dim", str(MODEL_ARCHITECTURE["transformer_cross_attention_dim"]),
        "--audio-num-attention-heads", str(MODEL_ARCHITECTURE["audio_num_attention_heads"]),
        "--audio-attention-head-dim", str(MODEL_ARCHITECTURE["audio_attention_head_dim"]),
        "--audio-cross-attention-dim", str(MODEL_ARCHITECTURE["audio_cross_attention_dim"]),
        "--text-feature-mode", "gemma_hidden",
        "--motion-source", "full",
        "--speak-source", "full",
        "--use-connectors",
        "--connector-residual",
        "--video-use-connector",
        "--audio-use-connector",
        "--no-audio-caption-projection-use-skip",
        "--audio-caption-projection-act-fn", "none",
        "--video-caption-projection-act-fn", "none",
        "--audio-phoneme-caption-projection-act-fn", "none",
        "--use-phoneme-encoder",
        "--phoneme-encoder-checkpoint", args.phoneme_encoder_path,
        "--phoneme-tokenizer-path", args.phoneme_tokenizer_path,
        "--phoneme-pad-length", str(args.phoneme_pad_length),
        "--phoneme-add-blank" if args.phoneme_add_blank else "--no-phoneme-add-blank",
        "--phoneme-pad-token-id", "0",
        "--phoneme-source", "speech_tags",
        "--phoneme-joiner", " ",
        "--prompt", "placeholder",
        "--negative-prompt", args.negative_prompt,
        "--num-frames", "129",
        "--fps", str(args.fps),
        "--num-inference-steps", str(args.num_inference_steps),
        "--cfg-guidance-scale", str(args.guidance_scale),
        *(["--cfg-guidance-scale-motion", str(args.motion_guidance_scale)]
          if args.motion_guidance_scale is not None else []),
        *(["--cfg-guidance-scale-audio", str(args.audio_guidance_scale)]
          if args.audio_guidance_scale is not None else []),
        "--seed", "0",
        "--output-path", str(args.output_dir / "placeholder.mp4"),
        "--output-name", "placeholder",
        "--no-render",
        "--device", args.device,
    ]
    if getattr(args, "ctc_guidance_scale", 0.0) > 0:
        synthetic_argv += [
            "--ctc-guidance-scale", str(args.ctc_guidance_scale),
            "--ctc-guidance-start-sigma", str(args.ctc_guidance_start_sigma),
            "--ctc-guidance-every", str(args.ctc_guidance_every),
            "--ctc-guidance-tap-block", str(args.ctc_guidance_tap_block),
        ]
    if args.use_correct_motion_time_coordinates:
        synthetic_argv.append("--use-correct-motion-time-coordinates")
    original_argv = sys.argv
    try:
        sys.argv = synthetic_argv
        engine_args = engine.parse_args()
    finally:
        sys.argv = original_argv
    return engine_args


def resolve_device(device: str) -> str:
    if not device.startswith("cuda:"):
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(int(device.split(":", 1)[1]))
    return device


def checkpoint_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def job_signature(job: Job, args: argparse.Namespace, checkpoint: dict[str, Any]) -> str:
    payload = {
        "sample_id": job.sample_id,
        "split": args.split,
        "seed": job.seed,
        "prompt_sha256": hashlib.sha256(job.prompt.encode("utf-8")).hexdigest(),
        "checkpoint": checkpoint,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "motion_guidance_scale": args.motion_guidance_scale,
        "audio_guidance_scale": args.audio_guidance_scale,
        "duration_mode": args.duration_mode,
        "negative_prompt": args.negative_prompt,
        "latent_fps": args.fps,
        "use_correct_motion_time_coordinates": args.use_correct_motion_time_coordinates,
        "architecture": "duoflow_joint_v1",
    }
    # Preserve signatures of existing false/500 results. Other phoneme formats
    # must not silently resume outputs generated with different conditioning.
    if args.phoneme_add_blank or args.phoneme_pad_length != 500:
        payload["phoneme_add_blank"] = args.phoneme_add_blank
        payload["phoneme_pad_length"] = args.phoneme_pad_length
    payload["model_architecture"] = MODEL_ARCHITECTURE
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_paths(output_root: Path, job: Job) -> dict[str, Path]:
    job_dir = output_root / job.sample_id / f"seed_{job.seed:04d}"
    stem = f"{job.sample_id}_seed{job.seed}"
    return {
        "dir": job_dir,
        "wav": job_dir / f"{stem}.wav",
        "smplx": job_dir / f"{stem}.npz",
        "mofea333": job_dir / f"{stem}.mofea333.npy",
        "motion_latent": job_dir / f"{stem}.motion_latent.npy",
        "mp4": job_dir / f"{stem}.mp4",
        "prompt": job_dir / "prompt.txt",
        "log": job_dir / "inference.log",
        "metadata": job_dir / "metadata.json",
    }


def expected_outputs(paths: dict[str, Path], render: bool) -> list[Path]:
    required = [
        paths["wav"], paths["smplx"], paths["mofea333"],
        paths["motion_latent"], paths["prompt"], paths["log"],
    ]
    if render:
        required.append(paths["mp4"])
    return required


def completed_job_matches(paths: dict[str, Path], signature: str, render: bool) -> bool:
    if not paths["metadata"].is_file():
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("status") == "complete"
        and metadata.get("job_signature") == signature
        and all(path.is_file() for path in expected_outputs(paths, render))
    )


def git_state() -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(
            command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def sync_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def flush_to_disk(path: Path) -> None:
    """Force a just-written artifact out of the page cache (best effort)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def partial_artifacts(output_root: Path, job: Job) -> dict[str, str]:
    """Artifacts of `job` that already exist on disk (empty for a fresh job)."""
    paths = output_paths(output_root, job)
    return {
        key: str(path)
        for key, path in paths.items()
        if key not in ("dir", "metadata") and path.is_file()
    }


def logged_call(log_path: Path, function, *args, **kwargs):
    """Append verbose engine diagnostics to a per-job log."""
    with log_path.open("a", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            return function(*args, **kwargs)


def run_job(
    job: Job,
    args: argparse.Namespace,
    base_engine_args: argparse.Namespace,
    motion_vae: torch.nn.Module,
    models: dict[str, Any],
    device: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    paths = output_paths(args.output_dir, job)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    signature = job_signature(job, args, checkpoint)
    if args.resume and completed_job_matches(paths, signature, args.render):
        return {
            "status": "skipped",
            "sample_id": job.sample_id,
            "base_id": job.base_id,
            "variant": job.variant,
            "seed": job.seed,
            "job_signature": signature,
            "metadata_path": str(paths["metadata"]),
        }

    if paths["metadata"].exists() and not args.overwrite:
        try:
            existing = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if existing.get("job_signature") not in (None, signature):
            raise FileExistsError(
                f"Existing job has another signature: {paths['metadata']} "
                "(use a new --output-dir or pass --overwrite)"
            )

    job_args = copy.copy(base_engine_args)
    job_args.prompt = job.prompt
    job_args.negative_prompt = args.negative_prompt
    job_args.motion_source = "full"  # joint caption: both branches consume the full caption
    job_args.reference_motion = str(job.reference_motion)
    job_args.num_frames = read_num_frames(job.reference_motion)
    duration_info = {"mode": args.duration_mode}
    if args.duration_mode == "phoneme":
        n_ph = read_phoneme_count(args.val_root, job.sample_id)
        frames, tokens = derive_num_frames_from_phonemes(
            load_duration_coeff(args.duration_regression), n_ph
        )
        job_args.num_frames = frames
        duration_info.update(
            phoneme_count=n_ph, audio_tokens=tokens,
            reference_frames=read_num_frames(job.reference_motion),
        )
    job_args.seed = job.seed
    job_args.no_render = not args.render
    job_args.keep_temp = args.keep_temp
    job_args.output_name = f"{job.sample_id}_seed{job.seed}"
    job_args.output_path = str(paths["mp4"])

    temp_dir = Path(tempfile.mkdtemp(prefix=f"duoflow_{job.sample_id}_{job.seed}_"))
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    else:
        baseline_allocated = 0

    try:
        sync_cuda(device)
        phase_start = time.perf_counter()
        motion_latents, audio_latents, target_duration = logged_call(
            paths["log"],
            engine.generate_from_text,
            prompt=job.prompt,
            negative_prompt=args.negative_prompt,
            models=models,
            args=job_args,
            device=device,
        )
        sync_cuda(device)
        sampling_seconds = time.perf_counter() - phase_start

        # Persist every artifact the moment it exists, so one already-generated
        # result is on disk in OUTPUT_DIR even if a later stage of this job fails.
        motion_latent_np = motion_latents.detach().float().cpu().numpy()
        np.save(paths["motion_latent"], motion_latent_np)
        flush_to_disk(paths["motion_latent"])
        print(f"    [saved] {paths['motion_latent'].name}")

        phase_start = time.perf_counter()
        motion_data = logged_call(
            paths["log"], engine.decode_motion_latents, motion_latents, motion_vae, device
        )
        sync_cuda(device)
        motion_decode_seconds = time.perf_counter() - phase_start

        motion_data_np = np.asarray(motion_data, dtype=np.float32)
        np.save(paths["mofea333"], motion_data_np)
        flush_to_disk(paths["mofea333"])
        print(f"    [saved] {paths['mofea333'].name}")
        logged_call(
            paths["log"],
            engine.save_motion_as_smplx_npz,
            str(paths["smplx"]), motion_data_np, fps=float(engine.MOTION_FPS)
        )
        flush_to_disk(paths["smplx"])
        print(f"    [saved] {paths['smplx'].name}")

        phase_start = time.perf_counter()
        audio_waveform = logged_call(
            paths["log"],
            engine.decode_audio_latents,
            audio_latents,
            models["audio_vae_decoder"],
            models["vocoder"],
            models["audio_sample_rate"],
            models["audio_hop_length"],
            models["audio_mel_bins"],
            device,
            target_duration=target_duration,
        )
        sync_cuda(device)
        audio_decode_seconds = time.perf_counter() - phase_start

        if audio_waveform.ndim == 2:
            audio_waveform = audio_waveform.T
        audio_normalized = np.clip(audio_waveform, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        from scipy.io import wavfile

        wavfile.write(paths["wav"], models["audio_sample_rate"], audio_int16)
        flush_to_disk(paths["wav"])
        print(f"    [saved] {paths['wav'].name}")
        paths["prompt"].write_text(job.prompt + "\n", encoding="utf-8")
        flush_to_disk(paths["prompt"])

        render_seconds = 0.0
        if args.render:
            phase_start = time.perf_counter()
            rendered = logged_call(
                paths["log"],
                engine.render_motion_to_video,
                str(paths["smplx"]),
                str(paths["wav"]),
                str(temp_dir),
                float(engine.MOTION_FPS),
                models["audio_sample_rate"],
            )
            if not rendered or not Path(rendered).is_file():
                raise RuntimeError("Motion renderer did not produce a video")
            logged_call(
                paths["log"],
                engine.merge_video_audio,
                rendered,
                str(paths["wav"]),
                str(paths["mp4"]),
            )
            render_seconds = time.perf_counter() - phase_start

        total_seconds = time.perf_counter() - started
        audio_duration = float(audio_int16.shape[0]) / float(models["audio_sample_rate"])
        peak_allocated = torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.startswith("cuda") else 0
        metadata = {
            "status": "complete",
            "split": args.split,
            "job_signature": signature,
            "sample_id": job.sample_id,
            "base_id": job.base_id,
            "variant": job.variant,
            "seed": job.seed,
            "prompt": job.prompt,
            "prompt_sha256": hashlib.sha256(job.prompt.encode("utf-8")).hexdigest(),
            "reference_motion": str(job.reference_motion),
            "gt_audio": str(job.gt_audio),
            "checkpoint": checkpoint,
            "num_motion_latent_frames": job_args.num_frames,
            "duration": duration_info,
            "latent_fps": args.fps,
            "use_correct_motion_time_coordinates": args.use_correct_motion_time_coordinates,
            "decoded_motion_fps": float(engine.MOTION_FPS),
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
        "motion_guidance_scale": args.motion_guidance_scale,
        "audio_guidance_scale": args.audio_guidance_scale,
            "negative_prompt": args.negative_prompt,
            "phoneme_add_blank": args.phoneme_add_blank,
            "phoneme_pad_length": args.phoneme_pad_length,
            "audio_sample_rate": int(models["audio_sample_rate"]),
            "audio_duration_seconds": audio_duration,
            "motion_latent_shape": list(motion_latent_np.shape),
            "mofea333_shape": list(motion_data_np.shape),
            "audio_shape": list(audio_int16.shape),
            "timing_seconds": {
                "sampling": sampling_seconds,
                "motion_decode": motion_decode_seconds,
                "audio_decode": audio_decode_seconds,
                "render": render_seconds,
                "total": total_seconds,
            },
            "rtf": {
                "sampling": sampling_seconds / audio_duration if audio_duration else None,
                "end_to_end_no_model_load": total_seconds / audio_duration if audio_duration else None,
            },
            "gpu_memory_bytes": {
                "baseline_allocated": baseline_allocated,
                "peak_allocated": peak_allocated,
                "peak_reserved": peak_reserved,
                "incremental_peak_allocated": max(0, peak_allocated - baseline_allocated),
            },
            "outputs": {
                key: str(value)
                for key, value in paths.items()
                if key not in ("dir", "metadata") and value.exists()
            },
            "completed_at_unix": time.time(),
        }
        atomic_json(paths["metadata"], metadata)
        return metadata
    finally:
        if not args.keep_temp:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def print_inventory(inventory: dict[str, Any], jobs: Iterable[Job]) -> None:
    print(json.dumps({key: value for key, value in inventory.items() if key != "rejected"}, indent=2))
    jobs = list(jobs)
    print("First selected jobs:")
    for job in jobs[:10]:
        print(f"  {job.sample_id} seed={job.seed} ref={job.reference_motion}")
    if inventory.get("rejected"):
        print("First rejected items:")
        for item in inventory["rejected"][:10]:
            print(" ", json.dumps(item, ensure_ascii=False))


def main() -> int:
    global MODEL_ARCHITECTURE
    args = parse_args()
    MODEL_ARCHITECTURE = read_model_architecture(args.trained_model)
    print(
        f"phoneme settings: add_blank={args.phoneme_add_blank}, "
        f"pad_length={args.phoneme_pad_length}"
    )

    registers = MODEL_ARCHITECTURE["connector_num_learnable_registers"]
    os.environ["MOTIONSPEECH_NUM_LEARNABLE_REGISTERS"] = (
        "none" if registers is None else str(registers)
    )
    print(f"model architecture: {json.dumps(MODEL_ARCHITECTURE, sort_keys=True)}")

    if getattr(args, "ctc_guidance_scale", 0.0) > 0:
        raise NotImplementedError("CTC-guided sampling is not included in this release")

    args.val_root = args.val_root.resolve()
    args.l1_json = args.l1_json.resolve()
    args.l2_json = args.l2_json.resolve()
    args.trained_model = args.trained_model.resolve()
    args.output_dir = args.output_dir.resolve()

    duration_coeff = None
    if args.duration_mode == "phoneme":
        if args.duration_regression is None or not args.duration_regression.is_file():
            raise FileNotFoundError(
                "--duration-mode phoneme requires --duration-regression JSON "
                f"(got {args.duration_regression})"
            )
        coeff = load_duration_coeff(args.duration_regression)
        print(f"duration: phoneme regression a={coeff['a_tokens_per_phoneme']:.4f} "
              f"b={coeff['b_tokens']:.2f} r2={coeff['r2']:.3f}")
    for path in (args.l1_json, args.l2_json, args.trained_model):
        if not path.is_file():
            raise FileNotFoundError(path)

    jobs, inventory = build_jobs(args)
    print_inventory(inventory, jobs)
    if not jobs:
        raise RuntimeError("No validation jobs selected")
    if args.dry_run:
        print("Dry run complete; models were not loaded and no generations were written.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_identity(args.trained_model)
    json_line(
        args.output_dir / "runs.jsonl",
        {
            "event": "run_start",
            "argv": sys.argv,
            "checkpoint": checkpoint,
            "inventory": {key: value for key, value in inventory.items() if key != "rejected"},
            "git": git_state(),
            "time_unix": time.time(),
        },
    )

    if args.resume and all(
        completed_job_matches(
            output_paths(args.output_dir, job),
            job_signature(job, args, checkpoint),
            args.render,
        )
        for job in jobs
    ):
        summary = {
            "status": "complete",
            "counts": {"complete": 0, "skipped": len(jobs), "failed": 0},
            "selected_jobs": len(jobs),
            "model_load_seconds": 0.0,
            "batch_seconds": 0.0,
            "checkpoint": checkpoint,
            "output_dir": str(args.output_dir),
            "model_architecture": MODEL_ARCHITECTURE,
            "completed_at_unix": time.time(),
        }
        atomic_json(args.output_dir / "summary.json", summary)
        json_line(args.output_dir / "runs.jsonl", {"event": "run_end", **summary})
        print("All selected jobs are already complete; model loading was skipped.")
        print(json.dumps(summary, indent=2))
        return 0

    engine_args = make_engine_args(args)
    device = resolve_device(args.device)
    print(f"Loading models once on {device} ...")
    load_started = time.perf_counter()
    torch.set_grad_enabled(False)
    motion_vae = engine.load_motion_vae(engine_args.motion_vae_path, device)
    models = engine.load_motionspeech_models(engine_args, device)
    load_seconds = time.perf_counter() - load_started
    print(f"Model load completed in {load_seconds:.2f}s")

    counts = {"complete": 0, "skipped": 0, "failed": 0}
    manifest_path = args.output_dir / "manifest.jsonl"
    batch_started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] {job.sample_id} seed={job.seed}")
        attempt_started = time.time()
        try:
            result = run_job(
                job, args, engine_args, motion_vae, models, device, checkpoint
            )
            status = str(result["status"])
            counts[status] += 1
            json_line(
                manifest_path,
                {"event": "job_result", "attempt_started_unix": attempt_started, **result},
            )
            timing = result.get("timing_seconds", {})
            total = timing.get("total")
            total_text = f" total={total:.2f}s" if isinstance(total, (int, float)) else ""
            print(f"[{status}] {job.sample_id} seed={job.seed}{total_text}")
        except Exception as exc:
            counts["failed"] += 1
            failure = {
                "event": "job_result",
                "status": "failed",
                "attempt_started_unix": attempt_started,
                "failed_at_unix": time.time(),
                "sample_id": job.sample_id,
                "base_id": job.base_id,
                "variant": job.variant,
                "seed": job.seed,
                "reference_motion": str(job.reference_motion),
                "gt_audio": str(job.gt_audio),
                "partial_artifacts": partial_artifacts(args.output_dir, job),
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            json_line(manifest_path, failure)
            json_line(args.output_dir / "failures.jsonl", failure)
            print(f"[failed] {job.sample_id} seed={job.seed}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if args.fail_fast:
                break

    summary = {
        "status": "complete" if counts["failed"] == 0 else "completed_with_failures",
        "counts": counts,
        "selected_jobs": len(jobs),
        "model_load_seconds": load_seconds,
        "batch_seconds": time.perf_counter() - batch_started,
        "checkpoint": checkpoint,
        "output_dir": str(args.output_dir),
        "model_architecture": MODEL_ARCHITECTURE,
        "completed_at_unix": time.time(),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    json_line(args.output_dir / "runs.jsonl", {"event": "run_end", **summary})
    print(json.dumps(summary, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
