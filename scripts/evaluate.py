#!/usr/bin/env python3
"""Evaluate DuoFlow held-out generations with reproducible metrics.

The motion protocol follows the public EMAGE/PantoMatrix evaluator:

* FGD consumes all 55 SMPL-X joints in Rot6D form (330 dimensions).
* Beat Consistency (BC) and L1 diversity consume SMPL-X joint positions.
* The first/last two seconds are excluded from BC, matching the reference code.
* BC Gap compares mean BC for generated/reference motions, both with reference audio.
* Foot skating uses world-space joints including root rotation and translation.

WER and speaker similarity are optional because they require pretrained models.
No model is downloaded unless ``--allow-model-download`` is explicitly passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import subprocess
import time
import traceback
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import torch
from scipy.io import wavfile

from data_splits import validate_preprocessed_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs/inference/test"
DEFAULT_GT_MOTION_DIR = REPO_ROOT / "outputs/preprocessed/motionspeech/test/gt_motion"
DEFAULT_EMAGE_ROOT = Path(os.environ.get("MOTIONSPEECH_EMAGE_ROOT", REPO_ROOT / "external/PantoMatrix"))
EVALUATOR_VERSION = "duoflow-eval-v4-pairacc-bc-gap-world-feet"


@dataclass(frozen=True)
class EvalJob:
    metadata_path: Path
    sample_id: str
    base_id: str
    variant: str
    seed: int
    prompt: str
    job_signature: str
    pred_motion: Path
    pred_audio: Path
    gt_motion: Path
    gt_audio: Path
    metadata: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.sample_id}|seed={self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate batch DuoFlow outputs produced by infer_batch.py."
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gt-motion-dir", type=Path, default=None)
    parser.add_argument("--emage-root", type=Path, default=DEFAULT_EMAGE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=("l1", "l2"), default=("l1", "l2"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--edge-seconds", type=float, default=2.0)
    parser.add_argument("--fk-chunk-size", type=int, default=512)
    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("MOTIONSPEECH_ASR_MODEL"),
        help="Optional local Hugging Face Whisper/CTC model path (or model ID with --allow-model-download).",
    )
    parser.add_argument("--asr-language", default="en")
    parser.add_argument("--asr-num-beams", type=int, default=5,
                        help="Whisper beam-search width; >1 avoids greedy repetition "
                             "loops on rhythmic/repeated audio (validated num_beams=5)")
    parser.add_argument("--asr-repetition-penalty", type=float, default=1.05,
                        help="Whisper repetition penalty (validated default: 1.05)")
    parser.add_argument(
        "--speaker-model",
        default=os.environ.get("MOTIONSPEECH_SPEAKER_MODEL"),
        help="Optional local Hugging Face WavLM/Wav2Vec2-style model for cosine similarity.",
    )
    parser.add_argument("--require-speech-metrics", action="store_true", help="Fail unless WER and SpkSim backends both load.")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face to download optional ASR/speaker models.",
    )
    parser.add_argument("--pairacc-checkpoint", type=Path, default=os.environ.get("DUOFLOW_PAIRACC_CHECKPOINT"))
    parser.add_argument("--pairacc-model-root", type=Path, default=os.environ.get("DUOFLOW_PAIRACC_MODEL_ROOT"))
    parser.add_argument("--pairacc-model-gender", default="neutral")
    parser.add_argument("--pairacc-conversion-batch-size", type=int, default=2)
    parser.add_argument("--require-pairacc", action="store_true")
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = DEFAULT_INPUT_DIR.parent / args.split
    if args.gt_motion_dir is None:
        env_name = "MOTIONSPEECH_TEST_ROOT" if args.split == "test" else "MOTIONSPEECH_VAL_ROOT"
        root = Path(os.environ.get(env_name, DEFAULT_GT_MOTION_DIR.parents[1] / args.split))
        args.gt_motion_dir = root / "gt_motion"
    validate_preprocessed_split(args.gt_motion_dir.parent, args.split)
    return args


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_latest_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"Warning: ignoring malformed {path}:{line_number}: {exc}")
                continue
            if isinstance(row, dict) and row.get("key"):
                latest[str(row["key"])] = row
    return latest


def resolve_output(metadata_path: Path, metadata: dict[str, Any], key: str, suffix: str) -> Path:
    recorded = metadata.get("outputs", {}).get(key)
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    candidates = sorted(metadata_path.parent.glob(f"*{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot resolve {key!r} for {metadata_path}; found {len(candidates)} *{suffix} files"
    )


def discover_jobs(args: argparse.Namespace) -> tuple[list[EvalJob], list[dict[str, Any]]]:
    jobs: list[EvalJob] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metadata_path in sorted(args.input_dir.rglob("metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete":
                raise ValueError(f"status={metadata.get('status')!r}, expected 'complete'")
            if metadata.get("split") is not None and metadata["split"] != args.split:
                raise ValueError(f"Output split {metadata['split']!r} does not match requested split {args.split!r}")
            sample_id = str(metadata["sample_id"])
            base_id = str(metadata["base_id"])
            variant = str(metadata["variant"]).lower()
            seed = int(metadata["seed"])
            key = f"{sample_id}|seed={seed}"
            if key in seen:
                raise ValueError(f"duplicate completed job {key}")
            seen.add(key)
            if variant not in args.variants or (args.seeds is not None and seed not in args.seeds):
                continue
            pred_motion = resolve_output(metadata_path, metadata, "mofea333", ".mofea333.npy")
            pred_audio = resolve_output(metadata_path, metadata, "wav", ".wav")
            gt_motion = args.gt_motion_dir / f"{base_id}.npy"
            gt_audio = Path(str(metadata["gt_audio"]))
            missing = [str(path) for path in (pred_motion, pred_audio, gt_motion, gt_audio) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing assets: {missing}")
            jobs.append(
                EvalJob(
                    metadata_path=metadata_path,
                    sample_id=sample_id,
                    base_id=base_id,
                    variant=variant,
                    seed=seed,
                    prompt=str(metadata.get("prompt", "")),
                    job_signature=str(metadata.get("job_signature", "")),
                    pred_motion=pred_motion,
                    pred_audio=pred_audio,
                    gt_motion=gt_motion,
                    gt_audio=gt_audio,
                    metadata=metadata,
                )
            )
        except Exception as exc:
            rejected.append(
                {
                    "metadata_path": str(metadata_path),
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
    jobs.sort(key=lambda item: (item.sample_id, item.seed))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        jobs = jobs[: args.limit]
    return jobs, rejected


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def evaluation_signature(job: EvalJob, args: argparse.Namespace, emage_weights: Path) -> str:
    value = {
        "version": EVALUATOR_VERSION,
        "split": args.split,
        "job_signature": job.job_signature,
        "pred_motion": file_identity(job.pred_motion),
        "pred_audio": file_identity(job.pred_audio),
        "gt_motion": file_identity(job.gt_motion),
        "gt_audio": file_identity(job.gt_audio),
        "emage_weights": file_identity(emage_weights),
        "edge_seconds": args.edge_seconds,
        "asr_model": args.asr_model,
        "asr_language": args.asr_language,
        "asr_num_beams": args.asr_num_beams,
        "asr_repetition_penalty": args.asr_repetition_penalty,
        "speaker_model": args.speaker_model,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_mofea333(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 333:
        raise ValueError(f"Expected [T,333+] mofea333 at {path}, got {value.shape}")
    if value.shape[0] < 3 or not np.isfinite(value[:, :333]).all():
        raise ValueError(f"Invalid/non-finite motion at {path}: {value.shape}")
    return value[:, :333]


def load_audio(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    sample_rate, waveform = wavfile.read(path)
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform.astype(np.float32).mean(axis=1)
    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        scale = float(max(abs(info.min), info.max))
        waveform = waveform.astype(np.float32) / scale
    else:
        waveform = waveform.astype(np.float32)
    if target_sr is not None and sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    return np.clip(waveform, -1.0, 1.0), int(sample_rate)


def audio_signal_metrics(path: Path) -> dict[str, float]:
    waveform, sample_rate = load_audio(path)
    eps = 1e-12
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    frame_length = max(1, int(round(sample_rate * 0.020)))
    complete = (waveform.size // frame_length) * frame_length
    if complete:
        frames = waveform[:complete].reshape(-1, frame_length)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + eps)
        silence_fraction = float(np.mean(20.0 * np.log10(frame_rms + eps) < -50.0))
    else:
        silence_fraction = float(rms < 10 ** (-50.0 / 20.0))
    return {
        "duration_seconds": float(waveform.size / sample_rate),
        "rms_dbfs": float(20.0 * math.log10(max(rms, eps))),
        "peak": peak,
        "clipping_fraction": float(np.mean(np.abs(waveform) >= 0.999)),
        "silence_fraction_below_minus50dbfs": silence_fraction,
    }


def normalize_transcript(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def reference_speech(prompt: str) -> str:
    match = re.search(r"<S>(.*?)</E>", prompt, flags=re.IGNORECASE | re.DOTALL)
    return normalize_transcript(match.group(1) if match else "")


def word_errors(reference: str, hypothesis: str) -> dict[str, int | float]:
    ref = reference.split()
    hyp = hypothesis.split()
    # Dynamic programming cells hold (total, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp, start=1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
                continue
            substitution = previous[j - 1]
            deletion = previous[j]
            insertion = current[j - 1]
            choices = [
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            current.append(min(choices, key=lambda item: (item[0], item[1], item[2], item[3])))
        previous = current
    total, substitutions, deletions, insertions = previous[-1]
    return {
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        "errors": total,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": float(total / len(ref)) if ref else (0.0 if not hyp else 1.0),
    }


class ASRBackend:
    def __init__(self, model_name: str, device: str, allow_download: bool, language: str,
                 num_beams: int = 1, repetition_penalty: float = 1.0):
        from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor

        local_only = not allow_download
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_only)
        self.device = torch.device(device)
        self.language = language
        # num_beams > 1 enables beam-search decoding, which breaks the greedy
        # repetition loop whisper falls into on rhythmic/repeated content
        # (e.g. "Dobby, Dobby, Dobby" transcribed as "Dummy" x150 until the
        # token cap); validated on whisper-large-v3-turbo with num_beams=5.
        self.num_beams = max(int(num_beams), 1)
        self.repetition_penalty = float(repetition_penalty)
        if not np.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be a finite positive number")
        self.kind = "seq2seq"
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, local_files_only=local_only
            )
        except (ValueError, OSError):
            self.model = AutoModelForCTC.from_pretrained(model_name, local_files_only=local_only)
            self.kind = "ctc"
        self.model.to(self.device).eval()
        self.sample_rate = int(getattr(self.processor.feature_extractor, "sampling_rate", 16000))

    def transcribe(self, path: Path) -> str:
        waveform, _ = load_audio(path, self.sample_rate)
        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            key: value.to(
                self.device,
                dtype=model_dtype if torch.is_floating_point(value) else value.dtype,
            )
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            if self.kind == "seq2seq":
                try:
                    token_ids = self.model.generate(
                        **inputs, language=self.language, task="transcribe",
                        num_beams=self.num_beams,
                        repetition_penalty=self.repetition_penalty,
                    )
                except (TypeError, ValueError):
                    token_ids = self.model.generate(**inputs)
            else:
                token_ids = self.model(**inputs).logits.argmax(dim=-1)
        return normalize_transcript(self.processor.batch_decode(token_ids, skip_special_tokens=True)[0])


class SpeakerBackend:
    def __init__(self, model_name: str, device: str, allow_download: bool):
        from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioXVector

        local_only = not allow_download
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=local_only)
        try:
            # Speaker-verification checkpoints such as WavLM-base-plus-SV have
            # a trained x-vector projection head that should not be discarded.
            self.model = AutoModelForAudioXVector.from_pretrained(
                model_name, local_files_only=local_only
            )
        except (ValueError, OSError):
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.sample_rate = int(getattr(self.extractor, "sampling_rate", 16000))

    def embed(self, path: Path) -> np.ndarray:
        waveform, _ = load_audio(path, self.sample_rate)
        inputs = self.extractor(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
        if getattr(output, "embeddings", None) is not None:
            vector = output.embeddings[0]
        else:
            vector = output.last_hidden_state[0].mean(dim=0)
        vector = torch.nn.functional.normalize(vector, dim=0)
        return vector.float().cpu().numpy()

    def cosine(self, first: Path, second: Path) -> float:
        return float(np.dot(self.embed(first), self.embed(second)))


class MotionBackend:
    def __init__(self, emage_root: Path, device: str, fk_chunk_size: int):
        if str(emage_root) not in sys.path:
            sys.path.insert(0, str(emage_root))
        # PantoMatrix imports wget unconditionally even when all evaluator assets
        # already exist.  Keep the evaluation environment dependency-free and
        # fail clearly only if its download path is unexpectedly reached.
        if importlib.util.find_spec("wget") is None:
            wget_stub = types.ModuleType("wget")

            def unavailable_download(*_args, **_kwargs):
                raise RuntimeError(
                    "PantoMatrix requested a download, but wget is not installed. "
                    "Provide the EMAGE evaluator assets locally."
                )

            wget_stub.download = unavailable_download
            sys.modules["wget"] = wget_stub
        from emage_evaltools.mertic import BC, FGD
        from emage_utils import rotation_conversions as rc
        import smplx

        self.device = torch.device(device)
        self.rc = rc
        self.fk_chunk_size = fk_chunk_size
        tools_dir = emage_root / "emage_evaltools"
        self.fgd = FGD(download_path=str(tools_dir) + os.sep, device=device)
        self.bc = BC(download_path=str(tools_dir) + os.sep, sigma=0.3, order=7)
        self.smplx = smplx.create(
            str(tools_dir / "smplx_models"),
            model_type="smplx",
            gender="NEUTRAL_2020",
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext="npz",
            use_pca=False,
        ).to(self.device).eval()

    def fgd_feature(self, motion: np.ndarray) -> np.ndarray:
        rotation = torch.from_numpy(motion[:, 3:333]).unsqueeze(0).float()
        return np.asarray(self.fgd.get_feature(rotation), dtype=np.float32)

    def positions(self, motion: np.ndarray, *, world_space: bool = False) -> np.ndarray:
        """Keep canonical joints for BC; restore full root transforms for skating."""
        rotations = torch.from_numpy(motion[:, 3:333]).float().reshape(-1, 55, 6)
        positions: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, rotations.shape[0], self.fk_chunk_size):
                rot6d = rotations[start : start + self.fk_chunk_size].to(self.device)
                axis_angle = self.rc.matrix_to_axis_angle(
                    self.rc.rotation_6d_to_matrix(rot6d)
                ).reshape(-1, 165)
                count = axis_angle.shape[0]
                zeros3 = torch.zeros(count, 3, device=self.device)
                output = self.smplx(
                    betas=torch.zeros(count, 300, device=self.device),
                    transl=torch.from_numpy(motion[start : start + count, :3]).to(self.device) if world_space else zeros3,
                    expression=torch.zeros(count, 100, device=self.device),
                    jaw_pose=zeros3,
                    global_orient=axis_angle[:, :3] if world_space else zeros3,
                    body_pose=axis_angle[:, 3 : 21 * 3 + 3],
                    left_hand_pose=axis_angle[:, 25 * 3 : 40 * 3],
                    right_hand_pose=axis_angle[:, 40 * 3 : 55 * 3],
                    return_joints=True,
                    leye_pose=zeros3,
                    reye_pose=zeros3,
                )
                positions.append(output.joints[:, :55].float().cpu().numpy())
        return np.concatenate(positions, axis=0)

    def audio_onsets(self, path: Path, start_sample: int, end_sample: int) -> np.ndarray:
        return self.bc.load_audio(str(path), t_start=start_sample, t_end=end_sample)

    def beat_consistency(
        self,
        onsets: np.ndarray,
        positions: np.ndarray,
        crop_start: int,
        crop_end: int,
        fps: float,
    ) -> float | None:
        if len(onsets) == 0 or crop_end - crop_start < 3:
            return None
        motion_beats = self.bc.load_motion(
            positions.reshape(positions.shape[0], -1),
            t_start=crop_start,
            t_end=crop_end,
            pose_fps=fps,
            without_file=True,
        )
        self.bc.reset()
        self.bc.compute(onsets, motion_beats, length=1, pose_fps=int(round(fps)))
        return float(self.bc.avg())


def l1_diversity(positions: np.ndarray) -> float:
    center = np.mean(positions, axis=0, keepdims=True)
    return float(np.sum(np.abs(positions - center), dtype=np.float64) / positions.shape[0])


def derivative_norms(positions: np.ndarray, fps: float) -> dict[str, float]:
    values: dict[str, float] = {}
    current = positions.astype(np.float64)
    for name in ("velocity", "acceleration", "jerk"):
        current = np.diff(current, axis=0) * fps
        norms = np.linalg.norm(current, axis=-1)
        values[f"{name}_mean"] = float(np.mean(norms))
        values[f"{name}_p95"] = float(np.percentile(norms, 95))
    return values


def foot_skating(world_positions: np.ndarray, fps: float) -> dict[str, float | None]:
    """Compute contact-frame skating from fully transformed SMPL-X world joints."""
    foot_indices = [7, 8, 10, 11]
    world = world_positions[:, foot_indices]
    if world.shape[0] < 2:
        return {"contact_fraction": None, "skating_ratio": None, "contact_horizontal_speed": None}
    ground = float(np.percentile(world[..., 1], 1))
    contact = world[:-1, :, 1] <= ground + 0.05
    horizontal_delta = np.diff(world[..., [0, 2]], axis=0) * fps
    horizontal_speed = np.linalg.norm(horizontal_delta, axis=-1)
    if not np.any(contact):
        return {"contact_fraction": 0.0, "skating_ratio": None, "contact_horizontal_speed": None}
    return {
        "contact_fraction": float(np.mean(contact)),
        "skating_ratio": float(np.mean(horizontal_speed[contact] > 0.10)),
        "contact_horizontal_speed": float(np.mean(horizontal_speed[contact])),
    }


def weighted_mean(rows: Iterable[dict[str, Any]], field: str, weight_field: str | None = None) -> float | None:
    total = 0.0
    weight_total = 0.0
    for row in rows:
        value = row.get("metrics", {}).get(field)
        if value is None or not math.isfinite(float(value)):
            continue
        weight = float(row.get("metrics", {}).get(weight_field, 1.0)) if weight_field else 1.0
        total += float(value) * weight
        weight_total += weight
    return total / weight_total if weight_total else None


def summarize_group(
    rows: list[dict[str, Any]], pred_features: list[np.ndarray], gt_features: list[np.ndarray], fgd_class
) -> dict[str, Any]:
    summary: dict[str, Any] = {"jobs": len(rows)}
    if pred_features and gt_features:
        pred = np.concatenate([value.reshape(-1, value.shape[-1]) for value in pred_features], axis=0)
        target = np.concatenate([value.reshape(-1, value.shape[-1]) for value in gt_features], axis=0)
        if pred.shape[0] >= 2 and target.shape[0] >= 2:
            summary["fgd"] = float(fgd_class.frechet_distance(pred, target))
            summary["fgd_feature_vectors_pred"] = int(pred.shape[0])
            summary["fgd_feature_vectors_gt"] = int(target.shape[0])
        else:
            summary["fgd"] = None
            summary["fgd_unavailable_reason"] = "fewer than two feature vectors"

    fields = [
        "bc_generated_audio_pred_motion",
        "bc_gt_audio_pred_motion",
        "bc_gt_audio_gt_motion",
        "diversity_l1_pred",
        "diversity_l1_gt",
        "velocity_mean_pred",
        "velocity_mean_gt",
        "acceleration_mean_pred",
        "acceleration_mean_gt",
        "jerk_mean_pred",
        "jerk_mean_gt",
        "foot_skating_ratio_pred",
        "foot_skating_ratio_gt",
        "audio_rms_dbfs_pred",
        "audio_clipping_fraction_pred",
        "audio_silence_fraction_pred",
        "speaker_cosine",
        "rtf_sampling",
        "rtf_end_to_end_no_model_load",
        "peak_allocated_gib",
        "incremental_peak_allocated_gib",
    ]
    for field in fields:
        summary[field] = weighted_mean(rows, field)

    pred_div = summary.get("diversity_l1_pred")
    gt_div = summary.get("diversity_l1_gt")
    summary["diversity_l1_gap_abs"] = (
        abs(pred_div - gt_div) if pred_div is not None and gt_div is not None else None
    )
    summary["bc_gap"] = (
        abs(summary["bc_gt_audio_pred_motion"] - summary["bc_gt_audio_gt_motion"])
        if summary.get("bc_gt_audio_pred_motion") is not None
        and summary.get("bc_gt_audio_gt_motion") is not None
        else None
    )
    summary["bc_gt_audio_pred_motion_gap_from_human_abs"] = summary["bc_gap"]

    wer_rows = [row for row in rows if row.get("metrics", {}).get("asr_errors") is not None]
    summary["wer_samples_unavailable"] = len(rows) - len(wer_rows)
    if wer_rows:
        # Primary WER includes every transcribed row, including severe outliers.
        errors = sum(int(row["metrics"]["asr_errors"]) for row in wer_rows)
        words = sum(int(row["metrics"]["asr_reference_words"]) for row in wer_rows)
        summary["wer_full"] = float(errors / words) if words else None
        summary["asr_errors"] = errors
        summary["asr_reference_words"] = words
        for field in ("substitutions", "deletions", "insertions"):
            values = [row["metrics"].get(f"asr_{field}") for row in wer_rows]
            summary[f"asr_{field}"] = sum(int(value) for value in values) if all(value is not None for value in values) else None
        summary["wer"] = summary["wer_full"]
        summary["wer_samples_used"] = len(wer_rows)
        summary["wer_outliers_excluded"] = 0

        # Per-sample WER derived from errors/words: cached rows written before
        # the "wer" key existed in word_errors() only carry the two counters,
        # so deriving keeps resumed evaluations working.
        def _row_wer(row: dict[str, Any]) -> float:
            metrics = row["metrics"]
            if metrics.get("wer") is not None:
                return float(metrics["wer"])
            ref = int(metrics.get("asr_reference_words", 0))
            if ref > 0:
                return int(metrics["asr_errors"]) / ref
            hyp = int(metrics.get("asr_hypothesis_words", 0) or 0)
            return 1.0 if hyp > 0 else 0.0

        # Keep the historical filtered metric explicitly labelled for comparison.
        # A high ASR WER alone does not establish an audible repetition loop.
        kept_rows = [row for row in wer_rows if _row_wer(row) <= 1.0]
        summary["wer_outlier_count"] = len(wer_rows) - len(kept_rows)
        summary["wer_outlier_rate"] = summary["wer_outlier_count"] / len(wer_rows)
        summary["wer_filtered_samples_used"] = len(kept_rows)
        summary["wer_mean_per_sample"] = sum(_row_wer(row) for row in wer_rows) / len(wer_rows)
        if kept_rows:
            kept_errors = sum(int(row["metrics"]["asr_errors"]) for row in kept_rows)
            kept_words = sum(int(row["metrics"]["asr_reference_words"]) for row in kept_rows)
            summary["wer_filtered"] = float(kept_errors / kept_words) if kept_words else None
            summary["wer_filtered_mean_per_sample"] = float(
                sum(_row_wer(row) for row in kept_rows) / len(kept_rows)
            )
        else:
            summary["wer_filtered"] = None
            summary["wer_filtered_mean_per_sample"] = None
    else:
        summary["wer"] = None
        summary["wer_samples_used"] = 0
    return summary


def build_failure_report(
    input_dir: Path, rows: list[dict[str, Any]], rejected: list[dict[str, Any]],
    evaluation_failures: list[dict[str, Any]], jobs: list[EvalJob],
) -> dict[str, Any]:
    """Keep failures visible, including generation attempts with no metadata."""
    attempts: dict[str, dict[str, Any]] = {}
    malformed: list[dict[str, Any]] = []
    for name in ("failures.jsonl", "manifest.jsonl"):
        path = input_dir / name
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                key = f"{item['sample_id']}|seed={int(item['seed'])}"
                stamp = float(item.get("completed_at_unix", item.get("failed_at_unix", item.get("attempt_started_unix", 0))))
                old = attempts.get(key)
                if old is None or stamp >= old["_audit_timestamp"]:
                    attempts[key] = {**item, "_audit_timestamp": stamp}
            except (ValueError, KeyError, TypeError) as exc:
                malformed.append({"path": str(path), "line": number, "exception": str(exc)})
    completed = {job.key for job in jobs}
    evaluated = {row["key"] for row in rows} | {row["key"] for row in evaluation_failures}
    not_attempted = [
        {"key": job.key, "sample_id": job.sample_id, "seed": job.seed, "metadata_path": str(job.metadata_path)}
        for job in jobs if job.key not in evaluated
    ]
    inference_failures = []
    for key, item in attempts.items():
        if item.get("status") == "failed" and key not in completed:
            metadata = input_dir / str(item["sample_id"]) / f"seed_{int(item['seed']):04d}" / "metadata.json"
            try:
                recovered = json.loads(metadata.read_text(encoding="utf-8")).get("status") == "complete"
            except (OSError, ValueError):
                recovered = False
            if not recovered:
                inference_failures.append(item)
    outliers = []
    for row in rows:
        metrics = row.get("metrics", {})
        words = metrics.get("asr_reference_words")
        errors = metrics.get("asr_errors")
        if errors is not None and words is not None and int(errors) > int(words):
            outliers.append({
                "key": row["key"], "sample_id": row["sample_id"], "seed": row["seed"],
                "paths": row.get("paths", {}), "metrics": metrics,
                "reason": "ASR errors exceed reference words; inspect audio to confirm repetition",
            })
    return {
        "inference_scope": "entire input directory; evaluation selection may be narrower",
        "inference_failures": inference_failures, "evaluation_failures": evaluation_failures,
        "evaluation_not_attempted": not_attempted,
        "rejected_metadata": rejected, "malformed_manifest_records": malformed,
        "wer_over_100_percent": outliers,
        "counts": {
            "inference_failed": len(inference_failures), "evaluation_failed": len(evaluation_failures),
            "evaluation_not_attempted": len(not_attempted),
            "rejected_metadata": len(rejected), "malformed_manifest_records": len(malformed),
            "wer_over_100_percent": len(outliers),
        },
    }


def flatten_for_csv(group: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {"group": group, **{key: value for key, value in summary.items() if not isinstance(value, (dict, list))}}


def write_summary_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = [flatten_for_csv(group, value) for group, value in summaries.items()]
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key != "group", key))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.gt_motion_dir = args.gt_motion_dir.resolve()
    args.emage_root = args.emage_root.resolve()
    args.output_dir = (args.output_dir or (args.input_dir / "evaluation")).resolve()
    if args.require_pairacc and not (args.pairacc_checkpoint and args.pairacc_model_root):
        raise ValueError("--require-pairacc needs --pairacc-checkpoint and --pairacc-model-root")
    if bool(args.pairacc_checkpoint) != bool(args.pairacc_model_root):
        raise ValueError("Supply both PairAcc checkpoint and SMPL-X model root")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    if not args.gt_motion_dir.is_dir():
        raise FileNotFoundError(args.gt_motion_dir)
    if args.fk_chunk_size < 1:
        raise ValueError("--fk-chunk-size must be >= 1")

    tools_dir = args.emage_root / "emage_evaltools"
    emage_weights = tools_dir / "AESKConv_240_100.bin"
    required = [
        emage_weights,
        tools_dir / "mean_vel_smplxflame_30.npy",
        tools_dir / "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
    ]
    missing_required = [str(path) for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(f"Missing EMAGE evaluator assets: {missing_required}")

    jobs, rejected = discover_jobs(args)
    if not jobs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = build_failure_report(args.input_dir, [], rejected, [], [])
        atomic_json(args.output_dir / "failure_report.json", report)
        atomic_json(args.output_dir / "metrics_summary.json", {
            "status": "no_completed_jobs", "evaluator_version": EVALUATOR_VERSION,
            "counts": {"selected": 0, "successful": 0}, "summary": {},
            "failure_report": str(args.output_dir / "failure_report.json"),
            "failure_counts": report["counts"],
        })
        print(f"No completed evaluation jobs under {args.input_dir}; see failure_report.json")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.output_dir / "per_sample_metrics.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    latest = read_latest_jsonl(per_sample_path) if args.resume else {}

    print(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "selected_jobs": len(jobs),
                "rejected_metadata": len(rejected),
                "variants": args.variants,
                "seeds": args.seeds,
            },
            indent=2,
        )
    )
    print(f"Loading EMAGE FGD and SMPL-X on {args.device} ...")
    motion_backend = MotionBackend(args.emage_root, args.device, args.fk_chunk_size)

    availability: dict[str, Any] = {
        "fgd": {"available": True, "protocol": "EMAGE/PantoMatrix 55-joint Rot6D"},
        "bc": {"available": True, "protocol": "EMAGE sigma=0.3, order=7"},
        "diversity_l1": {"available": True, "protocol": "EMAGE joint-position L1 diversity"},
        "wer": {"available": False, "reason": "--asr-model was not supplied"},
        "speaker_similarity": {"available": False, "reason": "--speaker-model was not supplied"},
        "utmos_or_nisqa": {
            "available": False,
            "reason": "no pretrained UTMOS/NISQA backend is installed; signal diagnostics are reported instead",
        },
    }
    asr_backend = None
    if args.asr_model:
        try:
            asr_backend = ASRBackend(
                args.asr_model, args.device, args.allow_model_download, args.asr_language,
                num_beams=args.asr_num_beams,
                repetition_penalty=args.asr_repetition_penalty,
            )
            availability["wer"] = {
                "available": True,
                "model": args.asr_model,
                "num_beams": args.asr_num_beams,
                "repetition_penalty": args.asr_repetition_penalty,
            }
        except Exception as exc:
            availability["wer"] = {
                "available": False,
                "model": args.asr_model,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"Warning: ASR disabled: {availability['wer']['reason']}")

    speaker_backend = None
    if args.speaker_model:
        try:
            speaker_backend = SpeakerBackend(
                args.speaker_model, args.device, args.allow_model_download
            )
            availability["speaker_similarity"] = {
                "available": True,
                "model": args.speaker_model,
            }
        except Exception as exc:
            availability["speaker_similarity"] = {
                "available": False,
                "model": args.speaker_model,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"Warning: speaker similarity disabled: {availability['speaker_similarity']['reason']}")

    if args.require_speech_metrics and (asr_backend is None or speaker_backend is None):
        raise RuntimeError(f"Required speech metrics are unavailable: {availability}")

    rows: list[dict[str, Any]] = []
    features_by_group: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"pred": [], "gt": []}
    )
    counts = {"computed": 0, "resumed": 0, "failed": 0}
    evaluation_failures: list[dict[str, Any]] = []
    started = time.time()

    for index, job in enumerate(jobs, start=1):
        signature = evaluation_signature(job, args, emage_weights)
        cache_path = cache_dir / f"{job.sample_id}__seed_{job.seed:04d}.npz"
        previous = latest.get(job.key)
        if (
            args.resume
            and previous is not None
            and previous.get("evaluation_signature") == signature
            and cache_path.is_file()
        ):
            try:
                cached = np.load(cache_path)
                pred_feature = np.asarray(cached["pred_fgd"], dtype=np.float32)
                gt_feature = np.asarray(cached["gt_fgd"], dtype=np.float32)
                rows.append(previous)
                for group in ("all", job.variant):
                    features_by_group[group]["pred"].append(pred_feature)
                    features_by_group[group]["gt"].append(gt_feature)
                counts["resumed"] += 1
                print(f"[{index}/{len(jobs)}] resumed {job.key}")
                continue
            except Exception as exc:
                print(f"Warning: ignoring invalid cache {cache_path}: {exc}")

        print(f"[{index}/{len(jobs)}] evaluating {job.key}")
        try:
            pred_motion = load_mofea333(job.pred_motion)
            gt_motion = load_mofea333(job.gt_motion)
            frames = min(pred_motion.shape[0], gt_motion.shape[0])
            pred_motion = pred_motion[:frames]
            gt_motion = gt_motion[:frames]
            fps = float(job.metadata.get("decoded_motion_fps", 30.0))
            if fps <= 0:
                raise ValueError(f"Invalid decoded_motion_fps={fps}")

            pred_feature = motion_backend.fgd_feature(pred_motion)
            gt_feature = motion_backend.fgd_feature(gt_motion)
            pred_positions = motion_backend.positions(pred_motion)
            gt_positions = motion_backend.positions(gt_motion)
            crop = int(round(args.edge_seconds * fps))
            crop_start = crop
            crop_end = frames - crop
            if crop_end - crop_start < 3:
                raise ValueError(
                    f"Motion length {frames} is too short for {args.edge_seconds}s edge crop"
                )

            pred_waveform, pred_sr = load_audio(job.pred_audio)
            gt_waveform, gt_sr = load_audio(job.gt_audio)
            pred_audio_end = min(
                pred_waveform.size,
                int(round((crop_end / fps) * pred_sr)),
            )
            gt_audio_end = min(
                gt_waveform.size,
                int(round((crop_end / fps) * gt_sr)),
            )
            pred_onsets = motion_backend.audio_onsets(
                job.pred_audio, int(round(args.edge_seconds * pred_sr)), pred_audio_end
            )
            gt_onsets = motion_backend.audio_onsets(
                job.gt_audio, int(round(args.edge_seconds * gt_sr)), gt_audio_end
            )

            pred_derivatives = derivative_norms(pred_positions, fps)
            gt_derivatives = derivative_norms(gt_positions, fps)
            pred_world_positions = motion_backend.positions(pred_motion, world_space=True)
            gt_world_positions = motion_backend.positions(gt_motion, world_space=True)
            pred_skating = foot_skating(pred_world_positions, fps)
            gt_skating = foot_skating(gt_world_positions, fps)
            pred_audio_metrics = audio_signal_metrics(job.pred_audio)
            gt_audio_metrics = audio_signal_metrics(job.gt_audio)
            timing = job.metadata.get("timing_seconds", {})
            rtf = job.metadata.get("rtf", {})
            memory = job.metadata.get("gpu_memory_bytes", {})

            metrics: dict[str, Any] = {
                "frames": frames,
                "fps": fps,
                "duration_seconds": frames / fps,
                "bc_crop_frames": crop_end - crop_start,
                "bc_generated_audio_pred_motion": motion_backend.beat_consistency(
                    pred_onsets, pred_positions, crop_start, crop_end, fps
                ),
                "bc_gt_audio_pred_motion": motion_backend.beat_consistency(
                    gt_onsets, pred_positions, crop_start, crop_end, fps
                ),
                "bc_gt_audio_gt_motion": motion_backend.beat_consistency(
                    gt_onsets, gt_positions, crop_start, crop_end, fps
                ),
                "diversity_l1_pred": l1_diversity(pred_positions),
                "diversity_l1_gt": l1_diversity(gt_positions),
                "foot_skating_ratio_pred": pred_skating["skating_ratio"],
                "foot_skating_ratio_gt": gt_skating["skating_ratio"],
                "foot_contact_fraction_pred": pred_skating["contact_fraction"],
                "foot_contact_fraction_gt": gt_skating["contact_fraction"],
                "foot_contact_horizontal_speed_pred": pred_skating["contact_horizontal_speed"],
                "foot_contact_horizontal_speed_gt": gt_skating["contact_horizontal_speed"],
                "audio_rms_dbfs_pred": pred_audio_metrics["rms_dbfs"],
                "audio_rms_dbfs_gt": gt_audio_metrics["rms_dbfs"],
                "audio_clipping_fraction_pred": pred_audio_metrics["clipping_fraction"],
                "audio_silence_fraction_pred": pred_audio_metrics[
                    "silence_fraction_below_minus50dbfs"
                ],
                "audio_duration_seconds_pred": pred_audio_metrics["duration_seconds"],
                "audio_duration_seconds_gt": gt_audio_metrics["duration_seconds"],
                "audio_duration_error_seconds": abs(
                    pred_audio_metrics["duration_seconds"] - gt_audio_metrics["duration_seconds"]
                ),
                "rtf_sampling": rtf.get("sampling"),
                "rtf_end_to_end_no_model_load": rtf.get("end_to_end_no_model_load"),
                "sampling_seconds": timing.get("sampling"),
                "peak_allocated_gib": (
                    float(memory["peak_allocated"]) / (1024**3)
                    if memory.get("peak_allocated") is not None
                    else None
                ),
                "incremental_peak_allocated_gib": (
                    float(memory["incremental_peak_allocated"]) / (1024**3)
                    if memory.get("incremental_peak_allocated") is not None
                    else None
                ),
            }
            for name, value in pred_derivatives.items():
                metrics[f"{name}_pred"] = value
            for name, value in gt_derivatives.items():
                metrics[f"{name}_gt"] = value

            if asr_backend is not None:
                reference = reference_speech(job.prompt)
                hypothesis = asr_backend.transcribe(job.pred_audio)
                errors = word_errors(reference, hypothesis)
                metrics.update({f"asr_{key}": value for key, value in errors.items()})
                metrics["asr_reference"] = reference
                metrics["asr_hypothesis"] = hypothesis
            else:
                metrics.update(
                    {
                        "asr_wer": None,
                        "asr_errors": None,
                        "asr_reference_words": None,
                    }
                )
            metrics["speaker_cosine"] = (
                speaker_backend.cosine(job.pred_audio, job.gt_audio)
                if speaker_backend is not None
                else None
            )

            record = {
                "key": job.key,
                "sample_id": job.sample_id,
                "base_id": job.base_id,
                "variant": job.variant,
                "seed": job.seed,
                "status": "complete",
                "evaluation_signature": signature,
                "job_signature": job.job_signature,
                "paths": {
                    "metadata": str(job.metadata_path),
                    "pred_motion": str(job.pred_motion),
                    "pred_audio": str(job.pred_audio),
                    "gt_motion": str(job.gt_motion),
                    "gt_audio": str(job.gt_audio),
                    "fgd_feature_cache": str(cache_path),
                },
                "metrics": metrics,
                "completed_at_unix": time.time(),
            }
            temporary_cache = cache_path.with_name(cache_path.name + ".tmp.npz")
            np.savez_compressed(temporary_cache, pred_fgd=pred_feature, gt_fgd=gt_feature)
            os.replace(temporary_cache, cache_path)
            append_jsonl(per_sample_path, record)
            rows.append(record)
            for group in ("all", job.variant):
                features_by_group[group]["pred"].append(pred_feature)
                features_by_group[group]["gt"].append(gt_feature)
            counts["computed"] += 1
        except Exception as exc:
            counts["failed"] += 1
            failure = {
                "key": job.key,
                "sample_id": job.sample_id,
                "variant": job.variant,
                "seed": job.seed,
                "status": "failed",
                "evaluation_signature": signature,
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "failed_at_unix": time.time(),
            }
            append_jsonl(failures_path, failure)
            evaluation_failures.append({**failure, "metadata_path": str(job.metadata_path)})
            print(f"FAILED {job.key}: {failure['exception']}")
            if args.fail_fast:
                break

    rows_by_group = {
        "all": rows,
        "l1": [row for row in rows if row["variant"] == "l1"],
        "l2": [row for row in rows if row["variant"] == "l2"],
    }
    summaries: dict[str, dict[str, Any]] = {}
    for group, group_rows in rows_by_group.items():
        if not group_rows:
            continue
        summaries[group] = summarize_group(
            group_rows,
            features_by_group[group]["pred"],
            features_by_group[group]["gt"],
            motion_backend.fgd,
        )

    report = build_failure_report(args.input_dir, rows, rejected, evaluation_failures, jobs)
    atomic_json(args.output_dir / "failure_report.json", report)
    has_failures = any(report["counts"][key] for key in (
        "inference_failed", "evaluation_failed", "evaluation_not_attempted", "rejected_metadata", "malformed_manifest_records"
    ))
    result = {
        "status": "completed_with_failures" if has_failures else "complete",
        "evaluator_version": EVALUATOR_VERSION,
        "counts": {**counts, "selected": len(jobs), "successful": len(rows)},
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "protocol": {
            "motion_fps": "read per sample (expected 30 FPS)",
            "fgd": "official EMAGE/PantoMatrix AESKConv, all 55 joints, 330D Rot6D",
            "bc": (
                f"official EMAGE/PantoMatrix, sigma=0.3, order=7, "
                f"exclude first/last {args.edge_seconds:g}s"
            ),
            "bc_gap": "abs(mean BC(reference audio, generated motion) - mean BC(reference audio, reference motion))",
            "diversity": "official EMAGE L1 deviation over SMPL-X joint positions",
            "kinematics": "finite differences over body-relative SMPL-X joint positions",
            "foot_skating": "SMPL-X world joints with root rotation and translation; contact<=5cm above per-sequence ground; speed>0.10m/s",
            "wer_normalization": "lowercase ASCII alphanumeric/apostrophe tokens; micro-average",
            "wer_aggregation": "wer/wer_full include every ASR row; wer_filtered is the legacy WER<=1 subset",
        },
        "availability": availability,
        "summary": summaries,
        "rejected_metadata": rejected,
        "failure_report": str(args.output_dir / "failure_report.json"),
        "failure_counts": report["counts"],
        "elapsed_seconds": time.time() - started,
        "completed_at_unix": time.time(),
    }
    if args.pairacc_checkpoint:
        pairacc_path = args.output_dir / "pairacc.json"
        command = [sys.executable, str(REPO_ROOT / "scripts/evaluate_pairacc.py"),
                   "--input-root", str(args.input_dir), "--output", str(pairacc_path),
                   "--checkpoint", str(args.pairacc_checkpoint), "--model-root", str(args.pairacc_model_root),
                   "--model-gender", args.pairacc_model_gender, "--device", args.device,
                   "--conversion-batch-size", str(args.pairacc_conversion_batch_size),
                   "--variants", *args.variants]
        if args.seeds:
            command += ["--seeds", *map(str, args.seeds)]
        if args.limit:
            command += ["--limit", str(args.limit)]
        subprocess.run(command, check=True)
        result["pairacc"] = json.loads(pairacc_path.read_text())
        for level, values in result["pairacc"]["levels"].items():
            if level in summaries:
                summaries[level]["global_pairacc"] = values["global_pairacc"]
            summaries.setdefault("all", {})[f"global_pairacc_{level}"] = values["global_pairacc"]
        result["availability"]["pairacc"] = {"available": True, "protocol": "CLaM-H Global PairAcc; percent"}
    else:
        result["availability"]["pairacc"] = {"available": False, "reason": "PairAcc resources were not supplied"}
    atomic_json(args.output_dir / "metrics_summary.json", result)
    write_summary_csv(args.output_dir / "metrics_summary.csv", summaries)
    print(json.dumps(result, indent=2))
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
