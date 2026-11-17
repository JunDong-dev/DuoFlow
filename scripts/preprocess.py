#!/usr/bin/env python3
"""Preprocess raw DuoFlow clips into training-ready latent caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import torch

from data_splits import claim_preprocessing_scope

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(REPO_ROOT / "packages/motionspeech-core/src"),
    str(REPO_ROOT / "packages/motionspeech-trainer/src"),
    str(REPO_ROOT / "third_party/motion_vae"),
]

VARIANTS = ("l1", "l2")
PHONEME_LENGTH = 500
TEXT_LENGTH = 320
SPEECH_PATTERN = re.compile(r"<S>(.*?)</E>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--audio-only", action="store_true", help="Preprocess speech-pretraining data without motion files.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("motion", "audio", "phoneme", "text"),
        default=("motion", "audio", "phoneme", "text"),
    )
    parser.add_argument(
        "--motion-vae-checkpoint",
        type=Path,
        default=Path(os.environ.get("DUOFLOW_MOTION_VAE", REPO_ROOT / "weights/motion_vae.pth")),
    )
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--audioldm2-model", default="anhnct/audioldm2_gigaspeech")
    parser.add_argument("--audioldm2-cache", type=Path, default=None)
    parser.add_argument("--phoneme-tokenizer", type=Path, default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.variants = ("speech",) if args.audio_only else VARIANTS
    if args.audio_only:
        args.stages = tuple(stage for stage in args.stages if stage != "motion")
    return args


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_ids(data_root: Path, split: str, limit: int | None) -> list[str]:
    path = data_root / "split" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    sample_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    sample_ids = list(dict.fromkeys(sample_ids))
    return sample_ids[:limit]


def validate_raw_files(data_root: Path, sample_ids: Iterable[str], audio_only: bool = False) -> None:
    missing = []
    sources = [("audio_wav", ".wav"), ("caption", ".txt")]
    if not audio_only:
        sources.append(("motion_smplx", ".npz"))
    for sample_id in sample_ids:
        for directory, suffix in sources:
            path = data_root / directory / f"{sample_id}{suffix}"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing raw assets:\n" + "\n".join(missing[:20]))


def variant_caption(text: str, variant: str) -> str:
    drop = "L2" if variant == "l1" else "L1"
    value = re.sub(fr"<{drop}>.*?</{drop}>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return value.strip()


def speech_caption(text: str) -> str:
    speech = SPEECH_PATTERN.search(text)
    audcap = re.search(r"<AUDCAP>.*?</AUDCAP>", text, flags=re.IGNORECASE | re.DOTALL)
    if speech is None:
        raise ValueError("Caption does not contain <S>...</E>")
    fields = [f"<S>{speech.group(1).strip()}</E>"]
    if audcap is not None:
        fields.append(audcap.group(0))
    return "".join(fields)


def axis_angle_to_mofea333(path: Path) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    payload = np.load(path)
    poses = np.asarray(payload["poses"], dtype=np.float32)
    translations = np.asarray(payload["trans"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 165 or translations.shape != (poses.shape[0], 3):
        raise ValueError(f"Unexpected SMPL-X shape in {path}: poses={poses.shape}, trans={translations.shape}")
    matrices = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 55, 3, 3)
    rotations_6d = matrices[:, :, :2, :].reshape(poses.shape[0], 330)
    return np.concatenate((translations, rotations_6d), axis=1).astype(np.float32)


def load_motion_vae(checkpoint: Path, device: str) -> torch.nn.Module:
    from models.tae3_parts import Causal_HumanTAE3parts

    model = Causal_HumanTAE3parts(
        inputdim=333,
        hidden_size=2048,
        down_t=1,
        stride_t=3,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        latent_dim=64,
        clip_range=[-30, 20],
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["net"] if isinstance(payload, dict) and "net" in payload else payload
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocess_motion(args: argparse.Namespace, sample_ids: list[str]) -> None:
    output = args.output_root / "motion_latents"
    ground_truth = args.output_root / "gt_motion"
    output.mkdir(parents=True, exist_ok=True)
    ground_truth.mkdir(parents=True, exist_ok=True)
    mean = np.load(REPO_ROOT / "assets/motion_vae_stats/mean.npy").astype(np.float32)
    std = np.load(REPO_ROOT / "assets/motion_vae_stats/std.npy").astype(np.float32)
    if mean.shape != (333,) or std.shape != (333,):
        raise ValueError("Motion VAE normalization statistics must contain 333 values")
    pending = [
        sample_id for sample_id in sample_ids
        if not (ground_truth / f"{sample_id}.npy").is_file()
        or not all((output / f"{sample_id}_{variant}.pt").is_file() for variant in args.variants)
    ]
    if not pending:
        print(f"[motion] reused {len(sample_ids)} samples", flush=True)
        return
    model = load_motion_vae(args.motion_vae_checkpoint, args.device)
    torch.manual_seed(0)
    for index, sample_id in enumerate(pending, start=1):
        targets = [output / f"{sample_id}_{variant}.pt" for variant in args.variants]
        gt_path = ground_truth / f"{sample_id}.npy"
        if all(path.is_file() for path in targets) and gt_path.is_file():
            continue
        motion = axis_angle_to_mofea333(args.data_root / "motion_smplx" / f"{sample_id}.npz")
        np.save(gt_path, motion)
        normalized = (motion - mean) / np.maximum(std, 1e-8)
        with torch.inference_mode():
            latent, _, _ = model.encode(torch.from_numpy(normalized).unsqueeze(0).to(args.device))
        latent = latent.squeeze(0).transpose(0, 1).to(dtype=torch.bfloat16, device="cpu")
        value = {"latents": latent, "num_frames": latent.shape[1], "height": 1, "width": 1, "fps": 10.0}
        for target in targets:
            atomic_torch_save(value, target)
        print(f"[motion] {index}/{len(pending)} {sample_id}", flush=True)


def preprocess_audio(args: argparse.Namespace, sample_ids: list[str]) -> None:
    import soundfile as sf
    import torchaudio
    from diffusers import AutoencoderKL
    from transformers import SpeechT5HifiGan

    output = args.output_root / "audio_latents"
    output.mkdir(parents=True, exist_ok=True)
    pending = [
        sample_id for sample_id in sample_ids
        if not all((output / f"{sample_id}_{variant}.pt").is_file() for variant in args.variants)
    ]
    if not pending:
        print(f"[audio] reused {len(sample_ids)} samples", flush=True)
        return
    # Only the audio VAE and vocoder are used for latent preprocessing.
    # Loading the full pipeline needlessly requires text/language-model weights.
    from model_resources import local_model_path
    model_path = local_model_path(args.audioldm2_model, args.audioldm2_cache)
    options = dict(cache_dir=args.audioldm2_cache, local_files_only=args.local_files_only,
                   torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **options).to(args.device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained(model_path, subfolder="vocoder", **options).to(args.device).eval()
    sample_rate = int(getattr(vocoder.config, "sampling_rate", 16000))
    hop_length = int(np.prod(getattr(vocoder.config, "upsample_rates", [5, 4, 2, 2, 2, 2])))
    mel_bins = int(getattr(vocoder.config, "num_mel_bins", 64))
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=hop_length,
        n_mels=mel_bins,
        f_min=0,
        f_max=sample_rate // 2,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
    ).to(args.device)
    resamplers: dict[int, torch.nn.Module] = {}
    for index, sample_id in enumerate(pending, start=1):
        targets = [output / f"{sample_id}_{variant}.pt" for variant in args.variants]
        if all(path.is_file() for path in targets):
            continue
        waveform, source_rate = sf.read(args.data_root / "audio_wav" / f"{sample_id}.wav")
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = waveform.mean(axis=1) if waveform.ndim == 2 else waveform
        tensor = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0).to(args.device)
        if source_rate != sample_rate:
            resamplers.setdefault(source_rate, torchaudio.transforms.Resample(source_rate, sample_rate).to(args.device))
            tensor = resamplers[source_rate](tensor)
        mel = mel_transform(tensor.squeeze(1))
        mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2).unsqueeze(1)
        with torch.inference_mode():
            latent = vae.encode(mel).latent_dist.sample().squeeze(0)
        latent = latent.to(dtype=torch.bfloat16, device="cpu")
        value = {
            "latents": latent,
            "num_time_steps": latent.shape[1],
            "frequency_bins": latent.shape[2],
            "duration": tensor.shape[-1] / sample_rate,
        }
        for target in targets:
            atomic_torch_save(value, target)
        print(f"[audio] {index}/{len(pending)} {sample_id}", flush=True)


def resolve_tokenizer(args: argparse.Namespace) -> Path | str:
    if args.phoneme_tokenizer is not None:
        return args.phoneme_tokenizer
    if args.audioldm2_cache is not None:
        snapshots = sorted(args.audioldm2_cache.glob("models--anhnct--audioldm2_gigaspeech/snapshots/*/tokenizer_2"))
        if snapshots:
            return snapshots[-1]
    return args.audioldm2_model


def preprocess_phonemes(args: argparse.Namespace, sample_ids: list[str], captions: dict[str, str]) -> None:
    from transformers import VitsTokenizer

    output = args.output_root / "phonemes"
    output.mkdir(parents=True, exist_ok=True)
    pending = [
        sample_id for sample_id in sample_ids
        if not all((output / f"{sample_id}_{variant}.npy").is_file() for variant in args.variants)
    ]
    if not pending:
        print(f"[phoneme] reused {len(sample_ids)} samples", flush=True)
        return
    tokenizer = VitsTokenizer.from_pretrained(resolve_tokenizer(args), local_files_only=args.local_files_only)
    tokenizer.phonemize = True
    for index, sample_id in enumerate(pending, start=1):
        match = SPEECH_PATTERN.search(captions[sample_id])
        if match is None:
            raise ValueError(f"Caption for {sample_id} does not contain speech tags")
        encoded = tokenizer(
            match.group(1).strip(),
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=PHONEME_LENGTH,
            return_tensors="np",
        )["input_ids"][0].astype(np.int64)
        for variant in args.variants:
            np.save(output / f"{sample_id}_{variant}.npy", encoded)
        print(f"[phoneme] {index}/{len(pending)} {sample_id}", flush=True)


def load_text_encoder(args: argparse.Namespace) -> torch.nn.Module:
    from motionspeech_core.text_encoders.gemma.encoders.av_encoder import (
        AVGemmaTextEncoderModelConfigurator,
    )
    from motionspeech_core.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root

    os.environ["MOTIONSPEECH_TEXT_MAX_LENGTH"] = str(TEXT_LENGTH)
    config = {"transformer": {"num_learnable_registers": None}}
    encoder = AVGemmaTextEncoderModelConfigurator.from_config(config)
    for operation in module_ops_from_gemma_root(str(args.text_encoder)):
        if operation.matcher(encoder):
            encoder = operation.mutator(encoder)
    encoder.skip_feature_extractor = True
    encoder.gemma_num_layers = 3
    encoder.gemma_feature_selection = "embedding_first_last"
    return encoder.to(device=args.device, dtype=torch.bfloat16).eval()


def save_text_condition(encoder: torch.nn.Module, text: str, path: Path) -> None:
    if path.is_file():
        return
    tokenizer = encoder.tokenizer.tokenizer
    tokenized = tokenizer(
        text,
        padding="max_length",
        max_length=TEXT_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        features, mask = encoder._preprocess_text(text)
    atomic_torch_save(
        {
            "encoded_features": features.squeeze(0).to(dtype=torch.bfloat16, device="cpu"),
            "raw_attention_mask": mask.squeeze(0).to(dtype=torch.int64, device="cpu"),
            "prompt_token_mask": tokenized.attention_mask[0].to(dtype=torch.int64),
        },
        path,
    )


def preprocess_text(args: argparse.Namespace, sample_ids: list[str], captions: dict[str, str]) -> None:
    joint = args.output_root / "joint_conditions"
    speech = args.output_root / "speak_conditions"
    if not args.audio_only:
        joint.mkdir(parents=True, exist_ok=True)
    speech.mkdir(parents=True, exist_ok=True)
    pending = [
        sample_id for sample_id in sample_ids
        if not all(
            (args.audio_only or (joint / f"{sample_id}_{variant}.pt").is_file())
            and (speech / f"{sample_id}_{variant}.pt").is_file()
            for variant in args.variants
        )
    ]
    if not pending:
        print(f"[text] reused {len(sample_ids)} samples", flush=True)
        return
    encoder = load_text_encoder(args)
    for index, sample_id in enumerate(pending, start=1):
        full_caption = captions[sample_id]
        speech_text = speech_caption(full_caption)
        for variant in args.variants:
            if not args.audio_only:
                save_text_condition(encoder, variant_caption(full_caption, variant), joint / f"{sample_id}_{variant}.pt")
            save_text_condition(encoder, speech_text, speech / f"{sample_id}_{variant}.pt")
        print(f"[text] {index}/{len(pending)} {sample_id}", flush=True)


def write_manifests(args: argparse.Namespace, sample_ids: list[str], captions: dict[str, str]) -> None:
    manifests = args.output_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        rows = [
            {
                "sample_id": sample_id,
                "caption": speech_caption(captions[sample_id]) if args.audio_only else variant_caption(captions[sample_id], variant),
                "audio_path": str((args.data_root / "audio_wav" / f"{sample_id}.wav").resolve()),
                **({} if args.audio_only else {
                    "motion_path": str((args.data_root / "motion_smplx" / f"{sample_id}.npz").resolve()),
                }),
            }
            for sample_id in sample_ids
        ]
        (manifests / f"{args.split}_{variant}.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_root / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n")
    summary = {
        "status": "complete",
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "audio_only": args.audio_only,
        "samples": len(sample_ids),
        "variants": len(sample_ids) * len(args.variants),
        "stages": list(args.stages),
    }
    (args.output_root / "preprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve(strict=True)
    claim_preprocessing_scope(args.output_root, args.data_root, args.split, args.audio_only)
    sample_ids = read_ids(args.data_root, args.split, args.limit)
    if not sample_ids:
        raise RuntimeError("The selected split is empty")
    validate_raw_files(args.data_root, sample_ids, args.audio_only)
    captions = {
        sample_id: (args.data_root / "caption" / f"{sample_id}.txt").read_text().strip()
        for sample_id in sample_ids
    }
    if "motion" in args.stages:
        preprocess_motion(args, sample_ids)
    if "audio" in args.stages:
        preprocess_audio(args, sample_ids)
    if "phoneme" in args.stages:
        preprocess_phonemes(args, sample_ids, captions)
    if "text" in args.stages:
        preprocess_text(args, sample_ids, captions)
    write_manifests(args, sample_ids, captions)
    print(json.dumps({"status": "complete", "samples": len(sample_ids), "output": str(args.output_root)}))


if __name__ == "__main__":
    main()
