#!/usr/bin/env python3
"""Run the documented ten-clip Motion VAE and DuoFlow integration workflow."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="Reuse successful stages in this smoke work directory")
    args = parser.parse_args()
    required = ["MOTIONSPEECH_TEXT_ENCODER", "MOTIONSPEECH_AUDIOLDM2_CACHE",
                "MOTIONSPEECH_PHONEME_ENCODER", "MOTIONSPEECH_PHONEME_TOKENIZER",
                "MOTIONSPEECH_EMAGE_ROOT", "MOTIONSPEECH_ASR_MODEL", "MOTIONSPEECH_SPEAKER_MODEL",
                "DUOFLOW_MOTION_VAE", "DUOFLOW_PAIRACC_CHECKPOINT", "DUOFLOW_PAIRACC_MODEL_ROOT"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        parser.error("Missing environment variables: " + ", ".join(missing))
    checks = {"MOTIONSPEECH_ASR_MODEL": "config.json", "MOTIONSPEECH_SPEAKER_MODEL": "config.json",
              "MOTIONSPEECH_PHONEME_ENCODER": "config.json", "MOTIONSPEECH_TEXT_ENCODER": "config.json"}
    for key, name in checks.items():
        if not (Path(os.environ[key]) / name).is_file():
            raise FileNotFoundError(f"{key} must point directly to a model directory containing {name}")
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()) and not args.resume:
        raise ValueError("Use a new, empty work directory; existing artifacts are not accepted as a smoke pass")
    (work / "logs").mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "packages/motionspeech-core/src"),
                                         str(REPO_ROOT / "packages/motionspeech-trainer/src"), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    data = work / "data"
    pretrained_vae = env["DUOFLOW_MOTION_VAE"]
    trained_vae = work / "motion_vae_train/checkpoint_last.pth"
    preprocessed = work / "preprocessed"
    env.update(EMILIA_TRAIN_ROOT=str(preprocessed / "train"), EMILIA_VAL_ROOT=str(preprocessed / "val"),
               MOTIONSPEECH_TRAIN_ROOT=str(preprocessed / "train"), MOTIONSPEECH_VAL_ROOT=str(preprocessed / "val"),
               MOTIONSPEECH_TEST_ROOT=str(preprocessed / "test"),
               MOTIONSPEECH_STAGE1_OUTPUT=str(work / "configs/stage1"),
               MOTIONSPEECH_STAGE2_OUTPUT=str(work / "configs/stage2"))
    env["MOTIONSPEECH_STAGE1_CHECKPOINT"] = str(work / "configs/stage1/checkpoints/best/model_weights_best.safetensors")
    report = {"status": "running", "purpose": "integration smoke test; not paper reproduction",
              "dataset": str(args.data_root), "splits": {"train": 6, "val": 2, "test": 2}, "stages": []}
    if args.resume and (work / "smoke_summary.json").is_file():
        report = json.loads((work / "smoke_summary.json").read_text())
        report["status"] = "running"
    def save():
        (work / "smoke_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, script, arguments):
        command = [sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, arguments)]
        if args.resume and any(stage["name"] == name and stage["returncode"] == 0 for stage in report["stages"]):
            print(f"REUSE {name}", flush=True)
            return
        started = time.time()
        print(f"START {name}", flush=True)
        save()
        with (work / "logs" / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["stages"].append({"name": name, "command": command, "returncode": result.returncode,
                                 "seconds": time.time() - started})
        if result.returncode:
            report["status"] = "failed"
            save()
            raise RuntimeError(f"{name} failed; see {work / 'logs' / (name + '.log')}")
        save()
        print(f"PASS {name} ({time.time() - started:.1f}s)", flush=True)

    run("select_data", "prepare_smoke_data.py", ["--data-root", args.data_root, "--output-root", data])
    run("motion_vae_train", "train_motion_vae.py", ["--data-root", data, "--output-dir", work / "motion_vae_train",
        "--checkpoint", pretrained_vae, "--split", "train", "--limit", 6, "--max-steps", 3, "--batch-size", 2, "--device", args.device])
    run("motion_vae_reconstruct", "infer_motion_vae.py", ["--data-root", data, "--checkpoint", trained_vae,
        "--mode", "reconstruct", "--split", "all", "--limit", 10, "--output-dir", work / "motion_vae_reconstruct", "--device", args.device])
    run("motion_vae_decode", "infer_motion_vae.py", ["--latent-dir", work / "motion_vae_reconstruct/latents",
        "--checkpoint", trained_vae, "--mode", "decode", "--output-dir", work / "motion_vae_decode", "--device", args.device])
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        run("preprocess_" + split, "preprocess.py", ["--data-root", data, "--output-root", preprocessed / split,
            "--split", split, "--limit", count, "--device", args.device, "--text-encoder", env["MOTIONSPEECH_TEXT_ENCODER"],
            "--audioldm2-cache", env["MOTIONSPEECH_AUDIOLDM2_CACHE"], "--phoneme-tokenizer", env["MOTIONSPEECH_PHONEME_TOKENIZER"],
            "--motion-vae-checkpoint", trained_vae])
    run("smoke_configs", "make_smoke_configs.py", ["--preprocessed-root", preprocessed / "train",
        "--validation-root", preprocessed / "val", "--output-root", work / "configs", "--steps", 12])
    for stage in (1, 2):
        run(f"stage{stage}_train", "train.py", [work / f"configs/stage{stage}_smoke.yaml", "--disable-progress-bars"])
    run("inference", "infer_batch.py", ["--split", "test", "--val-root", preprocessed / "test",
        "--trained-model", work / "configs/stage2/checkpoints/best/model_weights_best.safetensors",
        "--output-dir", work / "inference", "--variants", "l1", "l2", "--all-samples", "--num-inference-steps", 30,
        "--motion-guidance-scale", 1.0, "--audio-guidance-scale", 5.0, "--seeds", 0, "--device", args.device,
        "--text-encoder-path", env["MOTIONSPEECH_TEXT_ENCODER"], "--motion-vae-path", trained_vae,
        "--audioldm2-cache-dir", env["MOTIONSPEECH_AUDIOLDM2_CACHE"], "--phoneme-encoder-path", env["MOTIONSPEECH_PHONEME_ENCODER"],
        "--phoneme-tokenizer-path", env["MOTIONSPEECH_PHONEME_TOKENIZER"], "--use-correct-motion-time-coordinates", "--fail-fast"])
    run("evaluation", "evaluate.py", ["--split", "test", "--input-dir", work / "inference", "--output-dir", work / "evaluation",
        "--gt-motion-dir", preprocessed / "test/gt_motion", "--emage-root", env["MOTIONSPEECH_EMAGE_ROOT"],
        "--asr-model", env["MOTIONSPEECH_ASR_MODEL"], "--speaker-model", env["MOTIONSPEECH_SPEAKER_MODEL"],
        "--require-speech-metrics", "--require-pairacc", "--pairacc-checkpoint", env["DUOFLOW_PAIRACC_CHECKPOINT"],
        "--pairacc-model-root", env["DUOFLOW_PAIRACC_MODEL_ROOT"], "--device", args.device, "--fail-fast"])
    metrics = json.loads((work / "evaluation/metrics_summary.json").read_text())
    if metrics["status"] != "complete" or metrics["counts"]["successful"] != 4:
        raise RuntimeError("Expected four successful L1/L2 test generations and evaluations")
    for level in ("l1", "l2"):
        if level not in metrics["pairacc"]["levels"]:
            raise RuntimeError("Missing PairAcc level: " + level)
    vae_report = json.loads((work / "motion_vae_train/training_report.json").read_text())
    if set(vae_report["seen_samples"]) != set(vae_report["selected_samples"]):
        raise RuntimeError("Motion VAE did not train on all selected training samples")
    report.update(status="complete", evaluated_jobs=4, metrics_summary=str(work / "evaluation/metrics_summary.json"))
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
