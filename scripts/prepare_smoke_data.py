#!/usr/bin/env python3
"""Create a ten-clip view using six training, two validation and two test clips."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.data_root.resolve(strict=True)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new, empty directory for the smoke dataset view")
    (output / "split").mkdir()
    selected, seen = {}, set()
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        ids = list(dict.fromkeys(line.strip() for line in (source / "split" / f"{split}.txt").read_text().splitlines() if line.strip()))
        chosen = []
        for sample in ids:
            if sample in seen:
                continue
            files = [(source / directory / f"{sample}{suffix}", output / directory / f"{sample}{suffix}")
                     for directory, suffix in (("motion_smplx", ".npz"), ("audio_wav", ".wav"), ("caption", ".txt"))]
            if not all(origin.is_file() for origin, _ in files):
                continue
            for origin, target in files:
                target.parent.mkdir(exist_ok=True)
                target.symlink_to(origin)
            chosen.append(sample)
            seen.add(sample)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Not enough complete {split} samples")
        selected[split] = chosen
        (output / "split" / f"{split}.txt").write_text("\n".join(chosen) + "\n")
    (output / "split/all.txt").write_text("\n".join(sample for group in selected.values() for sample in group) + "\n")
    report = {"source": str(source), "samples": 10, "splits": selected}
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
