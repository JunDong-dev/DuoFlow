"""Split provenance checks shared by preprocessing, training, and evaluation."""

import json
from pathlib import Path


def read_split_metadata(root: Path) -> dict | None:
    for name in ("preprocess_scope.json", "preprocess_summary.json"):
        path = root / name
        if path.is_file():
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"Invalid preprocessing metadata: {path}")
            return value
    return None


def validate_preprocessed_split(root: Path, expected_split: str) -> None:
    """Check provenance when available, while supporting explicitly selected legacy caches."""
    metadata = read_split_metadata(root)
    if metadata is not None and metadata.get("split") != expected_split:
        raise ValueError(
            f"Expected {expected_split!r} data at {root}, "
            f"but preprocessing metadata declares {metadata.get('split')!r}"
        )


def claim_preprocessing_scope(root: Path, data_root: Path, split: str, audio_only: bool) -> None:
    """Never reuse one latent-cache directory across source datasets or splits."""
    scope = {"data_root": str(data_root.resolve()), "split": split, "audio_only": audio_only}
    previous = read_split_metadata(root)
    if previous is not None:
        for key, value in scope.items():
            if key in previous and previous[key] != value:
                raise ValueError(f"Cache {root} already belongs to {key}={previous[key]!r}; choose another output root")
    elif root.exists():
        for name in ("audio_latents", "motion_latents", "joint_conditions", "speak_conditions"):
            if next((root / name).rglob("*.pt"), None) is not None:
                raise ValueError(f"Cache {root} has no split provenance; choose a fresh output root")
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "preprocess_scope.json.tmp"
    temporary.write_text(json.dumps(scope, indent=2) + "\n")
    temporary.replace(root / "preprocess_scope.json")


def validate_training_data(config: dict) -> None:
    data = config.get("data", {})
    train = Path(data["preprocessed_data_root"]).expanduser().resolve()
    validate_preprocessed_split(train, "train")
    roots = [train]
    if data.get("replay_batch_prob", 0) > 0:
        if not data.get("replay_preprocessed_data_root"):
            raise ValueError("Audio replay requires replay_preprocessed_data_root")
        replay = Path(data["replay_preprocessed_data_root"]).expanduser().resolve()
        validate_preprocessed_split(replay, "train")
        if replay == train or replay in train.parents or train in replay.parents:
            raise ValueError("Speech replay and paired training caches must be separate directories")
        roots.append(replay)
    validation = config.get("validation", {})
    val_root = validation.get("val_preprocessed_data_root")
    if not val_root:
        if validation.get("interval") is not None:
            raise ValueError("Enabled validation requires a separate validation data root")
        return
    val = Path(val_root).expanduser().resolve()
    validate_preprocessed_split(val, "val")
    for root in roots:
        if root == val or root in val.parents or val in root.parents:
            raise ValueError("Training/replay and validation caches must be disjoint directories")
    # IDs are comparable only within the same source corpus.
    train_meta, val_meta = read_split_metadata(train), read_split_metadata(val)
    if train_meta and val_meta and train_meta.get("data_root") == val_meta.get("data_root"):
        train_ids, val_ids = train / "sample_ids.txt", val / "sample_ids.txt"
        if train_ids.is_file() and val_ids.is_file():
            overlap = set(train_ids.read_text().split()) & set(val_ids.read_text().split())
            if overlap:
                raise ValueError(f"Training and validation IDs overlap: {len(overlap)} clips")
