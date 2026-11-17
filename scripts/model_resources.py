"""Resolve an existing Hugging Face snapshot without contacting the Hub."""

from pathlib import Path


def local_model_path(model_id: str, cache_dir=None) -> str:
    direct = Path(model_id).expanduser()
    if direct.is_dir():
        return str(direct)
    if cache_dir is None:
        return model_id
    name = "models--" + model_id.replace("/", "--")
    for base in (Path(cache_dir), Path(cache_dir) / "hub"):
        root = base / name
        main = root / "refs/main"
        if main.is_file():
            snapshot = root / "snapshots" / main.read_text().strip()
            if snapshot.is_dir():
                return str(snapshot)
        snapshots = sorted(path for path in (root / "snapshots").glob("*") if path.is_dir())
        if len(snapshots) == 1:
            return str(snapshots[0])
        if len(snapshots) > 1:
            raise ValueError(f"Multiple snapshots in {root}; supply the intended snapshot directory as model ID")
    return model_id
