# Modified for DuoFlow; see NOTICE.md.
import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

from motionspeech_trainer.datasets import MANIFEST_VERSION, PrecomputedDataset


def _create_sample(data_root: Path, sample_name: str) -> None:
    for source_name in ("latents", "conditions"):
        source_path = data_root / source_name
        source_path.mkdir(exist_ok=True)
        (source_path / f"{sample_name}.pt").touch()


def _create_dataset(data_root: Path) -> PrecomputedDataset:
    return PrecomputedDataset(
        str(data_root),
        data_sources={"latents": "latent_conditions", "conditions": "text_conditions"},
    )


def test_manifest_is_created_and_reused(tmp_path: Path) -> None:
    _create_sample(tmp_path, "sample_0")
    first_dataset = _create_dataset(tmp_path)

    assert first_dataset.manifest_path.is_file()
    assert len(first_dataset) == 1

    with patch.object(PrecomputedDataset, "_discover_samples", side_effect=AssertionError("unexpected scan")):
        cached_dataset = _create_dataset(tmp_path)

    assert cached_dataset.sample_files == first_dataset.sample_files


def test_manifest_is_invalidated_when_source_directory_changes(tmp_path: Path) -> None:
    _create_sample(tmp_path, "sample_0")
    _create_dataset(tmp_path)
    previous_mtime = (tmp_path / "latents").stat().st_mtime_ns

    _create_sample(tmp_path, "sample_1")
    os.utime(tmp_path / "latents", ns=(previous_mtime + 1, previous_mtime + 1))

    refreshed_dataset = _create_dataset(tmp_path)

    assert len(refreshed_dataset) == 2


def test_corrupt_manifest_is_rebuilt(tmp_path: Path) -> None:
    _create_sample(tmp_path, "sample_0")
    first_dataset = _create_dataset(tmp_path)
    first_dataset.manifest_path.write_text("not json", encoding="utf-8")

    rebuilt_dataset = _create_dataset(tmp_path)
    payload = json.loads(rebuilt_dataset.manifest_path.read_text(encoding="utf-8"))

    assert len(rebuilt_dataset) == 1
    assert payload["version"] == MANIFEST_VERSION


def test_non_main_process_waits_for_manifest_without_scanning(tmp_path: Path) -> None:
    _create_sample(tmp_path, "sample_0")
    result: list[PrecomputedDataset] = []

    def wait_for_manifest() -> None:
        result.append(
            PrecomputedDataset(
                str(tmp_path),
                data_sources={"latents": "latent_conditions", "conditions": "text_conditions"},
                wait_for_manifest=True,
            )
        )

    with patch("motionspeech_trainer.datasets.MANIFEST_POLL_INTERVAL_SECONDS", 0.01):
        waiter = threading.Thread(target=wait_for_manifest)
        waiter.start()
        builder = _create_dataset(tmp_path)
        waiter.join(timeout=5)

    assert not waiter.is_alive()
    assert builder.manifest_available
    assert len(result) == 1
    assert result[0].sample_files == builder.sample_files
