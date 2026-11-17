# Modified for DuoFlow; see NOTICE.md.
import hashlib
import json
import os
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from motionspeech_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"
MANIFEST_CACHE_DIR_NAME = ".motionspeech-trainer-cache"
MANIFEST_VERSION = 2
MANIFEST_POLL_INTERVAL_SECONDS = 5.0
MANIFEST_WAIT_TIMEOUT_SECONDS = 6 * 60 * 60


def _collate_audio_source(batch_list: list[dict]) -> dict:
    """Collate audio_latents dicts, padding the time dim to the max in the batch.

    Returns a dict with stacked tensors, plus `audio_loss_mask: (B, max_T)` bool
    tensor marking the valid (non-padded) time positions for each sample.
    """
    audio_dicts = [s["audio_latents"] for s in batch_list]
    latents_list = [d["latents"] for d in audio_dicts]  # each (C, T_i, F)
    channels = latents_list[0].shape[0]
    freq_bins = latents_list[0].shape[2]
    time_steps = [l.shape[1] for l in latents_list]
    max_t = max(time_steps)
    B = len(batch_list)

    padded = latents_list[0].new_zeros((B, channels, max_t, freq_bins))
    loss_mask = torch.zeros((B, max_t), dtype=torch.bool)
    for i, l in enumerate(latents_list):
        t = l.shape[1]
        padded[i, :, :t, :] = l
        loss_mask[i, :t] = True

    # Stack scalar / fixed-shape metadata
    out: dict = {"latents": padded, "audio_loss_mask": loss_mask}
    for meta_key in audio_dicts[0].keys():
        if meta_key == "latents":
            continue
        vals = [d[meta_key] for d in audio_dicts]
        if all(torch.is_tensor(v) for v in vals):
            try:
                out[meta_key] = torch.stack(vals, dim=0)
            except RuntimeError:
                # Variable-shape: keep as list
                out[meta_key] = vals
        elif all(isinstance(v, (int, float, bool)) for v in vals):
            # Scalar metadata (num_time_steps, fps, ...): mirror default_collate
            # and produce a stacked 1-d tensor so consumers can do [0].item().
            out[meta_key] = torch.tensor(vals)
        else:
            out[meta_key] = vals
    return out


def _collate_dict_source(batch_list: list[dict]) -> dict:
    """Collate a list of dict samples by stacking each value along dim 0.

    Used for source outputs (motion_latents, conditions) and other dict-shaped
    items where all samples share the same structure. When tensors differ in
    shape along dim 0 (e.g. Gemma-encoded conditions have variable token
    lengths across samples: 128 for short prompts, 320 for long ones),
    right-pad to the batch max along dim 0 with the correct fill value
    (zeros for feature tensors, False for boolean masks) before stacking.
    This avoids the silent list-of-tensors fallback that downstream code
    cannot consume.
    """
    sample = batch_list[0]
    out: dict = {}
    for k in sample.keys():
        vals = [s[k] for s in batch_list]
        if all(torch.is_tensor(v) for v in vals):
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                # Variable first-dim length. Pad to the batch max along dim 0.
                max_len = max(v.shape[0] for v in vals)
                feature_dim = vals[0].shape[1:]
                fill_value = False if vals[0].dtype == torch.bool else 0
                padded = torch.full(
                    (len(vals), max_len, *feature_dim),
                    fill_value=fill_value,
                    dtype=vals[0].dtype,
                    device=vals[0].device,
                )
                for i, v in enumerate(vals):
                    padded[i, : v.shape[0]] = v
                out[k] = padded
        elif all(isinstance(v, (int, float, bool)) for v in vals):
            # Scalar metadata (num_frames, height, width, fps, ...): mirror
            # default_collate and produce a stacked 1-d tensor so consumers can
            # do [0].item().
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),  # random text embeddings
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),  # random attention mask
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        data_sources: dict[str, str] | list[str] | None = None,
        optional_sources: set[str] | list[str] | None = None,
        use_manifest_cache: bool = True,
        wait_for_manifest: bool = False,
    ) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
            optional_sources: Directory names (must match keys of `data_sources`)
                whose existence is NOT required at construction time. When a source
                is listed here but its directory is missing on disk, it is silently
                dropped from the dataset. Use this for modality-specific sources
                that are unused in certain training modes (e.g. motion_conditions
                in audio_only_training).
            use_manifest_cache: Cache the discovered relative paths in the data
                root so later processes and training runs avoid repeating the
                recursive scan and per-sample existence checks.
            wait_for_manifest: Wait for another process to create a valid
                manifest instead of scanning sources in this process.
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
            # Audio-only mode: motion_conditions is allowed to be missing
            dataset = PrecomputedDataset("data/", data_sources, optional_sources={"motion_conditions"})
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.optional_sources = self._normalize_optional_sources(optional_sources)
        self.source_paths = self._setup_source_paths()
        self._phoneme_pad_length = self._infer_phoneme_pad_length()
        self.manifest_path = self._get_manifest_path()
        self.manifest_available = False
        if wait_for_manifest:
            if not use_manifest_cache:
                raise ValueError("wait_for_manifest requires use_manifest_cache=True")
            self.sample_files = self._wait_for_manifest()
        elif use_manifest_cache:
            self.sample_files = self._load_or_discover_samples()
        else:
            self.sample_files = self._discover_samples()
        self._validate_setup()

    def _get_manifest_path(self) -> Path:
        """Return the cache path for this ordered source-to-output schema."""
        schema_json = json.dumps(self._manifest_schema(), sort_keys=True, separators=(",", ":"))
        schema_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()[:16]
        return self.data_root / MANIFEST_CACHE_DIR_NAME / f"samples-{schema_hash}.json"

    def _manifest_schema(self) -> dict:
        """Describe inputs that change how sample paths are discovered."""
        return {
            "data_sources": [[dir_name, output_key] for dir_name, output_key in self.data_sources.items()],
            "optional_sources": sorted(self.optional_sources),
        }

    def _source_fingerprints(self) -> dict[str, dict[str, int | bool | str | None]]:
        """Capture cheap invalidation metadata for each declared source directory."""
        fingerprints = {}
        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            exists = source_path.is_dir()
            fingerprints[dir_name] = {
                "exists": exists,
                "mtime_ns": source_path.stat().st_mtime_ns if exists else None,
            }
        ids_path = self.data_root / "sample_ids.txt"
        fingerprints["sample_ids.txt"] = {
            "exists": ids_path.is_file(),
            "sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest() if ids_path.is_file() else None,
        }
        return fingerprints

    def _load_or_discover_samples(self) -> dict[str, list[Path]]:
        sample_files = self._load_manifest()
        if sample_files is not None:
            self.manifest_available = True
            logger.info(f"Loaded dataset manifest from {self.manifest_path}")
            return sample_files

        logger.info(f"Dataset manifest missing or stale; scanning sources under {self.data_root}")
        sample_files = self._discover_samples()
        self.manifest_available = self._save_manifest(sample_files)
        return sample_files

    def _wait_for_manifest(self) -> dict[str, list[Path]]:
        """Poll the shared filesystem without holding an NCCL collective open."""
        started_at = time.monotonic()
        next_log_at = started_at
        while True:
            sample_files = self._load_manifest()
            if sample_files is not None:
                self.manifest_available = True
                logger.info(f"Loaded dataset manifest from {self.manifest_path}")
                return sample_files

            now = time.monotonic()
            if now >= next_log_at:
                elapsed = now - started_at
                logger.info(f"Waiting for rank 0 dataset manifest ({elapsed:.0f}s): {self.manifest_path}")
                next_log_at = now + 60.0
            if now - started_at >= MANIFEST_WAIT_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"Timed out waiting for rank 0 to create dataset manifest after "
                    f"{MANIFEST_WAIT_TIMEOUT_SECONDS}s: {self.manifest_path}"
                )
            time.sleep(MANIFEST_POLL_INTERVAL_SECONDS)

    def _load_manifest(self) -> dict[str, list[Path]] | None:  # noqa: PLR0911
        if not self.manifest_path.is_file():
            return None

        try:
            with self.manifest_path.open(encoding="utf-8") as manifest_file:
                payload = json.load(manifest_file)

            if not isinstance(payload, dict):
                return None
            if payload.get("version") != MANIFEST_VERSION:
                return None
            if payload.get("schema") != self._manifest_schema():
                return None
            if payload.get("source_fingerprints") != self._source_fingerprints():
                return None

            serialized_files = payload.get("sample_files")
            expected_keys = {self.data_sources[dir_name] for dir_name in self.source_paths}
            if not isinstance(serialized_files, dict) or set(serialized_files) != expected_keys:
                return None

            sample_files: dict[str, list[Path]] = {}
            for output_key, paths in serialized_files.items():
                if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                    return None
                relative_paths = [Path(path) for path in paths]
                if any(path.is_absolute() or ".." in path.parts for path in relative_paths):
                    return None
                sample_files[output_key] = relative_paths
            return sample_files
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable dataset manifest {self.manifest_path}: {exc}")
            return None

    def _save_manifest(self, sample_files: dict[str, list[Path]]) -> bool:
        payload = {
            "version": MANIFEST_VERSION,
            "schema": self._manifest_schema(),
            "source_fingerprints": self._source_fingerprints(),
            "sample_files": {
                output_key: [path.as_posix() for path in paths] for output_key, paths in sample_files.items()
            },
        }
        temporary_path = self.manifest_path.with_name(f".{self.manifest_path.name}.{os.getpid()}.tmp")

        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("w", encoding="utf-8") as manifest_file:
                json.dump(payload, manifest_file, separators=(",", ":"))
                manifest_file.flush()
                os.fsync(manifest_file.fileno())
            temporary_path.replace(self.manifest_path)
            logger.info(f"Saved dataset manifest to {self.manifest_path}")
            return True
        except OSError as exc:
            logger.warning(f"Could not save dataset manifest {self.manifest_path}: {exc}")
            return False
        finally:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)

    def _infer_phoneme_pad_length(self) -> int:
        if "phonemes" not in self.data_sources:
            return 0

        env_value = os.getenv("MOTIONSPEECH_PHONEME_PAD_LENGTH")
        if env_value is not None:
            try:
                return max(int(env_value), 1)
            except ValueError as exc:
                raise ValueError("Invalid MOTIONSPEECH_PHONEME_PAD_LENGTH value. Use an integer.") from exc

        phoneme_dir = self.source_paths.get("phonemes")
        if phoneme_dir is None or not phoneme_dir.exists():
            return 310

        for file_path in phoneme_dir.glob("**/*.npy"):
            try:
                data = np.load(file_path)
                if data.ndim == 1 and data.size > 0:
                    return int(data.shape[0])
            except Exception:
                continue

        return 310

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    @staticmethod
    def _normalize_optional_sources(optional_sources: set[str] | list[str] | None) -> set[str]:
        """Normalize optional_sources input to a set of directory names."""
        if optional_sources is None:
            return set()
        if isinstance(optional_sources, (set, list, tuple)):
            return {str(s) for s in optional_sources}
        raise TypeError(f"optional_sources must be set, list, or None, got {type(optional_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths.

        Sources listed in `self.optional_sources` are allowed to be missing on
        disk; when so, they are silently dropped (the strategy that consumes
        `batch[output_key]` must be tolerant of the missing key — typically by
        checking `audio_only_training`).
        """
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name

            if not source_path.exists():
                if dir_name in self.optional_sources:
                    logger.info(
                        f"Optional {dir_name} directory not found at {source_path}; "
                        f"skipping (strategy must tolerate missing key)."
                    )
                    continue
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

            source_paths[dir_name] = source_path

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources."""
        # Pick a reference source that actually exists on disk; in audio-only
        # mode the reference may be the first source rather than 'latents' if
        # we don't want to depend on motion data being present.
        reference_key = self._pick_reference_source()
        data_path = self.source_paths[reference_key]
        data_files = list(data_path.glob("**/*.pt"))
        ids_path = self.data_root / "sample_ids.txt"
        if ids_path.is_file():
            allowed_ids = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
            selected = []
            for path in data_files:
                sample_id = path.relative_to(data_path).with_suffix("").as_posix()
                base_id = sample_id
                for suffix in ("_l1", "_l2", "_speech"):
                    if sample_id.endswith(suffix):
                        base_id = sample_id.removesuffix(suffix)
                        break
                if sample_id in allowed_ids or base_id in allowed_ids:
                    selected.append(path)
            data_files = selected

        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        # Initialize sample files dict — only for sources that actually exist on disk.
        sample_files = {
            self.data_sources[dir_name]: []
            for dir_name in self.source_paths
            if dir_name in self.data_sources
        }

        # For each data file, find corresponding files in other sources
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check if corresponding files exist in ALL present sources
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)

        return sample_files

    def _pick_reference_source(self) -> str:
        """Choose which data source to use as the sample-id reference.

        Prefers 'latents' (the canonical audio/motion latents). If 'latents' is
        declared but missing on disk (audio-only mode where motion_conditions
        was dropped), falls back to the first available source that exists.
        """
        if "latents" in self.source_paths:
            return "latents"
        return next(iter(self.source_paths))

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        """Check if corresponding files exist in all (present) data sources.

        Sources that were dropped because they were optional and missing are
        skipped here. Sources that are still declared but whose directory is
        absent at sample-discover time are reported as missing (defensive).
        """
        for dir_name in self.data_sources:
            if dir_name not in self.source_paths:
                # Already dropped optional source — treat as "exists" so the
                # sample is not filtered out just because the modality is off.
                continue
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                # Demoted to debug — this fires once per missing sample during
                # dataset construction (for Emilia: ~77 of 509k samples are
                # filtered out here); printing at WARNING spammed the log on
                # every rank.
                logger.debug(
                    f"No matching {dir_name} file found for: {data_file.name} (expected in: {expected_path})"
                )
                return False

        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For phonemes, handle .npy extension instead of .pt
        if dir_name == "phonemes":
            if data_file.name.startswith("latent_"):
                # latent_X.pt -> phoneme_X.npy
                return source_path / f"phoneme_{data_file.stem[7:]}.npy"
            else:
                # X.pt -> X.npy
                return source_path / rel_path.with_suffix(".npy")

        # For audcaps, handle legacy naming where latent_X.pt maps to audcap_X.pt
        # (audcaps/ stores Gemma-encoded voice/timbre features, one .pt per sample).
        if dir_name == "audcaps" and data_file.name.startswith("latent_"):
            return source_path / f"audcap_{data_file.stem[7:]}.pt"

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking.

        Sources that were dropped (optional and missing on disk) are skipped:
        their output_key is not appended to, so `sample_files` only contains
        the keys for sources we actually have on disk.
        """
        for dir_name, output_key in self.data_sources.items():
            if dir_name not in self.source_paths:
                continue
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")

        # Verify all output keys have the same number of samples
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            # Skip sources that were declared optional and missing on disk.
            if dir_name not in self.source_paths:
                continue
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                # Load phoneme data from .npy files
                if dir_name == "phonemes":
                    data = np.load(file_path)
                    # Convert to torch tensor and ensure int64 type for embedding lookup
                    data = torch.from_numpy(data).long()
                    if data.ndim != 1:
                        data = data.flatten()

                    if self._phoneme_pad_length > 0:
                        if data.numel() == 0:
                            data = torch.zeros(self._phoneme_pad_length, dtype=torch.long)
                        elif data.numel() < self._phoneme_pad_length:
                            pad_len = self._phoneme_pad_length - data.numel()
                            data = torch.nn.functional.pad(data, (0, pad_len), value=0)
                        elif data.numel() > self._phoneme_pad_length:
                            data = data[: self._phoneme_pad_length]
                else:
                    # Load other data from .pt files
                    data = torch.load(file_path, map_location="cpu", weights_only=True)

                    # Normalize video latent format if this is a latent source
                    if "latent" in dir_name.lower():
                        data = self._normalize_video_latents(data)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def collate_precomputed_batch(batch_list: list[dict]) -> dict:
        """Collate a list of PrecomputedDataset samples into a batched dict.

        Pads variable-length tensors (audio latents time dim) to the max length
        within the batch. For audio sources, attaches an `audio_loss_mask`
        boolean tensor of shape (B, max_T) marking valid tokens; padded
        positions are False.

        Each top-level sample key has a known layout:
          - `audio_latents`: dict with key `latents` (variable T) + scalar metadata
          - `motion_latents`: dict with key `latents` (fixed shape) + metadata
          - `conditions`: dict of fixed-shape tensors (encoded_features, masks, ...)
          - `phonemes`: tensor (B, pad_length) int64 — fixed shape
          - `idx`: int
        """
        if not batch_list:
            return {}

        result: dict = {}
        for key in batch_list[0].keys():
            sample = batch_list[0][key]

            if key == "audio_latents":
                result[key] = _collate_audio_source(batch_list)
            elif isinstance(sample, dict) and "latents" in sample and torch.is_tensor(sample["latents"]):
                # Other latent source (motion_latents) — stack via dict helper
                result[key] = _collate_dict_source([s[key] for s in batch_list])
            elif isinstance(sample, dict):
                # Conditions: dict of fixed-shape tensors — stack each value
                result[key] = _collate_dict_source([s[key] for s in batch_list])
            elif key == "idx":
                result[key] = torch.tensor([s[key] for s in batch_list], dtype=torch.long)
            elif torch.is_tensor(sample):
                result[key] = torch.stack([s[key] for s in batch_list], dim=0)
            else:
                result[key] = [s[key] for s in batch_list]

        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video/motion latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.

        Handles two cases:
        1. Video latents: legacy format [seq_len, C] -> [C, F, H, W]
        2. Motion latents: [C, T] -> [C, T, 1, 1] (motion has no spatial dimensions)

        Audio latents are NOT processed here (they have different format and metadata).
        """
        latents = data["latents"]

        # Check if this data has video/motion metadata (vs audio metadata)
        # Audio latents don't have 'num_frames', 'height', 'width' fields
        if "num_frames" not in data:
            # This is audio or other non-video data, return as-is
            return data

        num_frames = data["num_frames"]
        height = data["height"]
        width = data["width"]

        # Check if latents are in 2D format (need normalization)
        if latents.dim() == 2:
            # Check if this is motion data (no spatial dimensions)
            if height == 1 and width == 1:
                # Motion latents format: [C, T]
                # Already in correct channel-first format, just need to add spatial dims
                # [C, T] -> [C, T, 1, 1]
                latents = latents.unsqueeze(-1).unsqueeze(-1)
            else:
                # Legacy video format: [seq_len, C] where seq_len = F * H * W
                # Unpatchify: [seq_len, C] -> [C, F, H, W]
                latents = rearrange(
                    latents,
                    "(f h w) c -> c f h w",
                    f=num_frames,
                    h=height,
                    w=width,
                )

            # Update the data dict with normalized latents
            data = data.copy()
            data["latents"] = latents

        return data
