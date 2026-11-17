"""Regression tests for dataset provenance and train/validation isolation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from data_splits import claim_preprocessing_scope, validate_preprocessed_split, validate_training_data


class SplitTests(unittest.TestCase):
    def test_cache_cannot_change_source_split_or_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache, source = root / "cache", root / "raw"
            claim_preprocessing_scope(cache, source, "train", False)
            claim_preprocessing_scope(cache, source, "train", False)
            for other_source, split, audio_only in (
                (source, "test", False), (root / "other", "train", False), (source, "train", True),
            ):
                with self.assertRaises(ValueError):
                    claim_preprocessing_scope(cache, other_source, split, audio_only)

    def test_legacy_split_metadata_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "preprocess_summary.json").write_text(json.dumps({"split": "train"}))
            validate_preprocessed_split(root, "train")
            with self.assertRaises(ValueError):
                validate_preprocessed_split(root, "test")

    def test_recursive_training_root_cannot_contain_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for train, val in ((root, root), (root, root / "val"), (root / "train", root)):
                with self.assertRaises(ValueError):
                    validate_training_data({
                        "data": {"preprocessed_data_root": str(train)},
                        "validation": {"val_preprocessed_data_root": str(val)},
                    })

    def test_sibling_splits_and_id_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, val = root / "train", root / "val"
            claim_preprocessing_scope(train, root / "raw", "train", False)
            claim_preprocessing_scope(val, root / "raw", "val", False)
            (train / "sample_ids.txt").write_text("one\ntwo\n")
            (val / "sample_ids.txt").write_text("three\n")
            config = {"data": {"preprocessed_data_root": str(train)},
                      "validation": {"val_preprocessed_data_root": str(val)}}
            validate_training_data(config)
            (val / "sample_ids.txt").write_text("two\n")
            with self.assertRaises(ValueError):
                validate_training_data(config)

    def test_replay_requires_separate_training_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"data": {"preprocessed_data_root": str(root / "paired/train"), "replay_batch_prob": 0.1},
                      "validation": {"val_preprocessed_data_root": str(root / "paired/val")}}
            with self.assertRaises(ValueError):
                validate_training_data(config)
            config["data"]["replay_preprocessed_data_root"] = config["data"]["preprocessed_data_root"]
            with self.assertRaises(ValueError):
                validate_training_data(config)
            config["data"]["replay_preprocessed_data_root"] = str(root / "speech/train")
            claim_preprocessing_scope(root / "speech/train", root / "raw_speech", "train", True)
            validate_training_data(config)

    def test_unknown_existing_cache_is_not_silently_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "audio_latents").mkdir()
            (root / "audio_latents/one.pt").touch()
            with self.assertRaises(ValueError):
                claim_preprocessing_scope(root, root / "raw", "test", False)


if __name__ == "__main__":
    unittest.main()
