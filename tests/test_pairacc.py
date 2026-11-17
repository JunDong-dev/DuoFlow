"""Check ranking, duplicate masking, ties, chunking and representation shapes."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/clam_h"))
from motionspeech_eval.common import global_pair_accuracy, split_text


def test_correct_ranking_and_inversion():
    features = torch.eye(3)
    captions = ["walk", "wave", "sit"]
    assert global_pair_accuracy(features, features, captions) == 100
    assert global_pair_accuracy(features, features.roll(1, 0), captions) == 25


def test_ties_receive_half_credit():
    features = torch.ones(3, 2)
    assert global_pair_accuracy(features, features, ["a", "b", "c"]) == 50


def test_equivalent_captions_are_excluded():
    motion = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    captions = [" A walk  ", "a WALK", "a wave"]
    assert global_pair_accuracy(motion, motion, captions) == 100
    with pytest.raises(ValueError, match="No non-equivalent"):
        global_pair_accuracy(motion[:2], motion[:2], captions[:2])


def test_long_text_is_chunked_without_losing_words():
    text = " ".join(f"word{i}" for i in range(90)) + ". One more sentence."
    chunks = split_text(text)
    assert all(len(chunk.split()) <= 32 for chunk in chunks)
    assert " ".join(chunks) == text


def test_block_size_preserves_result():
    features = torch.eye(5)
    captions = [str(index) for index in range(5)]
    assert global_pair_accuracy(features, features, captions, block=2) == 100
