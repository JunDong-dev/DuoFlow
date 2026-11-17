"""Regression for CPU Gemma backbones in checkpoints with unused meta modules."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from motionspeech_trainer.model_loader import load_text_encoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_backbone_uses_requested_device_with_unused_meta_side_modules():
    encoder = SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        unused_projection=torch.nn.Linear(2, 2, device="meta"),
    )
    with tempfile.TemporaryDirectory() as folder:
        with patch("motionspeech_core.loader.single_gpu_model_builder.SingleGPUModelBuilder") as builder:
            builder.return_value.build.return_value = encoder
            with patch("motionspeech_core.text_encoders.gemma.encoders.base_encoder.module_ops_from_gemma_root", return_value=()):
                result = load_text_encoder(Path(folder) / "model.safetensors", folder,
                                           device="cuda:0", dtype=torch.float32)
    output = result.model(torch.ones(1, 2, device="cuda:0"))
    assert output.device.type == "cuda"
    assert result.unused_projection.weight.is_meta
