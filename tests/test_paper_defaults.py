"""Prevent default entry points from reverting to the earlier architecture."""

import torch

from motionspeech_core.model.transformer.model import MotionSpeechModel
from motionspeech_core.model.transformer.model_configurator import MotionSpeechModelConfigurator
from motionspeech_trainer.config import ModelConfig
from motionspeech_trainer.training_strategies.text_to_motion_audio import TextToMotionAudioConfig


def test_training_defaults_supervise_all_eight_blocks():
    model = ModelConfig()
    strategy = TextToMotionAudioConfig()
    assert model.num_transformer_layers == 8
    assert model.aux_ctc_tap_blocks == list(range(8))
    assert strategy.aux_ctc_weight == 1
    assert strategy.aux_ctc_hidden_dim == 1024
    assert ModelConfig(num_transformer_layers=1).aux_ctc_tap_blocks == [0]
    assert ModelConfig(aux_ctc_tap_block=3).aux_ctc_tap_blocks is None


def test_coupled_model_defaults_match_paper_widths_and_ctc_taps():
    with torch.device("meta"):
        direct = MotionSpeechModel()
        configured = MotionSpeechModelConfigurator.from_config({"transformer": {}})
    for model in (direct, configured):
        assert len(model.transformer_blocks) == 8
        assert model.inner_dim == 1024
        assert model.audio_inner_dim == 1024
        assert model.aux_ctc_tap_blocks == list(range(8))


def test_explicit_small_architecture_and_disabled_taps_remain_supported():
    with torch.device("meta"):
        model = MotionSpeechModel(num_layers=1, num_attention_heads=1, attention_head_dim=8,
                                  audio_num_attention_heads=1, audio_attention_head_dim=8,
                                  cross_attention_dim=8, audio_cross_attention_dim=8,
                                  aux_ctc_tap_blocks=None)
    assert model.audio_inner_dim == 8
    assert model.aux_ctc_tap_blocks is None
