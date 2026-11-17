# Modified for DuoFlow; see NOTICE.md.
"""Training strategies for different conditioning modes.
This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio)
- Video-to-video training (IC-LoRA mode with reference videos)
- Motion-to-audio training (motion-conditioned audio generation)
- Text-to-motion-audio training (joint motion and audio generation from text)
Each strategy encapsulates the specific logic for preparing model inputs and computing loss.
"""

from motionspeech_trainer import logger
from motionspeech_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from motionspeech_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
from motionspeech_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy
from motionspeech_trainer.training_strategies.motion_to_audio import MotionToAudioConfig, MotionToAudioStrategy
from motionspeech_trainer.training_strategies.text_to_motion_audio import TextToMotionAudioConfig, TextToMotionAudioStrategy

# Type alias for all strategy config types
TrainingStrategyConfig = TextToVideoConfig | VideoToVideoConfig | MotionToAudioConfig | TextToMotionAudioConfig

__all__ = [
    "DEFAULT_FPS",
    "VIDEO_SCALE_FACTORS",
    "ModelInputs",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "VideoToVideoConfig",
    "VideoToVideoStrategy",
    "MotionToAudioConfig",
    "MotionToAudioStrategy",
    "TextToMotionAudioConfig",
    "TextToMotionAudioStrategy",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.
    The strategy is determined by the `name` field in the configuration.
    Args:
        config: Strategy-specific configuration with a `name` field
    Returns:
        The appropriate training strategy instance
    Raises:
        ValueError: If strategy name is not supported
    """

    match config:
        case TextToVideoConfig():
            strategy = TextToVideoStrategy(config)
        case VideoToVideoConfig():
            strategy = VideoToVideoStrategy(config)
        case MotionToAudioConfig():
            strategy = MotionToAudioStrategy(config)
        case TextToMotionAudioConfig():
            strategy = TextToMotionAudioStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    audio_mode = "(audio enabled 🔈)" if getattr(config, "with_audio", False) else "(audio disabled 🔇)"
    logger.debug(f"🎯 Using {strategy.__class__.__name__} training strategy {audio_mode}")
    return strategy
