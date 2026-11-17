# Modified for DuoFlow; see NOTICE.md.
import torch

from motionspeech_core.loader.fuse_loras import fused_add_round_launch
from motionspeech_core.loader.module_ops import ModuleOps
from motionspeech_core.loader.sd_ops import KeyValueOperationResult, SDOps
from motionspeech_core.model.model_protocol import ModelConfigurator
from motionspeech_core.model.transformer.attention import AttentionFunction
from motionspeech_core.model.transformer.model import MotionSpeechModel, MotionSpeechModelType
from motionspeech_core.model.transformer.rope import MotionSpeechRopeType
from motionspeech_core.utils import check_config_value


class MotionSpeechModelConfigurator(ModelConfigurator[MotionSpeechModel]):
    """
    Configurator for DuoFlow model.
    Used to create an DuoFlow model from a configuration dictionary.
    """

    @classmethod
    def from_config(cls: type[MotionSpeechModel], config: dict) -> MotionSpeechModel:
        config = dict(config.get("transformer", {}))

        def check_default(key, expected):
            config.setdefault(key, expected)
            check_config_value(config, key, expected)

        check_default("dropout", 0.0)
        check_default("attention_bias", True)
        check_default("num_vector_embeds", None)
        check_default("activation_fn", "gelu-approximate")
        check_default("num_embeds_ada_norm", 1000)
        check_default("use_linear_projection", False)
        check_default("only_cross_attention", False)
        check_default("cross_attention_norm", True)
        check_default("double_self_attention", False)
        check_default("upcast_attention", False)
        check_default("standardization_norm", "rms_norm")
        check_default("norm_elementwise_affine", False)
        check_default("qk_norm", "rms_norm")
        check_default("positional_embedding_type", "rope")
        check_default("use_audio_video_cross_attention", True)
        check_default("share_ff", False)
        check_default("av_cross_ada_norm", True)
        check_default("use_middle_indices_grid", True)

        return MotionSpeechModel(
            model_type=MotionSpeechModelType.AudioVideo,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 32),
            in_channels=config.get("in_channels", 256),
            out_channels=config.get("out_channels", 256),
            num_layers=config.get("num_layers", 8),
            cross_attention_dim=config.get("cross_attention_dim", 1024),
            norm_eps=config.get("norm_eps", 1e-06),
            attention_type=AttentionFunction(config.get("attention_type", "default")),
            caption_channels=config.get("caption_channels", 3840),
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 32),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 1024),
            audio_positional_embedding_max_pos=config.get("audio_positional_embedding_max_pos", [20]),
            av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1),
            audio_caption_projection_use_skip=config.get("audio_caption_projection_use_skip", False),
            audio_caption_projection_act_fn=config.get("audio_caption_projection_act_fn", "gelu_tanh"),
            audio_phoneme_caption_projection_act_fn=config.get("audio_phoneme_caption_projection_act_fn", "gelu_tanh"),
            audio_audcap_caption_projection_act_fn=config.get("audio_audcap_caption_projection_act_fn", "gelu_tanh"),
            video_caption_projection_use_skip=config.get("video_caption_projection_use_skip", False),
            video_caption_projection_act_fn=config.get("video_caption_projection_act_fn", "gelu_tanh"),
            use_text_cross_attention=config.get("use_text_cross_attention", True),
            rope_type=MotionSpeechRopeType(config.get("rope_type", "interleaved")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            use_phoneme_time_anchor=config.get("use_phoneme_time_anchor", False),
            aux_ctc_tap_block=config.get("aux_ctc_tap_block", None),
            aux_ctc_tap_blocks=config.get("aux_ctc_tap_blocks", "all"),
        )


class MotionSpeechVideoOnlyModelConfigurator(ModelConfigurator[MotionSpeechModel]):
    """
    Configurator for DuoFlow video only model.
    Used to create an DuoFlow video only model from a configuration dictionary.
    """

    @classmethod
    def from_config(cls: type[MotionSpeechModel], config: dict) -> MotionSpeechModel:
        config = config.get("transformer", {})

        check_config_value(config, "dropout", 0.0)
        check_config_value(config, "attention_bias", True)
        check_config_value(config, "num_vector_embeds", None)
        check_config_value(config, "activation_fn", "gelu-approximate")
        check_config_value(config, "num_embeds_ada_norm", 1000)
        check_config_value(config, "use_linear_projection", False)
        check_config_value(config, "only_cross_attention", False)
        check_config_value(config, "cross_attention_norm", True)
        check_config_value(config, "double_self_attention", False)
        check_config_value(config, "upcast_attention", False)
        check_config_value(config, "standardization_norm", "rms_norm")
        check_config_value(config, "norm_elementwise_affine", False)
        check_config_value(config, "qk_norm", "rms_norm")
        check_config_value(config, "positional_embedding_type", "rope")
        check_config_value(config, "use_middle_indices_grid", True)

        return MotionSpeechModel(
            model_type=MotionSpeechModelType.VideoOnly,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            attention_type=AttentionFunction(config.get("attention_type", "default")),
            caption_channels=config.get("caption_channels", 3840),
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get("positional_embedding_max_pos", [20, 2048, 2048]),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            rope_type=MotionSpeechRopeType(config.get("rope_type", "interleaved")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
        )


def _naive_weight_or_bias_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """
    Downcast the weight or bias to the float8_e4m3fn dtype.
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


def _upcast_and_round(
    weight: torch.Tensor, dtype: torch.dtype, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.Tensor:
    """
    Upcast the weight to the given dtype and optionally apply stochastic rounding.
    Input weight needs to have float8_e4m3fn or float8_e5m2 dtype.
    """
    if not with_stochastic_rounding:
        return weight.to(dtype)
    return fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


def replace_fwd_with_upcast(layer: torch.nn.Linear, with_stochastic_rounding: bool = False, seed: int = 0) -> None:
    """
    Replace linear.forward and rms_norm.forward with a version that:
      - upcasts weight and bias to input's dtype
      - returns F.linear or F.rms_norm calculated in that dtype
    """

    layer.original_forward = layer.forward

    def new_linear_forward(*args, **_kwargs) -> torch.Tensor:
        # assume first arg is the input tensor
        x = args[0]
        w_up = _upcast_and_round(layer.weight, x.dtype, with_stochastic_rounding, seed)
        b_up = None

        if layer.bias is not None:
            b_up = _upcast_and_round(layer.bias, x.dtype, with_stochastic_rounding, seed)

        return torch.nn.functional.linear(x, w_up, b_up)

    layer.forward = new_linear_forward


def amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the model's Linear and RMSNorm layers to forward
    with upcast and optional stochastic rounding.
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear)):
            replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


MOTIONSPEECH_MODEL_COMFY_RENAMING_MAP = (
    SDOps("MOTIONSPEECH_MODEL_COMFY_PREFIX_MAP")
    .with_matching(prefix="model.diffusion_model.")
    .with_replacement("model.diffusion_model.", "")
)

MOTIONSPEECH_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP = (
    SDOps("MOTIONSPEECH_MODEL_COMFY_PREFIX_MAP")
    .with_matching(prefix="model.diffusion_model.")
    .with_replacement("model.diffusion_model.", "")
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.0.proj.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.0.proj.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.2.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.2.bias", operation=_naive_weight_or_bias_downcast
    )
)

UPCAST_DURING_INFERENCE = ModuleOps(
    name="upcast_fp8_during_linear_forward",
    matcher=lambda model: isinstance(model, MotionSpeechModel),
    mutator=lambda model: amend_forward_with_upcast(model, False),
)


class UpcastWithStochasticRounding(ModuleOps):
    """
    ModuleOps for upcasting the model's float8_e4m3fn weights and biases to the bfloat16 dtype
    and applying stochastic rounding during linear forward.
    """

    def __new__(cls, seed: int = 0):
        return super().__new__(
            cls,
            name="upcast_fp8_during_linear_forward_with_stochastic_rounding",
            matcher=lambda model: isinstance(model, MotionSpeechModel),
            mutator=lambda model: amend_forward_with_upcast(model, True, seed),
        )
