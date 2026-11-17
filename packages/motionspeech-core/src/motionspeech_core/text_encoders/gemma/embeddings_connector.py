# Modified for DuoFlow; see NOTICE.md.
import os
import torch

from motionspeech_core.model.model_protocol import ModelConfigurator
from motionspeech_core.model.transformer.attention import Attention
from motionspeech_core.model.transformer.feed_forward import FeedForward
from motionspeech_core.model.transformer.rope import (
    MotionSpeechRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)
from motionspeech_core.utils import rms_norm


class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        rope_type: MotionSpeechRopeType = MotionSpeechRopeType.INTERLEAVED,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
        )

        self.ff = FeedForward(
            dim,
            dim_out=dim,
            mult=ff_mult,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    """Transform serialized multimodal embeddings with 1D transformer blocks.

    Valid tokens are packed before a configurable number of learnable register
    tokens. Rotary position embeddings and causal positioning are optional.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        input_dim: int | None = None,
        output_dim: int | None = None,
        ff_mult: int = 4,
        zero_init_output: bool = False,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: MotionSpeechRopeType = MotionSpeechRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.input_dim = self.inner_dim if input_dim is None else input_dim
        self.output_dim = self.inner_dim if output_dim is None else output_dim
        if min(self.input_dim, self.output_dim, self.inner_dim, num_layers, ff_mult) < 1:
            raise ValueError("Connector dimensions, layers, and ff_mult must all be positive")

        self.input_proj = (
            torch.nn.Identity()
            if self.input_dim == self.inner_dim
            else torch.nn.Linear(self.input_dim, self.inner_dim)
        )
        self.output_proj = (
            torch.nn.Identity()
            if self.output_dim == self.inner_dim
            else torch.nn.Linear(self.inner_dim, self.output_dim)
        )
        if zero_init_output and isinstance(self.output_proj, torch.nn.Linear):
            torch.nn.init.zeros_(self.output_proj.weight)
            torch.nn.init.zeros_(self.output_proj.bias)
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        )
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope

        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ff_mult=ff_mult,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.float32) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack valid tokens, append K registers, and pad to the batch maximum."""
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            raise ValueError("thinking-token replacement requires an attention mask")

        flat_mask = attention_mask.reshape(batch_size, -1)
        if flat_mask.dtype == torch.bool or not torch.is_floating_point(flat_mask):
            binary = flat_mask > 0
        else:
            binary = flat_mask >= -9000.0
        if binary.shape[1] != seq_len:
            raise ValueError(
                "connector attention mask length does not match hidden states: "
                f"mask={tuple(binary.shape)}, hidden={tuple(hidden_states.shape)}"
            )
        valid_lengths = binary.sum(dim=1, dtype=torch.int64)
        packed_seq_len = int(valid_lengths.max().item()) + self.num_learnable_registers
        packed = hidden_states.new_zeros((batch_size, packed_seq_len, hidden_states.shape[-1]))
        packed_binary = torch.zeros(
            (batch_size, packed_seq_len), dtype=torch.bool, device=hidden_states.device
        )
        register_bank = self.learnable_registers.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for batch_idx in range(batch_size):
            valid_tokens = hidden_states[batch_idx, binary[batch_idx]]
            valid_count = valid_tokens.shape[0]
            if valid_count:
                packed[batch_idx, :valid_count] = valid_tokens
            register_end = valid_count + self.num_learnable_registers
            packed[batch_idx, valid_count:register_end] = register_bank
            packed_binary[batch_idx, :register_end] = True
        hidden_states = packed

        mask_dtype = hidden_states.dtype
        invalid_value = -torch.finfo(mask_dtype).max
        attention_mask = torch.where(
            packed_binary[:, None, None, :],
            torch.tensor(-1e-5, dtype=mask_dtype, device=hidden_states.device),
            torch.tensor(invalid_value, dtype=mask_dtype, device=hidden_states.device),
        )

        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformed embeddings and their attention mask."""
        if hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                "Connector input width mismatch: "
                f"got {hidden_states.shape[-1]}, expected {self.input_dim}"
            )
        hidden_states = self.input_proj(hidden_states)

        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :]
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)

        return hidden_states, attention_mask


class Embeddings1DConnectorConfigurator(ModelConfigurator[Embeddings1DConnector]):
    @classmethod
    def from_config(cls: type[Embeddings1DConnector], config: dict) -> Embeddings1DConnector:
        config = config.get("transformer", {})
        rope_type = MotionSpeechRopeType(config.get("rope_type", "interleaved"))
        double_precision_rope = config.get("frequencies_precision", False) == "float64"
        pe_max_pos = config.get("connector_positional_embedding_max_pos", [1])
        attention_head_dim = config.get("connector_attention_head_dim", 128)
        num_attention_heads = config.get("connector_num_attention_heads", 30)
        num_layers = config.get("connector_num_layers", 2)
        input_dim = config.get("connector_input_dim")
        output_dim = config.get("connector_output_dim")
        ff_mult = config.get("connector_ff_mult", 4)
        zero_init_output = config.get("connector_zero_init_output", False)

        # Support custom num_learnable_registers from config
        # Set to None to disable learnable registers and preserve all text tokens
        num_learnable_registers = config.get("num_learnable_registers", 128)

        env_override = os.getenv("MOTIONSPEECH_NUM_LEARNABLE_REGISTERS")
        if env_override is not None:
            lowered = env_override.strip().lower()
            if lowered in {"none", "null", "nil"}:
                num_learnable_registers = None
            else:
                try:
                    num_learnable_registers = int(lowered)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid MOTIONSPEECH_NUM_LEARNABLE_REGISTERS value. Use an integer or 'none'."
                    ) from exc

        connector = Embeddings1DConnector(
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            input_dim=input_dim,
            output_dim=output_dim,
            ff_mult=ff_mult,
            zero_init_output=zero_init_output,
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            num_learnable_registers=num_learnable_registers,
        )
        return connector
