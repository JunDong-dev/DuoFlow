# Modified for DuoFlow; see NOTICE.md.
import torch

class PixArtAlphaTextProjection(torch.nn.Module):
    """Project caption embeddings with an optional residual linear branch.

    Adapted from PixArt-alpha:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py

    With ``act_fn="none"``, use a single linear projection to ``out_features``.
    Otherwise, apply two linear layers around the selected activation, with
    an optional parallel linear projection when ``use_skip=True``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
        use_skip: bool = False,
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        if act_fn == "none":
            # Pure linear projection. Collapse linear_1 + linear_2 into a single
            # Linear(in_features -> out_features). use_skip is meaningless here.
            if use_skip:
                raise ValueError("use_skip=True is incompatible with act_fn='none' (no GELU to bypass)")
            self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
            self.act_1 = torch.nn.Identity()
            self.linear_2 = None
            self.linear_skip = None
            self.use_skip = False
            self._is_pure_linear = True
        else:
            self._is_pure_linear = False
            self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
            if act_fn == "gelu_tanh":
                self.act_1 = torch.nn.GELU(approximate="tanh")
            elif act_fn == "silu":
                self.act_1 = torch.nn.SiLU()
            else:
                raise ValueError(f"Unknown activation function: {act_fn}")
            self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)
            self.use_skip = use_skip
            self.linear_skip = (
                torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
                if use_skip
                else None
            )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        if self._is_pure_linear:
            return hidden_states
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        if self.use_skip:
            hidden_states = hidden_states + self.linear_skip(caption)
        return hidden_states
