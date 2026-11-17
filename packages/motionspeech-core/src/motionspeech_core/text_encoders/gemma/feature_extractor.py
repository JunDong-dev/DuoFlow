# Modified for DuoFlow; see NOTICE.md.
import torch

from motionspeech_core.model.model_protocol import ModelConfigurator


class GemmaFeaturesExtractorProjLinear(torch.nn.Module, ModelConfigurator["GemmaFeaturesExtractorProjLinear"]):
    """
    Feature extractor module for Gemma models.
    This module applies a single linear projection to the input tensor.
    It expects a flattened feature tensor of shape (batch_size, seq_len, 3840*num_layers)
    and projects it to (batch_size, seq_len, 3840).

    Args:
        num_layers: Number of Gemma hidden-state layers to aggregate.
            Default is 49 (all layers).  Set to 1 to use only the last layer.
    Attributes:
        aggregate_embed (torch.nn.Linear): Linear projection layer.
        num_layers (int): Number of layers being aggregated.
    """

    HIDDEN_DIM: int = 3840  # Gemma per-layer hidden dimension

    def __init__(self, num_layers: int = 49) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.aggregate_embed = torch.nn.Linear(
            self.HIDDEN_DIM * num_layers, self.HIDDEN_DIM, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature extractor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, 3840 * num_layers).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, 3840).
        """
        return self.aggregate_embed(x)

    @classmethod
    def from_config(cls: type["GemmaFeaturesExtractorProjLinear"], _config: dict) -> "GemmaFeaturesExtractorProjLinear":
        num_layers = _config.get("num_layers", 49)
        return cls(num_layers=num_layers)
