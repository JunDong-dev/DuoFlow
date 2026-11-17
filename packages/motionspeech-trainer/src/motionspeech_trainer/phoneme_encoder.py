# Modified for DuoFlow; see NOTICE.md.
"""
Phoneme Encoder for text-to-audio generation (ported from AudioLDM2)

Based on VITS-style phoneme encoder with multi-head attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers import VitsModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention for phoneme encoding"""
    def __init__(self, n_channels, n_heads, p_dropout=0.0):
        super().__init__()
        assert n_channels % n_heads == 0

        self.n_channels = n_channels
        self.n_heads = n_heads
        self.k_channels = n_channels // n_heads

        self.conv_q = nn.Conv1d(n_channels, n_channels, 1)
        self.conv_k = nn.Conv1d(n_channels, n_channels, 1)
        self.conv_v = nn.Conv1d(n_channels, n_channels, 1)
        self.conv_o = nn.Conv1d(n_channels, n_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        # Split heads
        bs, c, t = q.size()
        q = q.view(bs, self.n_heads, self.k_channels, t).transpose(2, 3)
        k = k.view(bs, self.n_heads, self.k_channels, t).transpose(2, 3)
        v = v.view(bs, self.n_heads, self.k_channels, t).transpose(2, 3)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.k_channels ** 0.5)

        if attn_mask is not None:
            # attn_mask from TextEncoder: [B, 1, T] → need [B, 1, 1, T] for
            # broadcasting with scores [B, n_heads, T, T] (mask padding keys)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(2, 3).contiguous().view(bs, c, t)
        output = self.conv_o(output)

        return output


class FFN(nn.Module):
    """Feed-forward network"""
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask=None):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer with attention and FFN"""
    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size=3, p_dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_channels, n_heads, p_dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.ffn = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x_mask)
        x = x + self.drop(attn_output)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)

        # FFN
        ffn_output = self.ffn(x, x_mask)
        x = x + self.drop(ffn_output)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)

        return x


class TextEncoder(nn.Module):
    """VITS-style text encoder for phoneme sequences"""
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout)
            for _ in range(n_layers)
        ])

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths=None):
        # x: [bs, seq_len] phoneme IDs
        x = self.emb(x) * (self.hidden_channels ** 0.5)  # [bs, seq_len, hidden_channels]
        x = x.transpose(1, 2)  # [bs, hidden_channels, seq_len]

        # Create mask
        x_mask = None
        if x_lengths is not None:
            x_mask = torch.unsqueeze(self._sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # Encoder layers
        for layer in self.encoder:
            x = layer(x, x_mask)

        # Project to output dimension
        x = self.proj(x)

        if x_mask is not None:
            x = x * x_mask

        return x  # [bs, out_channels, seq_len]

    def _sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class PhonemeEncoder(nn.Module):
    """Phoneme encoder wrapper with learnable positional embeddings"""
    def __init__(
        self,
        vocabs_size: int = 183,
        pad_length: int = 310,
        pad_token_id: int = 0,
        out_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        add_blank: bool = False,
    ):
        super().__init__()
        self.pad_length = pad_length
        self.pad_token_id = pad_token_id
        self.out_channels = out_channels
        self.add_blank = add_blank

        self.text_encoder = TextEncoder(
            n_vocab=vocabs_size,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

        # Learnable positional embedding
        self.learnable_positional_embedding = nn.Parameter(
            torch.zeros(1, out_channels, pad_length)
        )
        nn.init.normal_(self.learnable_positional_embedding, 0.0, 0.02)

    def get_unconditional_condition(self, batch_size: int):
        """Return unconditional (null) phoneme embeddings"""
        device = self.learnable_positional_embedding.device
        unconditional_tokens = torch.full(
            (batch_size, self.pad_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )
        return self(unconditional_tokens)

    def _compute_sequence_lengths(self, phoneme_idx: torch.Tensor) -> torch.Tensor:
        """Compute actual sequence lengths by finding the last non-pad token.

        For add_blank sequences: [0, p1, 0, p2, ..., pN, 0, PAD, PAD, ...]
          → length = last_nonzero_pos + 2  (include trailing blank after last phoneme)
        For regular sequences:   [p1, p2, ..., pN, PAD, PAD, ...]
          → length = last_nonzero_pos + 1
        For all-padding (unconditional):
          → length = 0
        """
        B, T = phoneme_idx.shape
        nonzero_mask = (phoneme_idx != self.pad_token_id)  # [B, T]
        has_content = nonzero_mask.any(dim=-1)  # [B]

        # Find last non-zero position: flip and argmax finds the first True in reversed order
        reversed_mask = nonzero_mask.flip(dims=[-1])
        first_in_rev = reversed_mask.to(torch.int64).argmax(dim=-1)  # [B]
        last_nonzero_pos = T - 1 - first_in_rev  # [B]

        if self.add_blank:
            # Include the trailing blank after the last phoneme
            lengths = (last_nonzero_pos + 2).clamp(max=T)
        else:
            lengths = last_nonzero_pos + 1

        # All-padding sequences (unconditional) → length = 0
        lengths = lengths * has_content.to(torch.int64)
        return lengths

    def _get_src_mask(self, phoneme_idx: torch.Tensor) -> torch.Tensor:
        """Create positional mask: True for content positions, False for trailing padding.

        Unlike the old ``phoneme != pad_token_id`` approach, this correctly keeps
        interleaved blank tokens (id=0) as valid content when add_blank=True.
        """
        lengths = self._compute_sequence_lengths(phoneme_idx)  # [B]
        positions = torch.arange(phoneme_idx.size(1), device=phoneme_idx.device).unsqueeze(0)  # [1, T]
        return positions < lengths.unsqueeze(1)  # [B, T]

    def forward(self, phoneme_idx: torch.Tensor, *args, **kwargs):
        """
        Args:
            phoneme_idx: [bs, pad_length] phoneme ID sequences

        Returns:
            Tuple of (embeddings, attention_mask):
                embeddings: [bs, pad_length, out_channels]
                attention_mask: [bs, pad_length]
        """
        device = self.learnable_positional_embedding.device
        phoneme_idx = phoneme_idx.to(device)

        # Get sequence lengths (correctly handles interleaved blanks)
        src_length = self._compute_sequence_lengths(phoneme_idx)

        # Encode phonemes
        text_emb = self.text_encoder(phoneme_idx, src_length)  # [bs, out_channels, pad_length]

        # Add positional embedding
        text_emb = text_emb + self.learnable_positional_embedding

        # Create attention mask from positional lengths (int64 so that
        # _prepare_attention_mask applies the -1e-5 fix)
        attention_mask = self._get_src_mask(phoneme_idx).long()  # [bs, pad_length]

        # Apply mask to embeddings (use float view for multiplication)
        text_emb = text_emb * attention_mask.unsqueeze(1).float()

        # Return as list [embeddings, mask] for compatibility with training framework
        return [text_emb.transpose(1, 2), attention_mask]  # [bs, pad_length, out_channels], [bs, pad_length]


class AudioLDM2PhonemeEncoder(nn.Module):
    """AudioLDM2-compatible phoneme encoder using VitsModel weights."""

    def __init__(self, pretrained_dir: str, pad_token_id: int = 0, add_blank: bool = False):
        super().__init__()
        self.model = VitsModel.from_pretrained(pretrained_dir)
        self.pad_token_id = pad_token_id
        self.add_blank = add_blank

    @property
    def text_encoder(self) -> nn.Module:
        # Avoid registering a duplicate module name that aliases model.text_encoder.
        return self.model.text_encoder

    def _compute_sequence_lengths(self, phoneme_idx: torch.Tensor) -> torch.Tensor:
        """Compute actual sequence lengths by finding the last non-pad token.

        Same logic as PhonemeEncoder._compute_sequence_lengths.
        """
        B, T = phoneme_idx.shape
        nonzero_mask = (phoneme_idx != self.pad_token_id)  # [B, T]
        has_content = nonzero_mask.any(dim=-1)  # [B]

        reversed_mask = nonzero_mask.flip(dims=[-1])
        first_in_rev = reversed_mask.to(torch.int64).argmax(dim=-1)  # [B]
        last_nonzero_pos = T - 1 - first_in_rev  # [B]

        if self.add_blank:
            lengths = (last_nonzero_pos + 2).clamp(max=T)
        else:
            lengths = last_nonzero_pos + 1

        lengths = lengths * has_content.to(torch.int64)
        return lengths

    def get_unconditional_condition(self, batch_size: int):
        device = next(self.model.parameters()).device
        seq_len = self.model.config.max_text_length
        if seq_len is None:
            seq_len = 310
        unconditional_tokens = torch.full(
            (batch_size, seq_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        return self(unconditional_tokens)

    def forward(self, phoneme_idx: torch.Tensor, *args, **kwargs):
        device = next(self.model.parameters()).device
        phoneme_idx = phoneme_idx.to(device)

        # Positional mask: True for content (including interleaved blanks), False for trailing padding
        lengths = self._compute_sequence_lengths(phoneme_idx)  # [B]
        positions = torch.arange(phoneme_idx.size(1), device=device).unsqueeze(0)  # [1, T]
        attention_mask = (positions < lengths.unsqueeze(1))  # [B, T]

        # VitsTextEncoder expects padding_mask broadcastable to [B, T, H]
        padding_mask = attention_mask.unsqueeze(-1).to(dtype=phoneme_idx.dtype)

        outputs = self.text_encoder(
            input_ids=phoneme_idx,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
        )

        if isinstance(outputs, (list, tuple)):
            hidden = outputs[0]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden = outputs.hidden_states[-1]
        else:
            raise AttributeError("Vits text encoder output does not contain usable hidden states")

        return [hidden, attention_mask.long()]
