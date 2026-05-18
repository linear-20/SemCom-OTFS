"""DD-domain Perceiver receiver for visual-token logits.

The model consumes a complex delay-Doppler grid [B, M, N], embeds each DD bin
from real/imag plus normalized 2D position, then uses learned token queries to
cross-attend to the DD tokens. It intentionally contains no CNN and has no
dependency on the mapper, modem, or channel modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in module.parameters())


class DDTokenPerceiverReceiver(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        dd_shape: tuple[int, int] = (32, 32),
        token_shape: tuple[int, int] = (16, 16),
        embed_dim: int = 256,
        num_heads: int = 8,
        self_attn_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not isinstance(codebook_size, int) or codebook_size <= 0:
            raise ValueError("codebook_size must be a positive integer.")
        if (
            not isinstance(dd_shape, tuple)
            or len(dd_shape) != 2
            or not all(isinstance(x, int) and x > 0 for x in dd_shape)
        ):
            raise ValueError("dd_shape must be a tuple of two positive integers.")
        if (
            not isinstance(token_shape, tuple)
            or len(token_shape) != 2
            or not all(isinstance(x, int) and x > 0 for x in token_shape)
        ):
            raise ValueError("token_shape must be a tuple of two positive integers.")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if not isinstance(self_attn_layers, int) or self_attn_layers < 0:
            raise ValueError("self_attn_layers must be a non-negative integer.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.codebook_size = codebook_size
        self.dd_shape = dd_shape
        self.M = dd_shape[0]
        self.N = dd_shape[1]
        self.token_shape = token_shape
        self.num_output_tokens = token_shape[0] * token_shape[1]
        self.embed_dim = embed_dim

        self.dd_embed = nn.Linear(4, embed_dim)
        self.token_queries = nn.Parameter(torch.randn(self.num_output_tokens, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.token_self_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self_attn_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, codebook_size),
        )

        delay_pos = torch.linspace(-1.0, 1.0, self.M, dtype=torch.float32)
        doppler_pos = torch.linspace(-1.0, 1.0, self.N, dtype=torch.float32)
        delay_grid, doppler_grid = torch.meshgrid(delay_pos, doppler_pos, indexing="ij")
        positions = torch.stack([delay_grid, doppler_grid], dim=-1).reshape(self.M * self.N, 2)
        self.register_buffer("dd_positions", positions, persistent=False)

    def forward(self, y_dd: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(y_dd):
            raise TypeError("y_dd must be a torch.Tensor.")
        if y_dd.ndim != 3:
            raise ValueError(f"y_dd must have shape [B, M, N]; got {list(y_dd.shape)}.")
        if tuple(y_dd.shape[1:]) != self.dd_shape:
            raise ValueError(
                f"y_dd spatial shape must match dd_shape {self.dd_shape}; "
                f"got {tuple(y_dd.shape[1:])}."
            )
        if not torch.is_complex(y_dd):
            raise TypeError(f"y_dd must be a complex tensor; got {y_dd.dtype}.")

        batch_size = y_dd.shape[0]
        real_dtype = y_dd.real.dtype
        flat_dd = y_dd.reshape(batch_size, self.M * self.N)
        real_imag = torch.stack([flat_dd.real, flat_dd.imag], dim=-1)
        positions = self.dd_positions.to(device=y_dd.device, dtype=real_dtype)
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat([real_imag, positions], dim=-1)

        dd_tokens = self.dd_embed(features)
        queries = self.token_queries.unsqueeze(0).expand(batch_size, -1, -1)
        cross_output, _ = self.cross_attn(query=queries, key=dd_tokens, value=dd_tokens)
        token_hidden = self.cross_norm(queries + cross_output)
        token_hidden = self.token_self_attn(token_hidden)
        return self.head(token_hidden)
