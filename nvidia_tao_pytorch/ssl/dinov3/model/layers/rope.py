# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D axial Rotary Position Embedding (RoPE) for DINOv3.

DINOv3 replaces DINOv2's absolute positional embedding with 2D axial RoPE applied to
the query/key of *patch* tokens inside attention. The ``[CLS]`` token and register
tokens are non-spatial and receive an identity rotation (sin=0, cos=1).

Token layout (inherited from ``nvdinov2``): ``[CLS, patch_0 .. patch_{HW-1}, reg_0 .. reg_{R-1}]``
i.e. ``[CLS]`` at index 0 and register tokens appended at the **end** of the sequence.
This end-of-sequence placement is the documented v1 convention (see the module docstring
in ``vit.py``); the step-4 checkpoint remapper reorders Meta/timm's post-``[CLS]``
register layout to match.

Convention: rotary frequencies are built as a length-``head_dim//2`` vector
(``head_dim//4`` bands per spatial axis), then duplicated to full ``head_dim`` so the
``rotate_half`` (LLaMA / timm ``apply_rot_embed_cat``) convention applies. The exact
frequency base (``theta``) must match the timm DINOv3 reference and is verified by the
step-4 feature-parity smoke test; it is exposed via the ``rope_theta`` config field.
"""

import torch
from torch import nn


def rotate_half(x):
    """Rotate the last dimension by splitting it in half: ``[a, b] -> [-b, a]``.

    Args:
        x (torch.Tensor): Tensor with an even-sized last dimension.

    Returns:
        torch.Tensor: The half-rotated tensor, same shape as ``x``.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, sin, cos):
    """Apply rotary position embedding to a query/key tensor.

    Args:
        x (torch.Tensor): Tensor shaped ``[B, N, num_heads, head_dim]`` (xformers layout).
        sin (torch.Tensor): Sine table shaped ``[N, head_dim]``.
        cos (torch.Tensor): Cosine table shaped ``[N, head_dim]``.

    Returns:
        torch.Tensor: ``x`` with RoPE applied, same shape and dtype as ``x``.
    """
    # [N, head_dim] -> [1, N, 1, head_dim] to broadcast over batch and heads.
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.dtype)
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.dtype)
    return x * cos + rotate_half(x) * sin


class RoPE2D(nn.Module):
    """2D axial rotary position embedding generator.

    Produces per-token ``(sin, cos)`` tables for a patch grid, with identity rotation for
    the prefix ``[CLS]`` token and the trailing register tokens. The frequency bands are
    registered as a (non-persistent) buffer so FSDP does not attempt to shard them.
    """

    def __init__(self, head_dim: int, theta: float = 100.0,
                 num_prefix_tokens: int = 1, num_register_tokens: int = 4):
        """Initialize the axial RoPE generator.

        Args:
            head_dim (int): Per-head feature dimension. Must be divisible by 4 (two axes,
                each contributing ``head_dim//4`` rotary pairs).
            theta (float): Frequency base. Must match the timm DINOv3 reference.
            num_prefix_tokens (int): Number of non-spatial prefix tokens (``[CLS]`` -> 1).
            num_register_tokens (int): Number of trailing register tokens (identity rotation).
        """
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for 2D axial RoPE, got {head_dim}"
        self.head_dim = head_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.num_register_tokens = num_register_tokens

        n_freq = head_dim // 4  # rotary pairs per spatial axis
        bands = 1.0 / (theta ** (torch.arange(0, n_freq).float() / n_freq))
        self.register_buffer("bands", bands, persistent=False)

        # Small cache so repeated forwards at the same grid size avoid recompute.
        self._cache = {}

    def _grid_freqs(self, H: int, W: int, device, dtype):
        """Build axial frequencies for an ``H x W`` patch grid.

        Args:
            H (int): Grid height (number of patches along the first spatial axis).
            W (int): Grid width (number of patches along the second spatial axis).
            device: Target device.
            dtype: Target dtype (frequencies are computed in this dtype, typically fp32).

        Returns:
            torch.Tensor: Frequencies shaped ``[H*W, head_dim//2]`` in row-major order.
        """
        bands = self.bands.to(device=device, dtype=dtype)
        t_h = torch.arange(H, device=device, dtype=dtype)
        t_w = torch.arange(W, device=device, dtype=dtype)
        fh = torch.outer(t_h, bands)  # [H, n_freq]
        fw = torch.outer(t_w, bands)  # [W, n_freq]
        fh = fh[:, None, :].expand(H, W, -1)
        fw = fw[None, :, :].expand(H, W, -1)
        freqs = torch.cat([fh, fw], dim=-1).reshape(H * W, -1)  # [H*W, head_dim//2]
        return freqs

    def forward(self, H: int, W: int, device, dtype=torch.float32):
        """Build per-token sin/cos tables for one crop's token sequence.

        Args:
            H (int): Patch-grid height.
            W (int): Patch-grid width.
            device: Target device.
            dtype: Compute/return dtype for the tables (default fp32).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(sin, cos)`` each shaped
            ``[num_prefix + H*W + num_register, head_dim]``. Prefix and register rows are
            identity (sin=0, cos=1).
        """
        key = (H, W, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        n_patches = H * W
        seq_len = self.num_prefix_tokens + n_patches + self.num_register_tokens

        freqs = self._grid_freqs(H, W, device, dtype)        # [H*W, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)              # [H*W, head_dim]
        sin_p, cos_p = emb.sin(), emb.cos()

        sin = torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        cos = torch.ones(seq_len, self.head_dim, device=device, dtype=dtype)
        start = self.num_prefix_tokens
        sin[start:start + n_patches] = sin_p
        cos[start:start + n_patches] = cos_p

        self._cache[key] = (sin, cos)
        return sin, cos
