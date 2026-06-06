# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 ViT-B feature-parity smoke test vs the timm reference (GPU + weights).

This is the step-4 gate: load the timm ``vit_base_patch16_dinov3.lvd1689m`` weights into our
``DinoV3VisionTransformer`` via the checkpoint remapper and confirm the CLS + patch features
are cosine-close to timm's own forward. It validates the remapper key/shape coverage AND the
RoPE convention end-to-end (rope_theta, coord normalization, rotation layout).

Requires a CUDA GPU (xformers memory-efficient attention) and the local weights; skips
otherwise. Point ``DINOV3_VITB_WEIGHTS`` at the dir/file if not in a default location.
"""
import os

import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

_WEIGHT_CANDIDATES = [
    os.environ.get("DINOV3_VITB_WEIGHTS", ""),
    "/data/weights/dinov3/vitb16",
    os.path.expanduser("~/weights/dinov3/vitb16"),
]


def _find_weights():
    """Return the first existing weights path among the candidates, else None."""
    for cand in _WEIGHT_CANDIDATES:
        if cand and os.path.exists(cand):
            return cand
    return None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
_weights = _find_weights()
requires_weights = pytest.mark.skipif(_weights is None, reason="DINOv3 ViT-B weights not found")


def _cosine(a, b):
    """Mean per-row cosine similarity between two ``[..., C]`` tensors (fp32)."""
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


@requires_cuda
@requires_weights
@pytest.mark.ssl_unit
def test_dinov3_vitb_feature_parity_vs_timm():
    """Our remapped ViT-B matches timm's CLS/patch features (cosine > 0.99)."""
    timm = pytest.importorskip("timm")

    ckpt = os.path.join(_weights, "model.safetensors") if os.path.isdir(_weights) else _weights
    ref = timm.create_model(
        "vit_base_patch16_dinov3", pretrained=False, checkpoint_path=ckpt,
    ).cuda().half().eval()

    model = DinoV3VisionTransformer(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        init_values=1e-5, drop_path_schedule="linear", num_classes=0, drop_path_rate=0.0,
        register_tokens=4, use_custom_attention=True,
    )
    # Load via the same remapper the pl_model uses.
    timm_sd = DinoV3PlModel._load_pretrained_state_dict(_weights)
    remapped, unmapped = DinoV3PlModel._remap_dinov3_state_dict(timm_sd, model.state_dict())
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Remapper must cover the whole checkpoint and leave only mask_token uninitialized.
    assert unmapped == [], f"checkpoint keys not mapped: {unmapped}"
    assert unexpected == [], f"unexpected keys after remap: {unexpected}"
    assert set(missing) <= {"mask_token"}, f"unexpected missing keys: {missing}"

    model = model.cuda().half().eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 256, 256).cuda().half()
    with torch.no_grad():
        feats = ref.forward_features(x)
        np = ref.num_prefix_tokens
        cls_ref, patch_ref = feats[:, 0], feats[:, np:]
        out = model(x)
        cls_ours, patch_ours = out["x_norm_clstoken"], out["x_norm_patchtokens"]

    cls_cos = _cosine(cls_ours, cls_ref)
    patch_cos = _cosine(patch_ours, patch_ref)
    assert cls_cos > 0.99, f"CLS feature cosine too low: {cls_cos}"
    assert patch_cos > 0.99, f"patch feature cosine too low: {patch_cos}"
