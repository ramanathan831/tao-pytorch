# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 config unit tests (defaults, param map, inheritance from nvdinov2)."""
import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import (
    ExperimentConfig,
    map_params,
    SUPPORTED_BACKBONES,
)


@pytest.mark.config
@pytest.mark.ssl_unit
def test_default_model_name_is_dinov3():
    """The DINOv3 experiment config defaults its model name to 'dinov3'."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.model_name == "dinov3"


@pytest.mark.config
@pytest.mark.ssl_unit
def test_backbone_patch16_rope_defaults():
    """DINOv3 backbone defaults: patch-16, 4 register tokens, ViT-B, RoPE theta present."""
    cfg = OmegaConf.structured(ExperimentConfig())
    bb = cfg.model.backbone
    assert bb.patch_size == 16
    assert bb.num_register_tokens == 4
    assert bb.teacher_type == "vit_b"
    assert bb.student_type == "vit_b"
    assert bb.img_size == 256
    assert bb.rope_theta == 100.0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_gram_and_lora_present():
    """Gram and (disabled) LoRA configs are present on the model config."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert hasattr(cfg.model, "gram")
    assert hasattr(cfg.model, "lora")
    assert cfg.model.lora.enable is False


@pytest.mark.config
@pytest.mark.ssl_unit
def test_single_res_256_transform_defaults():
    """v1 is single-resolution 256 with patch-16-friendly local crops."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.dataset.transform.global_crops_size == 256
    assert cfg.dataset.transform.local_crops_size % 16 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_b():
    """The v3 param map carries the ViT-B (768/12/12, standard MLP) entry."""
    assert map_params["embed_dim"]["vit_b"] == 768
    assert map_params["depth"]["vit_b"] == 12
    assert map_params["num_heads"]["vit_b"] == 12
    assert map_params["mlp_layer"]["vit_b"] == "mlp"
    # ViT-H+ and ViT-7B use SwiGLU.
    assert map_params["mlp_layer"]["vit_h_plus"] == "swiglu"
    assert map_params["mlp_layer"]["vit_7b"] == "swiglu"
    assert set(SUPPORTED_BACKBONES) == {"vit_s", "vit_s_plus", "vit_b", "vit_l", "vit_h_plus", "vit_7b"}


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_l():
    """The v3 param map carries the ViT-L (1024/24/16, standard GELU MLP) entry (Phase 2)."""
    assert map_params["embed_dim"]["vit_l"] == 1024
    assert map_params["depth"]["vit_l"] == 24
    assert map_params["num_heads"]["vit_l"] == 16
    assert map_params["mlp_layer"]["vit_l"] == "mlp"
    # head_dim = 1024 / 16 = 64 (same as ViT-B) -> RoPE (needs head_dim % 4 == 0) works unchanged.
    assert (map_params["embed_dim"]["vit_l"] // map_params["num_heads"]["vit_l"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_s():
    """The v3 param map carries the ViT-S (384/12/6, standard MLP) entry."""
    assert map_params["embed_dim"]["vit_s"] == 384
    assert map_params["depth"]["vit_s"] == 12
    assert map_params["num_heads"]["vit_s"] == 6
    assert map_params["mlp_layer"]["vit_s"] == "mlp"
    assert map_params["mlp_ratio"]["vit_s"] == 4.0
    # head_dim = 384 / 6 = 64 (same as ViT-B/L) -> RoPE (needs head_dim % 4 == 0) works unchanged.
    assert (map_params["embed_dim"]["vit_s"] // map_params["num_heads"]["vit_s"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_s_plus():
    """The v3 param map carries the ViT-S+ (384/12/6, SwiGLU) entry."""
    assert map_params["embed_dim"]["vit_s_plus"] == 384
    assert map_params["depth"]["vit_s_plus"] == 12
    assert map_params["num_heads"]["vit_s_plus"] == 6
    assert map_params["mlp_layer"]["vit_s_plus"] == "swiglu"
    assert map_params["mlp_ratio"]["vit_s_plus"] == 4.0
    assert (map_params["embed_dim"]["vit_s_plus"] // map_params["num_heads"]["vit_s_plus"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_h_plus():
    """The v3 param map carries the ViT-H+ (1280/32/20, SwiGLU) entry."""
    assert map_params["embed_dim"]["vit_h_plus"] == 1280
    assert map_params["depth"]["vit_h_plus"] == 32
    assert map_params["num_heads"]["vit_h_plus"] == 20
    assert map_params["mlp_layer"]["vit_h_plus"] == "swiglu"
    assert map_params["mlp_ratio"]["vit_h_plus"] == 4.0
    assert (map_params["embed_dim"]["vit_h_plus"] // map_params["num_heads"]["vit_h_plus"]) % 4 == 0


@pytest.mark.config
@pytest.mark.ssl_unit
def test_param_map_vit_7b():
    """The v3 param map carries the ViT-7B (4096/40/32, SwiGLU) entry; the public 7B
    checkpoint uses the narrow SwiGLU (mlp_ratio 2.0)."""
    assert map_params["embed_dim"]["vit_7b"] == 4096
    assert map_params["depth"]["vit_7b"] == 40
    assert map_params["num_heads"]["vit_7b"] == 32
    assert map_params["mlp_layer"]["vit_7b"] == "swiglu"
    assert map_params["mlp_ratio"]["vit_7b"] == 2.0
    assert (map_params["embed_dim"]["vit_7b"] // map_params["num_heads"]["vit_7b"]) % 4 == 0
