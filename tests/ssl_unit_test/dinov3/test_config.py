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
    # ViT-H+ is reserved and uses SwiGLU.
    assert map_params["mlp_layer"]["vit_h_plus"] == "swiglu"
    assert set(SUPPORTED_BACKBONES) == {"vit_b", "vit_l", "vit_h_plus"}
