# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backbone unit tests."""

import pytest
import torch

from nvidia_tao_pytorch.cv.backbone_v2 import (
    convnext,
    convnext_v2,
    dino_v2,
    efficientvit,
    fan,
    fastervit,
    gcvit,
    hiera,
    open_clip,
    radio,
    resnet,
    swin,
    vit,
)
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "backbone_cls",
    [
        convnext.convnext_tiny,
        convnext_v2.convnextv2_atto,
        dino_v2.vit_large_patch14_dinov2_swiglu,
        dino_v2.vit_giant_patch14_reg4_dinov2_swiglu,
        efficientvit.efficientvit_b0,
        efficientvit.efficientvit_l0,
        fan.fan_tiny_12_p16_224,
        fan.fan_tiny_8_p4_hybrid,
        fan.fan_swin_tiny_patch4_window7_224,
        fastervit.faster_vit_0_224,
        gcvit.gc_vit_xxtiny,
        hiera.hiera_tiny_224,
        open_clip.vit_l_14_siglip_clipa_336,
        radio.c_radio_p1_vit_huge_patch16_mlpnorm,
        radio.c_radio_v2_vit_base_patch16,
        resnet.resnet_18,
        resnet.resnet_18d,
        swin.swin_tiny_patch4_window7_224,
        vit.vit_base_patch16,
    ],
)
@pytest.mark.parametrize("activation_checkpoint", [False, True])
@pytest.mark.parametrize("freeze_at", [[1], "all"])
def test_basic_usage(backbone_cls, activation_checkpoint, freeze_at):
    """Test the basic usage of the backbones."""
    # Common parameters.
    kwargs = {"in_chans": 3, "num_classes": 50, "freeze_at": freeze_at, "freeze_norm": True}
    if activation_checkpoint:
        kwargs["activation_checkpoint"] = activation_checkpoint
    # OpenCLIP and CRADIO require `num_classes` to be 0.
    if backbone_cls in (
        open_clip.vit_l_14_siglip_clipa_336,
        radio.c_radio_p1_vit_huge_patch16_mlpnorm,
        radio.c_radio_v2_vit_base_patch16,
    ):
        kwargs["num_classes"] = 0

    # Test the instantiation.
    try:
        backbone = backbone_cls(**kwargs)
    except TypeError:
        # Some backbones don't support activation checkpointing.
        pytest.skip(f"{backbone_cls.__name__} doesn't support activation checkpointing.")
    assert isinstance(backbone, BackboneBase), f"Expected BackboneBase, got {type(backbone)}"

    # Test the properties.
    assert backbone.in_chans == kwargs["in_chans"], (
        f"Expected in_chans to be {kwargs['in_chans']}, got {backbone.in_chans}"
    )
    assert backbone.num_classes == kwargs["num_classes"], (
        f"Expected num_classes to be {kwargs['num_classes']}, got {backbone.num_classes}"
    )
    if activation_checkpoint:
        assert backbone.activation_checkpoint == kwargs["activation_checkpoint"], (
            f"Expected activation_checkpoint to be {kwargs['activation_checkpoint']}, "
            f"got {backbone.activation_checkpoint}"
        )
    assert backbone.freeze_at == kwargs["freeze_at"], (
        f"Expected freeze_at to be {kwargs['freeze_at']}, got {backbone.freeze_at}"
    )
    assert backbone.freeze_norm is kwargs["freeze_norm"], (
        f"Expected freeze_norm to be {kwargs['freeze_norm']}, got {backbone.freeze_norm}"
    )

    # Test the freezing.
    if kwargs["freeze_at"] == "all":
        for p in backbone.parameters():
            assert p.requires_grad is False, f"Expected all parameters to be frozen, but {p} is not."
        assert backbone.training is False, "Expected backbone to be in eval mode, but it is not."
    if isinstance(kwargs["freeze_at"], list):
        stage_dict = backbone.get_stage_dict()
        for freeze_key in kwargs["freeze_at"]:
            module = stage_dict[freeze_key]
            for p in module.parameters():
                assert p.requires_grad is False, f"Expected {freeze_key} to be frozen, but it is not."
            assert module.training is False, f"Expected {freeze_key} to be in eval mode, but it is not."

    # Test the forward.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones(1, 3, 224, 224, device=device)
    backbone.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    y = backbone.forward_pre_logits(x)
    assert isinstance(y, torch.Tensor), f"Expected output to be a tensor, got {type(y)}"
