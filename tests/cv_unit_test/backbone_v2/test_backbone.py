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
        resnet.resnet_18,
        resnet.resnet_18d,
        swin.swin_tiny_patch4_window7_224,
        vit.vit_base_patch16,
    ],
)
def test_basic_usage(backbone_cls):
    """Test the basic usage of the backbones."""
    # Common parameters.
    kwargs = {"in_chans": 3, "num_classes": 50, "freeze_at": [1], "freeze_norm": True}

    # Test the instantiation.
    backbone = backbone_cls(**kwargs)
    assert isinstance(backbone, BackboneBase), f"Expected BackboneBase, got {type(backbone)}"

    # Test the properties.
    assert backbone.in_chans == kwargs["in_chans"], (
        f"Expected in_chans to be {kwargs['in_chans']}, got {backbone.in_chans}"
    )
    assert backbone.num_classes == kwargs["num_classes"], (
        f"Expected num_classes to be {kwargs['num_classes']}, got {backbone.num_classes}"
    )
    assert backbone.freeze_at == kwargs["freeze_at"], (
        f"Expected freeze_at to be {kwargs['freeze_at']}, got {backbone.freeze_at}"
    )
    assert backbone.freeze_norm is kwargs["freeze_norm"], (
        f"Expected freeze_norm to be {kwargs['freeze_norm']}, got {backbone.freeze_norm}"
    )

    # Test the forward.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones(1, 3, 224, 224, device=device)
    backbone.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    y = backbone.forward_pre_logits(x)
    assert isinstance(y, torch.Tensor), f"Expected output to be a tensor, got {type(y)}"
