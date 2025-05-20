# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Backbone Init Module."""

from nvidia_tao_pytorch.cv.backbone_v2.convnext_v2 import (
    convnextv2_atto,
    convnextv2_femto,
    convnextv2_pico,
    convnextv2_nano,
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large,
    convnextv2_huge,
)
from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import (
    vit_large_patch14_dinov2_swiglu,
    vit_giant_patch14_reg4_dinov2_swiglu,
)
from nvidia_tao_pytorch.cv.backbone_v2.open_clip import (
    vit_l_14_siglip_clipa_224,
    vit_l_14_siglip_clipa_336,
    vit_h_14_siglip_clipa_224,
)
from nvidia_tao_pytorch.cv.backbone_v2.radio import (
    c_radio_p1_vit_huge_patch16_mlpnorm,
    c_radio_p2_vit_huge_patch16_mlpnorm,
    c_radio_p3_vit_huge_patch16_mlpnorm,
    c_radio_v2_vit_base_patch16,
    c_radio_v2_vit_large_patch16,
    c_radio_v2_vit_huge_patch16,
)
from nvidia_tao_pytorch.cv.backbone_v2.fan import (
    fan_tiny_12_p16_224,
    fan_small_12_p16_224_se_attn,
    fan_small_12_p16_224,
    fan_base_18_p16_224,
    fan_large_24_p16_224,
    fan_tiny_8_p4_hybrid,
    fan_small_12_p4_hybrid,
    fan_base_16_p4_hybrid,
    fan_large_16_p4_hybrid,
    fan_xlarge_16_p4_hybrid,
)
from nvidia_tao_pytorch.cv.backbone_v2.fastervit import (
    faster_vit_0_224,
    faster_vit_1_224,
    faster_vit_2_224,
    faster_vit_3_224,
    faster_vit_4_224,
    faster_vit_5_224,
    faster_vit_6_224,
    faster_vit_4_21k_224,
    faster_vit_4_21k_384,
    faster_vit_4_21k_512,
    faster_vit_4_21k_768,
)
from nvidia_tao_pytorch.cv.backbone_v2.gcvit import (
    gc_vit_xxtiny,
    gc_vit_xtiny,
    gc_vit_tiny,
    gc_vit_small,
    gc_vit_base,
    gc_vit_large,
    gc_vit_large_384,
)


convnextv2_model_dict = {
    'convnextv2_atto': convnextv2_atto,
    'convnextv2_femto': convnextv2_femto,
    'convnextv2_pico': convnextv2_pico,
    'convnextv2_nano': convnextv2_nano,
    'convnextv2_tiny': convnextv2_tiny,
    'convnextv2_base': convnextv2_base,
    'convnextv2_large': convnextv2_large,
    'convnextv2_huge': convnextv2_huge,
}
fan_model_dict = {
    'fan_tiny_12_p16_224': fan_tiny_12_p16_224,
    'fan_small_12_p16_224_se_attn': fan_small_12_p16_224_se_attn,
    'fan_small_12_p16_224': fan_small_12_p16_224,
    'fan_base_18_p16_224': fan_base_18_p16_224,
    'fan_large_24_p16_224': fan_large_24_p16_224,
    'fan_tiny_8_p4_hybrid': fan_tiny_8_p4_hybrid,
    'fan_small_12_p4_hybrid': fan_small_12_p4_hybrid,
    'fan_base_16_p4_hybrid': fan_base_16_p4_hybrid,
    'fan_large_16_p4_hybrid': fan_large_16_p4_hybrid,
    'fan_Xlarge_16_p4_hybrid': fan_xlarge_16_p4_hybrid,
}

nvdino_model_dict = {
    'vit_large_patch14_dinov2_swiglu': vit_large_patch14_dinov2_swiglu,
    'vit_giant_patch14_reg4_dinov2_swiglu': vit_giant_patch14_reg4_dinov2_swiglu,
}

cradio_model_dict = {
    'c_radio_p1_vit_huge_patch16_mlpnorm': c_radio_p1_vit_huge_patch16_mlpnorm,
    'c_radio_p2_vit_huge_patch16_mlpnorm': c_radio_p2_vit_huge_patch16_mlpnorm,
    'c_radio_p3_vit_huge_patch16_mlpnorm': c_radio_p3_vit_huge_patch16_mlpnorm,
    'c_radio_v2_vit_base_patch16': c_radio_v2_vit_base_patch16,
    'c_radio_v2_vit_large_patch16': c_radio_v2_vit_large_patch16,
    'c_radio_v2_vit_huge_patch16': c_radio_v2_vit_huge_patch16,
}

faster_vit_model_dict = {
    'faster_vit_0_224': faster_vit_0_224,
    'faster_vit_1_224': faster_vit_1_224,
    'faster_vit_2_224': faster_vit_2_224,
    'faster_vit_3_224': faster_vit_3_224,
    'faster_vit_4_224': faster_vit_4_224,
    'faster_vit_5_224': faster_vit_5_224,
    'faster_vit_6_224': faster_vit_6_224,
    'faster_vit_4_21k_224': faster_vit_4_21k_224,
    'faster_vit_4_21k_384': faster_vit_4_21k_384,
    'faster_vit_4_21k_512': faster_vit_4_21k_512,
    'faster_vit_4_21k_768': faster_vit_4_21k_768,
}

gc_vit_model_dict = {
    'gc_vit_xxtiny': gc_vit_xxtiny,
    'gc_vit_xtiny': gc_vit_xtiny,
    'gc_vit_tiny': gc_vit_tiny,
    'gc_vit_small': gc_vit_small,
    'gc_vit_base': gc_vit_base,
    'gc_vit_large': gc_vit_large,
    'gc_vit_large_384': gc_vit_large_384,
}

clip_model_dict = {
    "ViT-L-14-SigLIP-CLIPA-224": vit_l_14_siglip_clipa_224,
    "ViT-L-14-SigLIP-CLIPA-336": vit_l_14_siglip_clipa_336,
    "ViT-H-14-SigLIP-CLIPA-224": vit_h_14_siglip_clipa_224,
}

# "fan_tiny_8_p4_hybrid": 192,  # FAN
# "fan_small_12_p4_hybrid": 384,
# "fan_base_16_p4_hybrid": 448,
# "fan_large_16_p4_hybrid": 480,
# "fan_Xlarge_16_p4_hybrid": 768,
# "fan_base_18_p16_224": 448,
# "fan_tiny_12_p16_224": 192,
# "fan_small_12_p16_224_se_attn": 384,
# "fan_small_12_p16_224": 384,
# "fan_large_24_p16_224": 480,
# "gc_vit_xxtiny": 512,  # GCViT
# "gc_vit_xtiny": 512,
# "gc_vit_tiny": 512,
# "gc_vit_small": 768,
# "gc_vit_base": 1024,
# "gc_vit_large": 1536,
# "gc_vit_large_384": 1536,
# "faster_vit_0_224": 512,  # FasterViT
# "faster_vit_1_224": 640,
# "faster_vit_2_224": 768,
# "faster_vit_3_224": 1024,
# "faster_vit_4_224": 1568,
# "faster_vit_5_224": 2560,
# "faster_vit_6_224": 2560,
# "faster_vit_4_21k_224": 1568,
# "faster_vit_4_21k_384": 1568,
# "faster_vit_4_21k_512": 1568,
# "faster_vit_4_21k_768": 1568,
# "vit_large_patch14_dinov2_swiglu": 1024,
# "vit_giant_patch14_reg4_dinov2_swiglu": 1536,
# "c_radio_p1_vit_huge_patch16_224_mlpnorm": 3840,
# "c_radio_p2_vit_huge_patch16_224_mlpnorm": 5120,
# "c_radio_p3_vit_huge_patch16_224_mlpnorm": 3840
# "ViT-H-14-SigLIP-CLIPA-224": 1024,
# "ViT-L-14-SigLIP-CLIPA-336": 768,
# "ViT-L-14-SigLIP-CLIPA-224": 768,

# not yet
# "ViT-L-14": 768,
# "ViT-B-16": 512,
# "ViT-L-14-336": 768,
# "ViT-g-14": 1024,
# "ViT-H-14": 1024,
# "EVA02-E-14-plus": 1024,
# "EVA02-E-14": 1024,
# "EVA02-L-14-336": 768,
# "EVA02-L-14": 768,
# "ViT-B-32": 512,
