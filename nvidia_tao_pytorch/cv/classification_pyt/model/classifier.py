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

"""Classification model builder"""

import torch.nn as nn

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO
from nvidia_tao_pytorch.cv.classification_pyt.model.backbones import (
    clip_model_dict,
    convnextv2_model_dict,
    cradio_model_dict,
    fan_model_dict,
    faster_vit_model_dict,
    gc_vit_model_dict,
    nvdino_model_dict,
)
from nvidia_tao_pytorch.cv.classification_pyt.model.decode_heads.tao_linear_head import TAOLinearClsHead


channels_map = {
    "convnextv2_atto": 320,
    "convnextv2_femto": 384,
    "convnextv2_pico": 512,
    "convnextv2_nano": 640,
    "convnextv2_tiny": 768,
    "convnextv2_base": 1024,
    "convnextv2_large": 1536,
    "convnextv2_huge": 2816,
    "fan_tiny_8_p4_hybrid": 192,  # FAN
    "fan_small_12_p4_hybrid": 384,
    "fan_base_16_p4_hybrid": 448,
    "fan_large_16_p4_hybrid": 480,
    "fan_Xlarge_16_p4_hybrid": 768,
    "fan_base_18_p16_224": 448,
    "fan_tiny_12_p16_224": 192,
    "fan_small_12_p16_224_se_attn": 384,
    "fan_small_12_p16_224": 384,
    "fan_large_24_p16_224": 480,
    "gc_vit_xxtiny": 512,  # GCViT
    "gc_vit_xtiny": 512,
    "gc_vit_tiny": 512,
    "gc_vit_small": 768,
    "gc_vit_base": 1024,
    "gc_vit_large": 1536,
    "gc_vit_large_384": 1536,
    "faster_vit_0_224": 512,  # FasterViT
    "faster_vit_1_224": 640,
    "faster_vit_2_224": 768,
    "faster_vit_3_224": 1024,
    "faster_vit_4_224": 1568,
    "faster_vit_5_224": 2560,
    "faster_vit_6_224": 2560,
    "faster_vit_4_21k_224": 1568,
    "faster_vit_4_21k_384": 1568,
    "faster_vit_4_21k_512": 1568,
    "faster_vit_4_21k_768": 1568,
    "vit_large_patch14_dinov2_swiglu": 1024,
    "vit_giant_patch14_reg4_dinov2_swiglu": 1536,
    "ViT-H-14-SigLIP-CLIPA-224": 1024,
    "ViT-L-14-SigLIP-CLIPA-336": 768,
    "ViT-L-14-SigLIP-CLIPA-224": 768,
    "c_radio_p1_vit_huge_patch16_mlpnorm": 3840,
    "c_radio_p2_vit_huge_patch16_mlpnorm": 5120,
    "c_radio_p3_vit_huge_patch16_mlpnorm": 3840,
    "c_radio_v2_vit_base_patch16": 2304,
    "c_radio_v2_vit_large_patch16": 3072,
    "c_radio_v2_vit_huge_patch16": 3840,
    "c_radio_v3_vit_large_patch16_reg4_dinov2": 3072,
}


class Classifier(nn.Module):
    """
    Classifier model for classification task using a Transformer backbone Encoder and a linear decoder.

    Args:
        input_nc (int): Number of input channels (default is 3).
        num_classes (int): Number of output classes (default is 2).
        decoder_softmax (bool): Whether to apply softmax to the decoder output (default is False).
        embed_dim (int): Embedding dimension for the Transformer Encoder (default is 256).
        model (str): Backbone model name (default is 'fan_tiny_8_p4_hybrid_256').
        img_size (int): Input image size (default is 256).
        embed_dims (list): List of embedding dimensions for the Transformer Decoder (default is [128, 256, 384, 384]).
        feature_strides (list): List of feature strides for the Transformer Decoder (default is [4, 8, 16, 16]).
        in_index (list): List of input indices for the Transformer Decoder (default is [0, 1, 2, 3]).
        feat_downsample: Flag to indicate whether to use feature downsampling for the transformer block for FAN-Hybrid
        learnable_difference_modules (int): Number of decoder difference modules for Architecture 2
        pretrained_backbone_path (str): Path to the pre-trained backbone weights.
        activation_checkpoint (bool): Enable activation checkpointing
        freeze_backbone: Flag to freeze backbone weights during training.
        return_interm_indices (list): list of layer indices to reutrn as backbone features.
    """

    def __init__(self, binary=True, num_classes=2, model='fan_tiny_8_p4_hybrid_256', img_size=256,
                 in_channels=384, pretrained_backbone_path=None, activation_checkpoint=False, freeze_backbone=False):
        """Initialize Classifier class"""
        super(Classifier, self).__init__()

        # Freeze the backbone without the pretrained weights is not allowed.
        if freeze_backbone and pretrained_backbone_path is None:
            raise AssertionError(
                "You shouldn't freeze a model without specifying pretrained_backbone_path."
            )

        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1
        self.model_name = model
        self.in_channels = in_channels
        self.binary = binary
        self.num_classes = num_classes
        self.resolution = (img_size, img_size)

        if get_global_rank() == 0:
            logging.info(f"Number of output classes: {num_classes}")

        # Initialize the backbone model.
        if 'fan' in self.model_name:
            freeze_at = "all" if freeze_backbone else None
            self.backbone = fan_model_dict[self.model_name](num_classes=0, freeze_at=freeze_at)
            if pretrained_backbone_path is not None:
                self.backbone.load_pretrained_weights(pretrained_backbone_path, strict=False)
        elif 'gc' in self.model_name:
            freeze_at = "all" if freeze_backbone else None
            self.backbone = gc_vit_model_dict[self.model_name](num_classes=0, freeze_at=freeze_at)
            if pretrained_backbone_path is not None:
                self.backbone.load_pretrained_weights(pretrained_backbone_path, strict=False)
        elif 'faster' in self.model_name:
            freeze_at = "all" if freeze_backbone else None
            self.backbone = faster_vit_model_dict[self.model_name](num_classes=0, freeze_at=freeze_at)
            if pretrained_backbone_path is not None:
                self.backbone.load_pretrained_weights(pretrained_backbone_path, strict=False)
        elif 'radio' in self.model_name:
            freeze_at = "all" if freeze_backbone else None
            self.backbone = cradio_model_dict[self.model_name](
                num_classes=0, resolution=self.resolution, freeze_at=freeze_at
            )
            if pretrained_backbone_path is not None:
                self.backbone.load_pretrained_weights(pretrained_backbone_path, strict=False)
        elif 'CLIP' in self.model_name:
            freeze_at = "all" if freeze_backbone else None
            self.backbone = clip_model_dict[self.model_name](num_classes=0, freeze_at=freeze_at)
            if pretrained_backbone_path is not None:
                self.backbone.load_pretrained_weights(pretrained_backbone_path, strict=False)
        elif 'convnextv2' in self.model_name:
            freeze_at = "all" if freeze_backbone else None
            self.backbone = convnextv2_model_dict[self.model_name](num_classes=0, freeze_at=freeze_at)
            if pretrained_backbone_path is not None:
                self.backbone.load_pretrained_weights(pretrained_backbone_path, strict=False)
        elif 'vit' in self.model_name:  # NVDINOV2
            freeze_at = "all" if freeze_backbone else None
            self.backbone = nvdino_model_dict[self.model_name](num_classes=0, freeze_at=freeze_at)
            if pretrained_backbone_path is not None:
                self.backbone.load_pretrained_weights(pretrained_backbone_path, strict=False)
        else:
            raise NotImplementedError('Bacbkbone name [%s] is not supported' % self.model_name)

        # Initialize the classification head.
        self.decoder = TAOLinearClsHead(
            binary=self.binary, num_classes=self.num_classes, in_channels=self.in_channels
        )

    def forward(self, x):
        """
        Forward pass of the Classifier model.

        Args:
            x (torch.Tensor): Input tensor for the image input.

        Returns:
            torch.Tensor: Output tensor representing the class prediction.
        """
        features = self.backbone.forward_pre_logits(x)
        if isinstance(self.backbone, RADIO):
            features = features[0]
        return self.decoder(features)


def build_model(experiment_config,
                export=False):
    """ Build Classifier model according to configuration

    Args:
        experiment_config: experiment configuration
        export: flag to indicate onnx export

    Returns:
        model

    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset

    backbone = model_config.backbone['type']
    freeze_backbone = model_config.backbone['freeze_backbone']
    pretrained_backbone_path = model_config.backbone.pretrained_backbone_path

    # Map input resolution for different backbones
    map_input_resolution = {
        "faster_vit_4_21k_384": 384,
        "faster_vit_4_21k_512": 512,
        "faster_vit_4_21k_768": 768,
        "gc_vit_large_384": 384,
        "ViT-L-14-SigLIP-CLIPA-336": 336
    }
    if backbone in channels_map:
        in_channels = channels_map[backbone]
        model_config.head.in_channels = in_channels
    else:
        raise NotImplementedError('Bacbkbone name [%s] is not supported' % backbone)
    if backbone in map_input_resolution:
        dataset_config.img_size = map_input_resolution[backbone]

    binary = model_config.head.binary
    in_channels = model_config.head.in_channels

    num_classes = dataset_config.num_classes
    img_size = dataset_config.img_size

    model = Classifier(
        binary=binary,
        num_classes=num_classes,
        model=backbone,
        img_size=img_size,
        in_channels=in_channels,
        pretrained_backbone_path=pretrained_backbone_path,
        freeze_backbone=freeze_backbone
    )

    return model
