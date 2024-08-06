# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""The build nn module model."""

import torch.nn as nn

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import load_pretrained_weights

from nvidia_tao_pytorch.cv.rtdetr.model.resnet import resnet_model_dict
from nvidia_tao_pytorch.cv.rtdetr.model.hybrid_encoder import HybridEncoder
from nvidia_tao_pytorch.cv.rtdetr.model.rtdetr_decoder import RTDETRTransformer
from nvidia_tao_pytorch.cv.rtdetr.model.rtdetr import RTDETR


class RTDETRModel(nn.Module):
    """RT-DETR model module."""

    def __init__(self,
                 backbone='resnet_50',
                 pretrained_backbone=None,
                 train_backbone=True,
                 num_classes=80,
                #  num_classes=4,
                 out_indices=[1, 2, 3],
                 # Encoder
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0,
                 enc_act='gelu',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1,
                 act='silu',
                #  eval_spatial_size=[544, 960],
                 eval_spatial_size=[640, 640],
                 # Decoder
                 feat_channels=[256, 256, 256],
                 num_levels=3,
                 num_queries=300,
                 num_decoder_layers=6,
                 num_denoising=100,
                 eval_idx=-1,
                #  multi_scale=[[480, 832], [512, 896], [544, 960], [544, 960], [544, 960], [576, 992], [608, 1056], [672, 1184], [704, 1216], [736, 1280], [768, 1344], [800, 1408]] # must be divisible by 32
                 multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
                 ):
        """Initialize RT-DETR Model.

        Args:

        """
        super().__init__()
        if backbone.startswith('resnet'):
            backbone = resnet_model_dict[backbone](
                out_indices,
            )

        pretrained_backbone_ckp = load_pretrained_weights(pretrained_backbone) if pretrained_backbone else None
        if pretrained_backbone_ckp:
            _tmp_st_output = backbone.load_state_dict(pretrained_backbone_ckp, strict=False)
            if get_global_rank() == 0:
                logging.info(f"Loaded pretrained weights from {pretrained_backbone}")
                logging.info(f"{_tmp_st_output}")

        encoder = HybridEncoder(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
        )

        decoder = RTDETRTransformer(
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_denoising=num_denoising,
            eval_idx=eval_idx,
            eval_spatial_size=eval_spatial_size,
            num_classes=num_classes,
        )

        self.model = RTDETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            multi_scale=multi_scale,
        )

    def forward(self, x, targets=None):
        """model forward function"""
        x = self.model(x, targets)
        return x


def build_model(experiment_config,
                export=False):
    """ Build dino model according to configuration.

    Args:
        experiment_config (OmegaConf): experiment configuration.
        export (bool): flag to indicate onnx export.

    Returns:
        model (nn.Module): DINO model.
    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset

    backbone = model_config.backbone
    train_backbone = model_config.train_backbone
    
    pretrained_backbone = model_config.pretrained_backbone_path
    return_interm_indices = model_config.return_interm_indices
    model = RTDETRModel(
        backbone=backbone,
        train_backbone=train_backbone,
        pretrained_backbone=pretrained_backbone,
        out_indices=return_interm_indices,
    )
    return model

# if __name__ == "__main__":
#     rtdetr = RTDETRModel(backbone="resnet_50",
#                          pretrained_backbone="/home/scratch.p3/sean/dino/resnet50-0676ba61.pth")

#     rtdetr.eval()
#     import torch
#     img = torch.randn(1, 3, 640, 640)

#     with torch.no_grad():
#         outs = rtdetr(img)

#     for k, v in outs.items():
#         print(k, v.shape)
