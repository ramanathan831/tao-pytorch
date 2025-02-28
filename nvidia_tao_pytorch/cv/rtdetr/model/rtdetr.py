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

""" RT-DETR model. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
torch.serialization.add_safe_globals([Namespace])


class RTDETR(nn.Module):
    """RT-DETR Module."""

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None, frozen_fm_cfg=None, export=False):
        """Init function."""
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.frozen_fm_cfg = frozen_fm_cfg
        self.export = export
        if frozen_fm_cfg and frozen_fm_cfg.enabled:
            if "radio" in frozen_fm_cfg.backbone:
                model_version = frozen_fm_cfg.checkpoint
                self.frozen_radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
                self.frozen_radio.float()
                self.frozen_radio.eval().cuda()
                self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                for _, param in self.frozen_radio.named_parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError("The backbone of the frozen FM must be `radio` for now.")

    def forward(self, x, targets=None):
        """Forward function."""
        if self.frozen_fm_cfg and self.frozen_fm_cfg.enabled:
            b, _, h, w = x.shape
            x_down = F.interpolate(x, size=[h // 2, w // 2])
            with torch.no_grad():
                summary, spatial_features = self.frozen_radio(x_down)
            spatial_features = spatial_features.view(b, 20, 20, -1).permute(0, 3, 1, 2)
            spatial_features = self.maxpool(spatial_features)

        feats = self.backbone(x)
        if self.frozen_fm_cfg and self.frozen_fm_cfg.enabled:
            feats.append(spatial_features)
            x, proj_feats = self.encoder(feats)
            x = x[:-1]
            x = self.decoder(x, targets, summary.view(b, 1, -1))
        else:
            x, proj_feats = self.encoder(feats)
            x = self.decoder(x, targets)
        if not self.export:
            x['bb_feats'] = feats
            x['srcs'] = proj_feats

        return x

    def deploy(self):
        """Convert to deploy mode."""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
