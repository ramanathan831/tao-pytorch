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

import torch.nn as nn 
import torch.nn.functional as F 

import random 


class RTDETR(nn.Module):

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = random.choice(self.multi_scale)
            if isinstance(sz, int):
                # square resize
                x = F.interpolate(x, size=[sz, sz])
            elif isinstance(sz, tuple) or isinstance(sz, list):
                x = F.interpolate(x, size=sz)
            else:
                raise TypeError(f"{sz} is {type(sz)}. Need to pass int / list / tuple for multi_scale")
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
