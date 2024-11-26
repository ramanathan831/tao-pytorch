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

"""Configuration hyperparameter schema for the model."""

from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_pytorch.ssl.nvdinov2.config.model_params_mapping import SUPPORTED_BACKBONES


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = STR_FIELD(
        value="vit_l",
        default_value="vit_l",
        display_name="backbone",
        description="""The backbone name of the model.
                    TAO implementation of NVDINOv2 support vit_l
                    """,
        valid_options=",".join(SUPPORTED_BACKBONES),
        popular="no"
    )
    num_register_tokens: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Number of register tokens",
        display_name="num register tokens",
        popular="yes"
    )
    drop_path_rate: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        description="Drop path rate for stochastic depth regularization",
        display_name="drop path rate",
        popular="yes"
    )
    patch_size: int = INT_FIELD(
        value=14,
        default_value=14,
        description="Size of patches",
        display_name="patch size",
        valid_options="14,16",
        popular="yes"
    )
    img_size: int = INT_FIELD(
        value=518,
        default_value=518,
        description="Size of images for the backbone",
        display_name="image size",
        valid_options="224,518",
        popular="yes"
    )


@dataclass
class NVDINOv2HeadConfig:
    """Configuration parameters for NVDINOv2 head."""

    num_layers: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="Number of layers in the NVDINOv2 head",
        display_name="number of Layers",
        popular="yes"
    )
    hidden_dim: int = INT_FIELD(
        value=2048,
        default_value=2048,
        valid_min=1,
        valid_max="inf",
        description="Dimension of the hidden layers in the NVDINOv2 head",
        display_name="hidden dimension",
        popular="yes"
    )
    bottleneck_dim: int = INT_FIELD(
        value=384,
        default_value=384,
        valid_min=1,
        valid_max="inf",
        description="Dimension of the bottleneck layer in the NVDINOv2 head",
        display_name="bottleneck dimension",
        popular="yes"
    )


@dataclass
class NVDINOv2ModelConfig:
    """NVDINOv2 Model config."""

    backbone: BackboneConfig = DATACLASS_FIELD(
        BackboneConfig(),
        description="Configuration for the NVDINOv2 backbone"
    )
    head: NVDINOv2HeadConfig = DATACLASS_FIELD(
        NVDINOv2HeadConfig(),
        description="Configuration for the NVDINOv2 head"
    )
