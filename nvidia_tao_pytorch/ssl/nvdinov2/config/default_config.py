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

"""Default config file"""

from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import DATACLASS_FIELD
from nvidia_tao_core.config.common.common_config import CommonExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.config.dataset import NVDINOv2DatasetConfig
from nvidia_tao_pytorch.ssl.nvdinov2.config.model import NVDINOv2ModelConfig
from nvidia_tao_pytorch.ssl.nvdinov2.config.train import NVDINOv2TrainExpConfig


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: NVDINOv2ModelConfig = DATACLASS_FIELD(
        NVDINOv2ModelConfig(),
        description="Configurable parameters to construct the model for a NVDINOv2 experiment.",
    )
    dataset: NVDINOv2DatasetConfig = DATACLASS_FIELD(
        NVDINOv2DatasetConfig(),
        description="Configurable parameters to construct the dataset for a NVDINOv2 experiment.",
    )
    train: NVDINOv2TrainExpConfig = DATACLASS_FIELD(
        NVDINOv2TrainExpConfig(),
        description="Configurable parameters to construct the trainer for a NVDINOv2 experiment.",
    )
