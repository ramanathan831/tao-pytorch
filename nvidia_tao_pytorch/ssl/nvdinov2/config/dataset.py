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

"""Configuration hyperparameter schema for the dataset."""

from typing import List
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    images_dir: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to images directory for dataset",
        display_name="image directory"
    )


@dataclass
class NVDINOv2TransformConfig:
    """NVDINOv2 Data Transform Config."""

    n_global_crops: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        description="Number of global crops to generate",
        display_name="Number of Global Crops",
        popular="yes"
    )
    global_crops_scale: List[float] = LIST_FIELD(
        arrList=[0.32, 1.0],
        default_value=[0.32, 1.0],
        description="Scale range for global crops",
        display_name="Global Crops Scale",
        popular="yes"
    )
    global_crops_size: int = INT_FIELD(
        value=224,
        default_value=224,
        valid_min=1,
        valid_max="inf",
        description="Size of global crops",
        display_name="Global Crops Size",
        popular="yes"
    )
    n_local_crops: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of local crops to generate",
        display_name="Number of Local Crops",
        popular="yes"
    )
    local_crops_scale: List[float] = LIST_FIELD(
        arrList=[0.05, 0.32],
        default_value=[0.05, 0.32],
        description="Scale range for local crops",
        display_name="Local Crops Scale",
        popular="yes"
    )
    local_crops_size: int = INT_FIELD(
        value=98,
        default_value=98,
        valid_min=1,
        valid_max="inf",
        description="Size of local crops",
        display_name="Local Crops Size",
        popular="yes"
    )


@dataclass
class NVDINOv2DatasetConfig:
    """NVDINOv2 Dataset Config."""

    train_dataset: DataPathFormat = DATACLASS_FIELD(
        DataPathFormat(),
        description="Configuration for the training dataset path",
        display_name="Training Dataset"
    )
    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="The batch size for training",
        automl_enabled="TRUE",
        display_name="batch size",
        popular="yes"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="""Flag to enable the dataloader to allocated pagelocked memory for faster
                    of data between the CPU and GPU.""",
        popular="yes"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="The number of parallel workers processing data",
        automl_enabled="TRUE",
        display_name="batch size",
        popular="yes"
    )
    transform: NVDINOv2TransformConfig = DATACLASS_FIELD(
        NVDINOv2TransformConfig(),
        description="Configuration parameters for data transformation",
        display_name="transform",
    )
