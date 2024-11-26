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

"""Configuration hyperparameter schema for the trainer."""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    DATACLASS_FIELD,
)

from nvidia_tao_core.config.common.common_config import TrainConfig


@dataclass
class NVDINOv2BaseSchedulerConfig:
    """Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=1e-8,
        default_value=1e-8,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=1e-6,
        default_value=1e-6,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )
    val_start: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Starting value for scheduler",
        display_name="starting value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=0,
        default_value=0,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )
    max_decay_steps: int = INT_FIELD(
        value=2500000,
        default_value=2500000,
        description="Maximum decay steps",
        display_name="max decay steps",
        popular="yes"
    )


@dataclass
class NVDINOv2LearningRateConfig(NVDINOv2BaseSchedulerConfig):
    """Learning Rate Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=7.07e-6,
        default_value=7.07e-6,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=100000,
        default_value=100000,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )


@dataclass
class NVDINOv2LastLayerLearningRateConfig(NVDINOv2BaseSchedulerConfig):
    """Last Layer Learning Rate Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=7.07e-6,
        default_value=7.07e-6,
        description="The value after warm-up for scheduler.",
        display_name="base value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=100000,
        default_value=100000,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )
    freeze_steps: int = INT_FIELD(
        value=1250,
        default_value=1250,
        description="Number of freeze steps",
        display_name="freeze steps",
        popular="yes"
    )


@dataclass
class NVDINOv2WeightDecayConfig(NVDINOv2BaseSchedulerConfig):
    """Weight Decay Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=0.04,
        default_value=0.04,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )


@dataclass
class NVDINOv2MomentumConfig(NVDINOv2BaseSchedulerConfig):
    """Momentum Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=0.994,
        default_value=0.994,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )


@dataclass
class NVDINOv2TeacherTemperatureConfig(NVDINOv2BaseSchedulerConfig):
    """Teacher Temperature Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=0.07,
        default_value=0.07,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=0.07,
        default_value=0.07,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )
    val_start: float = FLOAT_FIELD(
        value=0.04,
        default_value=0.04,
        description="Starting value for scheduler",
        display_name="starting value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=37500,
        default_value=37500,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )
    max_decay_steps: int = INT_FIELD(
        value=37500,
        default_value=37500,
        description="Maximum decay steps",
        display_name="max decay steps",
        popular="yes"
    )


@dataclass
class NVDINOv2SchedulerConfig:
    """Schedulers Config."""

    learning_rate: NVDINOv2LearningRateConfig = DATACLASS_FIELD(
        NVDINOv2LearningRateConfig(),
        description="Learning rate scheduler configuration"
    )
    last_layer_learning_rate: NVDINOv2LastLayerLearningRateConfig = DATACLASS_FIELD(
        NVDINOv2LastLayerLearningRateConfig(),
        description="Last layer learning rate scheduler configuration"
    )
    weight_decay: NVDINOv2WeightDecayConfig = DATACLASS_FIELD(
        NVDINOv2WeightDecayConfig(),
        description="Weight decay scheduler configuration"
    )
    momentum: NVDINOv2MomentumConfig = DATACLASS_FIELD(
        NVDINOv2MomentumConfig(),
        description="Momentum scheduler configuration"
    )
    teacher_temperature: NVDINOv2TeacherTemperatureConfig = DATACLASS_FIELD(
        NVDINOv2TeacherTemperatureConfig(),
        description="Teacher temperature scheduler configuration"
    )


@dataclass
class NVDINOv2OptimConfig:
    """Optimizer config."""

    optim: str = STR_FIELD(
        value="adamw",
        default_value="adamw",
        description="Optimizer type",
        display_name="optimizer",
        valid_options="adamw,",
        popular="yes"
    )


@dataclass
class NVDINOv2TrainExpConfig(TrainConfig):
    """Train Config."""

    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description="Path to a pre-trained NVDINOv2 model to initialize the current training from."
    )
    max_steps: int = INT_FIELD(
        value=2500000,
        default_value=2500000,
        valid_min=1,
        valid_max="inf",
        description="Maximum number of training steps",
        display_name="max training steps",
        popular="yes"
    )
    checkpoint_step_interval: int = INT_FIELD(
        value=1250,
        default_value=1250,
        valid_min=1,
        valid_max="inf",
        description="Interval steps to save checkpoint",
        display_name="checkpoint step interval",
        popular="yes"
    )
    layerwise_decay: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Layerwise decay factor",
        display_name="layerwise decay factor",
        popular="yes"
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Value to clip gradients norm",
        display_name="clip gradient norm",
        popular="yes"
    )
    num_prototypes: int = INT_FIELD(
        value=131072,
        default_value=131072,
        description="Number of prototypes",
        display_name="number of prototypes",
        popular="yes"
    )
    schedulers: NVDINOv2SchedulerConfig = DATACLASS_FIELD(
        NVDINOv2SchedulerConfig(),
        description="Schedulers configuration for NVDINOv2 training"
    )
    optim: NVDINOv2OptimConfig = DATACLASS_FIELD(
        NVDINOv2OptimConfig(),
        description="Optimizer configuration for NVDINOv2"
    )
