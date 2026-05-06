# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

""" Unit test for NVPanoptix3D 2D stage model. """

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.nvpanoptix3d.model import NVPanoptix3DModelConfig
from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.model_2d import MaskFormerModel


@pytest.fixture
def _test_experiment_spec():
    """Test experiment spec."""
    model_config = OmegaConf.structured(NVPanoptix3DModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("export", [False, True])
def test_2d_model(_test_experiment_spec, export):
    """Test 2D model."""
    model = MaskFormerModel(_test_experiment_spec, export=export)
    assert model is not None, "Model instantiation failed."
