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

"""Test cases for DINO quantization functionality."""

import pytest

from nvidia_tao_core.config.dino.dataset import DINODatasetConfig
from nvidia_tao_core.config.dino.default_config import ExperimentConfig


@pytest.mark.cv_unit
def test_dataset_config_has_quant_calibration_data_sources():
    """Test that DINODatasetConfig has quant_calibration_data_sources field."""
    config = DINODatasetConfig()
    assert hasattr(config, 'quant_calibration_data_sources'), (
        "DINODatasetConfig should have quant_calibration_data_sources field"
    )


@pytest.mark.cv_unit
def test_experiment_config_has_quantize():
    """Test that ExperimentConfig has quantize field."""
    config = ExperimentConfig()
    assert hasattr(config, 'quantize'), (
        "ExperimentConfig should have quantize field"
    )


@pytest.mark.cv_unit
def test_quantize_config_structure():
    """Test that quantize config has expected structure."""
    config = ExperimentConfig()
    quantize_config = config.quantize

    # Check required fields exist
    assert hasattr(quantize_config, 'backend'), "quantize should have backend field"
    assert hasattr(quantize_config, 'mode'), "quantize should have mode field"
    assert hasattr(quantize_config, 'model_path'), "quantize should have model_path field"
