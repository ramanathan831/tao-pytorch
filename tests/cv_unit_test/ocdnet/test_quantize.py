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

"""Test cases for OCDNet quantization functionality."""

import pytest

from nvidia_tao_core.config.ocdnet.default_config import (
    ExperimentConfig,
    OCDNetDataConfig,
    QuantCalibrationDataset,
)


@pytest.mark.cv_unit
def test_dataset_config_has_quant_calibration_dataset():
    """Test that OCDNetDataConfig has quant_calibration_dataset field."""
    data_config = OCDNetDataConfig()
    assert hasattr(data_config, "quant_calibration_dataset")
    assert isinstance(data_config.quant_calibration_dataset, QuantCalibrationDataset)


@pytest.mark.cv_unit
def test_quant_calibration_dataset_has_images_dir():
    """Test that QuantCalibrationDataset has images_dir field."""
    calib_config = QuantCalibrationDataset()
    assert hasattr(calib_config, "images_dir")
    assert calib_config.images_dir == ""


@pytest.mark.cv_unit
def test_experiment_config_has_quantize():
    """Test that ExperimentConfig has quantize field."""
    exp_config = ExperimentConfig()
    assert hasattr(exp_config, "quantize")


@pytest.mark.cv_unit
def test_quantize_config_has_required_fields():
    """Test that quantize config has all required fields."""
    exp_config = ExperimentConfig()
    quantize_config = exp_config.quantize
    assert hasattr(quantize_config, "backend")
    assert hasattr(quantize_config, "mode")
    assert hasattr(quantize_config, "algorithm")
    assert hasattr(quantize_config, "model_path")
    assert hasattr(quantize_config, "results_dir")
