# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
Visual ChangeNet-Segmentation/Classification Model builder Unit Tests
"""
import pytest
from omegaconf import OmegaConf

from nvidia_tao_core.config.visual_changenet.default_config import CNModelConfig, CNDatasetConfig, ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.changenet import build_model, ChangeNetClassify
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet import build_model as build_model_segment
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet import ChangeNetSegment

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
OUTPUT_SHAPE = 128

@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(CNDatasetConfig())
    model_config = OmegaConf.structured(CNModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone",
                         [("fan_tiny_8_p4_hybrid"),
                          ("fan_large_16_p4_hybrid"),
                          ("fan_small_12_p4_hybrid"),
                          ("fan_base_16_p4_hybrid"),
                          ("vit_large_nvdinov2")])
@pytest.mark.parametrize("export", [False, True])
@pytest.mark.parametrize("difference_module", ['learnable', 'euclidean'])
@pytest.mark.parametrize("task", ['classify'])
def test_changenet_model(_test_experiment_spec, backbone, export, task, difference_module):
    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec.task = task
    _test_experiment_spec["dataset"]['classify']["image_width"] = IMAGE_WIDTH
    _test_experiment_spec["dataset"]['classify']["image_height"] = IMAGE_HEIGHT
    _test_experiment_spec["model"]['classify'].difference_module = difference_module

    model = build_model(_test_experiment_spec, export)
    assert(isinstance(model, ChangeNetClassify))


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone",
                         [("fan_tiny_8_p4_hybrid"),
                          ("fan_large_16_p4_hybrid"),
                          ("fan_small_12_p4_hybrid"),
                          ("fan_base_16_p4_hybrid"),
                          ("vit_large_nvdinov2")])
@pytest.mark.parametrize("export", [False, True])
@pytest.mark.parametrize("task", ['segment'])
def test_changenet_model_segment(_test_experiment_spec, backbone, export, task):
    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec.task = task
    _test_experiment_spec["dataset"]['segment']["img_size"] = OUTPUT_SHAPE

    model = build_model_segment(_test_experiment_spec, export)
    assert(isinstance(model, ChangeNetSegment))
