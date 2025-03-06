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
SegFormer_pl Model builder Unit Tests
"""
import pytest
from omegaconf import OmegaConf

from nvidia_tao_core.config.segformer_pl.default_config import SFModelConfig, SFDatasetConfig, ExperimentConfig
from nvidia_tao_pytorch.cv.segformer_pl.model.segformer_pl_model import build_model
from nvidia_tao_pytorch.cv.segformer_pl.model.segformer import SegFormer

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
OUTPUT_SHAPE = 224

@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(SFDatasetConfig())
    model_config = OmegaConf.structured(SFModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone",
                         [("mit_b0"),
                          ("mit_b1"),
                          ("mit_b2"),
                          ("mit_b3"),
                          ("mit_b4"),
                          ("mit_b5"),
                          ("fan_tiny_8_p4_hybrid"),
                          ("fan_large_16_p4_hybrid"),
                          ("fan_small_12_p4_hybrid"),
                          ("fan_base_16_p4_hybrid"),
                          ("vit_large_nvdinov2"),
                          ("vit_giant_nvdinov2"),
                          ("vit_base_nvclip_16_siglip"),
                          ("vit_huge_nvclip_14_siglip"),
                          ("c_radio_v2_vit_base_patch16_224"),
                          ("c_radio_v2_vit_large_patch16_224"),
                          ("c_radio_v2_vit_huge_patch16_224")])
@pytest.mark.parametrize("export", [False, True])
def test_changenet_model_segment(_test_experiment_spec, backbone, export):
    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["dataset"]['segment']["img_size"] = OUTPUT_SHAPE

    model = build_model(_test_experiment_spec, export)
    assert(isinstance(model, SegFormer))
