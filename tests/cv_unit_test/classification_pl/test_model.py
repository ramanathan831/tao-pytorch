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
Classification_pl Model builder Unit Tests
"""
import pytest
from omegaconf import OmegaConf

from nvidia_tao_core.config.classification_pyt.default_config import ModelConfig, DatasetConfig, ExperimentConfig
from nvidia_tao_pytorch.cv.classification_pl.model.classifier_pl_model import build_model
from nvidia_tao_pytorch.cv.classification_pl.model.classifier import Classifier

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
OUTPUT_SHAPE = 224

@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DatasetConfig())
    model_config = OmegaConf.structured(ModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone",
                         [("fan_tiny_8_p4_hybrid"),
                          ("fan_small_12_p4_hybrid"),
                          ("fan_base_16_p4_hybrid"),
                          ("fan_large_16_p4_hybrid"),
                          ("fan_Xlarge_16_p4_hybrid"),
                          ("fan_base_18_p16_224"),
                          ("fan_tiny_12_p16_224"),
                          ("fan_small_12_p16_224_se_attn"),
                          ("fan_small_12_p16_224"),
                          ("fan_large_24_p16_224"),
                          ("vit_large_patch14_dinov2_swiglu"),
                          ("vit_giant_patch14_reg4_dinov2_swiglu"),
                          ("ViT-H-14-SigLIP-CLIPA-224"),
                          ("ViT-L-14-SigLIP-CLIPA-336"),
                          ("ViT-L-14-SigLIP-CLIPA-224"),
                          ("c_radio_p1_vit_huge_patch16_mlpnorm"),
                          ("c_radio_p2_vit_huge_patch16_mlpnorm"),
                          ("c_radio_p3_vit_huge_patch16_mlpnorm"),
                          ("c_radio_v2_vit_base_patch16"),
                          ("c_radio_v2_vit_large_patch16"),
                          ("c_radio_v2_vit_huge_patch16")])
# for classification_pl, export or not is not affecting anything
@pytest.mark.parametrize("export", [False, True])
def test_changenet_model(_test_experiment_spec, backbone, export):
    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["dataset"]["img_size"] = OUTPUT_SHAPE

    model = build_model(_test_experiment_spec, export)
    assert(isinstance(model, Classifier))
