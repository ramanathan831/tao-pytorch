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

import os
import pytest
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import tempfile

import torch
from mmengine.runner import Runner

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.mmlab.common.utils import get_latest_pth_model
from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import ExperimentConfig
from nvidia_tao_pytorch.core.mmlab.mmclassification.logistic_regression_trainer import LogisticRegressionTrainer as LRTrainer
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import MMPretrainConfig
from nvidia_tao_pytorch.core.utilities import check_and_create, check_and_delete
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614

tmp_top_dir = "tests/cv_unit_test/classification/tmp_test_data_dir/"
tmp_results_dir = "tests/cv_unit_test/classification/tmp_results/"
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
json_file = os.path.join(tmp_top_dir, "status.json")

@pytest.fixture
def _test_dir():
    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_foreground_dir = os.path.join(tmp_top_dir, "foreground")
    tmp_background_dir = os.path.join(tmp_top_dir, "background")
    classes_file = os.path.join(tmp_top_dir, "classes.txt")
    with open(classes_file, "w+") as f:
        f.write("%s \n %s \n" % ("foreground", "background"))
    check_and_create(tmp_foreground_dir)
    check_and_create(tmp_background_dir)
    test_data = np.random.rand(320, 320, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)
    im.save(os.path.join(tmp_foreground_dir, "test.jpg"))
    im.save(os.path.join(tmp_background_dir, "test.jpg"))
    yield tmp_top_dir #, log_file
    check_and_delete(tmp_top_dir)


@pytest.fixture
def _test_exp_spec():
    classes_file = os.path.join(tmp_top_dir, "classes.txt")
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]["data"]["train"]["data_prefix"] = tmp_top_dir
    experiment_config["dataset"]["data"]["val"]["data_prefix"] = tmp_top_dir
    experiment_config["dataset"]["data"]["test"]["data_prefix"] = tmp_top_dir
    experiment_config["results_dir"] = tmp_results_dir
    experiment_config["dataset"]["data"]["train"]["classes"] = classes_file 
    experiment_config["dataset"]["data"]["val"]["classes"] = classes_file
    experiment_config["dataset"]["data"]["test"]["classes"] = classes_file
    experiment_config["dataset"]["data"]["samples_per_gpu"] = 1
    experiment_config["dataset"]["data"]["workers_per_gpu"] = 1
    experiment_config["train"]["train_config"]["runner"]["max_epochs"] = 1
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, head, head_custom_args, backbone_custom_args, input_size",
                         [("fan_tiny_8_p4_hybrid", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_small_12_p4_hybrid","TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_base_16_p4_hybrid","TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_large_16_p4_hybrid", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_base_18_p16_224", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_tiny_12_p16_224", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_small_12_p16_224_se_attn", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_small_12_p16_224", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_large_24_p16_224", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224),
                          ("gc_vit_xxtiny", "TAOLinearClsHead", None, None, 224),
                          ("gc_vit_xtiny", "TAOLinearClsHead", None, None, 224),
                          ("gc_vit_tiny", "TAOLinearClsHead", None, None, 224),
                          ("gc_vit_small", "TAOLinearClsHead", None, None, 224),
                          ("gc_vit_base", "TAOLinearClsHead", None, None, 224),
                          ("gc_vit_large", "TAOLinearClsHead", None, None, 224),
                          ("vit_large_patch14_dinov2_swiglu", "TAOLinearClsHead", {"in_channels": 1024}, None, 224),
                          ("faster_vit_0_224", "TAOLinearClsHead", None, None, 224),
                          ("faster_vit_1_224", "TAOLinearClsHead", None, None, 224),
                          ("faster_vit_2_224", "TAOLinearClsHead", None, None, 224),
                          ("faster_vit_3_224", "TAOLinearClsHead", None, None, 224),
                          ("faster_vit_4_224", "TAOLinearClsHead", None, None, 224),
                          ("faster_vit_4_21k_224", "TAOLinearClsHead", None, None, 224),
                          ("faster_vit_4_21k_384", "TAOLinearClsHead", None, None, 384),
                          ("faster_vit_4_21k_512", "TAOLinearClsHead", None, None, 512),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "TAOLinearClsHead", {"in_channels": 1536}, None, 224),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "TAOLinearClsHead", {"in_channels": 1536}, None, 336),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 512}, {"model_name":"ViT-B-32"}, 224),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 512}, {"model_name":"ViT-B-32"}, 336),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 1024}, {"model_name":"ViT-H-14-SigLIP-CLIPA-224"}, 224),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-336"}, 336),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-224"}, 224),
                          ("fan_tiny_8_p4_hybrid", "LogisticRegressionHead",  {"head_init_scale": 1.0}, None, 224),
                          ("fan_small_12_p4_hybrid","LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_base_16_p4_hybrid","LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_large_16_p4_hybrid", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_base_18_p16_224", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_tiny_12_p16_224", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_small_12_p16_224_se_attn", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_small_12_p16_224", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("fan_large_24_p16_224", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224),
                          ("gc_vit_xxtiny", "LogisticRegressionHead", None, None, 224),
                          ("gc_vit_xtiny", "LogisticRegressionHead", None, None, 224),
                          ("gc_vit_tiny", "LogisticRegressionHead", None, None, 224),
                          ("gc_vit_small", "LogisticRegressionHead", None, None, 224),
                          ("gc_vit_base", "LogisticRegressionHead", None, None, 224),
                          ("gc_vit_large", "LogisticRegressionHead", None, None, 224),
                          ("vit_large_patch14_dinov2_swiglu", "LogisticRegressionHead", {"in_channels": 1024}, None, 224),
                          ("faster_vit_0_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_1_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_2_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_3_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_4_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_5_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_6_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_4_21k_224", "LogisticRegressionHead", None, None, 224),
                          ("faster_vit_4_21k_384", "LogisticRegressionHead", None, None, 384),
                          ("faster_vit_4_21k_512", "LogisticRegressionHead", None, None, 512),
                          ("faster_vit_4_21k_768", "LogisticRegressionHead", None, None, 768),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "LogisticRegressionHead", {"in_channels": 1536}, None, 224),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "LogisticRegressionHead", {"in_channels": 1536}, None, 336),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 512}, {"model_name":"ViT-B-32"}, 224),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 512}, {"model_name":"ViT-B-32"}, 336),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 1024}, {"model_name":"ViT-H-14-SigLIP-CLIPA-224"}, 224),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-336"}, 336),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-224"}, 224),
                          ])
def test_build_dataloader(_test_dir, _test_exp_spec, backbone, head,
                          head_custom_args, backbone_custom_args, input_size):
    results_dir = _test_exp_spec["results_dir"]
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=json_file,
            append=True
        )
    )
    _test_exp_spec["model"]["backbone"]["type"] = backbone
    _test_exp_spec["model"]["head"]["type"] = head

    _test_exp_spec["model"]["head"]["custom_args"] = head_custom_args
    _test_exp_spec["model"]["backbone"]["custom_args"] = backbone_custom_args

    _test_exp_spec["dataset"]["data"]["train"]["pipeline"] = [{"type": "Resize", "scale": int(input_size)}]
    _test_exp_spec["dataset"]["data"]["val"]["pipeline"] = [{"type": "Resize", "scale": int(input_size)}]
    _test_exp_spec["dataset"]["data"]["test"]["pipeline"] = [{"type": "Resize", "scale": int(input_size)}]
    
    mmpretrain_config = MMPretrainConfig(_test_exp_spec, phase="train")
    train_cfg = mmpretrain_config.updated_config
    train_cfg["work_dir"] = results_dir
    train_cfg.pop("launcher")
    if _test_exp_spec.model.head.type == "LogisticRegressionHead":
        lr_trainer = LRTrainer(train_cfg=_test_exp_spec,
                               updated_config=train_cfg,
                               status_logger=status_logging.get_status_logger())
        lr_trainer.fit()
        model = lr_trainer.model
        model.eval()
        with torch.no_grad():
            weights = lr_trainer.classifier.coef_
            model.head.fc.weight.data.copy_(torch.from_numpy(weights))
            biases = lr_trainer.classifier.intercept_
            model.head.fc.bias.data.copy_(torch.from_numpy(biases))

    else:
        resume_checkpoint = get_latest_pth_model(results_dir)
        if resume_checkpoint:
            train_cfg["load_from"] = resume_checkpoint
            train_cfg["resume"] = True
            train_cfg["model"]["backbone"]["init_cfg"] = None  # Disable pretrained weights if there are any
        train_cfg["work_dir"] = results_dir
        runner = Runner.from_cfg(train_cfg)
        runner.train()

    check_and_delete(tmp_top_dir)
    check_and_delete(tmp_results_dir)
    tmp_top_obj.cleanup()
