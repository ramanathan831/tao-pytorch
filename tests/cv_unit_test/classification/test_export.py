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
import subprocess
import sys
import pytest
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import tempfile

import torch
from mmengine.runner import Runner

from nvidia_tao_core.config.classification_pyt.default_config import ExperimentConfig
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.mmlab.mmclassification.logistic_regression_trainer import LogisticRegressionTrainer as LRTrainer
from nvidia_tao_pytorch.core.mmlab.mmclassification.utils import load_model, MMPretrainConfig
from nvidia_tao_pytorch.core.utilities import check_and_create, check_and_delete
from nvidia_tao_pytorch.cv.classification.heads import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.models import *  # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.classification.tools.onnx_utils import pytorch_to_onnx

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

    yield tmp_top_dir
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
@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.parametrize("model_config",
                         [("fan_tiny_12_p16_224", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224, None),
                          ("gc_vit_xxtiny", "TAOLinearClsHead", None, None, 224, None),
                          ("gc_vit_large_384", "TAOLinearClsHead", None, None, 384, None),
                          ("vit_large_patch14_dinov2_swiglu", "TAOLinearClsHead", {"in_channels": 1024}, None, 224, None),
                          ("c_radio_p1_vit_huge_patch16_224_mlpnorm", "TAOLinearClsHead", {"in_channels": 3840}, None, 224, None),
                          ("c_radio_p2_vit_huge_patch16_224_mlpnorm", "TAOLinearClsHead", {"in_channels": 5120}, None, 224, None),
                          ("c_radio_p3_vit_huge_patch16_224_mlpnorm", "TAOLinearClsHead", {"in_channels": 3840}, None, 224, None),
                          ("fan_tiny_8_p4_hybrid", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224, None),
                          ("open_clip", "TAOLinearClsHead", None, {"model_name":"ViT-B-32"}, 224, "laion2b_s34b_b79k"),
                          ("open_clip", "TAOLinearClsHead", None, {"model_name":"ViT-B-32"}, 336, "laion2b_s34b_b79k"),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 1024}, {"model_name":"ViT-H-14-SigLIP-CLIPA-224"}, 224, None),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-336"}, 336, None),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-224"}, 224, None),
                          ("faster_vit_0_224", "TAOLinearClsHead", None, None, 224, None),
                          ("faster_vit_4_21k_768", "TAOLinearClsHead", None, None, 768, None),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "TAOLinearClsHead", {"in_channels": 1536}, None, 224, None),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "TAOLinearClsHead", {"in_channels": 1536}, None, 336, None),
                          ("fan_tiny_12_p16_224", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224, None),
                          ("gc_vit_xxtiny", "LogisticRegressionHead", None, None, 224, None),
                          ("gc_vit_large_384", "LogisticRegressionHead", None, None, 384, None),
                          ("vit_large_patch14_dinov2_swiglu", "LogisticRegressionHead", {"in_channels": 1024}, None, 224, None),
                          ("c_radio_p1_vit_huge_patch16_224_mlpnorm", "LogisticRegressionHead", {"in_channels": 3840}, None, 224, None),
                          ("c_radio_p2_vit_huge_patch16_224_mlpnorm", "LogisticRegressionHead", {"in_channels": 5120}, None, 224, None),
                          ("c_radio_p3_vit_huge_patch16_224_mlpnorm", "LogisticRegressionHead", {"in_channels": 3840}, None, 224, None),
                          ("fan_tiny_8_p4_hybrid", "LogisticRegressionHead", {"head_init_scale": 1.0}, None, 224, None),
                          ("open_clip", "LogisticRegressionHead", None, {"model_name":"ViT-B-32"}, 224, "laion2b_s34b_b79k"),
                          ("open_clip", "LogisticRegressionHead", None, {"model_name":"ViT-B-32"}, 336, "laion2b_s34b_b79k"),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 1024}, {"model_name":"ViT-H-14-SigLIP-CLIPA-224"}, 224, None),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-336"}, 336, None),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-224"}, 224, None),
                          ("faster_vit_0_224", "LogisticRegressionHead", None, None, 224, None),
                          ("faster_vit_4_21k_768", "LogisticRegressionHead", None, None, 768, None),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "LogisticRegressionHead", {"in_channels": 1536}, None, 224, None),
                          ("vit_giant_patch14_reg4_dinov2_swiglu", "LogisticRegressionHead", {"in_channels": 1536}, None, 336, None),
                          ])
@pytest.mark.parametrize("opset_version", [15, 17])
def test_cls_onnx_export(_test_exp_spec, _test_dir, model_config, opset_version):
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=json_file,
            append=True
        )
    )
    check_and_create(tmp_top_dir)
    backbone, head, head_custom_args, bb_custom_args, input_resolution, pretrained = model_config
    _test_exp_spec["model"]["backbone"]["type"] = backbone
    _test_exp_spec["model"]["head"]["type"] = head
    _test_exp_spec["model"]["head"]["custom_args"] = head_custom_args
    _test_exp_spec["model"]["backbone"]["custom_args"] = bb_custom_args

    results_dir = _test_exp_spec["results_dir"]
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
        train_cfg["work_dir"] = results_dir
        runner = Runner.from_cfg(train_cfg)
        model = runner.model

    model_name = bb_custom_args.get("model_name", "") if bb_custom_args else ""
    onnx_out_dir = os.path.join(tmp_results_dir,
                                f"{backbone}_{head}_{model_name}_{input_resolution}_{opset_version}")
    check_and_create(onnx_out_dir)

    onnx_path = os.path.join(onnx_out_dir, f"{backbone}_opset{opset_version}.onnx")
    if head == "LogisticRegressionHead":
        checkpoint = {}
        with torch.no_grad():
            weights = lr_trainer.classifier.coef_
            model.head.fc.weight.data.copy_(torch.from_numpy(weights))
            biases = lr_trainer.classifier.intercept_
            model.head.fc.bias.data.copy_(torch.from_numpy(biases))
        checkpoint['state_dict'] = model.state_dict()
        torch.save(checkpoint, os.path.join(onnx_out_dir, "model_lrHead_0.pth"))
        mmpretrain_config = MMPretrainConfig(_test_exp_spec, phase="evaluate")
        export_cfg = mmpretrain_config.updated_config
        model = load_model(os.path.join(onnx_out_dir, "model_lrHead_0.pth"), export_cfg)
        onnx_path = os.path.join(onnx_out_dir, f"{backbone}_lrhead_opset{opset_version}.onnx")
        os.remove(os.path.join(onnx_out_dir, "model_lrHead_0.pth"))

    input_shape = [1, 3, input_resolution, input_resolution]
    # export binary model
    pytorch_to_onnx(
        model,
        input_shape,
        opset_version=opset_version,
        show=False,
        output_file=onnx_path,
        verify=False,
        num_classes=2)

    assert os.path.exists(onnx_path), f"Failed to generate ONNX file at {onnx_path}"


@pytest.mark.cv_unit
@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.parametrize("model_config",
                         [("fan_tiny_12_p16_224", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224, None),
                          ("gc_vit_xxtiny", "TAOLinearClsHead", None, None, 224, None),
                          ("vit_large_patch14_dinov2_swiglu", "TAOLinearClsHead", {"in_channels": 1024}, None, 224, None),
                          ("c_radio_p1_vit_huge_patch16_224_mlpnorm", "TAOLinearClsHead", {"in_channels": 3840}, None, 224, None),
                          ("c_radio_p2_vit_huge_patch16_224_mlpnorm", "TAOLinearClsHead", {"in_channels": 5120}, None, 224, None),
                          ("c_radio_p3_vit_huge_patch16_224_mlpnorm", "TAOLinearClsHead", {"in_channels": 3840}, None, 224, None),
                          ("open_clip", "TAOLinearClsHead", None, {"model_name":"ViT-B-32"}, 224, "laion2b_s34b_b79k"),
                          ("open_clip", "TAOLinearClsHead", None, {"model_name":"ViT-B-32"}, 336, "laion2b_s34b_b79k"),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 1024}, {"model_name":"ViT-H-14-SigLIP-CLIPA-224"}, 224, None),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-336"}, 336, None),
                          ("open_clip", "TAOLinearClsHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-224"}, 224, None),
                          ("faster_vit_0_224", "TAOLinearClsHead", None, None, 224, None),
                          ("fan_tiny_8_p4_hybrid", "TAOLinearClsHead", {"head_init_scale": 1.0}, None, 224, None),
                          ("gc_vit_xxtiny", "LogisticRegressionHead", None, None, 224, None),
                          ("vit_large_patch14_dinov2_swiglu", "LogisticRegressionHead", {"in_channels": 1024}, None, 224, None),
                          ("c_radio_p1_vit_huge_patch16_224_mlpnorm", "LogisticRegressionHead", {"in_channels": 3840}, None, 224, None),
                          ("c_radio_p2_vit_huge_patch16_224_mlpnorm", "LogisticRegressionHead", {"in_channels": 5120}, None, 224, None),
                          ("c_radio_p3_vit_huge_patch16_224_mlpnorm", "LogisticRegressionHead", {"in_channels": 3840}, None, 224, None),
                          ("open_clip", "LogisticRegressionHead", None, {"model_name":"ViT-B-32"}, 224, "laion2b_s34b_b79k"),
                          ("open_clip", "LogisticRegressionHead", None, {"model_name":"ViT-B-32"}, 336, "laion2b_s34b_b79k"),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 1024}, {"model_name":"ViT-H-14-SigLIP-CLIPA-224"}, 224, None),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-336"}, 336, None),
                          ("open_clip", "LogisticRegressionHead", {"in_channels": 768}, {"model_name":"ViT-L-14-SigLIP-CLIPA-224"}, 224, None),
                          ("faster_vit_0_224", "LogisticRegressionHead", None, None, 224, None),
                          ])
@pytest.mark.parametrize("opset_version", [15])  # TODO: Add 17 when we upgrade to DLFW 23.04+
@pytest.mark.parametrize("batch_size", [1, 4])
def test_cls_trtexec(model_config, opset_version, batch_size):
    check_and_create(tmp_top_dir)
    # backbone, head, input_resolution = model_config
    backbone, head, head_custom_args, bb_custom_args, input_resolution, pretrained = model_config

    model_name = bb_custom_args.get("model_name", "") if bb_custom_args else ""
    onnx_root_dir = os.path.join(tmp_results_dir,
                                 f"{backbone}_{head}_{model_name}_{input_resolution}_{opset_version}")
    onnx_path = os.path.join(onnx_root_dir, f"{backbone}_opset{opset_version}.onnx")

    if head == "LogisticRegressionHead":
        onnx_path = os.path.join(onnx_root_dir, f"{backbone}_lrhead_opset{opset_version}.onnx")

    # Test TensorRT engine generation for dynamic batch size ONNX
    call = (
        f"trtexec --onnx={onnx_path} "
        f"--minShapes=input_1:{batch_size}x3x{input_resolution}x{input_resolution} "
        f"--optShapes=input_1:{batch_size}x3x{input_resolution}x{input_resolution} "
        f"--maxShapes=input_1:{batch_size}x3x{input_resolution}x{input_resolution} "
    )
    print(call)

    # Run the call and capture output
    result = subprocess.run(call, shell=True, capture_output=True, text=True)

    # Assert to check if the call was successful
    assert result.returncode == 0, f"Subprocess failed with error: {result.stderr}"

    # Run the call as subprocess.
    #subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

@pytest.mark.cv_unit
def clean_tmp_dirs():
    check_and_delete(tmp_top_dir)
    check_and_delete(tmp_results_dir)
