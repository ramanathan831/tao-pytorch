# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from omegaconf import OmegaConf
import os
import tempfile
import torch
import onnx
import onnxruntime as ort
import numpy as np

from nvidia_tao_core.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.utils.onnx_export import Sparse4DExporter

ANCHOR_PATH = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/_ov_kmeans900_sample100_.npy"
CHECKPOINT_PATH = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/sparse4d/SURF_Booth_031325/sparse4d_tracking_aic25v0.3_moving_classes_iter_60900_v1.1.pth"

@pytest.fixture
def _test_experiment_spec():
    """Creates a minimal ExperimentConfig for testing export."""
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model.head.instance_bank.anchor = ANCHOR_PATH
    cfg.model.head.deformable_model.use_camera_embed = True
    cfg.dataset.classes = ['person', 'gr1_t2', 'agility_digit', 'nova_carter', 'transporter', 'forklift', 'pallet']
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.export
# @pytest.mark.parametrize("batch_size", [-1, 1]) # Parameterize if needed
def test_sparse4d_onnx_export(_test_experiment_spec, batch_size=1): # Fixed batch size for simplicity first
    """Unit test for ONNX export on Sparse4D model."""
    cfg = _test_experiment_spec

    model_path = CHECKPOINT_PATH
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    output_file = tmp_onnx_file
    on_cpu = cfg.export.on_cpu

    # Instantiate the model first
    model = Sparse4DPlModel(cfg)

    model.eval()
    if not on_cpu:
        model.cuda()

    sparse4d_exporter = Sparse4DExporter(model)
    sparse4d_exporter.export_model(
        cfg,
        model,
        output_file
    )
    sparse4d_exporter.check_onnx(output_file)
    assert os.path.exists(output_file), "ONNX file was not generated properly!"
