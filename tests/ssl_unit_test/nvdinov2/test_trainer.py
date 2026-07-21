# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVDINOv2 Trainer Unit Tests
"""
import os
import torch
import pytest
import tempfile
import numpy as np
from PIL import Image

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule

BATCH_SIZE = 2
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
SAMPLES = 100

@pytest.fixture()
def _test_dir_obj():
    tmp_obj = tempfile.TemporaryDirectory()
    check_and_create(tmp_obj.name)
    for sample in range(SAMPLES):
        test_data = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 3) * 255
        test_data = test_data.astype(np.uint8)
        im = Image.fromarray(test_data)
        save_name = f'test_{sample}.jpg'
        im.save(os.path.join(tmp_obj.name, save_name))

    yield tmp_obj

@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.train_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.results_dir = _test_dir_obj.name
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 3
    yield experiment_config

@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.train
@pytest.mark.ssl_unit
def test_trainer_fit(_test_dir_obj, _test_exp_spec):
    acc_flag = 'auto'
    precision = '16-mixed'

    dm = DinoV2DataModule(_test_exp_spec)
    model = DinoV2PlModel(_test_exp_spec)
    print(_test_exp_spec.train.num_gpus)
    trainer = Trainer(devices=_test_exp_spec.train.num_gpus,
                      num_nodes=_test_exp_spec.train.num_nodes,
                      max_epochs=_test_exp_spec.train.num_epochs,
                      check_val_every_n_epoch=_test_exp_spec.train.validation_interval,
                      default_root_dir=_test_exp_spec.results_dir,
                      accelerator='gpu',
                      strategy=acc_flag,
                      precision=precision,
                      )

    trainer.fit(model, dm, ckpt_path=None)

    _test_dir_obj.cleanup()


@pytest.mark.ssl_unit
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nvdinov2_nonfinite_train_loss_raises(bad_value):
    """A non-finite epoch train loss must raise so @monitor_status records FAILURE.

    Regression for the silent-PASS bug (6460915): train_loss is the sole in-loop AutoML KPI,
    written to status.json but previously never validated, so a NaN loss was reported as PASS.
    The guard lives on the shared base, so every SSL model (nvdinov2, dinov3, ...) inherits it.
    """
    from types import SimpleNamespace
    fake = SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics={"train_loss_epoch": torch.tensor(bad_value)})
    )
    with pytest.raises(ValueError, match="non-finite"):
        DinoV2PlModel._assert_train_loss_finite(fake)


@pytest.mark.ssl_unit
def test_nvdinov2_finite_train_loss_passes():
    """A finite epoch train loss must not raise."""
    from types import SimpleNamespace
    fake = SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics={"train_loss_epoch": torch.tensor(10.5)})
    )
    DinoV2PlModel._assert_train_loss_finite(fake)  # should not raise


@pytest.mark.ssl_unit
def test_nvdinov2_unsupported_backbone_type_raises():
    """The shared backbone-name validator raises a clear ValueError on an unsupported arch.

    Regression for bug 6460904: lives on the base so every SSL family inherits it. Uses the
    subclass's own param_map (its keys are the supported archs) as the single source of truth.
    """
    from nvidia_tao_pytorch.config.nvdinov2.default_config import map_params
    good = sorted(map_params["depth"].keys())[0]
    with pytest.raises(ValueError, match="Unsupported model.backbone"):
        DinoV2PlModel._validate_backbone_types(
            {"teacher_type": good, "student_type": "not_a_real_arch"}, map_params
        )
    # A fully-supported pair must not raise.
    DinoV2PlModel._validate_backbone_types(
        {"teacher_type": good, "student_type": good}, map_params
    )
