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

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.mmlab.mmclassification.classification_default_config import (
    ImgNormConfig,
    TrainData,
    ValData,
    TestData,
    DataConfig,
    DatasetConfig,
    DistParams,
    RunnerConfig,
    CheckpointConfig,
    LogConfig,
    ValidationConfig,
    ParamwiseConfig,
    EvaluationConfig,
    EnvConfig,
    TrainConfig,
    ExpConfig,
    TrainExpConfig,
    InferenceExpConfig,
    EvalExpConfig,
    TrtConfig,
    ExportExpConfig,
    LRHeadConfig,
    HeadConfig,
    InitCfg,
    BackboneConfig,
    TrainAugCfg,
    ModelConfig,
    GenTrtEngineExpConfig,
    ExperimentConfig
)
from nvidia_tao_pytorch.config.utils import create_json_schema, dataclass_to_json

sample_dataset_config = """
data:
    samples_per_gpu: 128
    workers_per_gpu: 8
    train:
        data_prefix: "/ImageNet2012/ImageNet2012/train"
        pipeline: # Augmentations alone
            - type: LoadImageFromFile
            - type: RandomResizedCrop
              scale: 224
            - type: ColorJitter
              brightness: 0.4
              contrast: 0.4
              saturation: 0.4
            - type: RandomFlip
              prob: 0.5
              direction: "horizontal"
    val:
        data_prefix: /ImageNet2012/ImageNet2012/val
    test:
        data_prefix: /ImageNet2012/ImageNet2012/val
"""


sample_model_config = """
backbone:
    type: "vit_large_patch14_dinov2_swiglu"
    freeze: True
head:
    type: "LogisticRegressionHead"
    loss:
        type: CrossEntropyLoss
        use_soft: False
    topk: [1, 5]
    lr_head:
        C: 0.316
        max_iter: 5000
        hpo: False
        cs_tune: [0.001, 0.01,  0.316, 1, 10, 1000, 10000]
        criteria: "accuracy"
    num_classes: ???
"""


sample_train_config = """
exp_config:
    manual_seed: 49
train_config:
    runner:
        max_epochs: 300
    checkpoint_config:
        interval: 1
    logging:
        interval: 5000
    validate: True
    evaluation:
        interval: 10
    optimizer:
        type: 'AdamW'
        lr: 10e-5
"""

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)
CONFIG_ROOT = os.path.join(
    ROOT_DIR,"nvidia_tao_pytorch/cv/classification/experiment_specs/"
)
train_config = os.path.join(CONFIG_ROOT, "train_lr_head_industrial.yaml")

with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_img_norm_spec():
    img_norm_config = ImgNormConfig()
    yield img_norm_config

@pytest.fixture
def _test_train_data_spec():
    train_data_config = TrainData()
    yield train_data_config

@pytest.fixture
def _test_val_data_spec():
    val_data_config = ValData()
    yield val_data_config

@pytest.fixture
def _test_test_data_spec():
    test_data_config = TestData()
    yield test_data_config

@pytest.fixture
def _test_data_config_spec():
    data_config = DataConfig()
    yield data_config

@pytest.fixture
def _test_dataset_config_spec():
    dataset_config = DatasetConfig()
    yield dataset_config

@pytest.fixture
def _test_dist_params_spec():
    dist_params_config = DistParams()
    yield dist_params_config

@pytest.fixture
def _test_runner_config_spec():
    runner_config = RunnerConfig()
    yield runner_config

@pytest.fixture
def _test_experiment_config_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config

@pytest.fixture
def _test_checkpoint_config_spec():
    checkpoint_config = CheckpointConfig()
    yield checkpoint_config

@pytest.fixture
def _test_log_config_spec():
    log_config = LogConfig()
    yield log_config

@pytest.fixture
def _test_validation_config_spec():
    validation_config = ValidationConfig()
    yield validation_config

@pytest.fixture
def _test_paramwise_config_spec():
    paramwise_config = ParamwiseConfig()
    yield paramwise_config

@pytest.fixture
def _test_evaluation_config_spec():
    evaluation_config = EvaluationConfig()
    yield evaluation_config

@pytest.fixture
def _test_env_config_spec():
    env_config = EnvConfig()
    yield env_config

@pytest.fixture
def _test_train_config_spec():
    train_config = TrainConfig()
    yield train_config

@pytest.fixture
def _test_exp_config_spec():
    exp_config = ExpConfig()
    yield exp_config

@pytest.fixture
def _test_train_exp_config_spec():
    train_exp_config = TrainExpConfig()
    yield train_exp_config

@pytest.fixture
def _test_inference_exp_config_spec():
    inference_exp_config = InferenceExpConfig()
    yield inference_exp_config

@pytest.fixture
def _test_eval_exp_config_spec():
    eval_exp_config = EvalExpConfig()
    yield eval_exp_config

@pytest.fixture
def _test_trt_config_spec():
    trt_config = TrtConfig()
    yield trt_config

@pytest.fixture
def _test_export_exp_config_spec():
    export_exp_config = ExportExpConfig()
    yield export_exp_config

@pytest.fixture
def _test_lr_head_config_spec():
    lr_head_config = LRHeadConfig()
    yield lr_head_config

@pytest.fixture
def _test_head_config_spec():
    head_config = HeadConfig()
    yield head_config

@pytest.fixture
def _test_init_cfg_spec():
    init_cfg = InitCfg()
    yield init_cfg

@pytest.fixture
def _test_backbone_config_spec():
    backbone_config = BackboneConfig()
    yield backbone_config

@pytest.fixture
def _test_train_aug_cfg_spec():
    train_aug_cfg = TrainAugCfg()
    yield train_aug_cfg

@pytest.fixture
def _test_model_config_spec():
    model_config = ModelConfig()
    yield model_config

@pytest.fixture
def _test_gen_trt_engine_spec():
    gen_trt_engine_config = GenTrtEngineExpConfig()
    yield gen_trt_engine_config


@pytest.mark.cv_unit
@pytest.mark.config
def test_img_norm_jsonschema_conversion(_test_img_norm_spec):
    """Test jsonschema conversion for img norm spec."""
    json_with_meta_config = dataclass_to_json(_test_img_norm_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_train_data_jsonschema_conversion(_test_train_data_spec):
    """Test jsonschema conversion for train data spec."""
    json_with_meta_config = dataclass_to_json(_test_train_data_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_val_data_jsonschema_conversion(_test_val_data_spec):
    """Test jsonschema conversion for val data spec."""
    json_with_meta_config = dataclass_to_json(_test_val_data_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_test_data_jsonschema_conversion(_test_test_data_spec):
    """Test jsonschema conversion for test data spec."""
    json_with_meta_config = dataclass_to_json(_test_test_data_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_data_config_jsonschema_conversion(_test_data_config_spec):
    """Test jsonschema conversion for data config spec."""
    json_with_meta_config = dataclass_to_json(_test_data_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_dataset_config_jsonschema_conversion(_test_dataset_config_spec):
    """Test jsonschema conversion for dataset config spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_dist_params_jsonschema_conversion(_test_dist_params_spec):
    """Test jsonschema conversion for dist params spec."""
    json_with_meta_config = dataclass_to_json(_test_dist_params_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_runner_config_jsonschema_conversion(_test_runner_config_spec):
    """Test jsonschema conversion for runner config spec."""
    json_with_meta_config = dataclass_to_json(_test_runner_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_config_jsonschema_conversion(_test_experiment_config_spec):
    """Test jsonschema conversion for experiment config spec."""
    json_with_meta_config = dataclass_to_json(_test_experiment_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_checkpoint_config_jsonschema_conversion(_test_checkpoint_config_spec):
    """Test jsonschema conversion for checkpoint config spec."""
    json_with_meta_config = dataclass_to_json(_test_checkpoint_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_log_config_jsonschema_conversion(_test_log_config_spec):
    """Test jsonschema conversion for log config spec."""
    json_with_meta_config = dataclass_to_json(_test_log_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_validation_config_jsonschema_conversion(_test_validation_config_spec):
    """Test jsonschema conversion for validation config spec."""
    json_with_meta_config = dataclass_to_json(_test_validation_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_paramwise_config_jsonschema_conversion(_test_paramwise_config_spec):
    """Test jsonschema conversion for paramwise config spec."""
    json_with_meta_config = dataclass_to_json(_test_paramwise_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_evaluation_config_jsonschema_conversion(_test_evaluation_config_spec):
    """Test jsonschema conversion for evaluation config spec."""
    json_with_meta_config = dataclass_to_json(_test_evaluation_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_env_config_jsonschema_conversion(_test_env_config_spec):
    """Test jsonschema conversion for env config spec."""
    json_with_meta_config = dataclass_to_json(_test_env_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_train_config_jsonschema_conversion(_test_train_config_spec):
    """Test jsonschema conversion for train config spec."""
    json_with_meta_config = dataclass_to_json(_test_train_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_exp_config_jsonschema_conversion(_test_exp_config_spec):
    """Test jsonschema conversion for exp config spec."""
    json_with_meta_config = dataclass_to_json(_test_exp_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_train_exp_config_jsonschema_conversion(_test_train_exp_config_spec):
    """Test jsonschema conversion for train exp config spec."""
    json_with_meta_config = dataclass_to_json(_test_train_exp_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_inference_exp_config_jsonschema_conversion(_test_inference_exp_config_spec):
    """Test jsonschema conversion for inference exp config spec."""
    json_with_meta_config = dataclass_to_json(_test_inference_exp_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_eval_exp_config_jsonschema_conversion(_test_eval_exp_config_spec):
    """Test jsonschema conversion for eval exp config spec."""
    json_with_meta_config = dataclass_to_json(_test_eval_exp_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_trt_config_jsonschema_conversion(_test_trt_config_spec):
    """Test jsonschema conversion for trt config spec."""
    json_with_meta_config = dataclass_to_json(_test_trt_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_export_exp_config_jsonschema_conversion(_test_export_exp_config_spec):
    """Test jsonschema conversion for export exp config spec."""
    json_with_meta_config = dataclass_to_json(_test_export_exp_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_lr_head_config_jsonschema_conversion(_test_lr_head_config_spec):
    """Test jsonschema conversion for lr head config spec."""
    json_with_meta_config = dataclass_to_json(_test_lr_head_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_head_config_jsonschema_conversion(_test_head_config_spec):
    """Test jsonschema conversion for head config spec."""
    json_with_meta_config = dataclass_to_json(_test_head_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_init_cfg_jsonschema_conversion(_test_init_cfg_spec):
    """Test jsonschema conversion for init cfg spec."""
    json_with_meta_config = dataclass_to_json(_test_init_cfg_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_backbone_config_jsonschema_conversion(_test_backbone_config_spec):
    """Test jsonschema conversion for backbone config spec."""
    json_with_meta_config = dataclass_to_json(_test_backbone_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_train_aug_cfg_jsonschema_conversion(_test_train_aug_cfg_spec):
    """Test jsonschema conversion for train aug cfg spec."""
    json_with_meta_config = dataclass_to_json(_test_train_aug_cfg_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_model_config_jsonschema_conversion(_test_model_config_spec):
    """Test jsonschema conversion for model config spec."""
    json_with_meta_config = dataclass_to_json(_test_model_config_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_gen_trt_engine_jsonschema_conversion(_test_gen_trt_engine_spec):
    """Test jsonschema conversion for gen trt engine spec."""
    json_with_meta_config = dataclass_to_json(_test_gen_trt_engine_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."



TEST_CONFIG_BLOCKS = [
    (sample_model_config, ModelConfig),
    (sample_dataset_config, DatasetConfig),
    (sample_train_config, TrainExpConfig),
    (sample_experiment_config, ExperimentConfig),
]

@pytest.mark.cv_unit
@pytest.mark.config
@pytest.mark.schema_validation
@pytest.mark.parametrize(
    "yaml_string, dataclass_class_name",
    TEST_CONFIG_BLOCKS                   
)
def test_load_experiment_spec(
    yaml_string,
    dataclass_class_name,
):
    """Simple function to load and validate the structure config from a yaml file."""
    schema = OmegaConf.structured(dataclass_class_name)
    config = OmegaConf.create(yaml_string)
    assert OmegaConf.merge(schema, config)
