# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""NVDINOv2 Train Script"""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.ssl.nvdinov2.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel


def run_experiment(experiment_config, key):
    """Start the training."""
    results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes
    max_steps = experiment_config.train.max_steps
    total_epochs = experiment_config.train.num_epochs
    checkpoint_interval = experiment_config.train.checkpoint_interval

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    precision = '16-mixed'
    sync_batchnorm = True

    dm = DinoV2DataModule(experiment_config)

    model = DinoV2PlModel(experiment_config)

    if pretrained_path is not None:
        model.pretrained_weights = pretrained_path
        model.restore_pretrained_weights()

    trainer = Trainer(logger=ptl_loggers,
                      devices=gpus,
                      num_nodes=num_nodes,
                      max_epochs=total_epochs,
                      max_steps=max_steps,
                      check_val_every_n_epoch=checkpoint_interval,
                      default_root_dir=results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      precision=precision,
                      use_distributed_sampler=False,
                      sync_batchnorm=sync_batchnorm,
                      enable_checkpointing=False,
                      )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="NVDINOv2", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
