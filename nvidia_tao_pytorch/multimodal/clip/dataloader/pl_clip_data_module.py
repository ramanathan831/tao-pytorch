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

"""CLIP Data Module"""

from typing import Optional
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.distributed.comm import is_dist_avail_and_initialized
from nvidia_tao_pytorch.multimodal.clip.dataloader.custom_loader import get_custom_dataloader
from nvidia_tao_pytorch.multimodal.clip.dataloader.wds import get_train_dataloader
from nvidia_tao_pytorch.multimodal.clip.dataloader.classification_loader import get_classification_dataloader


class CLIPDataModule(pl.LightningDataModule):
    """Lightning DataModule for CLIP."""

    def __init__(self, dataset_config, tokenizer, resume_step,
                 preprocess, mapping, world_size):
        """
        Initializes the CLIPDataModule.

        Args:
            dataset_config: Configuration for the dataset.
            tokenizer: Tokenizer to process input data.
            resume_step: Step to resume training from.
            preprocess: Tuple containing preprocessing functions for training and validation.
            mapping: Mapping for class names or other data processing.
            world_size: The number of processes for distributed training.
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.preprocess_train, self.preprocess_val = preprocess
        self.resume_step = resume_step
        self.mapping = mapping
        self.world_size = world_size

    def setup(self, stage: Optional[str] = None):
        """
        Prepares the dataloaders for each stage (fit, validate, test, predict).

        Args:
            stage (str): Stage options from fit, validate, test, predict or None.
        """
        # Skip if already setup (Lightning calls setup multiple times)
        if hasattr(self, '_setup_done') and self._setup_done:
            return
        self._setup_done = True

        is_distributed = is_dist_avail_and_initialized()

        train_dataloader_type = self.dataset_config.train.type
        val_dataloader_type = self.dataset_config.val.type
        if stage in ('fit', None):
            if train_dataloader_type == 'custom':
                train_dataset = get_custom_dataloader(
                    datasets=self.dataset_config.train.datasets,
                    transform=self.preprocess_train,
                    tokenizer=self.tokenizer,
                    batch_size=self.dataset_config.train.batch_size,
                    num_workers=self.dataset_config.train.num_workers,
                    seed=self.dataset_config.seed,
                    zero_shot_eval=False,
                    mapping=None,
                    shuffle=True,
                    pin_memory=self.dataset_config.pin_memory,
                    is_distributed=is_distributed,
                    mode='train'
                )
            elif train_dataloader_type == 'wds':
                # Setup dataset - wds
                train_dataset = get_train_dataloader(
                    root=self.dataset_config.train.wds.root_dir,
                    urls=self.dataset_config.train.wds.shard_list_file,
                    samples_per_file=self.dataset_config.train.wds.samples_per_shard,
                    batch_size=self.dataset_config.train.batch_size,
                    seed=self.dataset_config.seed,
                    num_workers=self.dataset_config.train.num_workers,
                    resume_step=self.resume_step,
                    transform=lambda data: (
                        self.preprocess_train(data[0]),
                        self.tokenizer(data[1])[0],
                    ),
                    world_size=self.world_size,
                    pin_memory=self.dataset_config.pin_memory
                )
            else:
                raise NotImplementedError(
                    'Wrong training dataset type (choose one from [custom,wds])')

            self.train_dataset = train_dataset

        # Setup validation/test dataset (used for both val and test stages)
        if stage in ('fit', 'test', None):
            if val_dataloader_type == 'custom':
                val_dataset = get_custom_dataloader(
                    datasets=[self.dataset_config.val.dataset],
                    transform=self.preprocess_val,
                    batch_size=self.dataset_config.val.batch_size,
                    num_workers=self.dataset_config.val.num_workers,
                    seed=self.dataset_config.seed,
                    zero_shot_eval=True,
                    mapping=self.mapping,
                    shuffle=False,
                    pin_memory=self.dataset_config.pin_memory,
                    is_distributed=None,
                    mode='val'
                )
            elif val_dataloader_type == 'classification':
                val_dataset = get_classification_dataloader(
                    root=self.dataset_config.val.dataset.root_dir,
                    split=self.dataset_config.val.split,
                    batch_size=self.dataset_config.val.batch_size,
                    transform=self.preprocess_val,
                    num_workers=self.dataset_config.val.num_workers
                )
            else:
                raise NotImplementedError(
                    'Wrong validation dataset type (choose one from [classification,custom])')

            self.val_dataset = val_dataset
            self.test_dataset = val_dataset  # Test uses same dataset as validation

    def train_dataloader(self):
        """Build the dataloader for training.

        Returns:
            train_loader: PyTorch DataLoader used for training.
        """
        return self.train_dataset

    def val_dataloader(self):
        """Build the dataloader for validation.

        Returns:
            val_loader: PyTorch DataLoader used for validation.
        """
        return self.val_dataset

    def test_dataloader(self):
        """Build the dataloader for evaluation.

        Returns:
            test_loader: PyTorch DataLoader used for evaluation.
        """
        return self.test_dataset
