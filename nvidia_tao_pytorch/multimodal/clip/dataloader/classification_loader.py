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

"""CLIP classification dataloader module for zero-shot evaluation."""

import os
from typing import Callable, Optional

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_classification_dataloader(
        root: str,
        split: str = "val",
        batch_size: int = 32,
        transform: Optional[Callable] = None,
        num_workers: int = 0
):
    """
    Creates a DataLoader for classification datasets (ImageFolder format).

    Supports any classification dataset organized as:
        root/
            split/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    img3.jpg
                    ...

    This is compatible with ImageNet, CIFAR, and other standard classification datasets.

    Args:
        root (str): The root directory of the dataset.
        split (str): The dataset split subfolder (e.g., 'train', 'val'). Default is 'val'.
        batch_size (int): The number of samples per batch. Default is 32.
        transform (Callable | None): A function/transform to apply to each sample.
            If None, the sample is returned unchanged. Default is None.
        num_workers (int): The number of subprocesses to use for data loading. Default is 0.

    Returns:
        DataLoader: A DataLoader for the specified classification dataset split.
    """
    if transform is None:

        def transform(x):
            return x

    dataset_path = os.path.join(root, split)
    dataset = ImageFolder(root=dataset_path, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        shuffle=split == "train",
    )
