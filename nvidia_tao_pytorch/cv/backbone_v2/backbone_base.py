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

"""Abstract base class for backbone models."""

import abc
from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import FrozenBatchNorm2d


class BackboneBase(nn.Module, abc.ABC):
    """Abstract base class for backbone models.

    This class defines the common interface and functionality for all backbone models.
    All backbone models should inherit from this class and implement the required methods:

    - `get_stage_dict`
    - `get_classifier` (optional if the model inherits from Timm library)
    - `reset_classifier` (optional if the model inherits from Timm library)
    - `forward_pre_logits`
    - `forward_feature_pyramid`

    Note that `_module_initialized` is an internal flag indicating whether the module has been initialized. This is
    used to avoid re-initialization of `nn.Module`.

    Examples:

        ```python
        # When using `TimmModel` as a base class, `nn.Module` should only be initialized once.
        class MyBackbone(TimmModel, BackboneBase):
            def __init__(self, *args, **kwargs):
                ...
                super().__init__(*args, **kwargs)  # `TimmModel` initialization. `nn.Module` is initialized here.
                self._module_initialized = True
                BackboneBase.__init__(self, ...)  # Skip the `nn.Module.__init__()` when initializing `BackboneBase`.
        ```
    """

    _module_initialized: bool = False

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        freeze_at: Optional[List[Union[int, str]]] = None,
        freeze_norm: bool = False,
    ):
        """Initialize the backbone base class.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head.
                Default: `1000`.
            freeze_at (list): List of keys corresponding to the stages or
                layers to freeze. If `None`, no specific layers are frozen.
                Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the
                backbone will be frozen. Default: `False`.
        """
        if not self._module_initialized:
            nn.Module.__init__(self)

        freeze_at = freeze_at or []
        if not isinstance(freeze_at, list):
            freeze_at = list(freeze_at)
        if not all(isinstance(i, (int, str)) for i in freeze_at):
            raise ValueError(f"Invalid freeze_at value: {freeze_at}. It should be a list of integers or strings.")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.freeze_norm = bool(freeze_norm)
        self.freeze_at = freeze_at

    def _init_weights(self, m):
        """initialize weights."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @abc.abstractmethod
    def get_stage_dict(self) -> Dict[Union[int, str], nn.Module]:
        """Get the stage dictionary.

        This method is useful for freezing specific stages of the backbone.

        Returns:
            dict: Dictionary containing the stages of the backbone. The keys are
                the stage names (e.g. the index of the stage) and the values are
                the corresponding modules.
        """
        pass

    def no_weight_decay(self) -> Set[str]:
        """Get the set of parameter names to exclude from weight decay.

        Returns:
            Set[str]: Set of parameter names to exclude from weight decay
        """
        return set()

    def no_weight_decay_keywords(self) -> Set[str]:
        """Get the set of parameter keywords to exclude from weight decay.

        Returns:
            Set[str]: Set of parameter keywords to exclude from weight decay
        """
        return set()

    @abc.abstractmethod
    def get_classifier(self) -> nn.Module:
        """Get the classifier module.

        Returns:
            nn.Module: The classifier module
        """
        pass

    @abc.abstractmethod
    def reset_classifier(self, num_classes: int = 0, **kwargs):
        """Reset the classifier head.

        Args:
            num_classes (int): Number of classes for the new classifier. Default: `0`.
            **kwargs: Additional arguments passed to the classifier implementation.
        """
        pass

    def _freeze_module(self, m: nn.Module):
        """Freeze the given module."""
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_bn_norm(self, m: nn.Module):
        """Recursively freeze the batch normalization layers in the given module."""
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_bn_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def freeze_backbone(self):
        """Freeze specific parts of the backbone and batch normalization layers.

        This method used `self.freeze_at` and `self.freeze_norm` to determine
        which parts of the backbone to freeze.

        The `get_stage_dict` method must be implemented in the subclass to
        provide the mapping of stage keys to modules.
        """
        if len(self.freeze_at) != 0:
            stage_dict = self.get_stage_dict()
            for key in self.freeze_at:
                if key in stage_dict:
                    self._freeze_module(stage_dict[key])
                    if get_global_rank() == 0:
                        logging.warning(f"Stage {key} is frozen.")
                else:
                    if get_global_rank() == 0:
                        logging.warning(f"Stage {key} not found. Freezing options: {stage_dict.keys()}")

        if self.freeze_norm:
            self._freeze_bn_norm(self)
            if get_global_rank() == 0:
                logging.warning("All batch normalization layers are frozen.")

    @abc.abstractmethod
    def forward_pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone, excluding the head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output pre-logits tensor.
        """
        pass

    @abc.abstractmethod
    def forward_feature_pyramid(
        self, x: torch.Tensor, indices: Optional[Union[int, List[int]]] = None, **kwargs
    ) -> List[torch.Tensor]:
        """Forward pass through the backbone to extract intermediate feature maps.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            **kwargs: Additional arguments.

        Returns:
            List[torch.Tensor]: List of intermediate feature maps.
        """
        pass
