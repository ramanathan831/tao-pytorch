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

"""Calibratable interface for PTQ calibration."""

from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import DataLoader


class Calibratable(ABC):
    """
    Calibratable extension for the quantizer, for PTQ calibration.

    This class provides an abstract interface for post-training quantization
    calibration. Subclasses must implement the `calibrate` method.

    Methods
    -------
    calibrate(model, data_loader)
        Collect statistics or perform any PTQ-style calibration.
    """

    @abstractmethod
    def calibrate(self, model: nn.Module, data_loader: DataLoader):
        """
        Collect statistics or perform any PTQ-style calibration.

        Parameters
        ----------
        model : torch.nn.Module
            The model to calibrate.
        data_loader : torch.utils.data.DataLoader
            DataLoader providing calibration data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Calling abstract method - calibrate. Subclass must implement this method.")
