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

"""Quantizer core for TAO Toolkit."""

from abc import ABC, abstractmethod
import torch.nn as nn

from nvidia_tao_core.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


class QuantizerBase(ABC):
    """
    Abstract base class for model quantization.

    This class provides an interface for model quantization. Subclasses must implement
    the `prepare` and `quantize` methods.

    Methods
    -------
    prepare(model, config)
        Insert Observer/FakeQuantize modules based on user-specified config.
    quantize(model, config)
        Convert or wrap the model to its quantized form.

    See Also
    --------
    Calibratable
        Mix-in interface that adds a ``calibrate`` method for PTQ back-ends.
    """

    @abstractmethod
    def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """
        Insert Observer/FakeQuantize modules based on user-specified config.

        Parameters
        ----------
        model : torch.nn.Module
            The model to prepare for quantization.
        config : ModelQuantizationConfig
            Configuration for quantization.

        Returns
        -------
        torch.nn.Module
            The prepared model with observers/fake quantize modules inserted.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Calling abstract method - prepare. Subclass must implement this method.")

    @abstractmethod
    def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """
        Convert or wrap the model to its quantized form.

        Parameters
        ----------
        model : torch.nn.Module
            The model to quantize.
        config : ModelQuantizationConfig
            Configuration for quantization.

        Returns
        -------
        torch.nn.Module
            The quantized model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Calling abstract method - quantize. Subclass must implement this method.")
