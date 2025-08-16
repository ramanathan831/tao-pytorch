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
"""Unit tests for the ModelOpt converter utility."""

import torch.nn as nn

from nvidia_tao_pytorch.core.quantization.backends.modelopt.utils import (
    convert_tao_to_modelopt_config,
)
from nvidia_tao_pytorch.core.quantization import (
    ModelQuantizationConfig,
    LayerQuantizationConfig,
    WeightQuantizationConfig,
    ActivationQuantizationConfig,
)


def _make_basic_layer(dtype: str = "int8"):
    """Helper to create a simple Linear-layer quantization spec."""
    return LayerQuantizationConfig(
        module_name="Linear",
        weights=WeightQuantizationConfig(
            dtype=dtype,
            observer_or_fake_quant="dummy_observer",
        ),
        activations=ActivationQuantizationConfig(
            dtype=dtype,
            observer_or_fake_quant="dummy_observer",
        ),
    )


def _make_toy_model():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.final_layer = nn.Linear(20, 5)

    return ToyModel()


def test_basic_int8_conversion():
    """Verify that a straightforward INT8 PTQ spec is converted correctly."""
    cfg = ModelQuantizationConfig(
        backend="modelopt",
        mode="static_ptq",
        algorithm="max",
        layers=[_make_basic_layer(dtype="int8")],
        skip_names=["final_layer"],
    )

    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)

    # Top-level keys
    assert set(result.keys()) == {
        "quant_cfg",
        "algorithm",
    }, "Converter should only emit 'quant_cfg' and 'algorithm' keys"
    assert (
        result["algorithm"] == "max"
    ), "Default algorithm should be 'max' for PTQ flows"

    quant_cfg = result["quant_cfg"]

    # Layer rule – pattern 'Linear' should expand to concrete name 'linear1' quantizers
    assert (
        "linear1.weight_quantizer" in quant_cfg
    ), "Weight quantizer rule missing for linear1"
    assert (
        "linear1.input_quantizer" in quant_cfg
    ), "Input quantizer rule missing for linear1"
    assert (
        quant_cfg["linear1.weight_quantizer"]["num_bits"] == 8
    ), "INT8 weights should map to 8 bits"
    assert (
        quant_cfg["linear1.input_quantizer"]["num_bits"] == 8
    ), "INT8 activations should map to 8 bits"

    # Skip rule must disable both quantizers
    assert (
        quant_cfg["final_layer.weight_quantizer"]["enable"] is False
    ), "Skip rule should disable weight quantizer"
    assert (
        quant_cfg["final_layer.input_quantizer"]["enable"] is False
    ), "Skip rule should disable input quantizer"


def test_default_disable_when_no_layers():
    """Without any layer/skip specification, conversion should disable quantization by default."""
    cfg = ModelQuantizationConfig()
    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)

    qc = result["quant_cfg"]
    assert qc == {
        "default": {"enable": False}
    }, "With no layers, default quant_cfg should disable quantization"
    # Default algorithm derives from mode when not set explicitly (static_ptq -> "max")
    assert (
        result["algorithm"] == "minmax"
    ), "Algorithm should default to 'minmax' for PTQ when not provided"


def test_algorithm_override_is_propagated():
    cfg = ModelQuantizationConfig(
        backend="modelopt",
        mode="static_ptq",
        algorithm="smoothquant",
        layers=[_make_basic_layer(dtype="int8")],
    )
    model = _make_toy_model()
    result = convert_tao_to_modelopt_config(cfg, model)
    assert (
        result["algorithm"] == "smoothquant"
    ), "Explicit algorithm should be propagated to ModelOpt config"
