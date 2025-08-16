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

from __future__ import annotations

import types
from unittest.mock import patch, MagicMock
import sys
import os
import pytest
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_PKG_ROOT = os.path.join(_REPO_ROOT, "tao-pytorch")
_CORE_ROOT = os.path.join(_PKG_ROOT, "tao-core")
for p in (_PKG_ROOT, _CORE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from nvidia_tao_pytorch.core.quantization import (  # noqa: E402
    ModelQuantizer,
    get_registry_manager,
    register_backend,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class _DummyMtqModule:
    def quantize(self, model, cfg, forward_loop):
        return model


def _patch_mtq():
    """Patch the entire modelopt module - BROAD AND SLOW"""
    dummy = _DummyMtqModule()
    mtq_mod = types.SimpleNamespace(quantize=dummy.quantize)
    return patch.dict(
        "sys.modules",
        {
            "modelopt": types.SimpleNamespace(
                torch=types.SimpleNamespace(quantization=mtq_mod)
            ),
            "modelopt.torch": types.SimpleNamespace(quantization=mtq_mod),
            "modelopt.torch.quantization": mtq_mod,
        },
    )


def _patch_modelopt_imports():
    """Targeted patch for modelopt imports - FAST"""

    def mock_quantize(model, cfg, forward_loop):
        # Return the actual model instead of a MagicMock
        return model

    return patch.multiple(
        "nvidia_tao_pytorch.core.quantization.backends.modelopt.modelopt",
        mtq=MagicMock(quantize=mock_quantize),
        mto=MagicMock(save=MagicMock()),
    )


@pytest.mark.unit
def test_quantize_model_accepts_dict_and_omegaconf():
    # Ensure backend is registered for this test and isolation maintained
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt.modelopt import (
        ModelOptBackend,
    )

    register_backend("modelopt")(ModelOptBackend)

    model = Tiny()
    cfg_dict = {
        "backend": "modelopt",
        "mode": "static_ptq",
        "algorithm": "max",
        "layers": [
            {
                "module_name": "Linear",
                "weights": {"dtype": "int8"},
                "activations": {"dtype": "int8"},
            }
        ],
    }
    with _patch_modelopt_imports():
        quantizer = ModelQuantizer(cfg_dict)
        out = quantizer.quantize_model(model)
        assert isinstance(
            out, nn.Module
        ), "quantize_model should return a torch.nn.Module"
