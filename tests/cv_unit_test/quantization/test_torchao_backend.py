"""Unit tests for the TorchAO backend integration."""

from __future__ import annotations

import types
from unittest.mock import patch
import os
import tempfile
import importlib
import sys
import pytest
import torch.nn as nn

from nvidia_tao_pytorch.core.quantization import (
    get_registry_manager,
    get_backend_class,
)
from nvidia_tao_pytorch.core.quantization.backends.modelopt.utils import (
    build_model_quant_config_from_omegaconf,
)


def _patch_torchao_imports():
    """Patch torchao.quantization symbols used by the backend.

    Creates light-weight stand-ins for ``Float8WeightOnlyConfig``, ``Int8WeightOnlyConfig``,
    ``AOPerModuleConfig`` and ``quantize_``. The dummy ``quantize_`` returns the model passed in
    to keep the test simple and focused.
    """

    class _DummyCfg:
        def __init__(self, kind: str):
            self.kind = kind

    class _DummyAOPerModuleConfig:
        def __init__(self, module_fqn_to_config):  # noqa: D401 - emulate torchao signature
            self.module_fqn_to_config = module_fqn_to_config

    def _dummy_quantize_(model, cfg):  # noqa: D401 - emulate torchao signature
        # In-place no-op; return the model
        return model

    taq_mod = types.SimpleNamespace(
        Float8WeightOnlyConfig=lambda: _DummyCfg("fp8"),
        Int8WeightOnlyConfig=lambda: _DummyCfg("int8"),
        AOPerModuleConfig=_DummyAOPerModuleConfig,
        quantize_=_dummy_quantize_,
    )
    return patch.dict(
        "sys.modules",
        {
            "torchao": types.SimpleNamespace(quantization=taq_mod),
            "torchao.quantization": taq_mod,
        },
    )


def _ensure_torchao_registered():
    """Ensure torchao backend module executes decorator to register backend.

    Reloads the module if it's already imported to re-run the decorator after a
    registry clear.
    """
    mod_name = "nvidia_tao_pytorch.core.quantization.backends.torchao.torchao"
    mod = sys.modules.get(mod_name)
    if mod is None:
        importlib.import_module(mod_name)
    else:
        importlib.reload(mod)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
def test_torchao_backend_prepare_and_quantize_int8():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()

    with _patch_torchao_imports():
        _ensure_torchao_registered()
        backend_cls = get_backend_class("torchao")
        q = backend_cls()

        model = ToyModel()
        cfg = build_model_quant_config_from_omegaconf(
            {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "layers": [
                    {
                        "module_name": "Linear",
                        "weights": {"dtype": "int8"},
                    }
                ],
            }
        )

        prepared = q.prepare(model, cfg)
        assert prepared is model, "prepare should be a no-op for the TorchAO backend"

        quantized = q.quantize(prepared, cfg)
        assert isinstance(quantized, nn.Module), "quantize should return a torch.nn.Module"


@pytest.mark.unit
def test_torchao_backend_quantize_fp8_and_skip():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()

    with _patch_torchao_imports():
        _ensure_torchao_registered()
        backend_cls = get_backend_class("torchao")
        q = backend_cls()

        model = ToyModel()
        cfg = build_model_quant_config_from_omegaconf(
            {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "layers": [
                    {
                        "module_name": "Linear",
                        "weights": {"dtype": "fp8_e4m3fn"},
                    }
                ],
                "skip_names": ["*linear*"],
            }
        )

        prepared = q.prepare(model, cfg)
        # With skip covering the only layer, quantize should still succeed
        quantized = q.quantize(prepared, cfg)
        assert isinstance(quantized, nn.Module)


@pytest.mark.unit
def test_torchao_backend_save_model(tmp_path=None):
    # Ensure clean registry across tests
    get_registry_manager().clear_all()

    with _patch_torchao_imports():
        _ensure_torchao_registered()
        backend_cls = get_backend_class("torchao")
        q = backend_cls()

        model = ToyModel()
        cfg = build_model_quant_config_from_omegaconf(
            {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "layers": [
                    {
                        "module_name": "Linear",
                        "weights": {"dtype": "int8"},
                    }
                ],
            }
        )

        prepared = q.prepare(model, cfg)
        quantized = q.quantize(prepared, cfg)

        # Use tmp directory from pytest or create a temp dir
        save_dir = tmp_path if tmp_path is not None else tempfile.mkdtemp()
        q.save_model(quantized, str(save_dir))
        expected_path = os.path.join(str(save_dir), "quantized_model_torchao.pth")
        assert os.path.exists(expected_path), "Expected saved file was not created"
