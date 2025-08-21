import types
import sys
import os

import pytest
import torch


class DummyLightning:
    def __init__(self, experiment_spec):
        self.experiment_spec = experiment_spec
        self.loaded_state_dict = None

    @classmethod
    def load_from_checkpoint(cls, model_path, map_location=None, experiment_spec=None):
        inst = cls(experiment_spec)
        inst.loaded_from_checkpoint = {
            "model_path": model_path,
            "map_location": map_location,
        }
        return inst

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


class DummyQuantizer:
    def __init__(self, cfg_like):
        self.cfg_like = cfg_like

    def quantize_model(self, model, calibration_loader=None):
        return model


def make_experiment_config(is_quantized=False, backend="modelopt"):
    evaluate = types.SimpleNamespace(is_quantized=is_quantized)
    inference = types.SimpleNamespace(is_quantized=is_quantized)
    export = types.SimpleNamespace(is_quantized=is_quantized)
    quantize = types.SimpleNamespace(backend=backend)
    return types.SimpleNamespace(evaluate=evaluate, inference=inference, export=export, quantize=quantize)


def install_stubs_for_classification(monkeypatch):
    # Stub the model module to avoid importing real heavy dependencies
    model_mod = types.ModuleType("nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model")
    model_mod.ClassifierPlModel = DummyLightning
    sys.modules["nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model"] = model_mod

    # Stub the quantizer
    q_mod = types.ModuleType("nvidia_tao_pytorch.core.quantization.quantizer")
    q_mod.ModelQuantizer = DummyQuantizer
    sys.modules["nvidia_tao_pytorch.core.quantization.quantizer"] = q_mod

    # Import after stubbing
    import importlib
    return importlib.import_module("nvidia_tao_pytorch.cv.classification_pyt.utils.model")


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason="TODO(nnagrajrao, vpraveen, hongyuc): Re-enable once the import issue is resolved.",
)
def test_create_model_from_config_non_quantized(monkeypatch, tmp_path):
    cl_utils = install_stubs_for_classification(monkeypatch)

    exp_cfg = make_experiment_config(is_quantized=False)
    ckpt_path = tmp_path / "model.pth"
    torch.save({"some": torch.tensor(1)}, ckpt_path)

    model = cl_utils.create_model_from_config(exp_cfg, str(ckpt_path), task="evaluate")

    assert isinstance(model, DummyLightning), f"Model type mismatch: expected DummyLightning, got {type(model).__name__}"
    assert hasattr(model, "loaded_from_checkpoint"), "Missing attribute: model.loaded_from_checkpoint not set"
    assert model.loaded_from_checkpoint["model_path"] == str(ckpt_path), (
        f"Checkpoint path mismatch: {model.loaded_from_checkpoint['model_path']} != {str(ckpt_path)}"
    )


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason="TODO(nnagrajrao, vpraveen, hongyuc): Re-enable once the import issue is resolved.",
)
def test_create_model_from_config_quantized_modelopt_loads_prefixed_keys(monkeypatch, tmp_path):
    cl_utils = install_stubs_for_classification(monkeypatch)

    exp_cfg = make_experiment_config(is_quantized=True, backend="modelopt")
    ckpt_path = tmp_path / "model_quantized.pth"
    # Save as ModelOpt-like artifact
    torch.save({"model_state_dict": {"weight": torch.tensor(42)}}, ckpt_path)

    model = cl_utils.create_model_from_config(exp_cfg, str(ckpt_path), task="inference")

    assert isinstance(model, DummyLightning), (
        f"Model type mismatch (quantized): expected DummyLightning, got {type(model).__name__}"
    )
    assert model.loaded_state_dict is not None, (
        "Expected loaded_state_dict to be populated for quantized model"
    )
    # Expect keys to be prefixed with 'model.'
    assert "model.weight" in model.loaded_state_dict, (
        f"Missing expected key 'model.weight' in loaded_state_dict. Keys: {list(model.loaded_state_dict.keys())}"
    )


