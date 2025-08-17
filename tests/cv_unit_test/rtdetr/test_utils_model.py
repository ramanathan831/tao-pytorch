import types
import sys
import torch


class DummyLightning:
    def __init__(self, experiment_spec, export=False):
        self.experiment_spec = experiment_spec
        self.export = export
        self.loaded_state_dict = None

    @classmethod
    def load_from_checkpoint(cls, model_path, map_location=None, experiment_spec=None, export=False):
        inst = cls(experiment_spec, export=export)
        inst.loaded_from_checkpoint = {
            "model_path": model_path,
            "map_location": map_location,
            "export": export,
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


def install_stubs_for_rtdetr(monkeypatch):
    # Stub RTDETR Lightning model
    model_mod = types.ModuleType("nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model")
    model_mod.RTDETRPlModel = DummyLightning
    sys.modules["nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model"] = model_mod

    # Stub quantizer
    q_mod = types.ModuleType("nvidia_tao_pytorch.core.quantization.quantizer")
    q_mod.ModelQuantizer = DummyQuantizer
    sys.modules["nvidia_tao_pytorch.core.quantization.quantizer"] = q_mod

    # Import after stubbing
    import importlib
    return importlib.import_module("nvidia_tao_pytorch.cv.rtdetr.utils.model")


def test_create_model_from_config_rtdetr_non_quantized(monkeypatch, tmp_path):
    rt_utils = install_stubs_for_rtdetr(monkeypatch)

    exp_cfg = make_experiment_config(is_quantized=False)
    ckpt_path = tmp_path / "rtdetr.pth"
    torch.save({"some": torch.tensor(1)}, ckpt_path)

    model = rt_utils.create_model_from_config(exp_cfg, str(ckpt_path), task="evaluate")

    assert isinstance(model, DummyLightning), f"Model type mismatch: expected DummyLightning, got {type(model).__name__}"
    assert hasattr(model, "loaded_from_checkpoint"), "Missing attribute: model.loaded_from_checkpoint not set"
    assert model.loaded_from_checkpoint["model_path"] == str(ckpt_path), (
        f"Checkpoint path mismatch: {model.loaded_from_checkpoint['model_path']} != {str(ckpt_path)}"
    )


def test_create_model_from_config_rtdetr_quantized_modelopt_loads_prefixed_keys(monkeypatch, tmp_path):
    rt_utils = install_stubs_for_rtdetr(monkeypatch)

    exp_cfg = make_experiment_config(is_quantized=True, backend="modelopt")
    ckpt_path = tmp_path / "rtdetr_quantized.pth"
    torch.save({"model_state_dict": {"weight": torch.tensor(7)}}, ckpt_path)

    model = rt_utils.create_model_from_config(exp_cfg, str(ckpt_path), task="inference")

    assert isinstance(model, DummyLightning), (
        f"Model type mismatch (quantized): expected DummyLightning, got {type(model).__name__}"
    )
    assert model.loaded_state_dict is not None, (
        "Expected loaded_state_dict to be populated for quantized model"
    )
    assert "model.weight" in model.loaded_state_dict, (
        f"Missing expected key 'model.weight' in loaded_state_dict. Keys: {list(model.loaded_state_dict.keys())}"
    )


def test_create_model_from_config_rtdetr_export_sets_flag(monkeypatch, tmp_path):
    rt_utils = install_stubs_for_rtdetr(monkeypatch)

    exp_cfg = make_experiment_config(is_quantized=False)
    ckpt_path = tmp_path / "rtdetr.pth"
    torch.save({"some": torch.tensor(1)}, ckpt_path)

    model = rt_utils.create_model_from_config(exp_cfg, str(ckpt_path), task="export")

    # When using non-quantized path, load_from_checkpoint is used with export=True
    assert isinstance(model, DummyLightning), (
        f"Model type mismatch (export): expected DummyLightning, got {type(model).__name__}"
    )
    assert model.loaded_from_checkpoint["export"] is True, (
        f"Expected export flag to be True, got {model.loaded_from_checkpoint['export']}"
    )


