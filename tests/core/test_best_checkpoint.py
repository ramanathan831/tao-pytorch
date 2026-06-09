# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for network-aware best-checkpoint saving."""

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import (
    TAOLightningModule, validate_monitor_metric,
)


class _DummyModule(TAOLightningModule):
    """Minimal concrete module to exercise configure_callbacks()."""

    def __init__(self, spec, monitor="val_acc", mode="max"):
        super().__init__(spec)
        self.checkpoint_filename = "dummy_model"
        self.monitor_metric = monitor
        self.monitor_mode = mode


def _spec(results_dir, checkpointer=None):
    train = {"checkpoint_interval": 1, "checkpoint_interval_unit": "epoch"}
    if checkpointer is not None:
        train["checkpointer"] = checkpointer
    return {"results_dir": str(results_dir), "dataset": {}, "model": {}, "train": train}


def _best_callbacks(callbacks):
    """ModelCheckpoints that actually monitor a metric (the 'best' ones)."""
    return [c for c in callbacks if isinstance(c, ModelCheckpoint) and c.monitor is not None]


def test_base_defaults_present():
    m = TAOLightningModule.__new__(TAOLightningModule)  # no model build
    TAOLightningModule.__init__(m, _spec("/tmp"))
    assert m.monitor_metric == "val_loss"
    assert m.monitor_mode == "min"


def test_no_checkpointer_block_is_backward_compatible(tmp_path):
    # Old spec: no 'checkpointer' key at all -> no monitored checkpoint appended.
    m = _DummyModule(_spec(tmp_path))
    cbs = m.configure_callbacks()
    assert _best_callbacks(cbs) == []


def test_disabled_topk_appends_nothing(tmp_path):
    m = _DummyModule(_spec(tmp_path, checkpointer={"enable_topk": False}))
    assert _best_callbacks(m.configure_callbacks()) == []


def test_enabled_uses_network_default_metric(tmp_path):
    m = _DummyModule(_spec(tmp_path, checkpointer={"enable_topk": True}),
                     monitor="val_acc", mode="max")
    best = _best_callbacks(m.configure_callbacks())
    assert len(best) == 1
    cb = best[0]
    assert cb.monitor == "val_acc"
    assert cb.mode == "max"
    # best callback must not fight the periodic callback over *_latest, and must
    # rank at validation end:
    assert cb.save_last is False
    assert cb._save_on_train_epoch_end is False
    assert str(cb.dirpath) == str(tmp_path)   # defaults to results_dir


def test_config_overrides_network_default(tmp_path):
    cfg = {"enable_topk": True, "monitor": "val_loss", "mode": "min", "save_top_k": 3}
    m = _DummyModule(_spec(tmp_path, checkpointer=cfg), monitor="val_acc", mode="max")
    cb = _best_callbacks(m.configure_callbacks())[0]
    assert cb.monitor == "val_loss"   # override wins over network default
    assert cb.mode == "min"
    assert cb.save_top_k == 3


def test_custom_dirpath_respected(tmp_path):
    sub = tmp_path / "best"
    cfg = {"enable_topk": True, "dirpath": str(sub)}
    m = _DummyModule(_spec(tmp_path, checkpointer=cfg))
    cb = _best_callbacks(m.configure_callbacks())[0]
    assert str(cb.dirpath) == str(sub)


def test_runtime_guard_raises_on_unlogged_metric():
    with pytest.raises(ValueError, match="is not logged"):
        validate_monitor_metric("val_miou", {"val_loss", "val_acc"})


def test_runtime_guard_passes_when_present():
    validate_monitor_metric("val_acc", {"val_loss", "val_acc"})  # no raise


def test_config_schema_default_disabled():
    from omegaconf import OmegaConf
    from nvidia_tao_pytorch.config.common.common_config import TrainConfig
    schema = OmegaConf.structured(TrainConfig)
    assert schema.checkpointer.enable_topk is False
    assert schema.checkpointer.monitor is None
    assert schema.checkpointer.mode is None
