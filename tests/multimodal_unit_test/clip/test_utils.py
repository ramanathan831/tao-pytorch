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

"""CLIP utils unit tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from nvidia_tao_pytorch.multimodal.clip.model.evaluation.zero_shot_classifier import (
    accuracy,
    batched,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    load_model_from_checkpoint,
    SUPPORTED_CHECKPOINT_EXTENSIONS,
)


@pytest.mark.multimodal_unit
class TestAccuracy:
    """Test accuracy function."""

    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        # Predictions exactly match targets (5 classes to support top-5)
        output = torch.tensor([
            [10.0, 0.0, 0.0, 0.0, 0.0],  # class 0
            [0.0, 10.0, 0.0, 0.0, 0.0],  # class 1
            [0.0, 0.0, 10.0, 0.0, 0.0],  # class 2
        ])
        target = torch.tensor([0, 1, 2])

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        assert acc1.item() == 3  # 3 correct
        assert acc5.item() == 3  # 3 correct in top-5

    def test_zero_accuracy(self):
        """Test with completely wrong predictions."""
        # 5 classes to support top-5
        output = torch.tensor([
            [0.0, 10.0, 0.0, 0.0, 0.0],  # predicts class 1
            [0.0, 0.0, 10.0, 0.0, 0.0],  # predicts class 2
            [10.0, 0.0, 0.0, 0.0, 0.0],  # predicts class 0
        ])
        target = torch.tensor([0, 1, 2])

        acc1, _ = accuracy(output, target, topk=(1, 5))
        assert acc1.item() == 0  # 0 correct

    def test_top5_accuracy(self):
        """Test top-5 accuracy includes lower ranked correct predictions."""
        # 10 classes
        output = torch.zeros(1, 10)
        output[0, 5] = 10.0  # 1st
        output[0, 3] = 8.0   # 2nd
        output[0, 7] = 6.0   # 3rd
        output[0, 1] = 4.0   # 4th
        output[0, 0] = 2.0   # 5th - correct answer
        target = torch.tensor([0])

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        assert acc1.item() == 0  # Not in top-1
        assert acc5.item() == 1  # In top-5


@pytest.mark.multimodal_unit
class TestBatched:
    """Test batched function."""

    def test_exact_batches(self):
        """Test when items divide evenly into batches."""
        items = [1, 2, 3, 4, 5, 6]
        batches = list(batched(items, 2))
        assert batches == [[1, 2], [3, 4], [5, 6]]

    def test_partial_last_batch(self):
        """Test when last batch is partial."""
        items = [1, 2, 3, 4, 5]
        batches = list(batched(items, 2))
        assert batches == [[1, 2], [3, 4], [5]]

    def test_single_item_batches(self):
        """Test with batch size of 1."""
        items = [1, 2, 3]
        batches = list(batched(items, 1))
        assert batches == [[1], [2], [3]]

    def test_batch_larger_than_items(self):
        """Test when batch size is larger than item count."""
        items = [1, 2, 3]
        batches = list(batched(items, 10))
        assert batches == [[1, 2, 3]]

    def test_empty_input(self):
        """Test with empty input."""
        items = []
        batches = list(batched(items, 2))
        assert not batches

    def test_generator_input(self):
        """Test with generator input."""
        def gen():
            yield from range(5)

        batches = list(batched(gen(), 2))
        assert batches == [[0, 1], [2, 3], [4]]


@pytest.mark.multimodal_unit
class TestSupportedCheckpointExtensions:
    """Test checkpoint extension constants."""

    def test_supported_extensions(self):
        """Test that expected extensions are supported."""
        assert '.pth' in SUPPORTED_CHECKPOINT_EXTENSIONS
        assert '.ckpt' in SUPPORTED_CHECKPOINT_EXTENSIONS


@pytest.mark.multimodal_unit
class TestLoadModelFromCheckpoint:
    """Test load_model_from_checkpoint function."""

    def test_unsupported_engine_format(self):
        """Test that .engine format raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="TensorRT inference"):
            load_model_from_checkpoint(
                "model.engine",
                MagicMock(),
                MagicMock()
            )

    def test_unsupported_format(self):
        """Test that unknown format raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not supported"):
            load_model_from_checkpoint(
                "model.xyz",
                MagicMock(),
                MagicMock()
            )

    def test_pth_extension_calls_load_from_checkpoint(self):
        """Test that .pth extension calls load_from_checkpoint."""
        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model_class.load_from_checkpoint.return_value = mock_model
        mock_config = MagicMock()

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name

        try:
            result = load_model_from_checkpoint(
                checkpoint_path,
                mock_config,
                mock_model_class
            )

            mock_model_class.load_from_checkpoint.assert_called_once_with(
                checkpoint_path,
                map_location="cpu",
                experiment_spec=mock_config
            )
            assert result == mock_model
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_ckpt_extension_calls_load_from_checkpoint(self):
        """Test that .ckpt extension calls load_from_checkpoint."""
        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model_class.load_from_checkpoint.return_value = mock_model
        mock_config = MagicMock()

        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            checkpoint_path = f.name

        try:
            result = load_model_from_checkpoint(
                checkpoint_path,
                mock_config,
                mock_model_class
            )

            mock_model_class.load_from_checkpoint.assert_called_once()
            assert result == mock_model
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
