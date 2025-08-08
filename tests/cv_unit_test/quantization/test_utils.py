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

"""Unit tests for quantization utility functions."""

import pytest
import torch.nn as nn

from nvidia_tao_pytorch.core.quantization.utils import match_layer


# A couple of common layer types to use in our tests
conv_layer = nn.Conv2d(3, 64, 3)
linear_layer = nn.Linear(10, 20)


@pytest.mark.parametrize(
    "module, module_name, pattern, expected, description",
    [
        (
            conv_layer,
            "features.conv1",
            "features.conv1",
            True,
            "Exact match on graph name",
        ),
        (
            linear_layer,
            "classifier.fc",
            "classifier.*",
            True,
            "Wildcard match on graph name",
        ),
        (
            conv_layer,
            "features.conv1",
            "*conv1",
            True,
            "Leading wildcard match on graph name",
        ),
        (linear_layer, "classifier.fc", "Conv2d", False, "Type mismatch on graph name"),
        (conv_layer, "irrelevant.path", "Conv2d", True, "Exact match on type name"),
        (linear_layer, "irrelevant.path", "Linear", True, "Exact match on type name"),
        (conv_layer, "irrelevant.path", "Conv*", True, "Wildcard match on type name"),
        (linear_layer, "irrelevant.path", "Lin*", True, "Wildcard match on type name"),
        (conv_layer, "features.conv1", "Linear", False, "No match for name or type"),
        (linear_layer, "classifier.fc", "Conv2d", False, "No match for name or type"),
        (
            conv_layer,
            "features.conv1",
            "*",
            True,
            "Global wildcard should always match",
        ),
    ],
)
def test_match_layer(module, module_name, pattern, expected, description):
    """Tests various scenarios for the match_layer function."""
    assert match_layer(module, module_name, pattern) == expected, description


def test_match_layer_precedence():
    """
    Tests that a match on the module's graph name is found, even if the type also matches.
    The implementation short-circuits, so this verifies the order of checks.
    """
    # This pattern matches the graph name but wouldn't match the type name.
    assert match_layer(conv_layer, "backbone.features.conv1", "backbone.features.*")
    # This pattern doesn't match the graph name but does match the type name.
    assert match_layer(conv_layer, "backbone.features.conv1", "Conv2d")


def test_match_layer_input_validation():
    """Tests the input validation for the match_layer function to ensure it handles bad inputs."""
    with pytest.raises(TypeError, match="module cannot be None"):
        match_layer(None, "some_name", "some_pattern")

    with pytest.raises(TypeError, match="module_name_in_graph must be a string"):
        match_layer(conv_layer, None, "some_pattern")

    with pytest.raises(TypeError, match="module_name_in_graph must be a string"):
        match_layer(conv_layer, 123, "some_pattern")

    with pytest.raises(TypeError, match="pattern must be a string"):
        match_layer(conv_layer, "some_name", None)

    with pytest.raises(TypeError, match="pattern must be a string"):
        match_layer(conv_layer, "some_name", ["a_list"])

    with pytest.raises(ValueError, match="pattern cannot be empty"):
        match_layer(conv_layer, "some_name", "")
