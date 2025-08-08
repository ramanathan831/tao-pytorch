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

"""Utility functions for quantization."""

import fnmatch
import torch.nn as nn


def match_layer(module: nn.Module, module_name_in_graph: str, pattern: str) -> bool:
    """
    Check if a module matches a given name or pattern, prioritizing the module's graph name.

    This function determines whether a module matches the provided pattern. It first attempts to match
    the pattern (including wildcards) against the module's name in the model graph (e.g., 'layers.0.conv1').
    If this matches, it returns True immediately. If not, it checks the module's type name (such as "Linear"
    or "Conv2d"), also supporting wildcards. If neither matches, it returns False.

    A match on the module's name in the graph always takes precedence over a match on the type name.

    Examples
    --------
    >>> import torch.nn as nn
    >>> linear_layer = nn.Linear(10, 20)
    >>> conv_layer = nn.Conv2d(3, 64, 3)
    >>>
    >>> # Quantize or skip quantization for only the first linear layer
    >>> match_layer(linear_layer, "backbone.classifier.fc", "backbone.classifier.fc")
    True
    >>>
    >>> # Quantize or skip quantization for all linear layers in the model
    >>> match_layer(linear_layer, "backbone.classifier.fc", "Linear")
    True
    >>> match_layer(conv_layer, "backbone.features.conv1", "Linear")
    False
    >>>
    >>> # Quantize or skip quantization for all layers in the classifier
    >>> match_layer(linear_layer, "backbone.classifier.fc", "backbone.classifier.*")
    True
    >>>
    >>> # Quantize or skip quantization for all convolution layers
    >>> match_layer(conv_layer, "backbone.features.conv1", "Conv2d")
    True
    >>> match_layer(linear_layer, "backbone.classifier.fc", "Conv2d")
    False
    >>>

    Parameters
    ----------
    module : torch.nn.Module
        The module instance to check.
    module_name_in_graph : str
        The module's name within the model's graph (e.g., from `node.target`).
    pattern : str
        The name or pattern (wildcards allowed) to match against.

    Returns
    -------
    bool
        True if the module matches the given name or pattern, False otherwise.

    Raises
    ------
    TypeError
        If any of the arguments are None or of incorrect type.
    ValueError
        If pattern is an empty string.

    """
    if module is None:
        raise TypeError("module cannot be None")
    if not isinstance(module_name_in_graph, str):
        raise TypeError("module_name_in_graph must be a string")
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")
    if not pattern:
        raise ValueError("pattern cannot be empty")

    if fnmatch.fnmatch(module_name_in_graph, pattern):
        return True

    module_type_name = module.__class__.__name__
    if fnmatch.fnmatch(module_type_name, pattern):
        return True

    return False
