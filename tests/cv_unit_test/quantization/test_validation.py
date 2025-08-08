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

"""Unit tests for quantization validation functions."""

from nvidia_tao_pytorch.core.quantization.validation import get_valid_dtype_options
from nvidia_tao_pytorch.core.quantization.constants import SupportedDtype


def test_get_valid_dtype_options():
    """
    Tests that get_valid_dtype_options returns the correct list of supported data types.
    It verifies the type, content, and consistency with the SupportedDtype enum.
    """
    valid_dtypes = get_valid_dtype_options()

    # It should return a list...
    assert isinstance(valid_dtypes, list), "Should return a list"
    # ...of strings.
    assert all(isinstance(item, str) for item in valid_dtypes), (
        "All items should be strings"
    )

    # And the content should be exactly what's in our enum
    expected_dtypes = [e.value for e in SupportedDtype]
    assert sorted(valid_dtypes) == sorted(expected_dtypes), (
        "The list of dtypes should match the source enum"
    )
