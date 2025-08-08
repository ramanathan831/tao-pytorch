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

"""Validation utilities for quantization framework."""

from typing import List

from .constants import SupportedDtype


def get_valid_dtype_options() -> List[str]:
    """Get valid dtype options from the enum values.

    Returns:
        List of valid data type strings (e.g., ["int8", "fp8_e4m3fn", "fp8_e5m2"])
    """
    return [dtype.value for dtype in SupportedDtype]
