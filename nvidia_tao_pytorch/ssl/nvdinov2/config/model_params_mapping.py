# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

""" Model Parameters Mapping Module """

SUPPORTED_BACKBONES = [
    *["vit_l"]
]

map_params = {
    'embed_dim': {
        'vit_l': 1024
    },
    'depth': {
        'vit_l': 24
    },
    'num_heads': {
        'vit_l': 16
    },
    'init_values': {
        'vit_l': 1e-5
    },
    'drop_path_schedule': {
        'vit_l': 'linear'
    },
    'num_classes': {
        'vit_l': 0
    },
}
