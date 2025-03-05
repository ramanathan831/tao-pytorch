# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""GST-Nvinfer config file for RT-DETR."""

from dataclasses import dataclass, is_dataclass
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseNVDSClassAttributes,
    BaseNvDSPropertyConfig,
    BaseDSType
)


@dataclass
class RTDETRNvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Structured configuration defining the schema for nvdsinfer property element for RT-DETR."""

    parse_bbox_func_name: str = "NvDsInferParseCustomDDETRTAO"
    custom_lib_path: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so"


@dataclass
class RTDETRNvDSClassAttribute(BaseNVDSClassAttributes):
    """Structured configuration defining the schema for nvdsinfer class-attr element for RT-DETR."""

    topk: int = 20


@dataclass
class RTDETRNvDSInferConfig(BaseDSType):
    """RTDETRNvDSInfer config element."""

    property: RTDETRNvDSPropertyConfig = RTDETRNvDSPropertyConfig(
        cluster_mode=4,
        net_scale_factor=0.0173520735728,
        network_type=0,
        network_mode=2
    )
    class_attrs_all: RTDETRNvDSClassAttribute = RTDETRNvDSClassAttribute()

    def validate(self):
        """Function to validate the dataclass."""
        pass


if __name__ == "__main__":
    rtdetr_config = RTDETRNvDSInferConfig()
    assert is_dataclass(rtdetr_config), "The instance of base_config is not a dataclass."
    print(str(rtdetr_config))
