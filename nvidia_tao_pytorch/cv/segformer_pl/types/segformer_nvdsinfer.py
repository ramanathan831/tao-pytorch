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

from dataclasses import dataclass, is_dataclass, field
from nvidia_tao_pytorch.core.types.nvdsinfer import (
    BaseNvDSPropertyConfig,
    BaseDSType
)


@dataclass
class SFNvDSPropertyConfig(BaseNvDSPropertyConfig):
    """Structured configuration defining the schema for nvdsinfer property element for SegFormer."""

    segmentation_output_order: int = 1

    def validate(self):
        """Validate the NVConfig."""
        super().validate()
        assert self.segmentation_output_order in [0, 1], (
            "Segmentation output order should be in 0, 1"
        )
        assert self.cluster_mode == 2, (
            "Cluster mode should be 2 since this is strictly a semantic segmentation model"
        )
        assert self.network_type == 100, (
            "Skip nvinfer post-processing, use pgie_pad_buffer_probe_network_type100() instead."
        )


@dataclass
class SFNvDSInferConfig(BaseDSType):
    """SFNvDSInfer config element."""

    property_field: SFNvDSPropertyConfig = field(default_factory=lambda: SFNvDSPropertyConfig(
        cluster_mode=2,
        net_scale_factor=0.0173520735728,
        offsets=[123.675, 116.28, 103.53],
        network_type=100,
        network_mode=2,
        output_blob_names=None,
        model_color_format=0,
        segmentation_output_order=1
    ))

    def validate(self):
        """Function to validate the dataclass."""
        self.property_field.validate()


if __name__ == "__main__":
    segformer_config = SFNvDSInferConfig()
    assert is_dataclass(segformer_config), "The instance of base_config is not a dataclass."
    print(str(segformer_config))
