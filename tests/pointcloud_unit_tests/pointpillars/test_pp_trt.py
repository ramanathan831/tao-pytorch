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

"""Unit test for PP TensorRT engines"""
from easydict import EasyDict
import os
import subprocess
import sys
import tempfile
import yaml
import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
import pytest
from nvidia_tao_pytorch.pointcloud.pointpillars.scripts.inference import CustomNMS


MODEL_CONFIG = """
model:
    name: PointPillar
    vfe:
        name: PillarVFE
        with_distance: False
        use_absolue_xyz: True
        use_norm: True
        num_filters: [64]
    map_to_bev:
        name: PointPillarScatter
        num_bev_features: 64
    backbone_2d:
        name: BaseBEVBackbone
        layer_nums: [3, 5, 5]
        layer_strides: [2, 2, 2]
        num_filters: [64, 128, 256]
        upsample_strides: [1, 2, 4]
        num_upsample_filters: [128, 128, 128]
    dense_head:
        name: AnchorHeadSingle
        class_agnostic: False
        use_direction_classifier: True
        dir_offset: 0.78539
        dir_limit_offset: 0.0
        num_dir_bins: 2
        anchor_generator_config: [
            {
                'class_name': 'Vehicle',
                'anchor_sizes': [[3.9, 1.6, 1.56]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': 'Pedestrian',
                'anchor_sizes': [[0.8, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': 'Cyclist',
                'anchor_sizes': [[1.76, 0.6, 1.73]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }
        ]
        target_assigner_config:
            name: AxisAlignedTargetAssigner
            pos_fraction: -1.0
            sample_size: 512
            norm_by_num_examples: False
            match_height: False
            box_coder: ResidualCoder
        loss_config:
            loss_weights: {
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
    post_processing:
        recall_thresh_list: [0.3, 0.5, 0.7]
        score_thresh: 0.1
        output_raw_score: False
        eval_metric: kitti
        nms_config:
            multi_classes_nms: False
            nms_type: nms_gpu
            nms_thresh: 0.01
            nms_pre_max_size: 4096
            nms_post_max_size: 500
    sync_bn: False
"""


@pytest.mark.pointcloud_unit
@pytest.mark.tensorrt
def test_trt_engine():
    """Test TensorRT Engines."""
    model_config = EasyDict(yaml.safe_load(MODEL_CONFIG)["model"])
    os_handle, tmp_fp32_file = tempfile.mkstemp(suffix=".fp32")
    os.close(os_handle)
    os_handle, tmp_fp16_file = tempfile.mkstemp(suffix=".fp16")
    os.close(os_handle)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_onnx_file = os.path.join(current_dir, "data/pointpillars_ptm.onnx")
    base_cmd = "trtexec --onnx={} --optShapes={} --workspace=1000 --saveEngine={}"
    fp32_cmd = base_cmd.format(
        tmp_onnx_file,
        "points:1x204800x4,num_points:1",
        tmp_fp32_file
    )
    fp16_cmd = base_cmd.format(
        tmp_onnx_file,
        "points:1x204800x4,num_points:1",
        tmp_fp16_file
    )
    fp16_cmd += " --fp16"
    rc = subprocess.call(fp32_cmd, stdout=sys.stderr, shell=True)
    assert rc == 0
    rc = subprocess.call(fp16_cmd, stdout=sys.stderr, shell=True)
    assert rc == 0
    # input_data = np.random.random((1, 64, 512, 512)).astype(np.float32)
    bin_file = os.path.join(current_dir, "data/102.bin")
    points = np.fromfile(bin_file, dtype=np.float32).reshape((1, 204800, 4))
    num_points = np.array([204800], dtype=np.int32)
    fp32_engine = EngineFromBytes(BytesFromPath(tmp_fp32_file))
    fp16_engine = EngineFromBytes(BytesFromPath(tmp_fp16_file))
    with TrtRunner(fp32_engine) as runner:
        fp32_out = runner.infer(
            feed_dict={
                "points": points,
                "num_points": num_points,
            }
        )
    with TrtRunner(fp16_engine) as runner:
        fp16_out = runner.infer(
            feed_dict={
                "points": points,
                "num_points": num_points,
            }
        )
    post_processor = CustomNMS(model_config.post_processing)
    fp32_boxes = post_processor(fp32_out["output_boxes"], fp32_out["num_boxes"])[0]
    fp16_boxes = post_processor(fp16_out["output_boxes"], fp16_out["num_boxes"])[0]
    fp32_boxes = fp32_boxes[fp32_boxes[:, -2] > 0.4]
    fp16_boxes = fp16_boxes[fp16_boxes[:, -2] > 0.4]
    num_boxes = min(fp32_boxes.shape[0], fp16_boxes.shape[0])
    fp32_boxes = fp32_boxes[:num_boxes, :]
    fp16_boxes = fp16_boxes[:num_boxes, :]
    assert np.allclose(fp32_boxes, fp16_boxes, rtol=1e-3, atol=1e-2), print(fp32_boxes, fp16_boxes)
    if os.path.exists(tmp_fp32_file):
        os.remove(tmp_fp32_file)
    if os.path.exists(tmp_fp16_file):
        os.remove(tmp_fp16_file)
