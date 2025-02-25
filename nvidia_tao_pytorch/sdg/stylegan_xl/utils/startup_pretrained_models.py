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

""" Download pretrained modules of StyleGAN-XL """

import pickle
import sys
import torch
import os
import shutil
import tempfile

from nvidia_tao_core.cloud_handlers import utils


def download_and_convert_pretrained_models():

    path_pretrained_modules = "/tao-pt/nvidia_tao_pytorch/sdg/stylegan_xl/pretrained_modules"
    if os.path.exists(os.path.join(path_pretrained_modules, "InceptionV3.pth")) and \
        os.path.exists(os.path.join(path_pretrained_modules, "tf_efficientnet_lite0_embed.pth")):
        return
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
            tmp_path = os.path.join(tmpdirname, "stylegan-xl")

            utils.download_huggingface_dataset("https://github.com/autonomousvision/stylegan-xl.git", tmp_path, token=False)
            utils.download_from_https_link("https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl", tmp_path)

            InceptionV3_file_path = os.path.join(tmp_path, "inception-2015-12-05.pkl")
            tf_efficientnet_lite0_embed_file_path = os.path.join(tmp_path, "in_embeddings/tf_efficientnet_lite0.pkl")

            # Try loading the checkpoint using pickle
            sys.path.append(tmp_path) # Add system path of stylegan-xl source repo to run pickle properly
            with open(InceptionV3_file_path, 'rb') as f:
                InceptionV3 = pickle.load(f)
            with open(tf_efficientnet_lite0_embed_file_path, 'rb') as f:
                tf_efficientnet_lite0_embed = pickle.load(f)

            torch.save(InceptionV3.state_dict(), os.path.join(path_pretrained_modules, "InceptionV3.pth"))
            torch.save(tf_efficientnet_lite0_embed['embed'].state_dict(), os.path.join(path_pretrained_modules, "tf_efficientnet_lite0_embed.pth"))

download_and_convert_pretrained_models()
