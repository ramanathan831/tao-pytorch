# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Augmentation Module."""

import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms


class CLDataAugmentation:
    """
    Class for applying data augmentation to images.

    Args:
        img_size (int): The size of the images after resizing.
        random_flip (dict, optional): A dictionary containing keys: enable (bool), hflip_probability (float), vflip_probability (float).
        random_rotate (dict, optional): A dictionary containing keys: enable (bool), rotate_probability (float), angle_list (List[float]).
        random_color (dict, optional): A dictionary containing keys: enable (bool), brightness (float), contrast (float), saturation (float), hue (float).
        with_scale_random_crop (dict, optional): A dictionary containing keys: enable (bool), scale_range (List[float]).
        with_random_crop (bool, optional): Apply random resized crop.
        with_random_blur (bool, optional): Apply random Gaussian blur.
        mean (list, optional): Mean values for normalization (default is [0.5, 0.5, 0.5]).
        std (list, optional): Standard deviation values for normalization (default is [0.5, 0.5, 0.5]).
    """

    def __init__(
            self,
            img_size,
            random_flip=None,
            random_rotate=None,
            random_color=None,
            with_scale_random_crop=None,
            with_random_crop=False,
            with_random_blur=False,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    ):
        """Initialize"""
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_color = random_color
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.mean = mean
        self.std = std
        # transform function
        if self.with_random_crop:
            self.randomcrop = transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0))
        if self.random_color is not None and self.random_color.enable:
            self.colorjitter = transforms.ColorJitter(
                brightness=self.random_color.brightness,
                contrast=self.random_color.contrast,
                saturation=self.random_color.saturation,
                hue=self.random_color.hue
            )

    def transform(self, imgs, to_tensor=True):
        """
        Apply a sequence of data augmentation routines to a list of images.

        Args:
            imgs (list): A list of PIL images to apply data augmentation to.
            to_tensor (bool, optional): Convert images to PyTorch tensors (default is True).

        Returns:
            tuple: A tuple containing augmented images.

        Notes:
            This function performs a series of image augmentation operations on the input images. The following steps are performed in sequence:

            1. Resize images to the specified image size (if not using dynamic resizing).
            3. Apply horizontal and vertical flips based on the given probabilities.
            4. Apply random rotation based on the given probability and angle list.
            5. Apply random resized crop if specified.
            6. Apply scale and random crop transformations if enabled.
            7. Apply random Gaussian blur if specified.
            8. Apply random color jitter transformations.
            9. Convert images to PyTorch tensors and normalize them.

        """
        # resize image and covert to tensor
        # input already is PIL image

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        if self.random_flip is not None and self.random_flip.enable:
            hflip_probability = 1 - self.random_flip.hflip_probability
            vflip_probability = 1 - self.random_flip.vflip_probability

            if random.random() > hflip_probability:
                imgs = [TF.hflip(img) for img in imgs]

            if random.random() > vflip_probability:
                imgs = [TF.vflip(img) for img in imgs]

        if self.random_rotate is not None and self.random_rotate.enable:
            random_base = 1 - self.random_rotate.rotate_probability

            if random.random() > random_base:
                angles = self.random_rotate.angle_list
                index = random.randint(0, 2)
                angle = angles[index]
                imgs = [TF.rotate(img, angle) for img in imgs]

        if self.with_random_crop:
            imgs = [self.randomcrop(img) for img in imgs]

        if self.with_scale_random_crop is not None and self.with_scale_random_crop.enable:
            # rescale
            scale_range = self.with_scale_random_crop.scale_range
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]

        if self.with_random_blur:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if self.random_color is not None and self.random_color.enable:

            imgs_tf = []
            for img in imgs:
                imgs_tf.append(self.colorjitter(img))
            imgs = imgs_tf

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            imgs = [TF.normalize(img, mean=self.mean, std=self.std)
                    for img in imgs]

        return imgs


def pil_crop(image, box, cropsize, default_value):
    """
    Crop an image using the specified box coordinates.

    Args:
        image (PIL.Image.Image): The input image to be cropped.
        box (Tuple[int]): A tuple containing the crop box coordinates.
        cropsize (int): The desired size of the crop.
        default_value (int): The default value to fill the cropped image.

    Returns:
        PIL.Image.Image: The cropped image.
    """
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    """
    Generate random crop box coordinates for cropping an image.

    Args:
        imgsize (Tuple[int, int]): The size of the original image (height, width).
        cropsize (int): The desired size of the crop.

    Returns:
        Tuple: A tuple containing the crop box coordinates.
    """
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    """
    Resize an image using a specified scale.

    Args:
        img (Image.Image): The input image to be rescaled.
        scale (float): The scaling factor.
        order (int): The interpolation order for resizing.

    Returns:
        Image.Image: The rescaled image.
    """
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    """
    Resize an image using a specified scale.

    Args:
        img (Image.Image): The input image to be resized.
        size (Tuple[int, int]): The target size (height, width) of the resized image.
        order (int): The interpolation order for resizing.

    Returns:
        Image.Image: The resized image.
    """
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
