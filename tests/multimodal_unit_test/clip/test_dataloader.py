# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for CLIP custom dataloader."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from nvidia_tao_pytorch.multimodal.clip.dataloader.custom_loader import (
    ImageTextDataset,
    get_custom_dataloader,
)


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset with images and text files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        image_dir = tmpdir / "images"
        label_dir = tmpdir / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        # Create sample images and corresponding text files
        for i in range(5):
            img = Image.new('RGB', (64, 64), color=(i * 50, i * 50, i * 50))
            img_path = image_dir / f"image_{i}.jpg"
            img.save(img_path)

            text_path = label_dir / f"image_{i}.txt"
            text_path.write_text(f"Caption for image {i}")

        # Create image list file
        image_list_file = tmpdir / "image_list.txt"
        image_list_file.write_text("\n".join([f"image_{i}.jpg" for i in range(5)]))

        yield {
            'root_dir': str(image_dir),
            'label_dir': str(label_dir),
            'image_list_file': str(image_list_file),
            'label_suffix': '.txt',
            'tmpdir': tmpdir,
            'image_dir': image_dir,
            'label_dir_path': label_dir,
        }


@pytest.mark.multimodal_unit
class TestImageTextDataset:
    """Test ImageTextDataset class."""

    def test_initialization(self, temp_dataset):
        """Test dataset initialization with valid config."""
        dataset_config = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        dataset = ImageTextDataset(datasets=dataset_config, mode='train')

        assert len(dataset) == 5

    def test_getitem_returns_image_and_text(self, temp_dataset):
        """Test that __getitem__ returns image and text."""
        dataset_config = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        dataset = ImageTextDataset(datasets=dataset_config, mode='train')

        image, text = dataset[0]

        assert isinstance(image, Image.Image)
        assert isinstance(text, str)
        assert "Caption for image" in text

    def test_transform_applied(self, temp_dataset):
        """Test that transform is applied to images."""
        dataset_config = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataset = ImageTextDataset(
            datasets=dataset_config,
            transform=dummy_transform,
            mode='train'
        )

        image, text = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)

    def test_tokenizer_applied(self, temp_dataset):
        """Test that tokenizer is applied to text."""
        dataset_config = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        def dummy_tokenizer(text):
            return [torch.tensor([1, 2, 3])]

        dataset = ImageTextDataset(
            datasets=dataset_config,
            tokenizer=dummy_tokenizer,
            mode='train'
        )

        image, text = dataset[0]

        assert isinstance(text, torch.Tensor)
        assert text.tolist() == [1, 2, 3]

    def test_multiple_datasets(self, temp_dataset):
        """Test initialization with multiple dataset configs."""
        # Create a second dataset
        image_dir2 = temp_dataset['tmpdir'] / "images2"
        label_dir2 = temp_dataset['tmpdir'] / "labels2"
        image_dir2.mkdir()
        label_dir2.mkdir()

        for i in range(3):
            img = Image.new('RGB', (64, 64), color=(100, 100, 100))
            img_path = image_dir2 / f"img2_{i}.jpg"
            img.save(img_path)

            text_path = label_dir2 / f"img2_{i}.txt"
            text_path.write_text(f"Caption for second dataset {i}")

        image_list_file2 = temp_dataset['tmpdir'] / "image_list2.txt"
        image_list_file2.write_text("\n".join([f"img2_{i}.jpg" for i in range(3)]))

        dataset_configs = [
            {
                'root_dir': temp_dataset['root_dir'],
                'label_dir': temp_dataset['label_dir'],
                'image_list_file': temp_dataset['image_list_file'],
                'label_suffix': '.txt',
            },
            {
                'root_dir': str(image_dir2),
                'label_dir': str(label_dir2),
                'image_list_file': str(image_list_file2),
                'label_suffix': '.txt',
            },
        ]

        dataset = ImageTextDataset(datasets=dataset_configs, mode='train')

        assert len(dataset) == 8  # 5 + 3

    def test_zero_shot_eval_single_dataset(self, temp_dataset):
        """Test zero-shot eval mode with single dataset."""
        dataset_config = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        mapping = {"Caption for image 0": "mapped_text_0"}

        dataset = ImageTextDataset(
            datasets=dataset_config,
            zero_shot_eval=True,
            mapping=mapping,
            mode='val'
        )

        image, text = dataset[0]
        assert text == "mapped_text_0"

    def test_zero_shot_eval_multiple_datasets_raises(self, temp_dataset):
        """Test that zero-shot eval with multiple datasets raises error."""
        dataset_configs = [
            {
                'root_dir': temp_dataset['root_dir'],
                'label_dir': temp_dataset['label_dir'],
                'image_list_file': temp_dataset['image_list_file'],
                'label_suffix': '.txt',
            },
            {
                'root_dir': temp_dataset['root_dir'],
                'label_dir': temp_dataset['label_dir'],
                'image_list_file': temp_dataset['image_list_file'],
                'label_suffix': '.txt',
            },
        ]

        with pytest.raises(NotImplementedError):
            ImageTextDataset(
                datasets=dataset_configs,
                zero_shot_eval=True,
                mapping={},
                mode='val'
            )

    def test_empty_dataset_raises(self):
        """Test that empty dataset raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_dir = tmpdir / "images"
            image_dir.mkdir()

            image_list_file = tmpdir / "image_list.txt"
            image_list_file.write_text("")  # Empty list

            dataset_config = [{
                'root_dir': str(image_dir),
                'image_list_file': str(image_list_file),
            }]

            with pytest.raises(ValueError, match="No valid image-text pairs"):
                ImageTextDataset(datasets=dataset_config, mode='train')

    def test_glob_fallback_without_image_list(self, temp_dataset):
        """Test that dataset globs for images without image_list_file."""
        dataset_config = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'label_suffix': '.txt',
        }]

        dataset = ImageTextDataset(datasets=dataset_config, mode='train')

        assert len(dataset) == 5

    def test_legacy_root_label_dir_support(self, temp_dataset):
        """Test backward compatibility with root_label_dir field name."""
        dataset_config = [{
            'root_dir': temp_dataset['root_dir'],
            'root_label_dir': temp_dataset['label_dir'],  # Legacy name
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        dataset = ImageTextDataset(datasets=dataset_config, mode='train')

        assert len(dataset) == 5


@pytest.mark.multimodal_unit
class TestGetCustomDataloader:
    """Test get_custom_dataloader function."""

    def test_creates_dataloader(self, temp_dataset):
        """Test that dataloader is created successfully."""
        dataset_configs = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            num_workers=0,
            mode='train'
        )

        assert dataloader is not None
        assert len(dataloader) > 0

    def test_batch_size_respected(self, temp_dataset):
        """Test that batch size is respected."""
        dataset_configs = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            mode='train'
        )

        batch = next(iter(dataloader))
        images, texts = batch
        assert len(images) == 2

    def test_val_mode_no_batch_sampler(self, temp_dataset):
        """Test that val mode doesn't use batch sampler."""
        dataset_configs = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            shuffle=False,
            mode='val'
        )

        assert dataloader is not None

    def test_seed_reproducibility(self, temp_dataset):
        """Test that seed produces reproducible results."""
        dataset_configs = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader1 = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            seed=42,
            mode='train'
        )

        dataloader2 = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            seed=42,
            mode='train'
        )

        # Both should have same length
        assert len(dataloader1) == len(dataloader2)

    def test_transform_and_tokenizer_passed(self, temp_dataset):
        """Test that transform and tokenizer are passed to dataset."""
        dataset_configs = [{
            'root_dir': temp_dataset['root_dir'],
            'label_dir': temp_dataset['label_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'label_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        def dummy_tokenizer(text):
            return [torch.tensor([1, 2, 3])]

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            tokenizer=dummy_tokenizer,
            num_workers=0,
            mode='train'
        )

        images, texts = next(iter(dataloader))
        assert images.shape == (2, 3, 32, 32)
        assert texts.shape == (2, 3)
