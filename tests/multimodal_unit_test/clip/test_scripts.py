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

"""CLIP scripts unit tests."""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from nvidia_tao_pytorch.multimodal.clip.scripts.inference import (
    get_image_files,
    load_and_preprocess_batch,
    save_embeddings,
    load_text_file,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    load_model_from_checkpoint,
    SUPPORTED_CHECKPOINT_EXTENSIONS,
)
from nvidia_tao_pytorch.multimodal.clip.scripts.export import (
    CLIPVisionEncoder,
    CLIPTextEncoder,
    VALID_ENCODER_TYPES,
)


@pytest.mark.multimodal_unit
class TestGetImageFiles:
    """Test get_image_files function."""

    def test_finds_supported_extensions(self):
        """Test that all supported image extensions are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images with different extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                img = Image.new('RGB', (100, 100), color='red')
                img.save(os.path.join(tmpdir, f'test{ext}'))

            found_files = get_image_files(tmpdir)
            assert len(found_files) == 6

    def test_ignores_unsupported_extensions(self):
        """Test that unsupported extensions are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(tmpdir, 'valid.jpg'))

            # Create files with unsupported extensions
            for ext in ['.txt', '.json', '.py', '.pdf']:
                with open(os.path.join(tmpdir, f'invalid{ext}'), 'w') as f:
                    f.write('test')

            found_files = get_image_files(tmpdir)
            assert len(found_files) == 1
            assert found_files[0].endswith('.jpg')

    def test_recursive_search(self):
        """Test that subdirectories are searched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            subdir1 = os.path.join(tmpdir, 'subdir1')
            subdir2 = os.path.join(tmpdir, 'subdir1', 'subdir2')
            os.makedirs(subdir2)

            # Create images at different levels
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(tmpdir, 'root.jpg'))
            img.save(os.path.join(subdir1, 'level1.jpg'))
            img.save(os.path.join(subdir2, 'level2.jpg'))

            found_files = get_image_files(tmpdir)
            assert len(found_files) == 3

    def test_returns_sorted_list(self):
        """Test that returned list is sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new('RGB', (100, 100), color='red')
            for name in ['z.jpg', 'a.jpg', 'm.jpg']:
                img.save(os.path.join(tmpdir, name))

            found_files = get_image_files(tmpdir)
            assert found_files == sorted(found_files)

    def test_empty_directory(self):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            found_files = get_image_files(tmpdir)
            assert not found_files


@pytest.mark.multimodal_unit
class TestSaveEmbeddings:
    """Test save_embeddings function."""

    def test_save_embeddings_creates_file(self):
        """Test that embeddings HDF5 file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
            embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            assert os.path.exists(output_path)

    def test_save_embeddings_correct_content(self):
        """Test that HDF5 file has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/img1.jpg"]
            embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            with h5py.File(output_path, 'r') as f:
                loaded_embeddings = f['embeddings'][:]
                # Decode bytes to string if needed
                loaded_paths = [
                    p.decode('utf-8') if isinstance(p, bytes) else p
                    for p in f['image_paths'][:]
                ]

            assert len(loaded_paths) == 1
            assert loaded_paths[0] == "/path/to/img1.jpg"
            np.testing.assert_array_almost_equal(loaded_embeddings[0], [0.1, 0.2, 0.3])

    def test_save_embeddings_handles_unicode(self):
        """Test that embeddings with unicode paths are saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/图片.jpg"]
            embeddings = np.array([[0.1, 0.2]], dtype=np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            with h5py.File(output_path, 'r') as f:
                # Decode bytes to string if needed
                loaded_paths = [
                    p.decode('utf-8') if isinstance(p, bytes) else p
                    for p in f['image_paths'][:]
                ]

            assert loaded_paths[0] == "/path/to/图片.jpg"

    def test_save_embeddings_metadata(self):
        """Test that HDF5 file has correct metadata attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg", "/path/to/img3.jpg"]
            embeddings = np.random.randn(3, 768).astype(np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            with h5py.File(output_path, 'r') as f:
                assert f.attrs['num_images'] == 3
                assert f.attrs['embedding_dim'] == 768

    def test_save_text_embeddings(self):
        """Test saving text embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts = ["a photo of a cat", "a photo of a dog"]
            embeddings = np.random.randn(2, 768).astype(np.float32)

            save_embeddings(texts, embeddings, tmpdir, embedding_type='text')

            output_path = os.path.join(tmpdir, "text_embeddings.h5")
            assert os.path.exists(output_path)

            with h5py.File(output_path, 'r') as f:
                assert f.attrs['num_texts'] == 2
                assert f.attrs['embedding_type'] == 'text'
                loaded_texts = [
                    t.decode('utf-8') if isinstance(t, bytes) else t
                    for t in f['texts'][:]
                ]
                assert loaded_texts[0] == "a photo of a cat"


@pytest.mark.multimodal_unit
class TestSupportedExtensions:
    """Test supported extension constants."""

    def test_image_extensions(self):
        """Test that common image extensions are supported."""
        expected = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        assert SUPPORTED_IMAGE_EXTENSIONS == expected

    def test_checkpoint_extensions(self):
        """Test that checkpoint extensions are consistent."""
        expected = {'.pth', '.ckpt'}
        assert SUPPORTED_CHECKPOINT_EXTENSIONS == expected


@pytest.mark.multimodal_unit
class TestLoadAndPreprocessBatch:
    """Test load_and_preprocess_batch function."""

    def test_returns_none_for_empty_batch(self):
        """Test that None is returned when no images can be loaded."""
        device = torch.device('cpu')

        # Non-existent files
        batch_files = ['/nonexistent/path/image1.jpg', '/nonexistent/path/image2.jpg']

        # Simple identity preprocess
        def preprocess(img):
            return torch.zeros(3, 224, 224)

        batch, valid_paths = load_and_preprocess_batch(batch_files, preprocess, device)
        assert batch is None
        assert not valid_paths

    def test_loads_valid_images(self):
        """Test that valid images are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            img_paths = []
            for i in range(3):
                img = Image.new('RGB', (224, 224), color='red')
                path = os.path.join(tmpdir, f'img{i}.jpg')
                img.save(path)
                img_paths.append(path)

            device = torch.device('cpu')

            def preprocess(img):
                return torch.zeros(3, 224, 224)

            batch, valid_paths = load_and_preprocess_batch(img_paths, preprocess, device)

            assert batch is not None
            assert batch.shape == (3, 3, 224, 224)
            assert len(valid_paths) == 3

    def test_skips_invalid_images(self):
        """Test that invalid images are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one valid image
            valid_path = os.path.join(tmpdir, 'valid.jpg')
            img = Image.new('RGB', (224, 224), color='red')
            img.save(valid_path)

            # Create one invalid file
            invalid_path = os.path.join(tmpdir, 'invalid.jpg')
            with open(invalid_path, 'w') as f:
                f.write('not an image')

            device = torch.device('cpu')

            def preprocess(img):
                return torch.zeros(3, 224, 224)

            batch, valid_paths = load_and_preprocess_batch(
                [valid_path, invalid_path], preprocess, device
            )

            assert batch is not None
            assert batch.shape == (1, 3, 224, 224)
            assert len(valid_paths) == 1
            assert valid_paths[0] == valid_path


@pytest.mark.multimodal_unit
class TestLoadModelFromCheckpoint:
    """Test load_model_from_checkpoint function."""

    def test_raises_for_unsupported_format(self):
        """Test that unsupported formats raise NotImplementedError."""
        # Mock model class (won't be used since format check happens first)
        mock_model_class = type('MockModel', (), {})
        with pytest.raises(NotImplementedError) as excinfo:
            load_model_from_checkpoint('/path/to/model.xyz', None, mock_model_class)
        assert "not supported" in str(excinfo.value)

    def test_raises_for_engine_format(self):
        """Test that engine format raises NotImplementedError with tao-deploy message."""
        # Mock model class (won't be used since format check happens first)
        mock_model_class = type('MockModel', (), {})
        with pytest.raises(NotImplementedError) as excinfo:
            load_model_from_checkpoint('/path/to/model.engine', None, mock_model_class)
        assert "tao-deploy" in str(excinfo.value)


@pytest.mark.multimodal_unit
class TestCLIPVisionEncoder:
    """Test CLIPVisionEncoder wrapper class."""

    def test_forward_with_dict_output(self):
        """Test forward pass when model returns dict output."""
        class MockModel:
            def __call__(self, image):
                batch_size = image.shape[0]
                return {"image_features": torch.randn(batch_size, 768)}

        encoder = CLIPVisionEncoder(MockModel())
        dummy_input = torch.randn(2, 3, 224, 224)
        output = encoder(dummy_input)

        assert output.shape == (2, 768)

    def test_forward_with_tuple_output(self):
        """Test forward pass when model returns tuple output."""
        class MockModel:
            def __call__(self, image):
                batch_size = image.shape[0]
                return (torch.randn(batch_size, 768), torch.randn(batch_size, 768))

        encoder = CLIPVisionEncoder(MockModel())
        dummy_input = torch.randn(2, 3, 224, 224)
        output = encoder(dummy_input)

        assert output.shape == (2, 768)

    def test_forward_preserves_batch_size(self):
        """Test that batch size is preserved through forward pass."""
        class MockModel:
            def __call__(self, image):
                batch_size = image.shape[0]
                return {"image_features": torch.randn(batch_size, 512)}

        encoder = CLIPVisionEncoder(MockModel())

        for batch_size in [1, 4, 16]:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            output = encoder(dummy_input)
            assert output.shape[0] == batch_size


@pytest.mark.multimodal_unit
class TestCLIPTextEncoder:
    """Test CLIPTextEncoder wrapper class."""

    def test_forward_with_dict_output(self):
        """Test forward pass when model returns dict output."""
        class MockModel:
            def __call__(self, text):
                batch_size = text['input_ids'].shape[0]
                return {"text_features": torch.randn(batch_size, 768)}

        encoder = CLIPTextEncoder(MockModel())
        dummy_input_ids = torch.zeros(2, 64, dtype=torch.long)
        dummy_attention_mask = torch.ones(2, 64, dtype=torch.long)
        output = encoder(dummy_input_ids, dummy_attention_mask)

        assert output.shape == (2, 768)

    def test_forward_with_tuple_output(self):
        """Test forward pass when model returns tuple output."""
        class MockModel:
            def __call__(self, text):
                batch_size = text['input_ids'].shape[0]
                return (torch.randn(batch_size, 768), torch.randn(batch_size, 768))

        encoder = CLIPTextEncoder(MockModel())
        dummy_input_ids = torch.zeros(2, 64, dtype=torch.long)
        dummy_attention_mask = torch.ones(2, 64, dtype=torch.long)
        output = encoder(dummy_input_ids, dummy_attention_mask)

        assert output.shape == (2, 768)


@pytest.mark.multimodal_unit
class TestValidEncoderTypes:
    """Test encoder type validation."""

    def test_valid_encoder_types(self):
        """Test that valid encoder types are defined correctly."""
        assert VALID_ENCODER_TYPES == {'combined', 'separate'}


@pytest.mark.multimodal_unit
class TestLoadTextFile:
    """Test load_text_file function."""

    def test_loads_text_file(self):
        """Test that text file is loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = os.path.join(tmpdir, 'texts.txt')
            with open(text_path, 'w') as f:
                f.write("a photo of a cat\n")
                f.write("a photo of a dog\n")
                f.write("a sunset over the ocean\n")

            texts = load_text_file(text_path)

            assert len(texts) == 3
            assert texts[0] == "a photo of a cat"
            assert texts[2] == "a sunset over the ocean"

    def test_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = os.path.join(tmpdir, 'texts.txt')
            with open(text_path, 'w') as f:
                f.write("line one\n")
                f.write("\n")
                f.write("   \n")
                f.write("line two\n")

            texts = load_text_file(text_path)

            assert len(texts) == 2
            assert texts[0] == "line one"
            assert texts[1] == "line two"

    def test_handles_unicode(self):
        """Test that unicode text is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = os.path.join(tmpdir, 'texts.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("一只猫的照片\n")
                f.write("日落\n")

            texts = load_text_file(text_path)

            assert len(texts) == 2
            assert texts[0] == "一只猫的照片"
