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

"""Unit tests for tokenizer utilities."""

import pytest
import torch

from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import (
    canonicalize_text,
    SigLIP2WrappedTokenizer,
    CLIPCompatibleTokenizer,
)


@pytest.mark.multimodal_unit
class TestCanonicalizeText:
    """Test canonicalize_text function."""

    def test_lowercase(self):
        """Test that text is lowercased."""
        assert canonicalize_text("Hello World") == "hello world"
        assert canonicalize_text("UPPERCASE") == "uppercase"

    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        assert canonicalize_text("Hello, World!") == "hello world"
        assert canonicalize_text("What's up?") == "whats up"
        assert canonicalize_text("test...test") == "testtest"

    def test_underscore_to_space(self):
        """Test that underscores are converted to spaces."""
        assert canonicalize_text("hello_world") == "hello world"
        assert canonicalize_text("this_is_a_test") == "this is a test"

    def test_whitespace_normalization(self):
        """Test that multiple whitespaces are normalized to single space."""
        assert canonicalize_text("hello   world") == "hello world"
        assert canonicalize_text("  leading and trailing  ") == "leading and trailing"
        assert canonicalize_text("tabs\tand\nnewlines") == "tabs and newlines"

    def test_combined_transformations(self):
        """Test combined transformations."""
        assert canonicalize_text("Hello, World! This_is_a_test.") == "hello world this is a test"
        assert canonicalize_text("  Multiple   Spaces_AND_Punctuation!!!  ") == "multiple spaces and punctuation"

    def test_keep_punctuation_exact_string(self):
        """Test keeping specific punctuation strings."""
        result = canonicalize_text("Hello {} World", keep_punctuation_exact_string="{}")
        assert result == "hello {} world"

    def test_empty_string(self):
        """Test empty string input."""
        assert canonicalize_text("") == ""

    def test_only_punctuation(self):
        """Test string with only punctuation."""
        assert canonicalize_text("...!!!???") == ""

    def test_unicode_text(self):
        """Test unicode text is preserved (except punctuation)."""
        # Unicode letters should be preserved
        assert canonicalize_text("Café") == "café"


@pytest.mark.multimodal_unit
class TestSigLIP2WrappedTokenizer:
    """Test SigLIP2WrappedTokenizer class."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        class MockProcessor:
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor())
        assert tokenizer._max_length == 64

    def test_custom_max_length(self):
        """Test tokenizer with custom max length."""
        class MockProcessor:
            def __call__(self, text, **kwargs):
                max_len = kwargs.get('max_length', 64)
                return {
                    'input_ids': torch.zeros(len(text), max_len, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), max_len, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor(), max_length=128)
        assert tokenizer._max_length == 128

    def test_canonicalization_applied(self):
        """Test that canonicalization is applied to input text."""
        received_text = []

        class MockProcessor:
            def __call__(self, text, **kwargs):
                received_text.extend(text)
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor())
        tokenizer(["Hello, World!", "Test_String"])

        # Text should be canonicalized before reaching processor
        assert received_text == ["hello world", "test string"]

    def test_returns_dict(self):
        """Test that tokenizer returns dict with expected keys."""
        class MockProcessor:
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor())
        result = tokenizer(["test"])

        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert 'attention_mask' in result


@pytest.mark.multimodal_unit
class TestCLIPCompatibleTokenizer:
    """Test CLIPCompatibleTokenizer class."""

    def test_single_text_returns_list(self):
        """Test that single text input returns list format."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())
        result = tokenizer("single text")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_single_text_squeezed(self):
        """Test that single text has batch dimension squeezed."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())
        result = tokenizer("single text")

        # Should be squeezed from (1, 64) to (64,)
        assert result[0]['input_ids'].shape == (64,)
        assert result[0]['attention_mask'].shape == (64,)

    def test_list_text_keeps_batch(self):
        """Test that list of texts keeps batch dimension."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())
        result = tokenizer(["text1", "text2", "text3"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['input_ids'].shape == (3, 64)
        assert result[0]['attention_mask'].shape == (3, 64)

    def test_clip_dataloader_compatibility(self):
        """Test that result[0] gives the dict (CLIP dataloader pattern)."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())

        # CLIP dataloader pattern: tokenizer(text)[0]
        result = tokenizer(["test"])[0]

        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert 'attention_mask' in result
