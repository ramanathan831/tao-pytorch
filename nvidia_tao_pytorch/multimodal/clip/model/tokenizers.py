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

"""Tokenizer utilities for CLIP-compatible training.

This module provides the canonical implementations for text processing
utilities used across C-RADIO, SigLIP2, and other CLIP-compatible model
implementations.

Classes:
    SigLIP2WrappedTokenizer: Tokenizer wrapper for SigLIP2 with text
        canonicalization
    OpenCLIPWrappedTokenizer: Tokenizer wrapper for DFN CLIP with text
        canonicalization
    CLIPCompatibleTokenizer: Wrapper for CLIP dataloader compatibility

Functions:
    canonicalize_text: Text normalization (lowercase, punctuation removal)
"""

from typing import List

from nvidia_tao_pytorch.cv.backbone_v2.text_utils import canonicalize_text


class SigLIP2WrappedTokenizer:
    """Tokenizer wrapper for SigLIP2 with text canonicalization.

    This wrapper applies text canonicalization before tokenization to improve
    zero-shot classification performance.

    Args:
        processor: The underlying processor from HuggingFace.
        max_length: Maximum sequence length for tokenization. Default: 64.
    """

    def __init__(self, processor, max_length: int = 64):
        """Initialize the tokenizer wrapper."""
        self._processor = processor
        self._max_length = max_length

    def __call__(self, text: List[str]):
        """Tokenize text with canonicalization.

        Args:
            text: List of strings to tokenize.

        Returns:
            BatchEncoding dict with 'input_ids' and 'attention_mask'.
        """
        text = [canonicalize_text(t) for t in text]
        ret = self._processor(
            text=text,
            return_tensors='pt',
            max_length=self._max_length,
            padding='max_length',
            truncation=True
        )
        return ret


class CLIPCompatibleTokenizer:
    """Wrapper to make tokenizers compatible with CLIP dataloader.

    The CLIP dataloader expects tokenizer(text)[0] to return a tensor or
    dict. This wrapper normalizes the interface across different tokenizer
    types:
    - SigLIP2WrappedTokenizer: returns dict with 'input_ids', 'attention_mask'
    - OpenCLIPWrappedTokenizer: returns dict with 'input_ids'

    This wrapper makes tokenizer(text) return a list where [0] gives the
    dict, allowing it to work with the existing dataloader pattern.

    Args:
        wrapped_tokenizer: A SigLIP2WrappedTokenizer or
            OpenCLIPWrappedTokenizer instance.
    """

    def __init__(self, wrapped_tokenizer):
        """Initialize the CLIP-compatible tokenizer wrapper."""
        self._tokenizer = wrapped_tokenizer

    def __call__(self, text):
        """Tokenize text and return in CLIP-compatible format.

        Args:
            text: Single string or list of strings.

        Returns:
            List where [0] is the tokenized dict.
        """
        if isinstance(text, str):
            # Single text - wrap in list, then squeeze the batch dimension
            result = self._tokenizer([text])
            result = {k: v.squeeze(0) for k, v in result.items()}
        else:
            # List of texts - keep batch dimension
            result = self._tokenizer(text)

        return [result]


class OpenCLIPWrappedTokenizer:
    """Tokenizer wrapper for OpenCLIP/DFN-CLIP with text canonicalization.

    This wrapper applies text canonicalization before tokenization and
    converts the output to a dict format matching SigLIP2/RADIO for
    consistency.

    Used for:
    - RADIO 'clip' adaptor (DFN CLIP)
    - backbone_v2 OpenCLIP models

    Args:
        tokenizer: The raw OpenCLIP tokenizer (callable that returns tensor).
    """

    def __init__(self, tokenizer):
        """Initialize the tokenizer wrapper."""
        self._tokenizer = tokenizer

    def __call__(self, text: List[str]):
        """Tokenize text with canonicalization and return dict format.

        Args:
            text: List of strings to tokenize.

        Returns:
            Dict with 'input_ids' key containing the tokenized tensor.
        """
        # Apply canonicalization (same as SigLIP2)
        text = [canonicalize_text(t) for t in text]

        # OpenCLIP tokenizer returns tensor directly
        result = self._tokenizer(text)

        # Wrap tensor in dict for consistency with SigLIP2/RADIO tokenizers
        return {'input_ids': result}
