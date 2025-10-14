# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Text tokenization utilities for OneFormer model.

This module provides tokenization functionality for processing text queries
in the OneFormer model, including BPE encoding and text preprocessing.
"""

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import torch


@lru_cache()
def default_bpe():
    """Get the default BPE vocabulary file path.

    Returns:
        str: Path to the BPE vocabulary file
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz"
    )


@lru_cache()
def bytes_to_unicode():
    """Returns list of utf-8 byte and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for decent
    coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables
    between utf-8 bytes and unicode strings. And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """Clean and normalize text using ftfy and html unescaping.

    Args:
        text (str): Input text to clean

    Returns:
        str: Cleaned text
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Clean whitespace in text by normalizing multiple spaces to single spaces.

    Args:
        text (str): Input text to clean

    Returns:
        str: Text with normalized whitespace
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class Tokenize:
    """Text tokenizer class for OneFormer text processing.

    This class handles tokenization of text inputs for the OneFormer model,
    including special tokens and length constraints.
    """

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        """Initialize the tokenizer.

        Args:
            tokenizer: The underlying tokenizer to use
            max_seq_len (int): Maximum sequence length
            truncate (bool): Whether to truncate sequences that are too long
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        """Tokenize input text(s).

        Args:
            texts (str or list): Text or list of texts to tokenize

        Returns:
            torch.Tensor: Tokenized text as tensor
        """
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts
        ]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[: self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {self.max_seq_len}"
                    )
            result[i, : len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result


class SimpleTokenizer(object):
    """Simple BPE tokenizer for text processing.

    This class implements Byte-Pair Encoding (BPE) tokenization for text inputs
    used in the OneFormer model's text processing pipeline.
    """

    def __init__(self, bpe_path: str = default_bpe()):
        """Initialize the SimpleTokenizer with BPE vocabulary.

        Args:
            bpe_path (str): Path to the BPE vocabulary file
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        """Apply BPE encoding to a token.

        Args:
            token (str): Token to encode

        Returns:
            str: BPE encoded token
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:  # noqa: E722
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """Encode text to token IDs.

        Args:
            text (str): Text to encode

        Returns:
            list: List of token IDs
        """
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        """Decode token IDs back to text.

        Args:
            tokens (list): List of token IDs to decode

        Returns:
            str: Decoded text
        """
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text
