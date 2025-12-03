"""Lightweight tokenization helpers used by legacy utilities.

Provides a fallback SimpleTokenizer (word-level) and a TokenTensorAdapter
that convert text into padded tensors for small experiments or diagnostic
runs. The HF tokenizer path is preferred for production, but these classes
remain for backward compatibility and quick smoke tests.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import torch


class SimpleTokenizer:
    """Minimal whitespace tokenizer with flexible vocab initialization.

    Accepts either a dict mapping token->id or a list of tokens defining
    the complete vocabulary order. Unit tests pass a list and expect the
    resulting vocab size to match exactly and expose `pad_token_id` and
    `unk_token_id` attributes.
    """

    def __init__(self, vocab: Dict[str, int] | List[str] | None = None, max_vocab_size: int = 50000) -> None:
        self.max_vocab_size = max_vocab_size
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

        if isinstance(vocab, list):
            # Direct list initialization (tests expect exact size)
            self.vocab = {tok: idx for idx, tok in enumerate(vocab)}
        else:
            # Start with default specials then optionally extend with provided dict
            self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            if isinstance(vocab, dict):
                # Merge user-provided mapping (assumed consistent/non-overlapping)
                for tok, idx in vocab.items():
                    self.vocab[tok] = idx

        # Inverse vocab
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

        # Determine special token IDs (fallbacks if absent)
        self.pad_id = self.vocab.get("<pad>", 0)
        self.bos_id = self.vocab.get("<bos>", self.pad_id + 1)
        self.eos_id = self.vocab.get("<eos>", self.bos_id + 1)
        self.unk_id = self.vocab.get("<unk>", self.eos_id + 1)

    def build_vocab(self, texts: Iterable[str]) -> None:
        """Build a vocabulary from the provided texts."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        for token, _ in counter.most_common(self.max_vocab_size - len(self.vocab)):
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().split()

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        words: List[str] = []
        for idx in token_ids:
            if skip_special_tokens and idx < len(self.special_tokens):
                continue
            words.append(self.inv_vocab.get(idx, "<unk>"))
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # Test-friendly attribute names
    @property
    def pad_token_id(self) -> int:  # type: ignore[override]
        return self.pad_id

    @property
    def unk_token_id(self) -> int:  # type: ignore[override]
        return self.unk_id


class TokenTensorAdapter:
    """Convert batches of token IDs to padded tensors."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def batch_texts(self, encoded: List[List[int]]) -> torch.Tensor:
        max_len = max(len(seq) for seq in encoded) if encoded else 1
        batch = torch.full((len(encoded), max_len), self.pad_token_id, dtype=torch.long)
        for idx, seq in enumerate(encoded):
            batch[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return batch

    def batch_tokens_to_tensor(self, encoded: List[List[int]]) -> torch.Tensor:
        """Compatibility wrapper used by legacy training code."""
        return self.batch_texts(encoded)

    def texts_to_tensor(self, texts: List[str], tokenizer: SimpleTokenizer) -> torch.Tensor:
        encoded = [tokenizer.encode(text) for text in texts]
        return self.batch_texts(encoded)

    def tensor_to_texts(self, tensor: torch.Tensor, tokenizer: SimpleTokenizer) -> List[str]:
        return [tokenizer.decode(seq.tolist()) for seq in tensor]
