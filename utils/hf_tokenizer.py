from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

try:
    from tokenizers import Tokenizer  # type: ignore  # nosec B404
except ImportError:
    Tokenizer = None  # type: ignore


logger = logging.getLogger(__name__)


class HFTokenizer:
    """Minimal tokenizer wrapper loading from `tokenizer.json`.

    Provides encode/decode, vocab_size, and best-effort eos_id/pad_id.
    """

    def __init__(self, tokenizer_path: Path):
        if Tokenizer is None:
            raise RuntimeError(
                "HuggingFace 'tokenizers' is required. Install via 'pip install tokenizers'."
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        # Attempt to resolve special token IDs
        self.eos_id: Optional[int] = None
        self.pad_id: int = 0  # Default to 0 if not found
        try:
            # Best effort: look for common EOS markers
            for cand in ("</s>", "<|endoftext|>", "<eos>"):
                try:
                    tid = self._tok.token_to_id(cand)
                    if tid is not None:
                        self.eos_id = int(tid)
                        break
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug("Token to id lookup failed for %s: %s", cand, exc)
            # Look for pad token
            for cand in ("<pad>", "<|pad|>", "[PAD]"):
                try:
                    tid = self._tok.token_to_id(cand)
                    if tid is not None:
                        self.pad_id = int(tid)
                        break
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug("Token to id lookup failed for %s: %s", cand, exc)
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to infer special token ids: %s", exc)

    @property
    def vocab_size(self) -> int:
        try:
            return int(self._tok.get_vocab_size())
        except AttributeError:
            # Fallback if method name differs
            return int(len(getattr(self._tok, "vocab", {})))

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        enc = self._tok.encode(text)
        return list(enc.ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids)


def load_tokenizer(model_dir: Path):  # return type flexible (HFTokenizer or fallback)
    """Load tokenizer, falling back to SimpleTokenizer if file missing.

    This keeps API tests functional even without a trained tokenizer artifact.
    """
    tok_path = Path(model_dir) / "tokenizer.json"
    try:
        return HFTokenizer(tok_path)
    except FileNotFoundError:
        try:
            from utils.tokenization import SimpleTokenizer
        except ImportError:
            raise
        # Minimal alphabet vocab
        vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + [chr(i) for i in range(ord('a'), ord('z')+1)]
        simple = SimpleTokenizer(vocab=vocab)

        class _Wrapper:
            def __init__(self, inner):
                self._inner = inner
                self.eos_id = getattr(inner, 'eos_id', 2)
                self.pad_id = getattr(inner, 'pad_id', 0)
            @property
            def vocab_size(self):
                return self._inner.vocab_size
            def encode(self, text: str, add_special_tokens: bool = True):
                return self._inner.encode(text, add_special_tokens=add_special_tokens)
            def decode(self, ids: List[int], skip_special_tokens: bool = True):
                return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)
        return _Wrapper(simple)
