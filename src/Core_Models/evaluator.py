"""Shared evaluation utilities for puzzle models.

Centralizes inference-time evaluation for both text-only and multimodal
models so individual scripts can focus on dataset/config wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import evaluate
import torch
from torch.utils.data import DataLoader

from utils import SimpleTokenizer, TokenTensorAdapter


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    loss: Optional[float] = None


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        task_type: str,
        device: torch.device,
        tokenizer: Optional[SimpleTokenizer] = None,
        adapter: Optional[TokenTensorAdapter] = None,
        metric_name: str = "accuracy",
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.device = device
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.metric_name = metric_name

    def _new_metric(self):
        return evaluate.load(self.metric_name)

    def _require_text_components(self) -> None:
        if self.tokenizer is None or self.adapter is None:
            raise RuntimeError("Evaluator requires tokenizer and adapter for text tasks")

    def load_checkpoint(self, path: Path) -> bool:
        if not path.exists():
            return False
        # SECURITY: Use safetensors exclusively to avoid arbitrary code execution
        from safetensors.torch import load_file
        
        safetensors_path = path.with_suffix('.safetensors') if path.suffix != '.safetensors' else path
        if not safetensors_path.exists():
            raise ValueError(
                f"Only safetensors checkpoints are supported for security. "
                f"Expected: {safetensors_path}"
            )
        
        state_dict = load_file(safetensors_path)
        self.model.load_state_dict(state_dict)
        return True

    def evaluate_text(self, pairs: Sequence[Tuple[str, str | int]], batch_size: int) -> EvalResult:
        self._require_text_components()
        tokenizer = self.tokenizer  # type: ignore[assignment]
        adapter = self.adapter  # type: ignore[assignment]

        self.model.eval()
        metric = self._new_metric()
        total_loss = 0.0
        criterion = None

        if not pairs:
            return EvalResult(metrics={self.metric_name: 0.0})

        if self.task_type != "classification":
            criterion = torch.nn.CrossEntropyLoss(ignore_index=getattr(tokenizer, "pad_id", 0))

        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                if self.task_type == "classification":
                    texts = [text for text, _ in batch]
                    labels = torch.tensor([int(label) for _, label in batch], dtype=torch.long, device=self.device)

                    token_ids = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
                    x = adapter.batch_tokens_to_tensor(token_ids).to(self.device)

                    logits = self.model(x)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    metric.add_batch(predictions=preds, references=labels.cpu().numpy())
                else:
                    inp_batch_tokens = [tokenizer.encode(inp, add_special_tokens=True) for inp, _ in batch]
                    out_batch_tokens = [tokenizer.encode(out, add_special_tokens=True) for _, out in batch]

                    x = adapter.batch_tokens_to_tensor(inp_batch_tokens).to(self.device)
                    targets = adapter.batch_tokens_to_tensor(out_batch_tokens).to(self.device)

                    logits = self.model(x)
                    if logits.shape[1] != targets.shape[1]:
                        min_len = min(logits.shape[1], targets.shape[1])
                        logits = logits[:, :min_len, :]
                        targets = targets[:, :min_len]

                    N, T, V = logits.shape
                    preds = logits.argmax(dim=-1).reshape(N * T).cpu().numpy()
                    metric.add_batch(
                        predictions=preds,
                        references=targets.reshape(N * T).cpu().numpy(),
                    )

                    if criterion is not None:
                        loss = criterion(logits.reshape(N * T, V), targets.reshape(N * T))
                        total_loss += float(loss.item())

        return EvalResult(metrics=metric.compute(), loss=total_loss if total_loss > 0 else None)

    def evaluate_multimodal(self, loader: DataLoader) -> EvalResult:
        self.model.eval()
        metric = self._new_metric()
        with torch.no_grad():
            for batch in loader:
                tokens = batch["tokens"].to(self.device)
                images = batch["images"].to(self.device)
                image_mask = batch["image_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(tokens, images, image_mask)
                preds = logits.argmax(dim=-1).cpu().numpy()
                metric.add_batch(predictions=preds, references=labels.cpu().numpy())
        return EvalResult(metrics=metric.compute())
