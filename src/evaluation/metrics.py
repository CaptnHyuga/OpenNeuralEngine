"""Basic evaluation metrics for quick local evaluation.

These are simpler metrics that don't require lm-evaluation-harness,
useful for quick validation during training or when the full
harness isn't installed.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def compute_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    stride: int = 512,
) -> float:
    """Compute perplexity of model on given input.
    
    Lower perplexity = better language modeling.
    
    Args:
        model: Language model with forward() returning logits.
        input_ids: Input token IDs [batch, seq_len] or [seq_len].
        attention_mask: Attention mask (optional).
        stride: Stride for sliding window (handles long sequences).
        
    Returns:
        Perplexity score (float).
    """
    model.eval()
    device = next(model.parameters()).device
    
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    seq_len = input_ids.size(1)
    max_length = getattr(model.config, "max_position_embeddings", 2048)
    
    nlls = []
    prev_end = 0
    
    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_len = end - prev_end  # Only score new tokens
            
            input_chunk = input_ids[:, begin:end]
            target_chunk = input_ids[:, begin + 1:end + 1] if end < seq_len else input_ids[:, begin + 1:]
            
            outputs = model(input_chunk)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_chunk[:, 1:].contiguous()
            
            # Compute loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            
            nlls.append(loss.item() * (end - begin - 1))
            prev_end = end
            
            if end == seq_len:
                break
    
    total_nll = sum(nlls)
    total_tokens = seq_len - 1
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    
    return ppl


def compute_accuracy(
    predictions: Union[List, torch.Tensor],
    labels: Union[List, torch.Tensor],
) -> float:
    """Compute accuracy score.
    
    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        
    Returns:
        Accuracy (0.0 to 1.0).
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    predictions = list(predictions)
    labels = list(labels)
    
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(labels)}")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def compute_f1(
    predictions: Union[List, torch.Tensor],
    labels: Union[List, torch.Tensor],
    pos_label: int = 1,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.
    
    Args:
        predictions: Model predictions.
        labels: Ground truth labels.
        pos_label: Positive class label.
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    predictions = list(predictions)
    labels = list(labels)
    
    true_positives = sum(p == pos_label and l == pos_label for p, l in zip(predictions, labels))
    false_positives = sum(p == pos_label and l != pos_label for p, l in zip(predictions, labels))
    false_negatives = sum(p != pos_label and l == pos_label for p, l in zip(predictions, labels))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> float:
    """Compute BLEU score for text generation.
    
    Simplified BLEU implementation for quick evaluation.
    For production, use sacrebleu or similar.
    
    Args:
        predictions: Generated texts.
        references: Reference texts (can have multiple refs per prediction).
        max_n: Maximum n-gram size.
        
    Returns:
        BLEU score (0.0 to 1.0).
    """
    from collections import Counter
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    total_matches = [0] * max_n
    total_counts = [0] * max_n
    ref_lengths = []
    pred_lengths = []
    
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        pred_lengths.append(len(pred_tokens))
        
        # Find closest reference length
        ref_lens = [len(r.lower().split()) for r in refs]
        closest_len = min(ref_lens, key=lambda x: abs(x - len(pred_tokens)))
        ref_lengths.append(closest_len)
        
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            
            # Get max counts from all references
            max_ref_counts = Counter()
            for ref in refs:
                ref_tokens = ref.lower().split()
                ref_ngrams = get_ngrams(ref_tokens, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
            
            # Clipped counts
            clipped = {ng: min(count, max_ref_counts[ng]) for ng, count in pred_ngrams.items()}
            total_matches[n-1] += sum(clipped.values())
            total_counts[n-1] += sum(pred_ngrams.values())
    
    # Brevity penalty
    total_pred_len = sum(pred_lengths)
    total_ref_len = sum(ref_lengths)
    if total_pred_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_pred_len) if total_pred_len > 0 else 0.0
    
    # Geometric mean of n-gram precisions
    precisions = []
    for i in range(max_n):
        if total_counts[i] > 0:
            precisions.append(total_matches[i] / total_counts[i])
        else:
            precisions.append(0.0)
    
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precision = sum(math.log(p) for p in precisions) / max_n
    bleu = bp * math.exp(log_precision)
    
    return bleu
