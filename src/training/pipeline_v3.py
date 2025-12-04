#!/usr/bin/env python3
"""
ONN Training Pipeline v3 - Optimized for Low VRAM
=================================================

Improvements over v2:
1. Mixed precision training with autocast
2. Gradient accumulation for larger effective batches
3. Chunked cross-entropy to avoid full vocab logits
4. Batched AIM logging to reduce Docker overhead
5. Memory-efficient embedding access
6. Optional gradient checkpointing
"""

import argparse
import json
import time
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.hf_tokenizer import load_tokenizer


@dataclass
class TrainingConfigV3:
    """Training configuration optimized for low VRAM."""
    model_path: str = "models/phi-4"
    output_dir: str = "output/training_v3"
    
    # Training params
    epochs: int = 1
    batch_size: int = 4  # Micro batch size
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    max_samples: int = 100
    max_seq_len: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # LoRA params
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_layers: List[int] = field(default_factory=lambda: [0, 10, 20, 30, 39])
    
    # Logging
    log_every: int = 5
    checkpoint_every: int = 50
    eval_every: int = 25
    aim_log_every: int = 10  # Batch AIM logs to reduce overhead
    
    # Validation split
    val_split: float = 0.1
    
    # Memory optimization
    empty_cache_every: int = 20  # Clear CUDA cache periodically
    
    # Resume from checkpoint
    resume_from: Optional[str] = None  # Path to checkpoint dir or "latest"
    
    # Early stopping
    early_stopping: bool = False  # Enable early stopping
    early_stopping_patience: int = 5  # Epochs to wait before stopping
    early_stopping_min_delta: float = 0.0  # Minimum improvement required
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "max_samples": self.max_samples,
            "max_seq_len": self.max_seq_len,
            "learning_rate": self.learning_rate,
            "use_amp": self.use_amp,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "resume_from": self.resume_from,
        }


class DatasetLoader:
    """Load datasets in various formats."""
    
    @staticmethod
    def load(path: str, max_samples: int = 100) -> List[Dict[str, Any]]:
        path = Path(path)
        
        if path.suffix == ".jsonl":
            return DatasetLoader._load_jsonl(path, max_samples)
        elif path.suffix == ".parquet":
            return DatasetLoader._load_parquet(path, max_samples)
        elif path.suffix == ".json":
            return DatasetLoader._load_json(path, max_samples)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    @staticmethod
    def _load_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(data) >= max_samples:
                    break
                data.append(json.loads(line.strip()))
        return data
    
    @staticmethod
    def _load_parquet(path: Path, max_samples: int) -> List[Dict[str, Any]]:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.head(max_samples).to_dict('records')
    
    @staticmethod
    def _load_json(path: Path, max_samples: int) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:max_samples] if isinstance(data, list) else [data]
    
    @staticmethod
    def extract_text(item: Dict[str, Any]) -> str:
        """Extract text from various dataset formats."""
        text_fields = ['text', 'content', 'input', 'prompt', 'question', 'instruction']
        
        for field in text_fields:
            if field in item:
                value = item[field]
                if isinstance(value, str):
                    return value
        
        # Handle multimodal prompts
        if 'prompt' in item:
            prompt = item['prompt']
            if isinstance(prompt, list):
                texts = []
                for elem in prompt:
                    if isinstance(elem, dict) and 'text' in elem:
                        texts.append(elem['text'])
                    elif isinstance(elem, str):
                        texts.append(elem)
                return " ".join(texts)
        
        # Fallback
        text_parts = [v for k, v in item.items() if isinstance(v, str) and len(v) > 0]
        return " ".join(text_parts)


class LoRALayerV3(nn.Module):
    """Optimized LoRA layer with FP16 support."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, 
                 alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        
        # LoRA matrices - use float16 for forward, but keep master copy in float32
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for LoRA computation
        x_float = x.float()
        x_drop = self.dropout(x_float)
        # Low-rank transform
        lora_out = F.linear(F.linear(x_drop, self.lora_A), self.lora_B)
        return x + (lora_out * self.scale).to(x.dtype)


class LoRAStackV3(nn.Module):
    """Stack of LoRA adapters."""
    
    def __init__(self, hidden_size: int, num_layers: int, 
                 rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoRALayerV3(hidden_size, hidden_size, rank, alpha, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx](hidden)
        return hidden


class EmbeddingManagerV3:
    """Memory-efficient embedding manager."""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.embeddings = None
        self.hidden_size = None
        self.vocab_size = None
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embedding weights from model."""
        index_path = self.model_path / "model.safetensors.index.json"
        
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            
            for key, file in weight_map.items():
                if "embed_tokens" in key and "weight" in key:
                    embed_file = self.model_path / file
                    print(f"📦 Loading embeddings from {embed_file.name}...")
                    weights = load_safetensors(str(embed_file))
                    self.embeddings = weights[key].to(self.device)
                    break
        else:
            shard_files = sorted(self.model_path.glob("model-*.safetensors"))
            if shard_files:
                weights = load_safetensors(str(shard_files[0]))
                for key, tensor in weights.items():
                    if "embed_tokens" in key and "weight" in key:
                        self.embeddings = tensor.to(self.device)
                        break
        
        if self.embeddings is None:
            raise RuntimeError("Could not find embedding weights")
        
        self.vocab_size, self.hidden_size = self.embeddings.shape
        print(f"   Embeddings: vocab={self.vocab_size}, hidden={self.hidden_size}")
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.clamp(0, self.vocab_size - 1)
        return F.embedding(token_ids, self.embeddings)


class ChunkedCrossEntropyLoss:
    """Memory-efficient cross-entropy that chunks over vocab dimension."""
    
    def __init__(self, vocab_size: int, chunk_size: int = 8192):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
    
    def __call__(self, hidden: torch.Tensor, lm_head_weight: torch.Tensor,
                 labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy in chunks to avoid materializing full logits.
        
        Args:
            hidden: [batch, seq, hidden]
            lm_head_weight: [vocab, hidden]
            labels: [batch, seq]
            mask: [batch, seq]
        """
        batch_size, seq_len, hidden_size = hidden.shape
        
        # Flatten for efficiency
        hidden_flat = hidden.view(-1, hidden_size)  # [batch*seq, hidden]
        labels_flat = labels.view(-1)  # [batch*seq]
        mask_flat = mask.view(-1)  # [batch*seq]
        
        total_loss = 0.0
        valid_count = mask_flat.sum()
        
        # Process in chunks over sequence positions
        chunk_size = min(self.chunk_size, hidden_flat.shape[0])
        
        for start in range(0, hidden_flat.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_flat.shape[0])
            
            chunk_hidden = hidden_flat[start:end]  # [chunk, hidden]
            chunk_labels = labels_flat[start:end]  # [chunk]
            chunk_mask = mask_flat[start:end]  # [chunk]
            
            # Compute logits for this chunk
            chunk_logits = F.linear(chunk_hidden.float(), lm_head_weight.float())  # [chunk, vocab]
            
            # Cross-entropy loss
            chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, reduction='none')
            chunk_loss = (chunk_loss * chunk_mask).sum()
            
            total_loss += chunk_loss
            
            # Free memory
            del chunk_logits
        
        return total_loss / (valid_count + 1e-8)


class CosineScheduler:
    """Cosine annealing LR scheduler with warmup."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, 
                 total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        lr_scale = self._get_lr_scale()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base_lr * lr_scale
    
    def _get_lr_scale(self) -> float:
        if self.step_count <= self.warmup_steps:
            return self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))


class AIMBatchLogger:
    """Batched AIM logger to reduce Docker overhead."""
    
    def __init__(self, experiment: str = "training_v3"):
        self.metrics_buffer = []
        self.buffer_size = 10
        self.aim = None
        self.experiment = experiment
        self._init_aim()
    
    def _init_aim(self):
        try:
            from src.tracking.aim_logger import AIMDockerLogger
            self.aim = AIMDockerLogger()
            self.aim.start_run(experiment=self.experiment)
        except Exception as e:
            print(f"   ⚠️ AIM not available: {e}")
            self.aim = None
    
    def log(self, metrics: Dict[str, float], step: int):
        """Buffer metrics for batch logging."""
        self.metrics_buffer.append((metrics, step))
        
        if len(self.metrics_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Send buffered metrics to AIM."""
        if not self.aim or not self.metrics_buffer:
            return
        
        for metrics, step in self.metrics_buffer:
            for name, value in metrics.items():
                try:
                    self.aim.log_metric(name, value, step)
                except Exception:
                    pass  # Silently ignore logging errors
        
        self.metrics_buffer = []
    
    def close(self):
        self.flush()
        if self.aim:
            try:
                self.aim.finish_run()
            except Exception:
                pass


class OptimizedTrainer:
    """Optimized trainer with mixed precision and gradient accumulation."""
    
    def __init__(self, config: TrainingConfigV3):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.metrics = {
            "train_loss": [], "val_loss": [],
            "train_ppl": [], "val_ppl": [],
            "learning_rate": [], "grad_norm": [],
        }
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if config.use_amp else None
        
        # Initialize components
        print(f"\n🔧 Initializing optimized trainer...")
        self._init_tokenizer()
        self._init_embeddings()
        self._init_lora()
        self._init_optimizer()
        self._init_loss_fn()
        self._init_logging()
        
        # Print config
        print(f"   Batch size: {config.batch_size} (micro) x {config.gradient_accumulation_steps} = {config.effective_batch_size} (effective)")
        print(f"   Mixed precision: {'ON' if config.use_amp else 'OFF'}")
    
    def _init_tokenizer(self):
        print(f"   Loading tokenizer...")
        self.tokenizer = load_tokenizer(Path(self.config.model_path))
        self.pad_id = self.tokenizer.pad_id
        self.vocab_size = self.tokenizer.vocab_size
    
    def _init_embeddings(self):
        self.embed_manager = EmbeddingManagerV3(Path(self.config.model_path), self.device)
        self.hidden_size = self.embed_manager.hidden_size
        # Use embeddings as LM head (tied weights)
        self.lm_head_weight = self.embed_manager.embeddings
    
    def _init_lora(self):
        num_layers = len(self.config.target_layers)
        self.lora = LoRAStackV3(
            self.hidden_size, num_layers,
            self.config.lora_rank, self.config.lora_alpha, self.config.lora_dropout
        ).to(self.device)
        
        params = sum(p.numel() for p in self.lora.parameters())
        print(f"   LoRA: {params:,} trainable params")
    
    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.lora.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = None
    
    def _init_loss_fn(self):
        self.loss_fn = ChunkedCrossEntropyLoss(self.vocab_size)
    
    def _init_logging(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            print(f"   📊 TensorBoard: {self.output_dir / 'tensorboard'}")
        except ImportError:
            self.writer = None
        
        self.aim = AIMBatchLogger(f"v3_{time.strftime('%Y%m%d_%H%M%S')}")
    
    def tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = self.config.max_seq_len
        batch_ids = []
        
        for text in texts:
            ids = self.tokenizer.encode(text)[:max_len]
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))
            batch_ids.append(ids)
        
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        attention_mask = (input_ids != self.pad_id).float()
        
        return input_ids, attention_mask
    
    def compute_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compute LM loss with chunked cross-entropy."""
        # Get embeddings
        hidden = self.embed_manager.get_embeddings(input_ids)
        
        # Apply LoRA
        for i in range(len(self.config.target_layers)):
            hidden = self.lora(hidden, i)
        
        # Shift for next-token prediction
        shift_hidden = hidden[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Chunked cross-entropy
        loss = self.loss_fn(shift_hidden, self.lm_head_weight, shift_labels, shift_mask)
        
        # Perplexity
        with torch.no_grad():
            ppl = torch.exp(loss.detach()).item()
        
        return loss, ppl
    
    def train_step(self, batch_texts: List[str], accumulation_step: int) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        self.lora.train()
        
        input_ids, attention_mask = self.tokenize_batch(batch_texts)
        
        # Mixed precision forward/backward
        if self.config.use_amp:
            with autocast(device_type='cuda'):
                loss, ppl = self.compute_loss(input_ids, attention_mask)
                loss = loss / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            loss, ppl = self.compute_loss(input_ids, attention_mask)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
        
        # Only update weights at end of accumulation
        if (accumulation_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.use_amp:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.lora.parameters(), self.config.grad_clip
            )
            
            if self.config.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
        else:
            grad_norm = 0.0
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "ppl": ppl,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": current_lr
        }
    
    @torch.no_grad()
    def eval_step(self, batch_texts: List[str]) -> Dict[str, float]:
        self.lora.eval()
        input_ids, attention_mask = self.tokenize_batch(batch_texts)
        
        if self.config.use_amp:
            with autocast(device_type='cuda'):
                loss, ppl = self.compute_loss(input_ids, attention_mask)
        else:
            loss, ppl = self.compute_loss(input_ids, attention_mask)
        
        return {"loss": loss.item(), "ppl": ppl}
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None):
        """Full training loop with optimizations."""
        print(f"\n🚀 Starting optimized training...")
        print(f"   Train: {len(train_data)} samples, Val: {len(val_data) if val_data else 0} samples")
        
        train_texts = [DatasetLoader.extract_text(item) for item in train_data]
        val_texts = [DatasetLoader.extract_text(item) for item in val_data] if val_data else []
        
        # Calculate steps
        steps_per_epoch = len(train_texts) // self.config.batch_size
        total_optimizer_steps = (steps_per_epoch * self.config.epochs) // self.config.gradient_accumulation_steps
        warmup_steps = int(total_optimizer_steps * self.config.warmup_ratio)
        
        self.scheduler = CosineScheduler(self.optimizer, warmup_steps, total_optimizer_steps)
        
        print(f"   Steps/epoch: {steps_per_epoch}, Total opt steps: {total_optimizer_steps}")
        
        # Initialize early stopping if enabled
        early_stopper = None
        if self.config.early_stopping and val_texts:
            from src.training.early_stopping import EarlyStopping
            early_stopper = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                restore_best_weights=True,
            )
            print(f"   Early stopping: patience={self.config.early_stopping_patience}")
        
        # Resume from checkpoint if specified
        start_epoch = 0
        start_global_step = 0
        optimizer_step = 0
        best_val_loss = float('inf')
        
        if self.config.resume_from:
            resume_state = self.load_checkpoint(self.config.resume_from)
            if resume_state:
                start_epoch = resume_state.get("epoch", 0)
                start_global_step = resume_state.get("global_step", 0)
                optimizer_step = resume_state.get("optimizer_step", 0)
                best_val_loss = resume_state.get("best_val_loss", float('inf'))
                
                # Restore scheduler state
                if "scheduler_step_count" in resume_state:
                    self.scheduler.step_count = resume_state["scheduler_step_count"]
                
                print(f"   📂 Resuming from epoch {start_epoch}, step {start_global_step}")
        
        start_time = time.time()
        global_step = start_global_step
        stopped_early = False
        
        for epoch in range(start_epoch, self.config.epochs):
            print(f"\n📚 Epoch {epoch + 1}/{self.config.epochs}")
            epoch_losses = []
            
            indices = np.random.permutation(len(train_texts))
            self.optimizer.zero_grad()
            
            # Calculate step offset for resume (skip already done steps in current epoch)
            step_offset_in_epoch = 0
            if epoch == start_epoch and start_global_step > 0:
                step_offset_in_epoch = start_global_step % steps_per_epoch
                print(f"   Resuming from step {step_offset_in_epoch} in epoch {epoch + 1}")
            
            for batch_idx, batch_start in enumerate(range(0, len(train_texts), self.config.batch_size)):
                # Skip already processed batches when resuming
                if epoch == start_epoch and batch_idx < step_offset_in_epoch:
                    continue
                batch_indices = indices[batch_start:batch_start + self.config.batch_size]
                if len(batch_indices) == 0:
                    continue
                    
                batch_texts = [train_texts[i] for i in batch_indices]
                
                step_start = time.time()
                metrics = self.train_step(batch_texts, batch_idx)
                step_time = (time.time() - step_start) * 1000
                
                global_step += 1
                epoch_losses.append(metrics["loss"])
                
                # Update optimizer step counter
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer_step += 1
                
                # Record metrics
                self.metrics["train_loss"].append(metrics["loss"])
                self.metrics["train_ppl"].append(metrics["ppl"])
                self.metrics["learning_rate"].append(metrics["lr"])
                if metrics["grad_norm"] > 0:
                    self.metrics["grad_norm"].append(metrics["grad_norm"])
                
                # TensorBoard
                if self.writer:
                    self.writer.add_scalar("train/loss", metrics["loss"], global_step)
                    self.writer.add_scalar("train/ppl", metrics["ppl"], global_step)
                    self.writer.add_scalar("train/lr", metrics["lr"], global_step)
                
                # AIM (batched)
                if global_step % self.config.aim_log_every == 0:
                    self.aim.log({
                        "train/loss": metrics["loss"],
                        "train/ppl": metrics["ppl"],
                    }, global_step)
                
                # Progress
                if global_step % self.config.log_every == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.log_every:])
                    print(f"   Step {global_step:4d} | loss={avg_loss:.4f} | ppl={metrics['ppl']:.0f} | "
                          f"lr={metrics['lr']:.2e} | {step_time:.0f}ms")
                
                # Validation
                if val_texts and global_step % self.config.eval_every == 0:
                    val_metrics = self._evaluate(val_texts)
                    print(f"   📊 Val: loss={val_metrics['loss']:.4f} | ppl={val_metrics['ppl']:.0f}")
                    
                    self.metrics["val_loss"].append(val_metrics["loss"])
                    self.metrics["val_ppl"].append(val_metrics["ppl"])
                    
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        self.save_checkpoint("best", epoch, global_step, optimizer_step, best_val_loss)
                
                # Memory cleanup
                if global_step % self.config.empty_cache_every == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Checkpoint
                if global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(f"step_{global_step}", epoch, global_step, optimizer_step, best_val_loss)
            
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"\n   📈 Epoch {epoch + 1}: avg_loss={avg_epoch_loss:.4f}")
            
            # End-of-epoch early stopping check
            if early_stopper is not None and val_texts:
                # Run final validation for this epoch
                val_metrics = self._evaluate(val_texts)
                
                if early_stopper(val_metrics["loss"], self, epoch):
                    stopped_early = True
                    print(f"   🛑 Early stopping triggered at epoch {epoch + 1}")
                    break
        
        total_time = time.time() - start_time
        self.save_checkpoint("final", epoch, global_step, optimizer_step, best_val_loss)
        
        summary = {
            "total_time_seconds": total_time,
            "total_steps": global_step,
            "optimizer_steps": optimizer_step,
            "final_train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else 0,
            "best_val_loss": best_val_loss if best_val_loss < float('inf') else None,
            "stopped_early": stopped_early,
            "stopped_epoch": epoch + 1,
            "config": self.config.to_dict(),
        }
        
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.writer:
            self.writer.close()
        self.aim.close()
        
        print(f"\n✅ Training complete!")
        print(f"   Time: {total_time:.1f}s")
        print(f"   Final loss: {summary['final_train_loss']:.4f}")
        print(f"   Output: {self.output_dir}")
        
        return summary
    
    def _evaluate(self, texts: List[str]) -> Dict[str, float]:
        losses, ppls = [], []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            if batch:
                metrics = self.eval_step(batch)
                losses.append(metrics["loss"])
                ppls.append(metrics["ppl"])
        
        return {"loss": np.mean(losses), "ppl": np.mean(ppls)}
    
    def save_checkpoint(self, name: str, epoch: int = 0, global_step: int = 0, 
                        optimizer_step: int = 0, best_val_loss: float = float('inf')):
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save LoRA weights
        state_dict = {k: v.contiguous() for k, v in self.lora.state_dict().items()}
        save_safetensors(state_dict, checkpoint_dir / f"{name}.safetensors")
        
        # Save config
        with open(checkpoint_dir / f"{name}_config.json", 'w') as f:
            json.dump({
                "target_layers": self.config.target_layers,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
            }, f, indent=2)
        
        # Save training state for resume
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "best_val_loss": best_val_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step_count": self.scheduler.step_count if self.scheduler else 0,
            "metrics": {k: v[-100:] for k, v in self.metrics.items()},  # Last 100 values
        }
        
        if self.scaler:
            training_state["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(training_state, checkpoint_dir / f"{name}_state.pt")
        
        print(f"   💾 Checkpoint: {name} (epoch={epoch}, step={global_step})")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint for resume training.
        
        Args:
            checkpoint_path: Path to checkpoint dir, checkpoint name (e.g. "step_100"), 
                           or "latest" to find the most recent checkpoint.
        
        Returns:
            Training state dict with epoch, global_step, etc.
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        
        if checkpoint_path == "latest":
            # Find most recent checkpoint
            state_files = list(checkpoint_dir.glob("*_state.pt"))
            if not state_files:
                print(f"   ⚠️ No checkpoints found in {checkpoint_dir}")
                return {}
            
            # Sort by modification time
            state_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            state_file = state_files[0]
            name = state_file.stem.replace("_state", "")
        else:
            # Check if it's a full path or just a name
            if Path(checkpoint_path).is_dir():
                # It's a directory - find latest in it
                state_files = list(Path(checkpoint_path).glob("*_state.pt"))
                if state_files:
                    state_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    state_file = state_files[0]
                    name = state_file.stem.replace("_state", "")
                    checkpoint_dir = Path(checkpoint_path)
                else:
                    print(f"   ⚠️ No checkpoints found in {checkpoint_path}")
                    return {}
            elif Path(checkpoint_path).exists():
                # It's a state file path
                state_file = Path(checkpoint_path)
                name = state_file.stem.replace("_state", "")
                checkpoint_dir = state_file.parent
            else:
                # It's a checkpoint name
                name = checkpoint_path
                state_file = checkpoint_dir / f"{name}_state.pt"
        
        weights_file = checkpoint_dir / f"{name}.safetensors"
        
        if not weights_file.exists():
            print(f"   ⚠️ Checkpoint weights not found: {weights_file}")
            return {}
        
        if not state_file.exists():
            # Weights exist but no state - just load weights
            print(f"   📦 Loading LoRA weights from {name}...")
            weights = load_safetensors(str(weights_file))
            self.lora.load_state_dict(weights)
            return {"global_step": 0, "epoch": 0, "optimizer_step": 0}
        
        print(f"   📦 Loading checkpoint: {name}")
        
        # Load LoRA weights
        weights = load_safetensors(str(weights_file))
        self.lora.load_state_dict(weights)
        
        # Load training state
        training_state = torch.load(state_file, map_location=self.device)
        
        # Restore optimizer
        self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
        
        # Restore scaler if using AMP
        if self.scaler and "scaler_state_dict" in training_state:
            self.scaler.load_state_dict(training_state["scaler_state_dict"])
        
        # Restore metrics history
        if "metrics" in training_state:
            for k, v in training_state["metrics"].items():
                if k in self.metrics:
                    self.metrics[k] = v
        
        print(f"   ✓ Resumed from epoch {training_state.get('epoch', 0)}, "
              f"step {training_state.get('global_step', 0)}")
        
        return {
            "epoch": training_state.get("epoch", 0),
            "global_step": training_state.get("global_step", 0),
            "optimizer_step": training_state.get("optimizer_step", 0),
            "best_val_loss": training_state.get("best_val_loss", float('inf')),
            "scheduler_step_count": training_state.get("scheduler_step_count", 0),
        }


def main():
    parser = argparse.ArgumentParser(description="ONN Training Pipeline v3 - Optimized")
    parser.add_argument("--dataset", required=True, help="Dataset path")
    parser.add_argument("--model", default="models/phi-4", help="Model path")
    parser.add_argument("--output", default="output/training_v3", help="Output dir")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Resume from checkpoint: 'latest', checkpoint name, or path")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ONN TRAINING PIPELINE v3 - OPTIMIZED FOR LOW VRAM")
    print("="*70)
    
    config = TrainingConfigV3(
        model_path=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        use_amp=not args.no_amp,
        resume_from=args.resume,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
    )
    
    print(f"\n📁 Loading dataset: {args.dataset}")
    all_data = DatasetLoader.load(args.dataset, config.max_samples)
    print(f"   Loaded {len(all_data)} samples")
    
    if config.val_split > 0 and len(all_data) > 10:
        split_idx = int(len(all_data) * (1 - config.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    else:
        train_data, val_data = all_data, None
    
    trainer = OptimizedTrainer(config)
    summary = trainer.train(train_data, val_data)
    
    return summary


if __name__ == "__main__":
    main()
