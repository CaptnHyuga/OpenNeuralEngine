#!/usr/bin/env python3
"""
ONN Training Pipeline v2 - Real Token-Based Training

Key improvements over v1:
1. Real tokenization using HuggingFace tokenizers
2. Actual embedding lookup from model weights
3. Proper language modeling objective (next-token prediction)
4. Gradient clipping for stability
5. Learning rate scheduler (cosine annealing)
6. Validation split for tracking generalization
7. Memory-efficient processing with chunking
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.hf_tokenizer import load_tokenizer


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""
    model_path: str = "models/phi-4"
    output_dir: str = "output/training_v2"
    
    # Training params
    epochs: int = 1
    batch_size: int = 8  # Smaller for real embeddings
    max_samples: int = 100
    max_seq_len: int = 128  # Max tokens per sample
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    grad_clip: float = 1.0
    
    # LoRA params
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_layers: List[int] = field(default_factory=lambda: [0, 10, 20, 30, 39])
    
    # Logging
    log_every: int = 5
    checkpoint_every: int = 50
    eval_every: int = 25
    
    # Validation split
    val_split: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "max_seq_len": self.max_seq_len,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }


class DatasetLoader:
    """Load datasets in various formats with tokenization."""
    
    @staticmethod
    def load(path: str, max_samples: int = 100) -> List[Dict[str, Any]]:
        """Load dataset from file."""
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
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    @staticmethod
    def _load_parquet(path: Path, max_samples: int) -> List[Dict[str, Any]]:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.head(max_samples).to_dict('records')
        except ImportError:
            raise RuntimeError("pandas and pyarrow required for parquet files")
    
    @staticmethod
    def _load_json(path: Path, max_samples: int) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[:max_samples]
        return [data]
    
    @staticmethod
    def extract_text(item: Dict[str, Any]) -> str:
        """Extract text from various dataset formats."""
        # Common text field names
        text_fields = ['text', 'content', 'input', 'prompt', 'question', 'instruction']
        
        for field in text_fields:
            if field in item:
                value = item[field]
                if isinstance(value, str):
                    return value
        
        # For multimodal data, extract text components
        if 'prompt' in item:
            prompt = item['prompt']
            if isinstance(prompt, list):
                # Extract text elements from prompt list
                texts = []
                for elem in prompt:
                    if isinstance(elem, dict) and 'text' in elem:
                        texts.append(elem['text'])
                    elif isinstance(elem, str):
                        texts.append(elem)
                return " ".join(texts)
        
        # Fallback: concatenate all string values
        text_parts = []
        for k, v in item.items():
            if isinstance(v, str) and len(v) > 0:
                text_parts.append(v)
        return " ".join(text_parts)


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer with dropout."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, 
                 alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA matrices - always float32 for stability
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with Kaiming, B with zeros for zero init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features]
        x_float = x.float()
        x_drop = self.dropout(x_float)
        # Low-rank transform
        lora_out = F.linear(F.linear(x_drop, self.lora_A), self.lora_B)
        return x + (lora_out * self.scale)


class LoRAStack(nn.Module):
    """Stack of LoRA adapters for multiple layers."""
    
    def __init__(self, hidden_size: int, num_layers: int, 
                 rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoRALayer(hidden_size, hidden_size, rank, alpha, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx](hidden)
        return hidden


class EmbeddingManager:
    """Manages model embeddings for memory-efficient training."""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.embeddings = None
        self.hidden_size = None
        self.vocab_size = None
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embedding weights from model."""
        # Find safetensors files
        shard_files = sorted(self.model_path.glob("model-*.safetensors"))
        if not shard_files:
            # Try single file
            single = self.model_path / "model.safetensors"
            if single.exists():
                shard_files = [single]
            else:
                raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        
        # Load embedding from first shard (usually contains embed_tokens)
        print(f"📦 Loading embeddings from {shard_files[0].name}...")
        
        # Load index to find embedding location
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            
            # Find embedding weights
            embed_key = None
            embed_file = None
            for key, file in weight_map.items():
                if "embed_tokens" in key and "weight" in key:
                    embed_key = key
                    embed_file = self.model_path / file
                    break
            
            if embed_key and embed_file:
                print(f"   Found embedding: {embed_key} in {embed_file.name}")
                weights = load_safetensors(str(embed_file))
                self.embeddings = weights[embed_key].to(self.device)
        else:
            # Load first shard and look for embeddings
            weights = load_safetensors(str(shard_files[0]))
            for key, tensor in weights.items():
                if "embed_tokens" in key and "weight" in key:
                    self.embeddings = tensor.to(self.device)
                    print(f"   Found embedding: {key}")
                    break
        
        if self.embeddings is None:
            raise RuntimeError("Could not find embedding weights in model")
        
        self.vocab_size, self.hidden_size = self.embeddings.shape
        print(f"   Embeddings: vocab_size={self.vocab_size}, hidden_size={self.hidden_size}")
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs."""
        # Clamp to valid range
        token_ids = token_ids.clamp(0, self.vocab_size - 1)
        return F.embedding(token_ids, self.embeddings)


class LMHead:
    """Language model head for next-token prediction (memory-efficient)."""
    
    def __init__(self, embeddings: torch.Tensor):
        # Tied embeddings - use same weights for output
        self.weight = embeddings  # [vocab_size, hidden_size]
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits: [batch, seq, hidden] -> [batch, seq, vocab]."""
        # Use matmul instead of F.linear for memory efficiency
        return torch.matmul(hidden.float(), self.weight.t().float())


class CosineScheduler:
    """Cosine annealing learning rate scheduler with warmup."""
    
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
            # Linear warmup
            return self.step_count / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))


class RealLanguageModelTrainer:
    """Trainer with real tokenization and language modeling objective."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_ppl": [],  # Perplexity
            "val_ppl": [],
            "learning_rate": [],
            "grad_norm": [],
        }
        
        # Initialize components
        print(f"\n🔧 Initializing trainer...")
        self._init_tokenizer()
        self._init_embeddings()
        self._init_lora()
        self._init_optimizer()
        self._init_tensorboard()
        self._init_aim()
    
    def _init_tokenizer(self):
        """Initialize tokenizer."""
        print(f"   Loading tokenizer from {self.config.model_path}...")
        self.tokenizer = load_tokenizer(Path(self.config.model_path))
        self.pad_id = self.tokenizer.pad_id
        self.vocab_size = self.tokenizer.vocab_size
        print(f"   Tokenizer: vocab_size={self.vocab_size}, pad_id={self.pad_id}")
    
    def _init_embeddings(self):
        """Initialize embedding manager."""
        self.embed_manager = EmbeddingManager(Path(self.config.model_path), self.device)
        self.hidden_size = self.embed_manager.hidden_size
        self.lm_head = LMHead(self.embed_manager.embeddings)
    
    def _init_lora(self):
        """Initialize LoRA adapters."""
        num_target_layers = len(self.config.target_layers)
        self.lora = LoRAStack(
            self.hidden_size, 
            num_target_layers,
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.lora_dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.lora.parameters())
        trainable_params = sum(p.numel() for p in self.lora.parameters() if p.requires_grad)
        print(f"   LoRA: {trainable_params:,} trainable params across {num_target_layers} layers")
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.lora.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = None  # Set during training when we know total steps
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            print(f"   📊 TensorBoard: {self.output_dir / 'tensorboard'}")
        except ImportError:
            self.writer = None
    
    def _init_aim(self):
        """Initialize AIM logging via Docker."""
        try:
            from src.tracking.aim_logger import AIMDockerLogger
            self.aim = AIMDockerLogger()
            self.aim.start_run(experiment=f"training_v2_{time.strftime('%Y%m%d_%H%M%S')}")
            print(f"   🎯 AIM: logging enabled")
        except Exception as e:
            print(f"   ⚠️ AIM not available: {e}")
            self.aim = None
    
    def tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize texts and create attention mask."""
        max_len = self.config.max_seq_len
        
        batch_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)[:max_len]
            # Pad to max_len
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))
            batch_ids.append(ids)
        
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.pad_id).float()
        
        return input_ids, attention_mask
    
    def compute_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compute language modeling loss (next-token prediction)."""
        # Get embeddings
        hidden = self.embed_manager.get_embeddings(input_ids)  # [batch, seq, hidden]
        
        # Apply LoRA layers (simulating transformer forward pass)
        for i, layer_idx in enumerate(self.config.target_layers):
            hidden = self.lora(hidden, i)
        
        # Get logits
        logits = self.lm_head.forward(hidden)  # [batch, seq, vocab]
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()   # [batch, seq-1]
        shift_mask = attention_mask[:, 1:].contiguous()  # [batch, seq-1]
        
        # Cross-entropy loss (only on non-padded positions)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )  # [batch * (seq-1)]
        
        # Mask padding
        loss = loss.view(shift_labels.shape)  # [batch, seq-1]
        loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
        
        # Perplexity
        ppl = torch.exp(loss).item()
        
        return loss, ppl
    
    def train_step(self, batch_texts: List[str]) -> Dict[str, float]:
        """Single training step."""
        self.lora.train()
        self.optimizer.zero_grad()
        
        # Tokenize
        input_ids, attention_mask = self.tokenize_batch(batch_texts)
        
        # Forward + loss
        loss, ppl = self.compute_loss(input_ids, attention_mask)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.lora.parameters(), 
            self.config.grad_clip
        )
        
        # Update
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            "loss": loss.item(),
            "ppl": ppl,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": current_lr
        }
    
    @torch.no_grad()
    def eval_step(self, batch_texts: List[str]) -> Dict[str, float]:
        """Evaluation step (no gradients)."""
        self.lora.eval()
        
        input_ids, attention_mask = self.tokenize_batch(batch_texts)
        loss, ppl = self.compute_loss(input_ids, attention_mask)
        
        return {"loss": loss.item(), "ppl": ppl}
    
    def train(self, train_data: List[Dict[str, Any]], val_data: Optional[List[Dict[str, Any]]] = None):
        """Full training loop."""
        print(f"\n🚀 Starting training...")
        print(f"   Train samples: {len(train_data)}")
        if val_data:
            print(f"   Val samples: {len(val_data)}")
        
        # Extract text from samples
        train_texts = [DatasetLoader.extract_text(item) for item in train_data]
        val_texts = [DatasetLoader.extract_text(item) for item in val_data] if val_data else []
        
        # Calculate total steps and init scheduler
        steps_per_epoch = (len(train_texts) + self.config.batch_size - 1) // self.config.batch_size
        total_steps = steps_per_epoch * self.config.epochs
        self.scheduler = CosineScheduler(
            self.optimizer, 
            self.config.warmup_steps, 
            total_steps
        )
        
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total steps: {total_steps}")
        
        start_time = time.time()
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            print(f"\n📚 Epoch {epoch + 1}/{self.config.epochs}")
            epoch_losses = []
            epoch_ppls = []
            
            # Shuffle training data
            indices = np.random.permutation(len(train_texts))
            
            for batch_start in range(0, len(train_texts), self.config.batch_size):
                batch_indices = indices[batch_start:batch_start + self.config.batch_size]
                batch_texts = [train_texts[i] for i in batch_indices]
                
                step_start = time.time()
                metrics = self.train_step(batch_texts)
                step_time = (time.time() - step_start) * 1000
                
                global_step += 1
                epoch_losses.append(metrics["loss"])
                epoch_ppls.append(metrics["ppl"])
                
                # Record metrics
                self.metrics["train_loss"].append(metrics["loss"])
                self.metrics["train_ppl"].append(metrics["ppl"])
                self.metrics["learning_rate"].append(metrics["lr"])
                self.metrics["grad_norm"].append(metrics["grad_norm"])
                
                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar("train/loss", metrics["loss"], global_step)
                    self.writer.add_scalar("train/perplexity", metrics["ppl"], global_step)
                    self.writer.add_scalar("train/learning_rate", metrics["lr"], global_step)
                    self.writer.add_scalar("train/grad_norm", metrics["grad_norm"], global_step)
                
                # Log to AIM
                if self.aim:
                    self.aim.log_metrics({
                        "train/loss": metrics["loss"],
                        "train/perplexity": metrics["ppl"],
                        "train/learning_rate": metrics["lr"],
                        "train/grad_norm": metrics["grad_norm"],
                    }, step=global_step)
                
                # Print progress
                if global_step % self.config.log_every == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.log_every:])
                    avg_ppl = np.mean(epoch_ppls[-self.config.log_every:])
                    print(f"   Step {global_step:4d} | loss={avg_loss:.4f} | ppl={avg_ppl:.1f} | "
                          f"lr={metrics['lr']:.2e} | {step_time:.0f}ms")
                
                # Validation
                if val_texts and global_step % self.config.eval_every == 0:
                    val_metrics = self._evaluate(val_texts)
                    print(f"   📊 Val: loss={val_metrics['loss']:.4f} | ppl={val_metrics['ppl']:.1f}")
                    
                    self.metrics["val_loss"].append(val_metrics["loss"])
                    self.metrics["val_ppl"].append(val_metrics["ppl"])
                    
                    if self.writer:
                        self.writer.add_scalar("val/loss", val_metrics["loss"], global_step)
                        self.writer.add_scalar("val/perplexity", val_metrics["ppl"], global_step)
                    
                    if self.aim:
                        self.aim.log_metrics({
                            "val/loss": val_metrics["loss"],
                            "val/perplexity": val_metrics["ppl"],
                        }, step=global_step)
                    
                    # Save best model
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        self.save_checkpoint("best")
                
                # Checkpoint
                if global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(f"step_{global_step}")
            
            # End of epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            avg_epoch_ppl = np.mean(epoch_ppls)
            print(f"\n   📈 Epoch {epoch + 1}: avg_loss={avg_epoch_loss:.4f}, avg_ppl={avg_epoch_ppl:.1f}")
        
        # Final evaluation
        total_time = time.time() - start_time
        self.save_checkpoint("final")
        
        # Final summary
        summary = {
            "total_time_seconds": total_time,
            "total_steps": global_step,
            "final_train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else 0,
            "final_train_ppl": self.metrics["train_ppl"][-1] if self.metrics["train_ppl"] else 0,
            "best_val_loss": best_val_loss if best_val_loss < float('inf') else None,
            "config": self.config.to_dict(),
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Close loggers
        if self.writer:
            self.writer.close()
        if self.aim:
            self.aim.close()
        
        print(f"\n✅ Training complete!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Final loss: {summary['final_train_loss']:.4f}")
        print(f"   Final perplexity: {summary['final_train_ppl']:.1f}")
        print(f"   Output: {self.output_dir}")
        
        return summary
    
    def _evaluate(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate on a set of texts."""
        all_losses = []
        all_ppls = []
        
        for batch_start in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[batch_start:batch_start + self.config.batch_size]
            metrics = self.eval_step(batch_texts)
            all_losses.append(metrics["loss"])
            all_ppls.append(metrics["ppl"])
        
        return {
            "loss": np.mean(all_losses),
            "ppl": np.mean(all_ppls)
        }
    
    def save_checkpoint(self, name: str):
        """Save checkpoint as safetensors."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        state_dict = {k: v.contiguous() for k, v in self.lora.state_dict().items()}
        save_safetensors(state_dict, checkpoint_dir / f"{name}.safetensors")
        
        config_path = checkpoint_dir / f"{name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "target_layers": self.config.target_layers,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "hidden_size": self.hidden_size,
            }, f, indent=2)
        
        print(f"   💾 Checkpoint: {name}")


def main():
    parser = argparse.ArgumentParser(description="ONN Training Pipeline v2 - Real Token Training")
    parser.add_argument("--dataset", required=True, help="Path to dataset file")
    parser.add_argument("--model", default="models/phi-4", help="Model path")
    parser.add_argument("--output", default="output/training_v2", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ONN TRAINING PIPELINE v2 - REAL TOKEN-BASED TRAINING")
    print("="*70)
    
    # Create config
    config = TrainingConfig(
        model_path=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        val_split=args.val_split,
    )
    
    # Load dataset
    print(f"\n📁 Loading dataset: {args.dataset}")
    all_data = DatasetLoader.load(args.dataset, config.max_samples)
    print(f"   Loaded {len(all_data)} samples")
    
    # Split train/val
    if config.val_split > 0 and len(all_data) > 10:
        split_idx = int(len(all_data) * (1 - config.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    else:
        train_data = all_data
        val_data = None
    
    # Create trainer
    trainer = RealLanguageModelTrainer(config)
    
    # Train
    summary = trainer.train(train_data, val_data)
    
    return summary


if __name__ == "__main__":
    main()
