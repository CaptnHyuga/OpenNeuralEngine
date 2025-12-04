#!/usr/bin/env python3
"""
Learning Rate Finder
====================

Implements the learning rate range test to find optimal learning rate.
Based on the paper "Cyclical Learning Rates for Training Neural Networks".

Usage:
    from src.training.lr_finder import LRFinder
    
    finder = LRFinder(trainer)
    optimal_lr = finder.find(train_data, min_lr=1e-7, max_lr=1.0, num_steps=100)
    
    # Or via CLI:
    python -m src.training.lr_finder --dataset data/Dataset/math.jsonl --model models/phi-4
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from pathlib import Path


class LRFinder:
    """
    Learning Rate Range Test.
    
    Trains the model with exponentially increasing learning rate over a short run
    and records the loss at each step. The optimal learning rate is where the 
    loss decreases the most (steepest descent), typically 1-10x before the minimum.
    """
    
    def __init__(self, trainer):
        """
        Initialize with a trainer object.
        
        Args:
            trainer: A trainer object with train_step method and optimizer.
                    Expected attributes: optimizer, lora (model), tokenize_batch
        """
        self.trainer = trainer
        self.history = {"lr": [], "loss": [], "smoothed_loss": []}
        
    def find(self, 
             train_data: List[Dict[str, Any]],
             min_lr: float = 1e-7, 
             max_lr: float = 1.0, 
             num_steps: int = 100,
             smooth_factor: float = 0.98,
             divergence_threshold: float = 5.0) -> Dict[str, Any]:
        """
        Run the learning rate range test.
        
        Args:
            train_data: List of training samples.
            min_lr: Starting learning rate.
            max_lr: Maximum learning rate to test.
            num_steps: Number of steps for the test.
            smooth_factor: Exponential smoothing factor for loss.
            divergence_threshold: Stop if loss exceeds this multiple of minimum.
            
        Returns:
            Dict with optimal_lr, suggested_lr_range, and history.
        """
        print(f"\n🔍 Running Learning Rate Finder...")
        print(f"   Range: {min_lr:.2e} → {max_lr:.2e} over {num_steps} steps")
        
        # Save initial state
        initial_state = {k: v.clone() for k, v in self.trainer.lora.state_dict().items()}
        initial_opt_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v 
            for k, v in self.trainer.optimizer.state_dict().items()
        }
        
        # Reset
        self.history = {"lr": [], "loss": [], "smoothed_loss": []}
        
        # Calculate multiplicative factor for exponential LR growth
        gamma = (max_lr / min_lr) ** (1.0 / num_steps)
        
        # Set initial LR
        current_lr = min_lr
        for group in self.trainer.optimizer.param_groups:
            group['lr'] = current_lr
        
        # Get text data
        from src.training.pipeline_v3 import DatasetLoader
        texts = [DatasetLoader.extract_text(item) for item in train_data]
        
        # Training loop
        best_loss = float('inf')
        smoothed_loss = None
        batch_size = self.trainer.config.batch_size
        
        for step in range(num_steps):
            # Sample a batch
            indices = np.random.choice(len(texts), min(batch_size, len(texts)), replace=False)
            batch_texts = [texts[i] for i in indices]
            
            # Training step
            self.trainer.lora.train()
            self.trainer.optimizer.zero_grad()
            
            # Forward pass
            input_ids, attention_mask = self.trainer.tokenize_batch(batch_texts)
            hidden = self.trainer.embed_manager.get_embeddings(input_ids)
            
            for i in range(len(self.trainer.config.target_layers)):
                hidden = self.trainer.lora(hidden, i)
            
            # Loss
            shift_hidden = hidden[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            loss = self.trainer.loss_fn(
                shift_hidden, 
                self.trainer.lm_head_weight, 
                shift_labels, 
                shift_mask
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainer.lora.parameters(), 1.0)
            self.trainer.optimizer.step()
            
            loss_val = loss.item()
            
            # Exponential smoothing
            if smoothed_loss is None:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smooth_factor * smoothed_loss + (1 - smooth_factor) * loss_val
            
            # Record
            self.history["lr"].append(current_lr)
            self.history["loss"].append(loss_val)
            self.history["smoothed_loss"].append(smoothed_loss)
            
            # Track best
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Check for divergence
            if smoothed_loss > divergence_threshold * best_loss:
                print(f"   ⚠️ Loss diverged at step {step}, stopping early")
                break
            
            # Progress
            if step % 10 == 0:
                print(f"   Step {step:3d} | lr={current_lr:.2e} | loss={smoothed_loss:.4f}")
            
            # Increase LR
            current_lr *= gamma
            for group in self.trainer.optimizer.param_groups:
                group['lr'] = current_lr
        
        # Restore initial state
        self.trainer.lora.load_state_dict(initial_state)
        
        # Analyze results
        result = self._analyze_results()
        
        # Print recommendations
        print(f"\n📊 Learning Rate Finder Results:")
        print(f"   Minimum loss: {result['min_loss']:.4f} at lr={result['lr_at_min']:.2e}")
        print(f"   Steepest descent at lr={result['steepest_lr']:.2e}")
        print(f"   💡 Suggested lr: {result['optimal_lr']:.2e}")
        print(f"   💡 Suggested range: {result['suggested_min_lr']:.2e} → {result['suggested_max_lr']:.2e}")
        
        return result
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze the LR vs loss curve to find optimal LR."""
        lrs = np.array(self.history["lr"])
        losses = np.array(self.history["smoothed_loss"])
        
        # Find minimum loss
        min_idx = np.argmin(losses)
        min_loss = losses[min_idx]
        lr_at_min = lrs[min_idx]
        
        # Find steepest descent (largest negative gradient)
        # Use log scale for LR
        log_lrs = np.log10(lrs)
        
        # Compute gradient of loss w.r.t. log(lr)
        if len(losses) > 5:
            # Use window for smoothing gradient
            gradients = np.gradient(losses, log_lrs)
            
            # Find steepest negative gradient (exclude first and last few points)
            valid_range = slice(5, min(len(gradients)-5, min_idx))
            if valid_range.stop > valid_range.start:
                steepest_idx = valid_range.start + np.argmin(gradients[valid_range])
                steepest_lr = lrs[steepest_idx]
            else:
                steepest_lr = lr_at_min / 10
        else:
            steepest_lr = lr_at_min / 10
        
        # Optimal LR is typically a bit below the steepest descent point
        # or 1 order of magnitude below minimum
        optimal_lr = min(steepest_lr, lr_at_min / 3)
        
        # Suggest range for cyclical LR
        suggested_min_lr = optimal_lr / 10
        suggested_max_lr = optimal_lr * 3
        
        return {
            "optimal_lr": optimal_lr,
            "min_loss": min_loss,
            "lr_at_min": lr_at_min,
            "steepest_lr": steepest_lr,
            "suggested_min_lr": suggested_min_lr,
            "suggested_max_lr": suggested_max_lr,
            "history": self.history,
        }
    
    def plot(self, save_path: Optional[Path] = None):
        """Plot the LR vs loss curve."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(self.history["lr"], self.history["loss"], alpha=0.3, label="Raw Loss")
            ax.plot(self.history["lr"], self.history["smoothed_loss"], label="Smoothed Loss")
            
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Loss")
            ax.set_title("Learning Rate Finder")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   📊 Plot saved: {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            print("   ⚠️ matplotlib not available for plotting")


def find_lr(
    dataset_path: str,
    model_path: str = "models/phi-4",
    output_dir: str = "output/lr_finder",
    min_lr: float = 1e-7,
    max_lr: float = 1.0,
    num_steps: int = 100,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """
    Convenience function to run LR finder.
    
    Args:
        dataset_path: Path to training dataset.
        model_path: Path to model.
        output_dir: Directory for outputs.
        min_lr: Starting learning rate.
        max_lr: Maximum learning rate.
        num_steps: Number of steps.
        max_samples: Maximum samples to load.
        
    Returns:
        Result dict with optimal_lr and other metrics.
    """
    from src.training.pipeline_v3 import TrainingConfigV3, DatasetLoader, OptimizedTrainer
    
    # Create trainer
    config = TrainingConfigV3(
        model_path=model_path,
        output_dir=output_dir,
        max_samples=max_samples,
        batch_size=4,
    )
    
    trainer = OptimizedTrainer(config)
    
    # Load data
    data = DatasetLoader.load(dataset_path, max_samples)
    
    # Run finder
    finder = LRFinder(trainer)
    result = finder.find(data, min_lr, max_lr, num_steps)
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    finder.plot(output_path / "lr_finder.png")
    
    # Save results
    import json
    with open(output_path / "lr_finder_results.json", 'w') as f:
        # Convert numpy types for JSON
        result_json = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in result.items() if k != "history"}
        json.dump(result_json, f, indent=2)
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Learning Rate Finder")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--model", default="models/phi-4", help="Model path")
    parser.add_argument("--output", default="output/lr_finder", help="Output directory")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum LR")
    parser.add_argument("--max-lr", type=float, default=1.0, help="Maximum LR")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples")
    
    args = parser.parse_args()
    
    print("="*70)
    print("LEARNING RATE FINDER")
    print("="*70)
    
    result = find_lr(
        dataset_path=args.dataset,
        model_path=args.model,
        output_dir=args.output,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_steps=args.steps,
        max_samples=args.max_samples,
    )
    
    print(f"\n✅ Done! Optimal LR: {result['optimal_lr']:.2e}")


if __name__ == "__main__":
    main()
