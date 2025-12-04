#!/usr/bin/env python3
"""
Early Stopping Callback
=======================

Implements early stopping based on validation loss plateau detection.
Supports patience, minimum delta, and restore best weights.

Usage:
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    
    for epoch in range(epochs):
        train_loss = train()
        val_loss = validate()
        
        if early_stopping(val_loss):
            print("Early stopping triggered!")
            break
"""

from typing import Optional, Dict, Any
from pathlib import Path


class EarlyStopping:
    """
    Early stopping callback to terminate training when validation loss stops improving.
    
    Args:
        patience: Number of epochs with no improvement after which training will be stopped.
        min_delta: Minimum change in loss to qualify as an improvement.
        mode: One of "min" or "max". In "min" mode, training stops when monitored 
              quantity stops decreasing; in "max" mode stops when it stops increasing.
        restore_best_weights: Whether to restore model weights from the epoch with 
                             the best value of the monitored metric.
        verbose: Whether to print messages.
    
    Example:
        >>> early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = train_and_validate()
        ...     if early_stopping(val_loss, trainer):
        ...         break
        >>> print(f"Best loss: {early_stopping.best_loss}")
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        # For max mode, we negate the delta
        self._delta = -min_delta if mode == "max" else min_delta
    
    def __call__(
        self, 
        current_loss: float, 
        trainer: Optional[Any] = None,
        epoch: int = 0,
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_loss: Current validation loss/metric.
            trainer: Optional trainer object with lora attribute (for weight saving).
            epoch: Current epoch number.
            
        Returns:
            True if training should stop, False otherwise.
        """
        improved = False
        
        if self.best_loss is None:
            # First call
            self.best_loss = current_loss
            self.best_epoch = epoch
            improved = True
        elif self.mode == "min":
            if current_loss < self.best_loss - self._delta:
                improved = True
        else:  # max mode
            if current_loss > self.best_loss + abs(self._delta):
                improved = True
        
        if improved:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0
            
            # Save best weights
            if self.restore_best_weights and trainer is not None:
                self.best_weights = {
                    k: v.clone() 
                    for k, v in trainer.lora.state_dict().items()
                }
            
            if self.verbose:
                print(f"   ✓ Validation improved to {current_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"   ⚠️ No improvement for {self.counter}/{self.patience} epochs "
                      f"(best: {self.best_loss:.4f})")
        
        # Check if we should stop
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            
            if self.verbose:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                print(f"   Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
            
            # Restore best weights
            if self.restore_best_weights and self.best_weights is not None and trainer is not None:
                trainer.lora.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"   ✓ Restored weights from epoch {self.best_epoch}")
            
            return True
        
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_loss = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "stopped_epoch": self.stopped_epoch,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.best_loss = state.get("best_loss")
        self.best_epoch = state.get("best_epoch", 0)
        self.counter = state.get("counter", 0)
        self.stopped_epoch = state.get("stopped_epoch", 0)


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    
    Similar to PyTorch's ReduceLROnPlateau but works with our custom training loop.
    
    Args:
        optimizer: Optimizer whose learning rate will be reduced.
        factor: Factor by which the learning rate will be reduced (new_lr = lr * factor).
        patience: Number of epochs with no improvement after which LR will be reduced.
        min_lr: Lower bound on the learning rate.
        verbose: Whether to print messages.
    """
    
    def __init__(
        self,
        optimizer,
        factor: float = 0.1,
        patience: int = 3,
        min_lr: float = 1e-7,
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_loss = None
        self.counter = 0
        self.num_reductions = 0
    
    def __call__(self, current_loss: float) -> bool:
        """
        Check if LR should be reduced.
        
        Args:
            current_loss: Current validation loss.
            
        Returns:
            True if LR was reduced, False otherwise.
        """
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        
        self.counter += 1
        
        if self.counter >= self.patience:
            self.counter = 0
            self.num_reductions += 1
            
            # Reduce LR
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                for group in self.optimizer.param_groups:
                    group['lr'] = new_lr
                
                if self.verbose:
                    print(f"   📉 Reducing LR: {old_lr:.2e} → {new_lr:.2e}")
                
                return True
        
        return False
    
    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
