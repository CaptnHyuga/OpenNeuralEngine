"""Architecture Detector - Automatic model architecture analysis.

Analyzes PyTorch models to detect:
- Architecture type (Transformer, CNN, RNN, GNN, Hybrid)
- Task type (classification, generation, etc.)
- Optimal training strategies
- Memory characteristics

Enables automatic configuration of training settings based on
model architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


class ArchitectureType(Enum):
    """Model architecture categories."""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GNN = "gnn"
    MLP = "mlp"
    DIFFUSION = "diffusion"
    VAE = "vae"
    GAN = "gan"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class TaskType(Enum):
    """Task type categories."""
    CAUSAL_LM = "causal_lm"
    MASKED_LM = "masked_lm"
    SEQ2SEQ = "seq2seq"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    REINFORCEMENT = "reinforcement"
    UNKNOWN = "unknown"


@dataclass
class LayerStats:
    """Statistics about a layer type."""
    count: int = 0
    total_params: int = 0
    layer_names: List[str] = field(default_factory=list)


@dataclass
class ArchitectureInfo:
    """Complete architecture analysis result."""
    
    # Core classification
    architecture_type: ArchitectureType = ArchitectureType.UNKNOWN
    task_type: TaskType = TaskType.UNKNOWN
    
    # Parameter counts
    total_params: int = 0
    trainable_params: int = 0
    
    # Layer statistics
    layer_stats: Dict[str, LayerStats] = field(default_factory=dict)
    
    # Architecture details
    num_attention_layers: int = 0
    num_conv_layers: int = 0
    num_rnn_layers: int = 0
    num_linear_layers: int = 0
    num_norm_layers: int = 0
    
    # Memory characteristics
    has_gradient_checkpointing_support: bool = False
    estimated_activation_memory_mb: float = 0.0
    recommended_batch_size: int = 1
    
    # Training recommendations
    supports_mixed_precision: bool = True
    supports_flash_attention: bool = False
    supports_quantization: bool = True
    recommended_precision: str = "fp16"
    recommended_optimizer: str = "adamw"
    
    # Detected patterns
    detected_patterns: List[str] = field(default_factory=list)
    
    @property
    def total_params_human(self) -> str:
        """Human-readable parameter count."""
        if self.total_params >= 1e9:
            return f"{self.total_params / 1e9:.1f}B"
        elif self.total_params >= 1e6:
            return f"{self.total_params / 1e6:.1f}M"
        else:
            return f"{self.total_params / 1e3:.1f}K"
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Architecture Analysis:",
            f"  Type: {self.architecture_type.value}",
            f"  Task: {self.task_type.value}",
            f"  Parameters: {self.total_params_human} ({self.trainable_params:,} trainable)",
            f"",
            f"Layer Composition:",
            f"  Attention layers: {self.num_attention_layers}",
            f"  Conv layers: {self.num_conv_layers}",
            f"  RNN layers: {self.num_rnn_layers}",
            f"  Linear layers: {self.num_linear_layers}",
            f"",
            f"Recommendations:",
            f"  Precision: {self.recommended_precision}",
            f"  Optimizer: {self.recommended_optimizer}",
            f"  Mixed Precision: {'✓' if self.supports_mixed_precision else '✗'}",
            f"  Flash Attention: {'✓' if self.supports_flash_attention else '✗'}",
        ]
        
        if self.detected_patterns:
            lines.append(f"")
            lines.append(f"Detected Patterns:")
            for pattern in self.detected_patterns:
                lines.append(f"  • {pattern}")
        
        return "\n".join(lines)


class ArchitectureDetector:
    """Detects and analyzes model architecture."""
    
    # Layer type patterns
    ATTENTION_PATTERNS = {
        "attention", "multihead", "mha", "self_attn", "cross_attn",
        "qkv", "q_proj", "k_proj", "v_proj", "o_proj"
    }
    
    CONV_PATTERNS = {
        "conv1d", "conv2d", "conv3d", "convolution", "depthwise",
        "pointwise", "separable"
    }
    
    RNN_PATTERNS = {
        "lstm", "gru", "rnn", "recurrent", "bidirectional"
    }
    
    NORM_PATTERNS = {
        "layernorm", "batchnorm", "groupnorm", "instancenorm",
        "rmsnorm", "layer_norm", "batch_norm"
    }
    
    TRANSFORMER_PATTERNS = {
        "transformer", "encoder", "decoder", "block", "layer"
    }
    
    def __init__(self):
        self._layer_cache: Dict[str, str] = {}
    
    def analyze(self, model: nn.Module) -> ArchitectureInfo:
        """Analyze a model and return architecture info.
        
        Args:
            model: PyTorch model to analyze.
        
        Returns:
            ArchitectureInfo with analysis results.
        """
        info = ArchitectureInfo()
        
        # Count parameters
        info.total_params = sum(p.numel() for p in model.parameters())
        info.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Analyze layer composition
        self._analyze_layers(model, info)
        
        # Detect architecture type
        info.architecture_type = self._detect_architecture_type(info)
        
        # Detect task type
        info.task_type = self._detect_task_type(model, info)
        
        # Detect patterns
        info.detected_patterns = self._detect_patterns(model, info)
        
        # Generate recommendations
        self._generate_recommendations(model, info)
        
        return info
    
    def _analyze_layers(self, model: nn.Module, info: ArchitectureInfo):
        """Analyze layer composition of model."""
        
        for name, module in model.named_modules():
            module_type = type(module).__name__.lower()
            module_name = name.lower()
            
            # Count attention layers
            if self._matches_patterns(module_type, module_name, self.ATTENTION_PATTERNS):
                info.num_attention_layers += 1
                self._update_layer_stats(info, "attention", module, name)
            
            # Count conv layers
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                info.num_conv_layers += 1
                self._update_layer_stats(info, "conv", module, name)
            
            # Count RNN layers
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                info.num_rnn_layers += 1
                self._update_layer_stats(info, "rnn", module, name)
            
            # Count linear layers
            elif isinstance(module, nn.Linear):
                info.num_linear_layers += 1
                self._update_layer_stats(info, "linear", module, name)
            
            # Count norm layers
            elif self._matches_patterns(module_type, module_name, self.NORM_PATTERNS):
                info.num_norm_layers += 1
                self._update_layer_stats(info, "norm", module, name)
    
    def _matches_patterns(self, module_type: str, module_name: str, patterns: Set[str]) -> bool:
        """Check if module matches any pattern."""
        combined = f"{module_type} {module_name}"
        return any(p in combined for p in patterns)
    
    def _update_layer_stats(self, info: ArchitectureInfo, layer_type: str, module: nn.Module, name: str):
        """Update layer statistics."""
        if layer_type not in info.layer_stats:
            info.layer_stats[layer_type] = LayerStats()
        
        stats = info.layer_stats[layer_type]
        stats.count += 1
        stats.total_params += sum(p.numel() for p in module.parameters(recurse=False))
        stats.layer_names.append(name)
    
    def _detect_architecture_type(self, info: ArchitectureInfo) -> ArchitectureType:
        """Determine primary architecture type."""
        
        # Count dominant layer types
        attention_score = info.num_attention_layers * 3
        conv_score = info.num_conv_layers * 2
        rnn_score = info.num_rnn_layers * 3
        
        # Transformer: attention-dominant
        if attention_score > conv_score and attention_score > rnn_score:
            if info.num_conv_layers > 0:
                return ArchitectureType.HYBRID
            return ArchitectureType.TRANSFORMER
        
        # CNN: conv-dominant
        if conv_score > attention_score and conv_score > rnn_score:
            if info.num_attention_layers > 0:
                return ArchitectureType.HYBRID
            return ArchitectureType.CNN
        
        # RNN: recurrent-dominant
        if rnn_score > attention_score and rnn_score > conv_score:
            return ArchitectureType.RNN
        
        # MLP: only linear layers
        if info.num_linear_layers > 0 and attention_score == 0 and conv_score == 0 and rnn_score == 0:
            return ArchitectureType.MLP
        
        # Hybrid or unknown
        if attention_score > 0 or conv_score > 0 or rnn_score > 0:
            return ArchitectureType.HYBRID
        
        return ArchitectureType.UNKNOWN
    
    def _detect_task_type(self, model: nn.Module, info: ArchitectureInfo) -> TaskType:
        """Infer task type from model structure."""
        
        # Check model class name for hints
        class_name = type(model).__name__.lower()
        
        # HuggingFace naming conventions
        if "causallm" in class_name or "forlm" in class_name:
            return TaskType.CAUSAL_LM
        if "maskedlm" in class_name:
            return TaskType.MASKED_LM
        if "seq2seq" in class_name or "conditional" in class_name:
            return TaskType.SEQ2SEQ
        if "sequenceclassification" in class_name or "classification" in class_name:
            return TaskType.CLASSIFICATION
        if "tokenclassification" in class_name:
            return TaskType.CLASSIFICATION
        if "regression" in class_name:
            return TaskType.REGRESSION
        if "detection" in class_name:
            return TaskType.OBJECT_DETECTION
        if "segmentation" in class_name:
            return TaskType.SEGMENTATION
        if "embedding" in class_name or "encoder" in class_name:
            return TaskType.EMBEDDING
        
        # Infer from architecture
        if info.architecture_type == ArchitectureType.TRANSFORMER:
            # Check for decoder-only (causal) vs encoder-only vs encoder-decoder
            if self._has_causal_mask(model):
                return TaskType.CAUSAL_LM
            return TaskType.MASKED_LM
        
        if info.architecture_type == ArchitectureType.CNN:
            # Check output layer
            last_layer = self._get_last_layer(model)
            if last_layer is not None:
                if isinstance(last_layer, nn.Conv2d):
                    return TaskType.SEGMENTATION
            return TaskType.CLASSIFICATION
        
        return TaskType.UNKNOWN
    
    def _has_causal_mask(self, model: nn.Module) -> bool:
        """Check if model uses causal masking."""
        # Look for causal attention patterns
        for name, module in model.named_modules():
            name_lower = name.lower()
            if "causal" in name_lower or "decoder" in name_lower:
                return True
            
            # Check for triangular masks in buffers
            for buffer_name, buffer in module.named_buffers():
                if "mask" in buffer_name.lower():
                    if buffer.dim() == 2 and buffer.size(0) == buffer.size(1):
                        # Check if triangular
                        if torch.allclose(buffer, torch.tril(buffer)):
                            return True
        
        return False
    
    def _get_last_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Get the last layer of the model."""
        last_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                last_layer = module
        return last_layer
    
    def _detect_patterns(self, model: nn.Module, info: ArchitectureInfo) -> List[str]:
        """Detect architectural patterns."""
        patterns = []
        
        # Check for residual connections
        if self._has_residual_connections(model):
            patterns.append("Residual connections detected")
        
        # Check for positional encoding
        if self._has_positional_encoding(model):
            patterns.append("Positional encoding detected")
        
        # Check for gating mechanisms
        if self._has_gating(model):
            patterns.append("Gating mechanism detected")
        
        # Check for dropout
        if self._has_dropout(model):
            patterns.append("Dropout regularization")
        
        # Check for layer norm
        if info.num_norm_layers > 0:
            patterns.append(f"Normalization layers: {info.num_norm_layers}")
        
        # Check for embedding layers
        num_embeddings = sum(1 for m in model.modules() if isinstance(m, nn.Embedding))
        if num_embeddings > 0:
            patterns.append(f"Embedding layers: {num_embeddings}")
        
        return patterns
    
    def _has_residual_connections(self, model: nn.Module) -> bool:
        """Check for residual/skip connections."""
        for name, _ in model.named_modules():
            if "residual" in name.lower() or "skip" in name.lower() or "shortcut" in name.lower():
                return True
        return False
    
    def _has_positional_encoding(self, model: nn.Module) -> bool:
        """Check for positional encoding."""
        for name, module in model.named_modules():
            name_lower = name.lower()
            if "position" in name_lower or "pos_" in name_lower or "rotary" in name_lower:
                return True
            if isinstance(module, nn.Embedding):
                if "position" in name_lower:
                    return True
        return False
    
    def _has_gating(self, model: nn.Module) -> bool:
        """Check for gating mechanisms."""
        for name, _ in model.named_modules():
            name_lower = name.lower()
            if "gate" in name_lower or "glu" in name_lower or "swiglu" in name_lower:
                return True
        return False
    
    def _has_dropout(self, model: nn.Module) -> bool:
        """Check for dropout layers."""
        return any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) for m in model.modules())
    
    def _generate_recommendations(self, model: nn.Module, info: ArchitectureInfo):
        """Generate training recommendations based on architecture."""
        
        # Mixed precision support
        info.supports_mixed_precision = info.architecture_type not in [
            ArchitectureType.UNKNOWN
        ]
        
        # Flash attention support
        info.supports_flash_attention = (
            info.architecture_type == ArchitectureType.TRANSFORMER and
            info.num_attention_layers > 0
        )
        
        # Precision recommendation
        if info.total_params > 1_000_000_000:  # > 1B params
            info.recommended_precision = "bf16"  # Better for large models
        elif info.architecture_type == ArchitectureType.CNN:
            info.recommended_precision = "fp16"  # Good for CNNs
        else:
            info.recommended_precision = "fp16"
        
        # Optimizer recommendation
        if info.architecture_type == ArchitectureType.TRANSFORMER:
            info.recommended_optimizer = "adamw"
        elif info.architecture_type == ArchitectureType.CNN:
            info.recommended_optimizer = "sgd"  # Often better for CNNs
        else:
            info.recommended_optimizer = "adamw"
        
        # Gradient checkpointing support
        info.has_gradient_checkpointing_support = (
            info.architecture_type in [
                ArchitectureType.TRANSFORMER,
                ArchitectureType.CNN,
                ArchitectureType.HYBRID,
            ]
        )
        
        # Estimate activation memory (rough)
        # Activations roughly scale with hidden dimensions and sequence length
        info.estimated_activation_memory_mb = self._estimate_activation_memory(model, info)
        
        # Recommended batch size based on model size
        if info.total_params > 10_000_000_000:  # > 10B
            info.recommended_batch_size = 1
        elif info.total_params > 1_000_000_000:  # > 1B
            info.recommended_batch_size = 4
        elif info.total_params > 100_000_000:  # > 100M
            info.recommended_batch_size = 16
        else:
            info.recommended_batch_size = 32
    
    def _estimate_activation_memory(self, model: nn.Module, info: ArchitectureInfo) -> float:
        """Estimate activation memory in MB (rough approximation)."""
        # Very rough estimate: activations ~= 4 * num_params for transformers
        # Less for CNNs due to spatial reduction
        
        if info.architecture_type == ArchitectureType.TRANSFORMER:
            multiplier = 4.0
        elif info.architecture_type == ArchitectureType.CNN:
            multiplier = 2.0
        else:
            multiplier = 3.0
        
        # Estimate in bytes (assuming fp32 activations)
        activation_bytes = info.total_params * multiplier * 4
        
        return activation_bytes / (1024 ** 2)


# Global detector instance
_detector: Optional[ArchitectureDetector] = None


def get_detector() -> ArchitectureDetector:
    """Get global architecture detector instance."""
    global _detector
    if _detector is None:
        _detector = ArchitectureDetector()
    return _detector


def analyze_model(model: nn.Module) -> ArchitectureInfo:
    """Analyze a model's architecture.
    
    Convenience function using global detector.
    
    Args:
        model: PyTorch model to analyze.
    
    Returns:
        ArchitectureInfo with analysis results.
    """
    return get_detector().analyze(model)


def detect_architecture(model: nn.Module) -> ArchitectureType:
    """Detect the architecture type of a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        ArchitectureType enum value.
    """
    return analyze_model(model).architecture_type


def detect_task(model: nn.Module) -> TaskType:
    """Detect the task type of a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        TaskType enum value.
    """
    return analyze_model(model).task_type
