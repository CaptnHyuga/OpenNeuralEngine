"""Universal Model Loader - Load any model from any source.

Supports loading models from:
- HuggingFace Hub (by name or path)
- Local PyTorch files (.pt, .pth, .safetensors)
- Python files with model definitions
- timm (vision models)
- torchvision models
- Custom registry

Automatically detects model type and applies appropriate loading strategy.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


class ModelSource(Enum):
    """Source types for model loading."""
    HUGGINGFACE = "huggingface"
    LOCAL_WEIGHTS = "local_weights"
    LOCAL_PYTHON = "local_python"
    TIMM = "timm"
    TORCHVISION = "torchvision"
    REGISTRY = "registry"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    
    source: ModelSource
    name: str
    num_params: int
    architecture_type: str  # "transformer", "cnn", "rnn", "hybrid", "unknown"
    task_type: str  # "causal_lm", "seq2seq", "classification", "vision", "audio", etc.
    config: Dict[str, Any]
    
    @property
    def num_params_human(self) -> str:
        """Human-readable parameter count."""
        if self.num_params >= 1e9:
            return f"{self.num_params / 1e9:.1f}B"
        elif self.num_params >= 1e6:
            return f"{self.num_params / 1e6:.1f}M"
        else:
            return f"{self.num_params / 1e3:.1f}K"


# Model registry for custom models
_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a custom model class.
    
    Usage:
        @register_model("my-custom-model")
        class MyModel(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_registered_models() -> List[str]:
    """Get list of registered model names."""
    return list(_MODEL_REGISTRY.keys())


class UniversalModelLoader:
    """Loads models from any source with automatic detection."""
    
    def __init__(self):
        # Check available backends
        self._hf_available = self._check_hf()
        self._timm_available = self._check_timm()
        self._torchvision_available = self._check_torchvision()
    
    def _check_hf(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("transformers") is not None

    def _check_timm(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("timm") is not None

    def _check_torchvision(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("torchvision") is not None
    
    def detect_source(self, model_identifier: str) -> ModelSource:
        """Detect the source type of a model identifier.
        
        Args:
            model_identifier: Model name, path, or identifier.
        
        Returns:
            Detected ModelSource type.
        """
        # Check for explicit prefixes
        if model_identifier.startswith("timm:"):
            return ModelSource.TIMM
        if model_identifier.startswith("torchvision:"):
            return ModelSource.TORCHVISION
        
        # Check local paths
        path = Path(model_identifier)
        if path.exists():
            if path.suffix in (".py",):
                return ModelSource.LOCAL_PYTHON
            if path.suffix in (".pt", ".pth", ".safetensors", ".bin"):
                return ModelSource.LOCAL_WEIGHTS
            if path.is_dir():
                # Could be HF model directory
                if (path / "config.json").exists():
                    return ModelSource.HUGGINGFACE
                return ModelSource.LOCAL_WEIGHTS
        
        # Check registry
        if model_identifier in _MODEL_REGISTRY:
            return ModelSource.REGISTRY
        
        # Check if it looks like a HuggingFace model
        if "/" in model_identifier or self._looks_like_hf_model(model_identifier):
            return ModelSource.HUGGINGFACE
        
        # Check timm models
        if self._timm_available and self._is_timm_model(model_identifier):
            return ModelSource.TIMM
        
        return ModelSource.UNKNOWN
    
    def _looks_like_hf_model(self, name: str) -> bool:
        """Check if name looks like a HuggingFace model."""
        hf_patterns = [
            "gpt", "bert", "llama", "mistral", "falcon", "opt", "bloom",
            "t5", "bart", "roberta", "distil", "whisper", "clip", "vit",
            "stable", "diffusion", "smol"
        ]
        name_lower = name.lower()
        return any(p in name_lower for p in hf_patterns)
    
    def _is_timm_model(self, name: str) -> bool:
        """Check if name is a valid timm model."""
        if not self._timm_available:
            return False
        try:
            import timm
            return name in timm.list_models()
        except Exception:  # noqa: BLE001 - broad check for library availability
            return False
    
    def load(
        self,
        model_identifier: str,
        task: Optional[str] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        quantization: Optional[str] = None,  # "int4", "int8"
        trust_remote_code: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load a model from any source.
        
        Args:
            model_identifier: Model name, path, or identifier.
            task: Task type hint (helps with HF model selection).
            device: Device to load model to ("auto", "cuda", "cpu").
            torch_dtype: Data type for model weights.
            quantization: Quantization level ("int4", "int8", None).
            trust_remote_code: Trust remote code (HF only).
            **kwargs: Additional arguments for specific loaders.
        
        Returns:
            Tuple of (model, model_info).
        """
        source = self.detect_source(model_identifier)
        
        if source == ModelSource.HUGGINGFACE:
            return self._load_huggingface(
                model_identifier, task, device, torch_dtype, 
                quantization, trust_remote_code, **kwargs
            )
        
        elif source == ModelSource.LOCAL_PYTHON:
            return self._load_python_file(model_identifier, device, **kwargs)
        
        elif source == ModelSource.LOCAL_WEIGHTS:
            return self._load_local_weights(model_identifier, device, **kwargs)
        
        elif source == ModelSource.TIMM:
            name = model_identifier.replace("timm:", "")
            return self._load_timm(name, device, **kwargs)
        
        elif source == ModelSource.TORCHVISION:
            name = model_identifier.replace("torchvision:", "")
            return self._load_torchvision(name, device, **kwargs)
        
        elif source == ModelSource.REGISTRY:
            return self._load_from_registry(model_identifier, device, **kwargs)
        
        else:
            raise ValueError(
                f"Cannot determine how to load model: {model_identifier}\n"
                f"Supported sources: HuggingFace, local files, timm:*, torchvision:*, registry"
            )
    
    def _load_huggingface(
        self,
        model_name: str,
        task: Optional[str],
        device: str,
        torch_dtype: Optional[torch.dtype],
        quantization: Optional[str],
        trust_remote_code: bool,
        **kwargs,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load model from HuggingFace."""
        if not self._hf_available:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoConfig,
            BitsAndBytesConfig,
        )
        
        # Build load kwargs
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        
        if device != "cpu":
            load_kwargs["device_map"] = "auto"
        
        if torch_dtype:
            load_kwargs["torch_dtype"] = torch_dtype
        
        # Handle quantization
        if quantization == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        load_kwargs.update(kwargs)
        
        # Select model class based on task
        task_to_class = {
            "causal_lm": AutoModelForCausalLM,
            "seq2seq": AutoModelForSeq2SeqLM,
            "classification": AutoModelForSequenceClassification,
        }
        
        model_class = task_to_class.get(task, AutoModelForCausalLM)
        
        # Load model
        model = model_class.from_pretrained(model_name, **load_kwargs)
        
        # Get config for info
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        info = ModelInfo(
            source=ModelSource.HUGGINGFACE,
            name=model_name,
            num_params=sum(p.numel() for p in model.parameters()),
            architecture_type=self._detect_architecture_type(model),
            task_type=task or self._infer_task_type(config),
            config=config.to_dict() if hasattr(config, "to_dict") else {},
        )
        
        return model, info
    
    def _load_python_file(
        self,
        file_path: str,
        device: str,
        model_class_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load model from a Python file."""
        path = Path(file_path)
        
        # Load module
        spec = importlib.util.spec_from_file_location("custom_model", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_model"] = module
        spec.loader.exec_module(module)
        
        # Find model class
        if model_class_name:
            model_class = getattr(module, model_class_name)
        else:
            # Look for nn.Module subclass
            model_class = None
            for _name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj != nn.Module:
                    model_class = obj
                    break
            
            if model_class is None:
                raise ValueError(f"No nn.Module subclass found in {file_path}")
        
        # Instantiate
        model = model_class(**kwargs)
        
        if device != "cpu":
            model = model.to(device if device != "auto" else "cuda")
        
        info = ModelInfo(
            source=ModelSource.LOCAL_PYTHON,
            name=path.stem,
            num_params=sum(p.numel() for p in model.parameters()),
            architecture_type=self._detect_architecture_type(model),
            task_type="custom",
            config=kwargs,
        )
        
        return model, info
    
    def _load_local_weights(
        self,
        path: str,
        device: str,
        model: Optional[nn.Module] = None,
        **kwargs,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load model weights from local file."""
        path = Path(path)
        
        if model is None:
            raise ValueError(
                "Loading raw weights requires a model instance. "
                "Pass model=your_model_instance or use a Python file instead."
            )
        
        # Load weights
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        model.load_state_dict(state_dict, strict=False)
        
        if device != "cpu":
            model = model.to(device if device != "auto" else "cuda")
        
        info = ModelInfo(
            source=ModelSource.LOCAL_WEIGHTS,
            name=path.stem,
            num_params=sum(p.numel() for p in model.parameters()),
            architecture_type=self._detect_architecture_type(model),
            task_type="custom",
            config={},
        )
        
        return model, info
    
    def _load_timm(
        self,
        model_name: str,
        device: str,
        pretrained: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load model from timm."""
        if not self._timm_available:
            raise ImportError("timm not installed. Run: pip install timm")
        
        import timm
        
        model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        
        if device != "cpu":
            model = model.to(device if device != "auto" else "cuda")
        
        info = ModelInfo(
            source=ModelSource.TIMM,
            name=model_name,
            num_params=sum(p.numel() for p in model.parameters()),
            architecture_type="cnn",  # Most timm models are CNNs or ViTs
            task_type="vision",
            config={"pretrained": pretrained, **kwargs},
        )
        
        return model, info
    
    def _load_torchvision(
        self,
        model_name: str,
        device: str,
        pretrained: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load model from torchvision."""
        if not self._torchvision_available:
            raise ImportError("torchvision not installed. Run: pip install torchvision")
        
        import torchvision.models as models
        
        # Get model function
        if not hasattr(models, model_name):
            raise ValueError(f"Unknown torchvision model: {model_name}")
        
        model_fn = getattr(models, model_name)
        
        # Handle pretrained weights
        if pretrained:
            weights_name = model_name.upper() + "_Weights"
            if hasattr(models, weights_name):
                weights = getattr(models, weights_name).DEFAULT
                model = model_fn(weights=weights, **kwargs)
            else:
                model = model_fn(pretrained=True, **kwargs)
        else:
            model = model_fn(**kwargs)
        
        if device != "cpu":
            model = model.to(device if device != "auto" else "cuda")
        
        info = ModelInfo(
            source=ModelSource.TORCHVISION,
            name=model_name,
            num_params=sum(p.numel() for p in model.parameters()),
            architecture_type="cnn",
            task_type="vision",
            config={"pretrained": pretrained, **kwargs},
        )
        
        return model, info
    
    def _load_from_registry(
        self,
        name: str,
        device: str,
        **kwargs,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load model from custom registry."""
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Model not found in registry: {name}")
        
        model_class = _MODEL_REGISTRY[name]
        model = model_class(**kwargs)
        
        if device != "cpu":
            model = model.to(device if device != "auto" else "cuda")
        
        info = ModelInfo(
            source=ModelSource.REGISTRY,
            name=name,
            num_params=sum(p.numel() for p in model.parameters()),
            architecture_type=self._detect_architecture_type(model),
            task_type="custom",
            config=kwargs,
        )
        
        return model, info
    
    def _detect_architecture_type(self, model: nn.Module) -> str:
        """Detect architecture type from model structure."""
        module_names = [name for name, _ in model.named_modules()]
        module_str = " ".join(module_names).lower()
        
        # Check for transformers
        if any(kw in module_str for kw in ["attention", "transformer", "mha", "self_attn"]):
            if "conv" in module_str:
                return "hybrid"
            return "transformer"
        
        # Check for CNNs
        if any(kw in module_str for kw in ["conv2d", "conv1d", "resnet", "efficientnet"]):
            return "cnn"
        
        # Check for RNNs
        if any(kw in module_str for kw in ["lstm", "gru", "rnn"]):
            return "rnn"
        
        return "unknown"
    
    def _infer_task_type(self, config) -> str:
        """Infer task type from model config."""
        config_str = str(config).lower()
        
        if "causal" in config_str or "gpt" in config_str:
            return "causal_lm"
        if "seq2seq" in config_str or "t5" in config_str or "bart" in config_str:
            return "seq2seq"
        if "classification" in config_str or "num_labels" in config_str:
            return "classification"
        if "vision" in config_str or "vit" in config_str:
            return "vision"
        if "whisper" in config_str or "wav2vec" in config_str:
            return "audio"
        
        return "causal_lm"


# Convenience function
def load_model(
    model_identifier: str,
    **kwargs,
) -> Tuple[nn.Module, ModelInfo]:
    """Load a model from any source.
    
    Convenience wrapper around UniversalModelLoader.load().
    
    Examples:
        # HuggingFace
        model, info = load_model("gpt2")
        model, info = load_model("meta-llama/Llama-2-7b", quantization="int4")
        
        # timm
        model, info = load_model("timm:resnet50")
        
        # torchvision
        model, info = load_model("torchvision:resnet50")
        
        # Local Python file
        model, info = load_model("./my_model.py")
        
        # Registry
        @register_model("my-model")
        class MyModel(nn.Module): ...
        model, info = load_model("my-model")
    """
    loader = UniversalModelLoader()
    return loader.load(model_identifier, **kwargs)
