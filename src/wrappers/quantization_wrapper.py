"""Quantization Wrapper - Unified quantization interface.

Wraps bitsandbytes, GPTQ, and AWQ for easy model quantization.
Automatically selects best quantization strategy based on hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union

import torch
import torch.nn as nn


class QuantizationType(Enum):
    """Supported quantization methods."""
    NONE = "none"
    BNB_INT8 = "bnb_int8"       # bitsandbytes 8-bit
    BNB_INT4 = "bnb_int4"       # bitsandbytes 4-bit (NF4/FP4)
    GPTQ = "gptq"               # GPTQ quantization
    AWQ = "awq"                 # AWQ quantization
    DYNAMIC = "dynamic"         # PyTorch dynamic quantization


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    quant_type: QuantizationType = QuantizationType.BNB_INT4
    
    # bitsandbytes options
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"  # nf4 or fp4
    bnb_4bit_use_double_quant: bool = True
    
    # GPTQ options
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_desc_act: bool = False
    
    # AWQ options
    awq_bits: int = 4
    awq_group_size: int = 128


class QuantizationWrapper:
    """Handles model quantization with multiple backends."""
    
    def __init__(self):
        self._bnb_available = self._check_bnb()
        self._gptq_available = self._check_gptq()
        self._awq_available = self._check_awq()

    def _check_bnb(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("bitsandbytes") is not None

    def _check_gptq(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("auto_gptq") is not None

    def _check_awq(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("awq") is not None
    
    @property
    def available_methods(self) -> list:
        """List available quantization methods."""
        methods = [QuantizationType.NONE, QuantizationType.DYNAMIC]
        if self._bnb_available:
            methods.extend([QuantizationType.BNB_INT8, QuantizationType.BNB_INT4])
        if self._gptq_available:
            methods.append(QuantizationType.GPTQ)
        if self._awq_available:
            methods.append(QuantizationType.AWQ)
        return methods
    
    def get_bnb_config(
        self,
        config: QuantizationConfig,
    ) -> Dict[str, Any]:
        """Get bitsandbytes config for model loading."""
        if not self._bnb_available:
            raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")
        
        from transformers import BitsAndBytesConfig
        
        if config.quant_type == QuantizationType.BNB_INT8:
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        
        elif config.quant_type == QuantizationType.BNB_INT4:
            compute_dtype = (
                torch.float16 if config.bnb_4bit_compute_dtype == "float16" 
                else torch.bfloat16
            )
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                )
            }
        
        return {}
    
    def quantize_model_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """Apply PyTorch dynamic quantization.
        
        Good for CPU inference, doesn't require calibration data.
        """
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=dtype,
        )
    
    def estimate_quantized_size(
        self,
        num_params: int,
        quant_type: QuantizationType,
    ) -> float:
        """Estimate model size in MB after quantization."""
        bytes_per_param = {
            QuantizationType.NONE: 4.0,  # FP32
            QuantizationType.BNB_INT8: 1.0,
            QuantizationType.BNB_INT4: 0.5,
            QuantizationType.GPTQ: 0.5,
            QuantizationType.AWQ: 0.5,
            QuantizationType.DYNAMIC: 1.0,
        }
        
        bpp = bytes_per_param.get(quant_type, 4.0)
        return (num_params * bpp) / (1024 ** 2)
    
    def recommend_quantization(
        self,
        num_params: int,
        available_vram_mb: float,
        prefer_quality: bool = False,
    ) -> QuantizationType:
        """Recommend best quantization method for given constraints.
        
        Args:
            num_params: Model parameter count.
            available_vram_mb: Available VRAM in MB.
            prefer_quality: If True, prefer higher quality over memory savings.
        
        Returns:
            Recommended quantization type.
        """
        # Estimate sizes
        size_fp16 = self.estimate_quantized_size(num_params, QuantizationType.NONE) / 2
        size_int8 = self.estimate_quantized_size(num_params, QuantizationType.BNB_INT8)
        size_int4 = self.estimate_quantized_size(num_params, QuantizationType.BNB_INT4)
        
        # Add overhead for activations and optimizer (rough 3x multiplier)
        required_fp16 = size_fp16 * 3
        required_int8 = size_int8 * 2.5
        required_int4 = size_int4 * 2
        
        if available_vram_mb >= required_fp16:
            return QuantizationType.NONE  # FP16 fits
        
        if available_vram_mb >= required_int8:
            if self._bnb_available:
                return QuantizationType.BNB_INT8
        
        if available_vram_mb >= required_int4:
            if prefer_quality and self._gptq_available:
                return QuantizationType.GPTQ
            if self._bnb_available:
                return QuantizationType.BNB_INT4
            if self._awq_available:
                return QuantizationType.AWQ
        
        # Last resort: try INT4 even if tight
        if self._bnb_available:
            return QuantizationType.BNB_INT4
        
        return QuantizationType.DYNAMIC


def quantize_model(
    model: Union[str, nn.Module],
    quant_type: str = "int4",
    **kwargs,
) -> nn.Module:
    """Convenience function to quantize a model.
    
    Args:
        model: Model name (for HF loading) or nn.Module.
        quant_type: Quantization type ("int4", "int8", "dynamic").
        **kwargs: Additional arguments for loader.
    
    Returns:
        Quantized model.
    """
    wrapper = QuantizationWrapper()
    
    type_map = {
        "int4": QuantizationType.BNB_INT4,
        "int8": QuantizationType.BNB_INT8,
        "dynamic": QuantizationType.DYNAMIC,
        "gptq": QuantizationType.GPTQ,
        "awq": QuantizationType.AWQ,
    }
    
    qt = type_map.get(quant_type, QuantizationType.BNB_INT4)

    if isinstance(model, str):
        # Load with quantization
        from .model_loader import load_model
        qt_types = (QuantizationType.BNB_INT4, QuantizationType.GPTQ, QuantizationType.AWQ)
        quantization = "int4" if qt in qt_types else "int8"
        model, _ = load_model(model, quantization=quantization, **kwargs)
        return model
    
    elif qt == QuantizationType.DYNAMIC:
        return wrapper.quantize_model_dynamic(model)
    
    else:
        raise ValueError(
            f"Quantization type {quant_type} requires loading from HuggingFace. "
            "Pass model name instead of nn.Module."
        )
