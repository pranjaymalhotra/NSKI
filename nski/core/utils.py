"""
Utility functions for NSKI.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import Optional, Dict, Any
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(prefer_cuda: bool = True) -> str:
    """Get available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def memory_stats() -> Dict[str, float]:
    """Get GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def get_model_type(model_name: str) -> str:
    """Infer model type from model name."""
    model_name_lower = model_name.lower()
    
    if "llama" in model_name_lower:
        return "llama"
    elif "mistral" in model_name_lower:
        return "mistral"
    elif "phi" in model_name_lower:
        return "phi"
    elif "gpt2" in model_name_lower:
        return "gpt2"
    else:
        return "llama"  # Default


def get_num_layers(model: nn.Module) -> int:
    """Get number of transformer layers in model."""
    # Try different model architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    elif hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    else:
        # Count attention modules
        count = 0
        for name, _ in model.named_modules():
            if 'attn' in name.lower() and 'self' in name.lower():
                count += 1
        return count


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
