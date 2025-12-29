"""
Supported Models for NSKI

Configuration for supported model architectures.
"""

from typing import Dict, Any

# Supported models with their configurations
SUPPORTED_MODELS = {
    # Llama 3 family
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "type": "llama",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 16,
        "memory_4bit_gb": 5,
    },
    "meta-llama/Meta-Llama-3-8B": {
        "type": "llama",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 16,
        "memory_4bit_gb": 5,
    },
    
    # Llama 2 family
    "meta-llama/Llama-2-7b-chat-hf": {
        "type": "llama",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 14,
        "memory_4bit_gb": 4,
    },
    "meta-llama/Llama-2-7b-hf": {
        "type": "llama",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 14,
        "memory_4bit_gb": 4,
    },
    "meta-llama/Llama-2-13b-chat-hf": {
        "type": "llama",
        "n_layers": 40,
        "hidden_size": 5120,
        "n_heads": 40,
        "recommended_layer": 20,
        "memory_fp16_gb": 26,
        "memory_4bit_gb": 8,
    },
    
    # Mistral family
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "type": "mistral",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 14,
        "memory_4bit_gb": 4,
    },
    "mistralai/Mistral-7B-v0.1": {
        "type": "mistral",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 14,
        "memory_4bit_gb": 4,
    },
    
    # Phi family
    "microsoft/phi-3-mini-4k-instruct": {
        "type": "phi",
        "n_layers": 32,
        "hidden_size": 3072,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 8,
        "memory_4bit_gb": 2.5,
    },
    "microsoft/phi-2": {
        "type": "phi",
        "n_layers": 32,
        "hidden_size": 2560,
        "n_heads": 32,
        "recommended_layer": 15,
        "memory_fp16_gb": 5.5,
        "memory_4bit_gb": 2,
    },
    
    # GPT-2 family
    "gpt2": {
        "type": "gpt2",
        "n_layers": 12,
        "hidden_size": 768,
        "n_heads": 12,
        "recommended_layer": 6,
        "memory_fp16_gb": 0.5,
        "memory_4bit_gb": 0.2,
    },
    "gpt2-medium": {
        "type": "gpt2",
        "n_layers": 24,
        "hidden_size": 1024,
        "n_heads": 16,
        "recommended_layer": 12,
        "memory_fp16_gb": 1.5,
        "memory_4bit_gb": 0.5,
    },
    "gpt2-large": {
        "type": "gpt2",
        "n_layers": 36,
        "hidden_size": 1280,
        "n_heads": 20,
        "recommended_layer": 18,
        "memory_fp16_gb": 3,
        "memory_4bit_gb": 1,
    },
    "gpt2-xl": {
        "type": "gpt2",
        "n_layers": 48,
        "hidden_size": 1600,
        "n_heads": 25,
        "recommended_layer": 24,
        "memory_fp16_gb": 6,
        "memory_4bit_gb": 2,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a model.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Configuration dictionary
    """
    # Direct match
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    
    # Partial match
    model_name_lower = model_name.lower()
    
    for name, config in SUPPORTED_MODELS.items():
        if name.lower() in model_name_lower or model_name_lower in name.lower():
            return config
    
    # Infer from name
    if "llama" in model_name_lower:
        return {
            "type": "llama",
            "n_layers": 32,
            "hidden_size": 4096,
            "n_heads": 32,
            "recommended_layer": 15,
        }
    elif "mistral" in model_name_lower:
        return {
            "type": "mistral",
            "n_layers": 32,
            "hidden_size": 4096,
            "n_heads": 32,
            "recommended_layer": 15,
        }
    elif "phi" in model_name_lower:
        return {
            "type": "phi",
            "n_layers": 32,
            "hidden_size": 3072,
            "n_heads": 32,
            "recommended_layer": 15,
        }
    elif "gpt2" in model_name_lower:
        return {
            "type": "gpt2",
            "n_layers": 12,
            "hidden_size": 768,
            "n_heads": 12,
            "recommended_layer": 6,
        }
    
    # Default (assume Llama-like)
    return {
        "type": "llama",
        "n_layers": 32,
        "hidden_size": 4096,
        "n_heads": 32,
        "recommended_layer": 15,
    }


def get_recommended_layer(model_name: str) -> int:
    """Get recommended intervention layer for a model."""
    config = get_model_config(model_name)
    return config.get('recommended_layer', 15)


def estimate_memory(model_name: str, use_4bit: bool = True) -> float:
    """Estimate GPU memory requirement in GB."""
    config = get_model_config(model_name)
    if use_4bit:
        return config.get('memory_4bit_gb', 5)
    return config.get('memory_fp16_gb', 14)
