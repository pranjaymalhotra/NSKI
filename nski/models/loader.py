"""
Model Loader for NSKI

Handles loading various transformer models with appropriate configurations
for NSKI experiments. Supports quantization for memory-constrained GPUs.
"""

import torch
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

from .supported import SUPPORTED_MODELS, get_model_config


@dataclass
class LoadConfig:
    """Configuration for model loading."""
    model_name: str
    device: str = "cuda"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    use_flash_attention: bool = True


class ModelLoader:
    """
    Unified model loader for NSKI experiments.
    
    Handles:
    - Multiple model architectures (Llama, Mistral, GPT-2, Phi)
    - Quantization (4-bit, 8-bit)
    - Device placement
    - Tokenizer configuration
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        quantization: str = "4bit",
        config: Optional[LoadConfig] = None
    ):
        """
        Initialize ModelLoader.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load on ("cuda" or "cpu")
            quantization: Quantization type ("4bit", "8bit", or "none")
            config: Optional LoadConfig object (alternative to individual params)
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_type: str = "llama"
    
    def load(
        self,
        model_name: str = None,
        device: str = None,
        load_in_4bit: bool = None,
        load_in_8bit: bool = None,
        torch_dtype: str = "float16",
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and tokenizer.
        
        Args:
            model_name: HuggingFace model name or path (uses self.model_name if not provided)
            device: Device to load on (uses self.device if not provided)
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            torch_dtype: Torch dtype for model
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Use instance variables if parameters not provided
        if model_name is None:
            model_name = self.model_name
        if device is None:
            device = self.device
        
        # Determine quantization from instance variable if not specified
        if load_in_4bit is None and load_in_8bit is None:
            if self.quantization == "4bit":
                load_in_4bit = True
                load_in_8bit = False
            elif self.quantization == "8bit":
                load_in_4bit = False
                load_in_8bit = True
            else:
                load_in_4bit = False
                load_in_8bit = False
        
        logger.info(f"Loading model: {model_name}")
        
        # Get model configuration
        model_config = get_model_config(model_name)
        self.model_type = model_config.get('type', 'llama')
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        
        # Quantization configuration
        if load_in_4bit:
            logger.info("Using 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for large models
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        elif load_in_8bit:
            logger.info("Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for large models
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            logger.info("Loading full precision model")
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
        
        # Check if flash attention is available before using it
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            # Flash attention not installed, use default (sdpa)
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("Flash Attention not available, using SDPA")
        
        # Merge additional kwargs
        model_kwargs.update(kwargs)
        
        # Load model
        logger.info("Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Set to eval mode
        self.model.eval()
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Model loaded. GPU memory: {memory_gb:.2f} GB")
        
        return self.model, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if self.model is None:
            return 0
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        elif hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        return 32  # Default
    
    def get_hidden_size(self) -> int:
        """Get hidden dimension size."""
        if self.model is None:
            return 0
        return self.model.config.hidden_size
    
    def cleanup(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cleaned up")


def load_model(
    model_name: str,
    device: str = "cuda",
    load_in_4bit: bool = True,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """
    Convenience function to load a model.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load on
        load_in_4bit: Use 4-bit quantization
        **kwargs: Additional loading arguments
        
    Returns:
        Tuple of (model, tokenizer, model_type)
    """
    loader = ModelLoader()
    return loader.load(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit,
        **kwargs
    )
