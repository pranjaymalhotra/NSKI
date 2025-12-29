"""
NSKI Surgical Intervention

High-level interface for applying Neural Surgical KV-cache Intervention.
Combines refusal direction extraction and KV-cache hooks into a unified API.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger
import time

from .kv_hooks import (
    KVCacheHook,
    KVCacheHookV2,
    HookConfig,
    KVCacheInterventionManager,
    get_attention_layers,
    get_value_proj_layers
)
from .refusal_direction import (
    RefusalDirectionExtractor,
    DirectionConfig,
    save_direction,
    load_direction
)


@dataclass
class SurgeryConfig:
    """Configuration for NSKI surgery."""
    target_layer: int = 15
    surgery_strength: float = 1.0
    strategy: str = "value_projection"
    hook_type: str = "v_proj"  # v_proj or attention
    
    # Direction extraction
    direction_n_harmful: int = 100
    direction_n_harmless: int = 100
    direction_aggregation: str = "mean_diff"
    direction_batch_size: int = 4
    
    # Pre-extracted direction (optional)
    direction_path: Optional[str] = None


@dataclass
class SurgeryResult:
    """Result of a surgery operation."""
    success: bool
    direction: Optional[torch.Tensor] = None
    metadata: Dict = field(default_factory=dict)
    intervention_count: int = 0
    elapsed_time: float = 0.0


class NSKISurgeon:
    """
    Neural Surgical KV-cache Intervention Surgeon.
    
    Provides a complete pipeline for:
    1. Extracting the refusal direction
    2. Applying KV-cache intervention during generation
    3. Measuring intervention effects
    
    Example usage:
        surgeon = NSKISurgeon(model, tokenizer, model_type="llama")
        
        # Extract direction
        surgeon.extract_direction(harmful_prompts, harmless_prompts)
        
        # Generate with intervention
        with surgeon.intervene():
            output = model.generate(input_ids, max_new_tokens=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_type: str = "llama",
        device: str = "cuda",
        config: Optional[SurgeryConfig] = None
    ):
        """
        Initialize the surgeon.
        
        Args:
            model: The transformer model
            tokenizer: The tokenizer
            model_type: Type of model (llama, mistral, gpt2, phi)
            device: Device to run on
            config: Surgery configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.config = config or SurgeryConfig()
        
        # Direction extractor
        self.extractor = RefusalDirectionExtractor(
            model, tokenizer, model_type, device
        )
        
        # Intervention manager
        self.intervention_manager = KVCacheInterventionManager(
            model, model_type
        )
        
        # State
        self.direction: Optional[torch.Tensor] = None
        self.direction_metadata: Dict = {}
        self.is_prepared = False
        
        logger.info(f"NSKISurgeon initialized for {model_type} model")
    
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> SurgeryResult:
        """
        Extract the refusal direction from activations.
        
        Args:
            harmful_prompts: List of harmful prompts
            harmless_prompts: List of harmless prompts
            layer_idx: Layer to extract from (default: config.target_layer)
            save_path: Optional path to save direction
            
        Returns:
            SurgeryResult with direction and metadata
        """
        start_time = time.time()
        layer_idx = layer_idx or self.config.target_layer
        
        logger.info(f"Extracting refusal direction from layer {layer_idx}")
        
        direction_config = DirectionConfig(
            layer_idx=layer_idx,
            n_harmful_samples=self.config.direction_n_harmful,
            n_harmless_samples=self.config.direction_n_harmless,
            aggregation=self.config.direction_aggregation,
            batch_size=self.config.direction_batch_size
        )
        
        try:
            result = self.extractor.extract(
                harmful_prompts,
                harmless_prompts,
                direction_config
            )
            
            self.direction = result['direction']
            self.direction_metadata = result['metadata']
            self.is_prepared = True
            
            if save_path:
                save_direction(result, save_path)
            
            elapsed = time.time() - start_time
            
            return SurgeryResult(
                success=True,
                direction=self.direction,
                metadata=self.direction_metadata,
                elapsed_time=elapsed
            )
        
        except Exception as e:
            logger.error(f"Direction extraction failed: {e}")
            return SurgeryResult(
                success=False,
                metadata={'error': str(e)},
                elapsed_time=time.time() - start_time
            )
    
    def load_direction(self, path: str) -> SurgeryResult:
        """Load a pre-extracted direction."""
        start_time = time.time()
        
        try:
            data = load_direction(path, self.device)
            self.direction = data['direction']
            self.direction_metadata = data.get('metadata', {})
            self.is_prepared = True
            
            return SurgeryResult(
                success=True,
                direction=self.direction,
                metadata=self.direction_metadata,
                elapsed_time=time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"Failed to load direction: {e}")
            return SurgeryResult(
                success=False,
                metadata={'error': str(e)},
                elapsed_time=time.time() - start_time
            )
    
    def prepare(
        self,
        layer_idx: Optional[int] = None,
        strength: Optional[float] = None
    ) -> None:
        """
        Prepare intervention by registering hooks.
        
        Must be called after direction is extracted/loaded.
        
        Args:
            layer_idx: Target layer (default: config.target_layer)
            strength: Surgery strength (default: config.surgery_strength)
        """
        if not self.is_prepared or self.direction is None:
            raise RuntimeError("Direction not extracted. Call extract_direction() first.")
        
        layer_idx = layer_idx or self.config.target_layer
        strength = strength or self.config.surgery_strength
        
        # Remove any existing hooks
        self.intervention_manager.remove_all()
        
        # Add new intervention
        self.intervention_manager.add_intervention(
            layer_idx=layer_idx,
            refusal_direction=self.direction,
            strength=strength,
            strategy=self.config.strategy,
            hook_type=self.config.hook_type
        )
        
        logger.info(f"Prepared intervention at layer {layer_idx} with strength {strength}")
    
    def enable(self) -> None:
        """Enable intervention."""
        self.intervention_manager.enable()
    
    def disable(self) -> None:
        """Disable intervention (without removing hooks)."""
        self.intervention_manager.disable()
    
    def cleanup(self) -> None:
        """Remove all hooks and clean up."""
        self.intervention_manager.remove_all()
    
    def intervene(self):
        """
        Context manager for intervention.
        
        Usage:
            with surgeon.intervene():
                output = model.generate(input_ids)
        """
        return InterventionContext(self)
    
    def generate_with_intervention(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with NSKI intervention applied.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        if not self.is_prepared:
            raise RuntimeError("Surgeon not prepared. Call extract_direction() and prepare() first.")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Prepare if not already
        if not self.intervention_manager.active:
            self.prepare()
        
        # Generate with intervention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated
    
    def generate_without_intervention(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text without intervention (baseline)."""
        # Disable intervention
        was_active = self.intervention_manager.active
        self.intervention_manager.disable()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Restore intervention state
        if was_active:
            self.intervention_manager.enable()
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated
    
    def get_stats(self) -> Dict:
        """Get intervention statistics."""
        return {
            'is_prepared': self.is_prepared,
            'active': self.intervention_manager.active,
            'intervention_count': self.intervention_manager.get_intervention_count(),
            'n_hooks': len(self.intervention_manager.hooks),
            'direction_metadata': self.direction_metadata
        }


class InterventionContext:
    """Context manager for temporary intervention."""
    
    def __init__(self, surgeon: NSKISurgeon):
        self.surgeon = surgeon
        self.was_active = False
    
    def __enter__(self):
        # Prepare and enable if direction is ready
        if self.surgeon.is_prepared:
            self.was_active = self.surgeon.intervention_manager.active
            if not self.was_active:
                self.surgeon.prepare()
            self.surgeon.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state
        if not self.was_active:
            self.surgeon.disable()
        return False


def create_surgeon(
    model_name: str,
    model_type: str = "llama",
    device: str = "cuda",
    load_in_4bit: bool = True,
    config: Optional[SurgeryConfig] = None
) -> Tuple[NSKISurgeon, nn.Module, Any]:
    """
    Convenience function to create a surgeon with model loading.
    
    Args:
        model_name: HuggingFace model name
        model_type: Type of model
        device: Device to run on
        load_in_4bit: Whether to use 4-bit quantization
        config: Surgery configuration
        
    Returns:
        Tuple of (surgeon, model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    logger.info(f"Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model with optional quantization
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    model.eval()
    
    # Create surgeon
    surgeon = NSKISurgeon(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        device=device,
        config=config
    )
    
    return surgeon, model, tokenizer
