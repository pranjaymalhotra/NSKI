"""
Belitsky et al. (2025) - Attention Head Modulation

Implementation of safety intervention via selective attention head modulation.
This method identifies and modulates specific attention heads associated with
refusal behavior.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class BelitskyConfig:
    """Configuration for Belitsky modulation."""
    target_layer: int = 15
    head_selection: str = "top_k"  # "top_k", "threshold", "all"
    k_heads: int = 10  # Number of heads to modulate
    modulation_strength: float = 0.5  # How much to reduce head contribution
    threshold: float = 0.7  # For threshold-based selection


class AttentionHeadHook:
    """
    Hook that modulates specific attention heads.
    
    Complexity: O(T) - must process attention at every position.
    """
    
    def __init__(
        self,
        head_mask: torch.Tensor,  # [n_heads] binary or float mask
        modulation_strength: float = 0.5
    ):
        self.head_mask = head_mask
        self.strength = modulation_strength
        self.handle = None
        self.intervention_count = 0
    
    def _hook_fn(
        self,
        module: nn.Module,
        args: Tuple,
        kwargs: Dict,
        output: Tuple
    ) -> Tuple:
        """
        Modify attention weights for selected heads.
        """
        # Output is typically (attn_output, attn_weights, ...)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_output = output[0]
            attn_weights = output[1]
            
            if attn_weights is not None:
                # attn_weights: [batch, n_heads, seq, seq]
                mask = self.head_mask.to(attn_weights.device, attn_weights.dtype)
                
                # Modulate: reduce contribution of selected heads
                # mask values: 1 = keep, strength = reduce
                modulation = torch.ones_like(mask)
                modulation[mask == 0] = self.strength
                
                # Apply to attention weights
                modulation = modulation.view(1, -1, 1, 1)
                attn_weights = attn_weights * modulation
                
                self.intervention_count += attn_weights.shape[2]  # seq length
                
                return (attn_output, attn_weights) + output[2:]
        
        return output
    
    def register(self, module: nn.Module) -> None:
        self.handle = module.register_forward_hook(self._hook_fn, with_kwargs=True)
    
    def remove(self) -> None:
        if self.handle:
            self.handle.remove()
            self.handle = None


class BelitskyModulation:
    """
    Implementation of Belitsky et al. (2025) attention head modulation.
    
    Key differences from NSKI:
    1. Modulates attention heads, not KV-cache
    2. O(T) complexity for attention weight modification
    3. Requires head importance scoring
    
    This serves as a baseline demonstrating alternative intervention points.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_type: str = "llama",
        device: str = "cuda",
        config: Optional[BelitskyConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.config = config or BelitskyConfig()
        
        self.head_importance: Optional[torch.Tensor] = None
        self.head_mask: Optional[torch.Tensor] = None
        self.hooks: List[AttentionHeadHook] = []
        self.is_prepared = False
    
    def compute_head_importance(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer_idx: int = 15,
        batch_size: int = 4
    ) -> torch.Tensor:
        """
        Compute importance scores for attention heads.
        
        Heads that show high activation difference between harmful
        and harmless prompts are identified as refusal-related.
        """
        logger.info(f"Computing head importance scores at layer {layer_idx}")
        
        attn_module = self._get_attention_module(layer_idx)
        n_heads = self._get_n_heads()
        
        harmful_patterns = self._collect_attention_patterns(
            harmful_prompts, attn_module, batch_size
        )
        harmless_patterns = self._collect_attention_patterns(
            harmless_prompts, attn_module, batch_size
        )
        
        # Compute head-wise importance as activation difference
        # harmful_patterns: [n_samples, n_heads, seq, seq]
        harmful_mean = harmful_patterns.mean(dim=(0, 2, 3))  # [n_heads]
        harmless_mean = harmless_patterns.mean(dim=(0, 2, 3))  # [n_heads]
        
        importance = torch.abs(harmful_mean - harmless_mean)
        importance = importance / importance.max()  # Normalize to [0, 1]
        
        self.head_importance = importance
        
        # Create mask based on selection method
        if self.config.head_selection == "top_k":
            _, top_indices = torch.topk(importance, self.config.k_heads)
            self.head_mask = torch.zeros(n_heads)
            self.head_mask[top_indices] = 1
        elif self.config.head_selection == "threshold":
            self.head_mask = (importance > self.config.threshold).float()
        else:  # all
            self.head_mask = torch.ones(n_heads)
        
        self.is_prepared = True
        n_selected = int(self.head_mask.sum().item())
        
        logger.info(f"Selected {n_selected}/{n_heads} heads for modulation")
        
        return self.head_importance
    
    def _get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get attention module."""
        if self.model_type in ["llama", "mistral", "phi"]:
            return self.model.model.layers[layer_idx].self_attn
        elif self.model_type == "gpt2":
            return self.model.transformer.h[layer_idx].attn
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_n_heads(self) -> int:
        """Get number of attention heads."""
        if hasattr(self.model.config, 'num_attention_heads'):
            return self.model.config.num_attention_heads
        elif hasattr(self.model.config, 'n_head'):
            return self.model.config.n_head
        else:
            return 32  # Default
    
    def _collect_attention_patterns(
        self,
        prompts: List[str],
        attn_module: nn.Module,
        batch_size: int
    ) -> torch.Tensor:
        """Collect attention patterns."""
        patterns = []
        
        def hook_fn(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    patterns.append(attn_weights.detach().cpu())
        
        handle = attn_module.register_forward_hook(hook_fn, with_kwargs=True)
        
        try:
            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i:i + batch_size]
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    # Enable attention output
                    _ = self.model(**inputs, output_attentions=True)
        finally:
            handle.remove()
        
        if patterns:
            return torch.cat(patterns, dim=0)
        return torch.zeros(1, self._get_n_heads(), 1, 1)
    
    def prepare(
        self,
        layer_idx: Optional[int] = None,
        strength: Optional[float] = None
    ) -> None:
        """Prepare modulation by registering hooks."""
        if not self.is_prepared or self.head_mask is None:
            raise RuntimeError("Head importance not computed. Call compute_head_importance() first.")
        
        layer_idx = layer_idx or self.config.target_layer
        strength = strength or self.config.modulation_strength
        
        self.cleanup()
        
        attn_module = self._get_attention_module(layer_idx)
        hook = AttentionHeadHook(self.head_mask, strength)
        hook.register(attn_module)
        self.hooks.append(hook)
        
        logger.info(f"Belitsky modulation prepared at layer {layer_idx}")
    
    def cleanup(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate with modulation applied."""
        if not self.hooks:
            self.prepare()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
    
    def __enter__(self):
        self.prepare()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# Literature-reported performance
BELITSKY_REPORTED_METRICS = {
    'paper': "Belitsky et al. (2025)",
    'title': "Safety Alignment via Attention Head Modulation",
    'metrics': {
        'asr_reduction': 0.55,  # ~55% ASR reduction
        'utility_preserved': 0.90,  # ~90% utility
        'complexity': 'O(T)',
    },
    'key_finding': "Specific attention heads encode refusal behavior"
}
