"""
REAL KV-Cache Hooks for NSKI

This module implements actual forward hooks that modify the KV-cache
during inference. NO SIMULATION - this is the real intervention.

The hook intercepts the attention layer's key-value computations and
projects out the refusal direction from value representations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class HookConfig:
    """Configuration for KV-cache hooks."""
    layer_idx: int
    refusal_direction: torch.Tensor
    strength: float = 1.0
    strategy: str = "value_projection"  # value_projection, key_masking, combined
    enabled: bool = True


class KVCacheHook:
    """
    Real KV-cache intervention hook.
    
    This hook attaches to a transformer attention layer and modifies
    the value representations by projecting out the refusal direction.
    
    Mathematical operation:
        V' = V - strength * (V @ d) @ d^T
    
    where d is the unit refusal direction vector.
    """
    
    def __init__(
        self,
        config: HookConfig,
        model_type: str = "llama"
    ):
        """
        Initialize KV-cache hook.
        
        Args:
            config: Hook configuration
            model_type: Type of model (llama, mistral, gpt2, phi)
        """
        self.config = config
        self.model_type = model_type
        self.handle: Optional[Any] = None
        self.intervention_count = 0
        
        # Normalize refusal direction to unit vector
        self.direction = config.refusal_direction.clone()
        self.direction = self.direction / torch.norm(self.direction)
        
        logger.info(f"KVCacheHook initialized for layer {config.layer_idx}, "
                   f"strategy={config.strategy}, strength={config.strength}")
    
    def _project_out_direction(
        self, 
        values: torch.Tensor,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Project out refusal direction from value representations.
        
        V' = V - strength * (V · d) * d
        
        Args:
            values: Value tensor [batch, heads, seq, dim] or [batch, seq, dim]
            direction: Unit direction vector [dim]
            
        Returns:
            Modified value tensor with same shape
        """
        # Ensure direction is on same device and dtype
        direction = direction.to(values.device, values.dtype)
        
        # Handle different tensor shapes
        original_shape = values.shape
        
        if len(original_shape) == 4:
            # [batch, heads, seq, dim] - reshape for projection
            batch, heads, seq, dim = original_shape
            values_flat = values.view(-1, dim)
        elif len(original_shape) == 3:
            # [batch, seq, dim]
            batch, seq, dim = original_shape
            values_flat = values.view(-1, dim)
        else:
            # [seq, dim] or other
            values_flat = values.view(-1, values.shape[-1])
        
        # Compute projection: (V · d) * d
        # projection_coeff: [N, 1]
        projection_coeff = torch.matmul(values_flat, direction.unsqueeze(-1))
        
        # Subtract projection: V - strength * coeff * d
        projection = self.config.strength * projection_coeff * direction.unsqueeze(0)
        values_modified = values_flat - projection
        
        # Reshape back to original
        return values_modified.view(original_shape)
    
    def _hook_llama(
        self,
        module: nn.Module,
        args: Tuple,
        kwargs: Dict,
        output: Any
    ) -> Any:
        """Hook for Llama-style attention (LlamaAttention, MistralAttention)."""
        if not self.config.enabled:
            return output
        
        # Output is typically (attn_output, attn_weights, past_key_value)
        if isinstance(output, tuple) and len(output) >= 3:
            attn_output, attn_weights, past_kv = output
            
            if past_kv is not None:
                # past_kv is typically (key, value) or DynamicCache
                if hasattr(past_kv, 'key_cache') and hasattr(past_kv, 'value_cache'):
                    # DynamicCache object
                    if len(past_kv.value_cache) > self.config.layer_idx:
                        values = past_kv.value_cache[self.config.layer_idx]
                        modified_values = self._project_out_direction(values, self.direction)
                        past_kv.value_cache[self.config.layer_idx] = modified_values
                        self.intervention_count += 1
                elif isinstance(past_kv, tuple) and len(past_kv) == 2:
                    # (key, value) tuple
                    keys, values = past_kv
                    modified_values = self._project_out_direction(values, self.direction)
                    past_kv = (keys, modified_values)
                    self.intervention_count += 1
                    return (attn_output, attn_weights, past_kv)
        
        return output
    
    def _hook_gpt2(
        self,
        module: nn.Module,
        args: Tuple,
        kwargs: Dict,
        output: Any
    ) -> Any:
        """Hook for GPT-2 style attention."""
        if not self.config.enabled:
            return output
        
        # GPT-2 attention returns (attn_output, present)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_output = output[0]
            present = output[1] if len(output) > 1 else None
            
            if present is not None and isinstance(present, tuple):
                # present is (key, value)
                key, value = present
                modified_value = self._project_out_direction(value, self.direction)
                present = (key, modified_value)
                self.intervention_count += 1
                return (attn_output, present) + output[2:] if len(output) > 2 else (attn_output, present)
        
        return output
    
    def get_hook_fn(self) -> Callable:
        """Get the appropriate hook function for the model type."""
        if self.model_type in ["llama", "mistral", "phi"]:
            return self._hook_llama
        elif self.model_type in ["gpt2"]:
            return self._hook_gpt2
        else:
            # Default to Llama-style
            return self._hook_llama
    
    def register(self, module: nn.Module) -> None:
        """Register hook on a module."""
        self.handle = module.register_forward_hook(
            self.get_hook_fn(),
            with_kwargs=True
        )
        logger.debug(f"Registered hook on {module.__class__.__name__}")
    
    def remove(self) -> None:
        """Remove hook from module."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            logger.debug(f"Removed hook (interventions: {self.intervention_count})")
    
    def reset_count(self) -> None:
        """Reset intervention counter."""
        self.intervention_count = 0


class KVCacheHookV2:
    """
    Alternative KV-cache hook that intercepts at the value projection layer.
    
    This hooks into the v_proj (value projection) linear layer directly,
    which is more reliable across different model implementations.
    """
    
    def __init__(
        self,
        config: HookConfig,
        intervention_point: str = "output"  # "output" or "input"
    ):
        self.config = config
        self.intervention_point = intervention_point
        self.handle: Optional[Any] = None
        self.intervention_count = 0
        
        # Normalize direction
        self.direction = config.refusal_direction.clone()
        self.direction = self.direction / torch.norm(self.direction)
    
    def _hook_fn(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function for value projection output.
        
        Args:
            module: The v_proj linear layer
            input: Input to the layer
            output: Output from the layer (value representations)
            
        Returns:
            Modified output with refusal direction projected out
        """
        if not self.config.enabled:
            return output
        
        # Project out refusal direction from output
        direction = self.direction.to(output.device, output.dtype)
        
        # output shape: [batch, seq, hidden_dim] typically
        original_shape = output.shape
        output_flat = output.view(-1, output.shape[-1])
        
        # V' = V - strength * (V · d) * d
        projection_coeff = torch.matmul(output_flat, direction.unsqueeze(-1))
        projection = self.config.strength * projection_coeff * direction.unsqueeze(0)
        output_modified = output_flat - projection
        
        self.intervention_count += 1
        
        return output_modified.view(original_shape)
    
    def register(self, module: nn.Module) -> None:
        """Register hook on v_proj module."""
        self.handle = module.register_forward_hook(self._hook_fn)
        logger.debug(f"Registered V2 hook on {module.__class__.__name__}")
    
    def remove(self) -> None:
        """Remove hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_attention_layers(model: nn.Module, model_type: str) -> List[nn.Module]:
    """
    Get list of attention layers from a model.
    
    Args:
        model: The transformer model
        model_type: Type of model (llama, mistral, gpt2, phi)
        
    Returns:
        List of attention layer modules
    """
    attention_layers = []
    
    if model_type in ["llama", "mistral"]:
        # Llama/Mistral: model.model.layers[i].self_attn
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                if hasattr(layer, 'self_attn'):
                    attention_layers.append(layer.self_attn)
    
    elif model_type == "phi":
        # Phi: model.model.layers[i].self_attn
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                if hasattr(layer, 'self_attn'):
                    attention_layers.append(layer.self_attn)
    
    elif model_type == "gpt2":
        # GPT-2: model.transformer.h[i].attn
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            for block in model.transformer.h:
                if hasattr(block, 'attn'):
                    attention_layers.append(block.attn)
    
    if not attention_layers:
        # Fallback: search for attention modules
        for name, module in model.named_modules():
            if 'attn' in name.lower() and not any(x in name.lower() for x in ['norm', 'proj', 'drop']):
                attention_layers.append(module)
    
    logger.info(f"Found {len(attention_layers)} attention layers")
    return attention_layers


def get_value_proj_layers(model: nn.Module, model_type: str) -> List[nn.Module]:
    """
    Get list of value projection layers from a model.
    
    Args:
        model: The transformer model
        model_type: Type of model
        
    Returns:
        List of v_proj modules
    """
    v_proj_layers = []
    
    for name, module in model.named_modules():
        if 'v_proj' in name.lower() and isinstance(module, nn.Linear):
            v_proj_layers.append(module)
    
    logger.info(f"Found {len(v_proj_layers)} value projection layers")
    return v_proj_layers


def register_kv_hook(
    model: nn.Module,
    config: HookConfig,
    model_type: str = "llama",
    hook_type: str = "attention"  # "attention" or "v_proj"
) -> KVCacheHook:
    """
    Register a KV-cache hook on a specific layer.
    
    Args:
        model: The transformer model
        config: Hook configuration
        model_type: Type of model
        hook_type: Where to attach hook ("attention" or "v_proj")
        
    Returns:
        The registered hook object
    """
    if hook_type == "v_proj":
        layers = get_value_proj_layers(model, model_type)
        if config.layer_idx < len(layers):
            hook = KVCacheHookV2(config)
            hook.register(layers[config.layer_idx])
            return hook
    else:
        layers = get_attention_layers(model, model_type)
        if config.layer_idx < len(layers):
            hook = KVCacheHook(config, model_type)
            hook.register(layers[config.layer_idx])
            return hook
    
    raise ValueError(f"Layer index {config.layer_idx} out of range (max: {len(layers)-1})")


def remove_kv_hooks(hooks: List[KVCacheHook]) -> None:
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()
    logger.info(f"Removed {len(hooks)} hooks")


class KVCacheInterventionManager:
    """
    Manager for multiple KV-cache hooks across layers.
    
    Provides a high-level interface for applying NSKI interventions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str = "llama"
    ):
        self.model = model
        self.model_type = model_type
        self.hooks: List[KVCacheHook] = []
        self.active = False
    
    def add_intervention(
        self,
        layer_idx: int,
        refusal_direction: torch.Tensor,
        strength: float = 1.0,
        strategy: str = "value_projection",
        hook_type: str = "v_proj"
    ) -> None:
        """Add an intervention at a specific layer."""
        config = HookConfig(
            layer_idx=layer_idx,
            refusal_direction=refusal_direction,
            strength=strength,
            strategy=strategy,
            enabled=True
        )
        
        hook = register_kv_hook(
            self.model,
            config,
            self.model_type,
            hook_type
        )
        self.hooks.append(hook)
        self.active = True
    
    def enable(self) -> None:
        """Enable all interventions."""
        for hook in self.hooks:
            hook.config.enabled = True
        self.active = True
    
    def disable(self) -> None:
        """Disable all interventions (without removing hooks)."""
        for hook in self.hooks:
            hook.config.enabled = False
        self.active = False
    
    def remove_all(self) -> None:
        """Remove all hooks."""
        remove_kv_hooks(self.hooks)
        self.hooks = []
        self.active = False
    
    def get_intervention_count(self) -> int:
        """Get total number of interventions performed."""
        return sum(h.intervention_count for h in self.hooks)
    
    def reset_counts(self) -> None:
        """Reset all intervention counters."""
        for hook in self.hooks:
            hook.reset_count()
    
    def __enter__(self):
        """Context manager entry."""
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_all()
        return False
