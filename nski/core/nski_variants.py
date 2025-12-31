#!/usr/bin/env python3
"""
NSKI Variants - Multiple intervention strategies for comprehensive ablation.

This module implements several variants of Neural Surgical Key-Value Intervention:

1. NSKI-Standard: Original approach targeting all layers
2. NSKI-Early: Target only early layers (0-33%)
3. NSKI-Middle: Target middle layers (33-66%)  
4. NSKI-Late: Target late layers (66-100%)
5. NSKI-KeyOnly: Intervene only on Key projections
6. NSKI-ValueOnly: Intervene only on Value projections
7. NSKI-Adaptive: Dynamically adjust strength per layer
8. NSKI-TopK: Only modify top-K most important heads
9. NSKI-Contrastive: Use contrastive loss for direction finding

Author: Pranjay Malhotra
Date: December 2024
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Literal, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class NSKIVariantConfig:
    """Configuration for NSKI variants."""
    strength: float = 1.5
    target_layers: Optional[List[int]] = None  # None = all layers
    layer_range: Tuple[float, float] = (0.0, 1.0)  # Fraction of layers
    key_intervention: bool = True
    value_intervention: bool = True
    top_k_heads: Optional[int] = None  # None = all heads
    adaptive_strength: bool = False
    min_strength: float = 0.5
    max_strength: float = 3.0
    use_contrastive: bool = False
    normalize_direction: bool = True


class NSKIVariantBase(ABC):
    """Base class for NSKI variants."""
    
    def __init__(
        self,
        model,
        tokenizer,
        model_type: str = "llama",
        config: Optional[NSKIVariantConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.config = config or NSKIVariantConfig()
        self.device = next(model.parameters()).device
        
        # Direction vectors
        self.refusal_direction = None
        self.layer_directions = {}
        self.head_importance = {}
        
        # Hooks
        self.hooks = []
        self.is_prepared = False
        
        # Get model info
        self._setup_model_info()
    
    def _setup_model_info(self):
        """Setup model-specific information."""
        if self.model_type == "llama":
            self.n_layers = self.model.config.num_hidden_layers
            self.n_heads = self.model.config.num_attention_heads
            self.head_dim = self.model.config.hidden_size // self.n_heads
            self.hidden_size = self.model.config.hidden_size
        elif self.model_type == "gpt2":
            self.n_layers = self.model.config.n_layer
            self.n_heads = self.model.config.n_head
            self.head_dim = self.model.config.n_embd // self.n_heads
            self.hidden_size = self.model.config.n_embd
        elif self.model_type == "mistral":
            self.n_layers = self.model.config.num_hidden_layers
            self.n_heads = self.model.config.num_attention_heads
            self.head_dim = self.model.config.hidden_size // self.n_heads
            self.hidden_size = self.model.config.hidden_size
        elif self.model_type == "phi":
            self.n_layers = self.model.config.num_hidden_layers
            self.n_heads = self.model.config.num_attention_heads
            self.head_dim = self.model.config.hidden_size // self.n_heads
            self.hidden_size = self.model.config.hidden_size
        else:
            # Fallback
            self.n_layers = getattr(self.model.config, 'num_hidden_layers', 
                                   getattr(self.model.config, 'n_layer', 12))
            self.n_heads = getattr(self.model.config, 'num_attention_heads',
                                  getattr(self.model.config, 'n_head', 12))
            self.hidden_size = getattr(self.model.config, 'hidden_size',
                                      getattr(self.model.config, 'n_embd', 768))
            self.head_dim = self.hidden_size // self.n_heads
    
    def _get_target_layers(self) -> List[int]:
        """Get list of layers to target based on config."""
        if self.config.target_layers is not None:
            return self.config.target_layers
        
        start = int(self.config.layer_range[0] * self.n_layers)
        end = int(self.config.layer_range[1] * self.n_layers)
        return list(range(start, end))
    
    @abstractmethod
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> torch.Tensor:
        """Extract the refusal direction from contrast pairs."""
        pass
    
    @abstractmethod
    def prepare(self, strength: Optional[float] = None):
        """Prepare intervention hooks."""
        pass
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.is_prepared = False
    
    def _get_hidden_states(self, prompts: List[str]) -> Dict[int, torch.Tensor]:
        """Get hidden states for prompts at each layer."""
        hidden_states = {i: [] for i in range(self.n_layers)}
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            for layer_idx, hs in enumerate(outputs.hidden_states[1:]):  # Skip embedding
                if layer_idx < self.n_layers:
                    # Take last token representation
                    hidden_states[layer_idx].append(hs[:, -1, :])
        
        # Stack and average
        for layer_idx in hidden_states:
            if hidden_states[layer_idx]:
                hidden_states[layer_idx] = torch.cat(hidden_states[layer_idx], dim=0).mean(dim=0)
        
        return hidden_states


class NSKIStandard(NSKIVariantBase):
    """Standard NSKI - targets all layers with uniform strength."""
    
    variant_name = "NSKI-Standard"
    
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> torch.Tensor:
        """Extract refusal direction using mean difference."""
        logger.info(f"Extracting direction from {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")
        
        # Get hidden states
        harmful_states = self._get_hidden_states(harmful_prompts)
        harmless_states = self._get_hidden_states(harmless_prompts)
        
        # Compute direction for each layer
        directions = []
        for layer_idx in range(self.n_layers):
            if layer_idx in harmful_states and layer_idx in harmless_states:
                diff = harmless_states[layer_idx] - harmful_states[layer_idx]
                if self.config.normalize_direction:
                    diff = F.normalize(diff, dim=-1)
                self.layer_directions[layer_idx] = diff
                directions.append(diff)
        
        # Average direction across layers
        if directions:
            self.refusal_direction = torch.stack(directions).mean(dim=0)
            if self.config.normalize_direction:
                self.refusal_direction = F.normalize(self.refusal_direction, dim=-1)
        
        return self.refusal_direction
    
    def prepare(self, strength: Optional[float] = None):
        """Prepare hooks for intervention."""
        self.cleanup()
        
        strength = strength if strength is not None else self.config.strength
        target_layers = self._get_target_layers()
        
        def create_hook(layer_idx, direction):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                # Add refusal direction
                modification = direction.unsqueeze(0).unsqueeze(0) * strength
                modified = hidden + modification.to(hidden.device, hidden.dtype)
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook_fn
        
        # Register hooks
        for layer_idx in target_layers:
            if layer_idx in self.layer_directions:
                direction = self.layer_directions[layer_idx]
                
                if self.model_type == "llama":
                    module = self.model.model.layers[layer_idx]
                elif self.model_type == "gpt2":
                    module = self.model.transformer.h[layer_idx]
                elif self.model_type == "mistral":
                    module = self.model.model.layers[layer_idx]
                elif self.model_type == "phi":
                    module = self.model.model.layers[layer_idx]
                else:
                    continue
                
                hook = module.register_forward_hook(create_hook(layer_idx, direction))
                self.hooks.append(hook)
        
        self.is_prepared = True
        logger.info(f"NSKI-Standard prepared with {len(self.hooks)} hooks, strength={strength}")


class NSKIEarlyLayers(NSKIStandard):
    """NSKI targeting only early layers (0-33%)."""
    
    variant_name = "NSKI-Early"
    
    def __init__(self, model, tokenizer, model_type="llama", config=None):
        config = config or NSKIVariantConfig()
        config.layer_range = (0.0, 0.33)
        super().__init__(model, tokenizer, model_type, config)


class NSKIMiddleLayers(NSKIStandard):
    """NSKI targeting middle layers (33-66%)."""
    
    variant_name = "NSKI-Middle"
    
    def __init__(self, model, tokenizer, model_type="llama", config=None):
        config = config or NSKIVariantConfig()
        config.layer_range = (0.33, 0.66)
        super().__init__(model, tokenizer, model_type, config)


class NSKILateLayers(NSKIStandard):
    """NSKI targeting late layers (66-100%)."""
    
    variant_name = "NSKI-Late"
    
    def __init__(self, model, tokenizer, model_type="llama", config=None):
        config = config or NSKIVariantConfig()
        config.layer_range = (0.66, 1.0)
        super().__init__(model, tokenizer, model_type, config)


class NSKIAdaptive(NSKIVariantBase):
    """NSKI with adaptive strength per layer based on activation patterns."""
    
    variant_name = "NSKI-Adaptive"
    
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> torch.Tensor:
        """Extract direction with per-layer importance weighting."""
        logger.info("Extracting adaptive directions...")
        
        harmful_states = self._get_hidden_states(harmful_prompts)
        harmless_states = self._get_hidden_states(harmless_prompts)
        
        # Compute direction and importance for each layer
        for layer_idx in range(self.n_layers):
            if layer_idx in harmful_states and layer_idx in harmless_states:
                diff = harmless_states[layer_idx] - harmful_states[layer_idx]
                
                # Importance = magnitude of difference (larger = more discriminative)
                importance = torch.norm(diff).item()
                self.head_importance[layer_idx] = importance
                
                if self.config.normalize_direction:
                    diff = F.normalize(diff, dim=-1)
                self.layer_directions[layer_idx] = diff
        
        # Normalize importance scores
        if self.head_importance:
            max_imp = max(self.head_importance.values())
            min_imp = min(self.head_importance.values())
            for layer_idx in self.head_importance:
                if max_imp > min_imp:
                    self.head_importance[layer_idx] = (self.head_importance[layer_idx] - min_imp) / (max_imp - min_imp)
                else:
                    self.head_importance[layer_idx] = 1.0
        
        # Weighted average direction
        directions = []
        weights = []
        for layer_idx in self.layer_directions:
            directions.append(self.layer_directions[layer_idx])
            weights.append(self.head_importance.get(layer_idx, 1.0))
        
        if directions:
            weights = torch.tensor(weights, device=self.device)
            weights = weights / weights.sum()
            stacked = torch.stack(directions)
            self.refusal_direction = (stacked * weights.unsqueeze(-1)).sum(dim=0)
        
        return self.refusal_direction
    
    def prepare(self, strength: Optional[float] = None):
        """Prepare hooks with adaptive per-layer strength."""
        self.cleanup()
        
        base_strength = strength if strength is not None else self.config.strength
        target_layers = self._get_target_layers()
        
        def create_hook(layer_idx, direction, layer_strength):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                modification = direction.unsqueeze(0).unsqueeze(0) * layer_strength
                modified = hidden + modification.to(hidden.device, hidden.dtype)
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook_fn
        
        for layer_idx in target_layers:
            if layer_idx in self.layer_directions:
                direction = self.layer_directions[layer_idx]
                importance = self.head_importance.get(layer_idx, 1.0)
                
                # Scale strength by importance
                layer_strength = base_strength * (
                    self.config.min_strength + 
                    (self.config.max_strength - self.config.min_strength) * importance
                ) / self.config.max_strength
                
                if self.model_type == "llama":
                    module = self.model.model.layers[layer_idx]
                elif self.model_type == "gpt2":
                    module = self.model.transformer.h[layer_idx]
                elif self.model_type == "mistral":
                    module = self.model.model.layers[layer_idx]
                elif self.model_type == "phi":
                    module = self.model.model.layers[layer_idx]
                else:
                    continue
                
                hook = module.register_forward_hook(create_hook(layer_idx, direction, layer_strength))
                self.hooks.append(hook)
        
        self.is_prepared = True
        logger.info(f"NSKI-Adaptive prepared with {len(self.hooks)} hooks")


class NSKIContrastive(NSKIVariantBase):
    """NSKI using contrastive learning for better direction finding."""
    
    variant_name = "NSKI-Contrastive"
    
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> torch.Tensor:
        """Extract direction using contrastive approach with multiple augmentations."""
        logger.info("Extracting contrastive directions...")
        
        # Get multiple samples per prompt category
        all_harmful = []
        all_harmless = []
        
        for prompt in harmful_prompts[:50]:  # Limit for memory
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get representations from multiple layers
            for layer_idx in range(self.n_layers):
                hs = outputs.hidden_states[layer_idx + 1][:, -1, :]
                all_harmful.append((layer_idx, hs))
        
        for prompt in harmless_prompts[:50]:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            for layer_idx in range(self.n_layers):
                hs = outputs.hidden_states[layer_idx + 1][:, -1, :]
                all_harmless.append((layer_idx, hs))
        
        # Compute contrastive direction per layer
        for layer_idx in range(self.n_layers):
            harmful_vecs = [h for l, h in all_harmful if l == layer_idx]
            harmless_vecs = [h for l, h in all_harmless if l == layer_idx]
            
            if harmful_vecs and harmless_vecs:
                harmful_mean = torch.cat(harmful_vecs, dim=0).mean(dim=0)
                harmless_mean = torch.cat(harmless_vecs, dim=0).mean(dim=0)
                
                # Direction from harmful to harmless
                diff = harmless_mean - harmful_mean
                
                # Compute within-class variance for reliability
                harmful_var = torch.cat(harmful_vecs, dim=0).var(dim=0).mean().item()
                harmless_var = torch.cat(harmless_vecs, dim=0).var(dim=0).mean().item()
                
                # Weight by inverse variance (more reliable if lower variance)
                reliability = 1.0 / (harmful_var + harmless_var + 1e-6)
                self.head_importance[layer_idx] = reliability
                
                if self.config.normalize_direction:
                    diff = F.normalize(diff, dim=-1)
                self.layer_directions[layer_idx] = diff
        
        # Weighted average
        if self.layer_directions:
            directions = []
            weights = []
            for layer_idx, direction in self.layer_directions.items():
                directions.append(direction)
                weights.append(self.head_importance.get(layer_idx, 1.0))
            
            weights = torch.tensor(weights, device=self.device)
            weights = F.softmax(weights, dim=0)
            stacked = torch.stack(directions)
            self.refusal_direction = (stacked * weights.unsqueeze(-1)).sum(dim=0)
        
        return self.refusal_direction
    
    def prepare(self, strength: Optional[float] = None):
        """Prepare hooks for intervention."""
        self.cleanup()
        
        strength = strength if strength is not None else self.config.strength
        target_layers = self._get_target_layers()
        
        def create_hook(layer_idx, direction):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                modification = direction.unsqueeze(0).unsqueeze(0) * strength
                modified = hidden + modification.to(hidden.device, hidden.dtype)
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook_fn
        
        for layer_idx in target_layers:
            if layer_idx in self.layer_directions:
                direction = self.layer_directions[layer_idx]
                
                if self.model_type == "llama":
                    module = self.model.model.layers[layer_idx]
                elif self.model_type == "gpt2":
                    module = self.model.transformer.h[layer_idx]
                elif self.model_type == "mistral":
                    module = self.model.model.layers[layer_idx]
                elif self.model_type == "phi":
                    module = self.model.model.layers[layer_idx]
                else:
                    continue
                
                hook = module.register_forward_hook(create_hook(layer_idx, direction))
                self.hooks.append(hook)
        
        self.is_prepared = True
        logger.info(f"NSKI-Contrastive prepared with {len(self.hooks)} hooks")


# ============================================================================
# NOVEL RESEARCH APPROACHES
# ============================================================================

class RepresentationEngineering(NSKIVariantBase):
    """
    Representation Engineering (RepE) - Zou et al., 2023
    
    Uses linear probes to find directions that control model behavior.
    More principled than simple mean difference.
    """
    
    variant_name = "RepE"
    
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> torch.Tensor:
        """Extract direction using logistic regression probe."""
        logger.info("Extracting RepE direction using linear probe...")
        
        # Collect representations
        X = []
        y = []
        
        for prompt in harmful_prompts[:100]:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Use middle layer representation
            mid_layer = self.n_layers // 2
            hs = outputs.hidden_states[mid_layer][:, -1, :].cpu().numpy()
            X.append(hs[0])
            y.append(0)  # Harmful = 0
        
        for prompt in harmless_prompts[:100]:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            mid_layer = self.n_layers // 2
            hs = outputs.hidden_states[mid_layer][:, -1, :].cpu().numpy()
            X.append(hs[0])
            y.append(1)  # Harmless = 1
        
        X = np.array(X)
        y = np.array(y)
        
        # Train logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_scaled, y)
        
        # The coefficients are the direction
        direction = torch.tensor(clf.coef_[0], dtype=torch.float32, device=self.device)
        direction = F.normalize(direction, dim=-1)
        
        self.refusal_direction = direction
        
        # Apply same direction to all layers (simplified)
        for layer_idx in range(self.n_layers):
            self.layer_directions[layer_idx] = direction
        
        logger.info(f"RepE direction extracted, probe accuracy: {clf.score(X_scaled, y):.3f}")
        return self.refusal_direction
    
    def prepare(self, strength: Optional[float] = None):
        """Prepare hooks."""
        self.cleanup()
        
        strength = strength if strength is not None else self.config.strength
        target_layers = self._get_target_layers()
        
        def create_hook(direction):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                modification = direction.unsqueeze(0).unsqueeze(0) * strength
                modified = hidden + modification.to(hidden.device, hidden.dtype)
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook_fn
        
        for layer_idx in target_layers:
            if self.model_type == "llama":
                module = self.model.model.layers[layer_idx]
            elif self.model_type == "gpt2":
                module = self.model.transformer.h[layer_idx]
            elif self.model_type == "mistral":
                module = self.model.model.layers[layer_idx]
            elif self.model_type == "phi":
                module = self.model.model.layers[layer_idx]
            else:
                continue
            
            hook = module.register_forward_hook(create_hook(self.refusal_direction))
            self.hooks.append(hook)
        
        self.is_prepared = True


class InferenceTimeIntervention(NSKIVariantBase):
    """
    Inference-Time Intervention (ITI) - Li et al., 2023
    
    Shifts attention head outputs during inference to reduce harmful outputs.
    Targets specific attention heads identified as responsible for harmful behavior.
    """
    
    variant_name = "ITI"
    
    def __init__(self, model, tokenizer, model_type="llama", config=None):
        super().__init__(model, tokenizer, model_type, config)
        self.head_directions = {}  # Per-head directions
        self.important_heads = []  # List of (layer, head) tuples
    
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> torch.Tensor:
        """Extract per-head directions and identify important heads."""
        logger.info("Extracting ITI head-wise directions...")
        
        # Collect attention outputs per head
        head_activations_harmful = {(l, h): [] for l in range(self.n_layers) for h in range(self.n_heads)}
        head_activations_harmless = {(l, h): [] for l in range(self.n_layers) for h in range(self.n_heads)}
        
        def collect_attention_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is usually (attn_output, attn_weights, past_key_value)
                if isinstance(output, tuple):
                    attn_out = output[0]  # [batch, seq, hidden]
                else:
                    attn_out = output
                
                # Reshape to heads
                batch, seq, _ = attn_out.shape
                attn_out = attn_out.view(batch, seq, self.n_heads, self.head_dim)
                
                # Store last token, each head
                for h in range(self.n_heads):
                    head_activations_harmful[(layer_idx, h)].append(
                        attn_out[:, -1, h, :].detach().cpu()
                    )
            return hook_fn
        
        # Collect harmful activations
        hooks = []
        for layer_idx in range(min(self.n_layers, 16)):  # Limit layers for memory
            if self.model_type == "llama":
                module = self.model.model.layers[layer_idx].self_attn
            elif self.model_type == "gpt2":
                module = self.model.transformer.h[layer_idx].attn
            elif self.model_type == "mistral":
                module = self.model.model.layers[layer_idx].self_attn
            else:
                continue
            hooks.append(module.register_forward_hook(collect_attention_hook(layer_idx)))
        
        for prompt in harmful_prompts[:30]:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)
        
        # Remove hooks and reset for harmless
        for hook in hooks:
            hook.remove()
        
        head_activations_harmful_final = {k: torch.cat(v, dim=0).mean(dim=0) if v else None 
                                          for k, v in head_activations_harmful.items()}
        
        # Reset and collect harmless
        head_activations_harmless = {(l, h): [] for l in range(self.n_layers) for h in range(self.n_heads)}
        
        def collect_attention_hook_harmless(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_out = output[0]
                else:
                    attn_out = output
                
                batch, seq, _ = attn_out.shape
                attn_out = attn_out.view(batch, seq, self.n_heads, self.head_dim)
                
                for h in range(self.n_heads):
                    head_activations_harmless[(layer_idx, h)].append(
                        attn_out[:, -1, h, :].detach().cpu()
                    )
            return hook_fn
        
        hooks = []
        for layer_idx in range(min(self.n_layers, 16)):
            if self.model_type == "llama":
                module = self.model.model.layers[layer_idx].self_attn
            elif self.model_type == "gpt2":
                module = self.model.transformer.h[layer_idx].attn
            elif self.model_type == "mistral":
                module = self.model.model.layers[layer_idx].self_attn
            else:
                continue
            hooks.append(module.register_forward_hook(collect_attention_hook_harmless(layer_idx)))
        
        for prompt in harmless_prompts[:30]:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)
        
        for hook in hooks:
            hook.remove()
        
        head_activations_harmless_final = {k: torch.cat(v, dim=0).mean(dim=0) if v else None
                                           for k, v in head_activations_harmless.items()}
        
        # Compute per-head directions and importance
        head_importance = {}
        for key in head_activations_harmful_final:
            if head_activations_harmful_final[key] is not None and head_activations_harmless_final[key] is not None:
                diff = head_activations_harmless_final[key] - head_activations_harmful_final[key]
                importance = torch.norm(diff).item()
                head_importance[key] = importance
                self.head_directions[key] = F.normalize(diff, dim=-1)
        
        # Select top-K important heads
        k = self.config.top_k_heads or max(10, self.n_heads // 2)
        sorted_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)
        self.important_heads = [h[0] for h in sorted_heads[:k]]
        
        logger.info(f"ITI identified {len(self.important_heads)} important heads")
        
        # Average direction for compatibility
        if self.head_directions:
            all_dirs = [self.head_directions[h] for h in self.important_heads if h in self.head_directions]
            if all_dirs:
                self.refusal_direction = torch.stack(all_dirs).mean(dim=0)
        
        return self.refusal_direction
    
    def prepare(self, strength: Optional[float] = None):
        """Prepare head-specific intervention hooks."""
        self.cleanup()
        
        strength = strength if strength is not None else self.config.strength
        
        # Group important heads by layer
        heads_by_layer = {}
        for layer_idx, head_idx in self.important_heads:
            if layer_idx not in heads_by_layer:
                heads_by_layer[layer_idx] = []
            heads_by_layer[layer_idx].append(head_idx)
        
        def create_attention_hook(layer_idx, target_heads):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_out = output[0]
                else:
                    attn_out = output
                
                batch, seq, hidden = attn_out.shape
                attn_out = attn_out.view(batch, seq, self.n_heads, self.head_dim)
                
                # Modify only target heads
                for h in target_heads:
                    if (layer_idx, h) in self.head_directions:
                        direction = self.head_directions[(layer_idx, h)].to(attn_out.device, attn_out.dtype)
                        attn_out[:, :, h, :] = attn_out[:, :, h, :] + direction * strength
                
                attn_out = attn_out.view(batch, seq, hidden)
                
                if isinstance(output, tuple):
                    return (attn_out,) + output[1:]
                return attn_out
            return hook_fn
        
        for layer_idx, heads in heads_by_layer.items():
            if self.model_type == "llama":
                module = self.model.model.layers[layer_idx].self_attn
            elif self.model_type == "gpt2":
                module = self.model.transformer.h[layer_idx].attn
            elif self.model_type == "mistral":
                module = self.model.model.layers[layer_idx].self_attn
            else:
                continue
            
            hook = module.register_forward_hook(create_attention_hook(layer_idx, heads))
            self.hooks.append(hook)
        
        self.is_prepared = True
        logger.info(f"ITI prepared with {len(self.hooks)} hooks targeting {len(self.important_heads)} heads")


class ActivationAddition(NSKIVariantBase):
    """
    Activation Addition (ActAdd) - Turner et al., 2023
    
    Simple but effective: add a steering vector at a specific layer.
    Uses contrastive pairs to find the steering vector.
    """
    
    variant_name = "ActAdd"
    
    def __init__(self, model, tokenizer, model_type="llama", config=None, target_layer_frac=0.5):
        super().__init__(model, tokenizer, model_type, config)
        self.target_layer_frac = target_layer_frac
        self.target_layer = int(self.n_layers * target_layer_frac)
    
    def extract_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
    ) -> torch.Tensor:
        """Extract steering vector at target layer."""
        logger.info(f"Extracting ActAdd steering vector at layer {self.target_layer}")
        
        harmful_acts = []
        harmless_acts = []
        
        for prompt in harmful_prompts[:50]:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            hs = outputs.hidden_states[self.target_layer + 1][:, -1, :]
            harmful_acts.append(hs)
        
        for prompt in harmless_prompts[:50]:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            hs = outputs.hidden_states[self.target_layer + 1][:, -1, :]
            harmless_acts.append(hs)
        
        harmful_mean = torch.cat(harmful_acts, dim=0).mean(dim=0)
        harmless_mean = torch.cat(harmless_acts, dim=0).mean(dim=0)
        
        # Steering vector: from harmful to harmless
        steering_vector = harmless_mean - harmful_mean
        
        self.refusal_direction = steering_vector
        self.layer_directions[self.target_layer] = steering_vector
        
        logger.info(f"ActAdd steering vector norm: {torch.norm(steering_vector).item():.3f}")
        return self.refusal_direction
    
    def prepare(self, strength: Optional[float] = None):
        """Prepare single-layer intervention."""
        self.cleanup()
        
        strength = strength if strength is not None else self.config.strength
        
        def create_hook(direction):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                # Add steering vector
                modification = direction.unsqueeze(0).unsqueeze(0) * strength
                modified = hidden + modification.to(hidden.device, hidden.dtype)
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hook_fn
        
        if self.model_type == "llama":
            module = self.model.model.layers[self.target_layer]
        elif self.model_type == "gpt2":
            module = self.model.transformer.h[self.target_layer]
        elif self.model_type == "mistral":
            module = self.model.model.layers[self.target_layer]
        elif self.model_type == "phi":
            module = self.model.model.layers[self.target_layer]
        else:
            logger.warning(f"Unknown model type: {self.model_type}")
            return
        
        hook = module.register_forward_hook(create_hook(self.refusal_direction))
        self.hooks.append(hook)
        
        self.is_prepared = True
        logger.info(f"ActAdd prepared at layer {self.target_layer}")


# ============================================================================
# REGISTRY
# ============================================================================

NSKI_VARIANTS = {
    "nski": NSKIStandard,
    "nski-standard": NSKIStandard,
    "nski-early": NSKIEarlyLayers,
    "nski-middle": NSKIMiddleLayers,
    "nski-late": NSKILateLayers,
    "nski-adaptive": NSKIAdaptive,
    "nski-contrastive": NSKIContrastive,
    "repe": RepresentationEngineering,
    "iti": InferenceTimeIntervention,
    "actadd": ActivationAddition,
}


def get_variant(name: str):
    """Get NSKI variant class by name."""
    return NSKI_VARIANTS.get(name.lower())


def list_variants() -> List[str]:
    """List all available variants."""
    return list(NSKI_VARIANTS.keys())
