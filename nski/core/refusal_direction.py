"""
Refusal Direction Extraction

Computes the refusal direction by analyzing the difference in activations
between harmful and harmless prompts. This direction encodes the "refusal"
concept in the model's representation space.

Methods:
1. Mean Difference: d = mean(A_harmful) - mean(A_harmless)
2. PCA: First principal component of (A_harmful - A_harmless)
3. ICA: Independent component analysis for cleaner separation
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from loguru import logger


@dataclass
class DirectionConfig:
    """Configuration for refusal direction extraction."""
    layer_idx: int
    n_harmful_samples: int = 100
    n_harmless_samples: int = 100
    aggregation: str = "mean_diff"  # mean_diff, pca, ica
    extraction_point: str = "residual"  # residual, attention_output, mlp_output
    use_last_token: bool = True  # Use last token position or mean pool
    batch_size: int = 4


class ActivationCollector:
    """
    Collects activations from specific layers during forward passes.
    """
    
    def __init__(
        self,
        layer_idx: int,
        extraction_point: str = "residual"
    ):
        self.layer_idx = layer_idx
        self.extraction_point = extraction_point
        self.activations: List[torch.Tensor] = []
        self.handle: Optional[any] = None
    
    def _hook_fn(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Union[torch.Tensor, Tuple]
    ) -> None:
        """Collect activations from the layer."""
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        
        # Detach and move to CPU to save memory
        self.activations.append(activation.detach().cpu())
    
    def register(self, module: nn.Module) -> None:
        """Register the collection hook."""
        self.handle = module.register_forward_hook(self._hook_fn)
    
    def remove(self) -> None:
        """Remove the collection hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    
    def clear(self) -> None:
        """Clear collected activations."""
        self.activations = []
    
    def get_activations(self) -> torch.Tensor:
        """Get stacked activations."""
        if not self.activations:
            return None
        return torch.cat(self.activations, dim=0)


def get_layer_module(
    model: nn.Module,
    layer_idx: int,
    model_type: str,
    extraction_point: str = "residual"
) -> nn.Module:
    """
    Get the module for collecting activations.
    
    Args:
        model: The transformer model
        layer_idx: Index of the layer
        model_type: Type of model (llama, mistral, gpt2, phi)
        extraction_point: Where to extract (residual, attention_output, mlp_output)
        
    Returns:
        The target module
    """
    if model_type in ["llama", "mistral", "phi"]:
        # Llama/Mistral/Phi: model.model.layers[i]
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[layer_idx]
            if extraction_point == "attention_output":
                return layer.self_attn
            elif extraction_point == "mlp_output":
                return layer.mlp
            else:  # residual
                return layer
    
    elif model_type == "gpt2":
        # GPT-2: model.transformer.h[i]
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer = model.transformer.h[layer_idx]
            if extraction_point == "attention_output":
                return layer.attn
            elif extraction_point == "mlp_output":
                return layer.mlp
            else:  # residual
                return layer
    
    raise ValueError(f"Could not find layer {layer_idx} for model type {model_type}")


class RefusalDirectionExtractor:
    """
    Extracts the refusal direction from model activations.
    
    The refusal direction is computed by comparing activations when the model
    processes harmful vs harmless prompts. The direction captures the
    representation of "refusal" in the model's latent space.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_type: str = "llama",
        device: str = "cuda"
    ):
        """
        Initialize the extractor.
        
        Args:
            model: The transformer model
            tokenizer: The tokenizer
            model_type: Type of model (llama, mistral, gpt2, phi)
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def _get_activations(
        self,
        prompts: List[str],
        config: DirectionConfig
    ) -> torch.Tensor:
        """
        Collect activations for a set of prompts.
        
        Args:
            prompts: List of input prompts
            config: Extraction configuration
            
        Returns:
            Tensor of activations [n_prompts, hidden_dim]
        """
        layer_module = get_layer_module(
            self.model,
            config.layer_idx,
            self.model_type,
            config.extraction_point
        )
        
        collector = ActivationCollector(config.layer_idx, config.extraction_point)
        collector.register(layer_module)
        
        all_activations = []
        
        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(prompts), config.batch_size), desc="Collecting activations"):
                    batch_prompts = prompts[i:i + config.batch_size]
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Forward pass
                    collector.clear()
                    _ = self.model(**inputs)
                    
                    # Get activations
                    batch_acts = collector.activations[0]  # [batch, seq, hidden]
                    
                    if config.use_last_token:
                        # Get activation at last non-padding token
                        attention_mask = inputs['attention_mask']
                        last_token_idx = attention_mask.sum(dim=1) - 1
                        batch_size = batch_acts.shape[0]
                        acts = batch_acts[torch.arange(batch_size), last_token_idx]
                    else:
                        # Mean pool over sequence
                        attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                        acts = (batch_acts * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                    
                    all_activations.append(acts)
                    collector.clear()
        
        finally:
            collector.remove()
        
        return torch.cat(all_activations, dim=0)  # [n_prompts, hidden_dim]
    
    def _mean_diff_direction(
        self,
        harmful_acts: torch.Tensor,
        harmless_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute direction as mean difference.
        
        d = mean(A_harmful) - mean(A_harmless)
        """
        harmful_mean = harmful_acts.mean(dim=0)
        harmless_mean = harmless_acts.mean(dim=0)
        
        direction = harmful_mean - harmless_mean
        
        # Normalize to unit vector
        direction = direction / torch.norm(direction)
        
        return direction
    
    def _pca_direction(
        self,
        harmful_acts: torch.Tensor,
        harmless_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute direction as first principal component of difference.
        """
        # Stack all activations with labels
        all_acts = torch.cat([harmful_acts, harmless_acts], dim=0)
        labels = torch.cat([
            torch.ones(harmful_acts.shape[0]),
            torch.zeros(harmless_acts.shape[0])
        ])
        
        # Center the data
        mean = all_acts.mean(dim=0)
        centered = all_acts - mean
        
        # Compute covariance matrix
        cov = torch.mm(centered.T, centered) / (centered.shape[0] - 1)
        
        # Get eigenvectors (first principal component)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Direction is last eigenvector (largest eigenvalue)
        direction = eigenvectors[:, -1]
        
        # Ensure direction points from harmless to harmful
        harmful_mean = harmful_acts.mean(dim=0)
        harmless_mean = harmless_acts.mean(dim=0)
        if torch.dot(direction, harmful_mean - harmless_mean) < 0:
            direction = -direction
        
        return direction / torch.norm(direction)
    
    def extract(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        config: DirectionConfig
    ) -> Dict[str, torch.Tensor]:
        """
        Extract the refusal direction.
        
        Args:
            harmful_prompts: List of harmful prompts
            harmless_prompts: List of harmless prompts
            config: Extraction configuration
            
        Returns:
            Dictionary containing:
                - direction: The refusal direction vector
                - harmful_activations: Activations for harmful prompts
                - harmless_activations: Activations for harmless prompts
                - metadata: Additional information
        """
        logger.info(f"Extracting refusal direction from layer {config.layer_idx}")
        logger.info(f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")
        
        # Limit to configured number of samples
        harmful_prompts = harmful_prompts[:config.n_harmful_samples]
        harmless_prompts = harmless_prompts[:config.n_harmless_samples]
        
        # Collect activations
        logger.info("Collecting harmful activations...")
        harmful_acts = self._get_activations(harmful_prompts, config)
        
        logger.info("Collecting harmless activations...")
        harmless_acts = self._get_activations(harmless_prompts, config)
        
        # Compute direction based on aggregation method
        if config.aggregation == "mean_diff":
            direction = self._mean_diff_direction(harmful_acts, harmless_acts)
        elif config.aggregation == "pca":
            direction = self._pca_direction(harmful_acts, harmless_acts)
        else:
            # Default to mean diff
            direction = self._mean_diff_direction(harmful_acts, harmless_acts)
        
        # Compute separability metrics
        harmful_proj = torch.mm(harmful_acts, direction.unsqueeze(-1)).squeeze()
        harmless_proj = torch.mm(harmless_acts, direction.unsqueeze(-1)).squeeze()
        
        separability = (harmful_proj.mean() - harmless_proj.mean()) / (
            (harmful_proj.std() + harmless_proj.std()) / 2
        )
        
        logger.info(f"Direction extracted (separability: {separability.item():.3f})")
        
        return {
            'direction': direction.to(self.device),
            'harmful_activations': harmful_acts,
            'harmless_activations': harmless_acts,
            'metadata': {
                'layer_idx': config.layer_idx,
                'n_harmful': len(harmful_prompts),
                'n_harmless': len(harmless_prompts),
                'aggregation': config.aggregation,
                'separability': separability.item(),
                'harmful_proj_mean': harmful_proj.mean().item(),
                'harmless_proj_mean': harmless_proj.mean().item(),
                'hidden_dim': direction.shape[0]
            }
        }
    
    def extract_multi_layer(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layers: List[int],
        config: DirectionConfig
    ) -> Dict[int, Dict]:
        """
        Extract refusal directions from multiple layers.
        
        Useful for ablation studies.
        """
        results = {}
        
        for layer_idx in layers:
            layer_config = DirectionConfig(
                layer_idx=layer_idx,
                n_harmful_samples=config.n_harmful_samples,
                n_harmless_samples=config.n_harmless_samples,
                aggregation=config.aggregation,
                extraction_point=config.extraction_point,
                use_last_token=config.use_last_token,
                batch_size=config.batch_size
            )
            
            results[layer_idx] = self.extract(
                harmful_prompts,
                harmless_prompts,
                layer_config
            )
        
        return results


def save_direction(direction_result: Dict, path: str) -> None:
    """Save extracted direction to disk."""
    torch.save({
        'direction': direction_result['direction'].cpu(),
        'metadata': direction_result['metadata']
    }, path)
    logger.info(f"Direction saved to {path}")


def load_direction(path: str, device: str = "cuda") -> Dict:
    """Load direction from disk."""
    data = torch.load(path, map_location=device)
    logger.info(f"Direction loaded from {path}")
    return data
